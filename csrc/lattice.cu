// Lattice kernels: fused optimizer step for quantized and native formats, dequantization.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <limits>

#include "common.cuh"

namespace bl {

struct StepParams {
  float lr;
  float decay;
  double lr_d;     // exact values for fp32 native weights (computed in double)
  double decay_d;
  int pb;      // phase bits
  int pcb;     // phase container bits
  int rule;    // ScaleRule
  float r_lo, r_hi;
  RoundParams rp;
};

// shared lookup tables
struct Tables {
  float sorted[kMaxLevels];
  int rank[kMaxLevels];
  int code[kMaxLevels];
};

__device__ __forceinline__ void load_tables(const Lattice& L, Tables& T) {
  if (L.kind != kTable) return;
  const int ncodes = 1 << L.bits;
  for (int i = threadIdx.x; i < kMaxLevels; i += blockDim.x) {
    T.sorted[i] = i < L.levels ? L.sorted[i] : 0.0f;
    T.code[i] = i < L.levels ? L.sorted_code[i] : 0;
    T.rank[i] = i < ncodes ? L.rank_of_code[i] : 0;
  }
}

__device__ __forceinline__ float old_scale(const Lattice& L, int64_t g) {
  const float gs = L.global_scale ? L.global_scale[0] : 1.0f;
  return decode_scale(L.scales_in, g, L.scale_dt) * gs;
}

__device__ __forceinline__ float group_zero(const Lattice& L, int64_t g) {
  return L.zeros ? __half2float(L.zeros[g]) : 0.0f;
}

// per-chunk quantities shared by the reduction and the projection
struct Chunk {
  float a[kEPT], lo[kEPT], hi[kEPT], tau[kEPT], tf[kEPT];
};

__device__ __forceinline__ void load_chunk(const Lattice& L, const Tables& T, const StepParams& P,
                                           const uint8_t* codes, const uint8_t* phase, const float* D, int64_t chunk,
                                           float s_old, float zero, Chunk& C) {
  uint32_t raw[kEPT];
  int ph[kEPT];
  load_codes(codes, chunk, L.cb, raw);
  load_phase(phase, chunk, P.pb, P.pcb, ph);
  const float4 d0 = reinterpret_cast<const float4*>(D)[chunk * 2];
  const float4 d1 = reinterpret_cast<const float4*>(D)[chunk * 2 + 1];
  const float dv[kEPT] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
  const int nr = num_ranks(L);
#pragma unroll
  for (int i = 0; i < kEPT; ++i) {
    const int r = raw_to_rank(L, T.rank, raw[i]);
    const float a = rank_level(L, T.sorted, r, zero);
    const float lo = rank_level(L, T.sorted, max(r - 1, 0), zero);
    const float hi = rank_level(L, T.sorted, min(r + 1, nr - 1), zero);
    const float d = dv[i] / s_old;
    const float lam = latent(a, lo, hi, ph[i], P.pb);
    C.a[i] = a;
    C.lo[i] = lo;
    C.hi[i] = hi;
    C.tau[i] = lam * P.decay - P.lr * d;
    C.tf[i] = P.rule == kFitLatent ? C.tau[i] : a * P.decay - P.lr * d;
  }
}

__device__ __forceinline__ void project_chunk(const Lattice& L, const Tables& T, const StepParams& P,
                                              uint8_t* codes, uint8_t* phase, int64_t chunk, float s_old, float s_new,
                                              float zero, const Chunk& C) {
  const int nr = num_ranks(L);
  const float tmin = rank_level(L, T.sorted, 0, zero);
  const float tmax = rank_level(L, T.sorted, nr - 1, zero);
  uint32_t raw[kEPT];
  int ph[kEPT];
#pragma unroll
  for (int i = 0; i < kEPT; ++i) {
    float t = C.tau[i] * s_old / s_new;
    t = fminf(fmaxf(t, tmin), tmax);
    int r = nearest_rank(L, T.sorted, t, zero);
    const float w = rank_level(L, T.sorted, r, zero);
    const float lo = rank_level(L, T.sorted, max(r - 1, 0), zero);
    const float hi = rank_level(L, T.sorted, min(r + 1, nr - 1), zero);
    const float u = rounding_u(P.rp, static_cast<uint64_t>(chunk) * kEPT + i);
    const int move = project<float>(t, w, lo, hi, u, P.pb, &ph[i]);
    r = min(max(r + move, 0), nr - 1);
    raw[i] = rank_to_raw(L, T.code, r);
  }
  store_codes(codes, chunk, L.cb, raw);
  store_phase(phase, chunk, P.pb, P.pcb, ph);
}

// scale decision from group sums; returns the new effective scale and writes storage bits on change
__device__ __forceinline__ float decide_scale(const Lattice& L, const StepParams& P, float s_old, float saa, float sat,
                                              float stt, int64_t g, bool writer) {
  float r = saa > 0.0f ? sat / saa : 1.0f;
  r = fminf(fmaxf(r, P.r_lo), P.r_hi);
  const float gs = L.global_scale ? L.global_scale[0] : 1.0f;
  uint32_t bits;
  const float cand = encode_scale(s_old * r / gs, L.scale_dt, &bits) * gs;
  const float rr = cand / s_old;
  const float before = stt - 2.0f * sat + saa;
  const float after = rr * rr * saa - 2.0f * rr * sat + stt;
  const bool ok = (after <= before) && isfinite(cand) && cand > 0.0f;
  if (ok && writer) store_scale(L.scales_out, g, L.scale_dt, bits);
  return ok ? cand : s_old;
}

// group <= 256 elements (power of two, multiple of kEPT): one warp segment per group, butterfly reduction
__global__ void step_small_groups(Lattice L, StepParams P, uint8_t* codes, uint8_t* phase, const float* D,
                                  int64_t nchunks) {
  __shared__ Tables T;
  load_tables(L, T);
  __syncthreads();
  const int64_t chunk = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool active = chunk < nchunks;
  const int tpg = static_cast<int>(L.group / kEPT);
  const int64_t g = active ? (chunk * kEPT) / L.group : 0;
  const float s_old = active ? old_scale(L, g) : 1.0f;
  const float zero = active ? group_zero(L, g) : 0.0f;
  Chunk C;
  float saa = 0.0f, sat = 0.0f, stt = 0.0f;
  if (active) {
    load_chunk(L, T, P, codes, phase, D, chunk, s_old, zero, C);
    if (P.rule != kFixed) {
#pragma unroll
      for (int i = 0; i < kEPT; ++i) {
        saa += C.a[i] * C.a[i];
        sat += C.a[i] * C.tf[i];
        stt += C.tf[i] * C.tf[i];
      }
    }
  }
  float s_new = s_old;
  if (P.rule != kFixed) {
    for (int off = tpg >> 1; off > 0; off >>= 1) {
      saa += __shfl_xor_sync(0xffffffffu, saa, off);
      sat += __shfl_xor_sync(0xffffffffu, sat, off);
      stt += __shfl_xor_sync(0xffffffffu, stt, off);
    }
    if (active) s_new = decide_scale(L, P, s_old, saa, sat, stt, g, (threadIdx.x % tpg) == 0);
  }
  if (active) project_chunk(L, T, P, codes, phase, chunk, s_old, s_new, zero, C);
}

// fixed scales: pure element-wise, any group layout
__global__ void step_fixed(Lattice L, StepParams P, uint8_t* codes, uint8_t* phase, const float* D, int64_t nchunks) {
  __shared__ Tables T;
  load_tables(L, T);
  __syncthreads();
  for (int64_t chunk = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; chunk < nchunks;
       chunk += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t g = group_of(L, chunk * kEPT);
    const float s = old_scale(L, g);
    const float zero = group_zero(L, g);
    Chunk C;
    load_chunk(L, T, P, codes, phase, D, chunk, s, zero, C);
    project_chunk(L, T, P, codes, phase, chunk, s, s, zero, C);
  }
}

// one block per group (per-row, per-tensor or large groups): deterministic block reduction then projection
__global__ void step_segments(Lattice L, StepParams P, uint8_t* codes, uint8_t* phase, const float* D,
                              int64_t group_elems) {
  __shared__ Tables T;
  __shared__ float red[3][32];
  __shared__ float s_new_sh;
  load_tables(L, T);
  __syncthreads();
  const int64_t g = blockIdx.x;
  const int64_t c0 = g * (group_elems / kEPT), c1 = c0 + group_elems / kEPT;
  const float s_old = old_scale(L, g);
  const float zero = group_zero(L, g);
  float saa = 0.0f, sat = 0.0f, stt = 0.0f;
  for (int64_t chunk = c0 + threadIdx.x; chunk < c1; chunk += blockDim.x) {
    Chunk C;
    load_chunk(L, T, P, codes, phase, D, chunk, s_old, zero, C);
#pragma unroll
    for (int i = 0; i < kEPT; ++i) {
      saa += C.a[i] * C.a[i];
      sat += C.a[i] * C.tf[i];
      stt += C.tf[i] * C.tf[i];
    }
  }
  for (int off = 16; off > 0; off >>= 1) {
    saa += __shfl_xor_sync(0xffffffffu, saa, off);
    sat += __shfl_xor_sync(0xffffffffu, sat, off);
    stt += __shfl_xor_sync(0xffffffffu, stt, off);
  }
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = blockDim.x >> 5;
  if (lane == 0) {
    red[0][warp] = saa;
    red[1][warp] = sat;
    red[2][warp] = stt;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float a = 0.0f, b = 0.0f, c = 0.0f;
    for (int w = 0; w < nwarps; ++w) {
      a += red[0][w];
      b += red[1][w];
      c += red[2][w];
    }
    s_new_sh = decide_scale(L, P, s_old, a, b, c, g, true);
  }
  __syncthreads();
  const float s_new = s_new_sh;
  for (int64_t chunk = c0 + threadIdx.x; chunk < c1; chunk += blockDim.x) {
    Chunk C;
    load_chunk(L, T, P, codes, phase, D, chunk, s_old, zero, C);
    project_chunk(L, T, P, codes, phase, chunk, s_old, s_new, zero, C);
  }
}

// native floating formats

template <typename W> struct NativeTraits;
template <> struct NativeTraits<__nv_bfloat16> {
  using U = uint16_t;
  using C = float;
  static constexpr int kBits = 16;
  __device__ static C to_c(__nv_bfloat16 v) { return __bfloat162float(v); }
  __device__ static __nv_bfloat16 from_c(C v) { return __float2bfloat16_rn(v); }
  __device__ static C max_finite() { return 3.3895313892515355e38f; }
};
template <> struct NativeTraits<__half> {
  using U = uint16_t;
  using C = float;
  static constexpr int kBits = 16;
  __device__ static C to_c(__half v) { return __half2float(v); }
  __device__ static __half from_c(C v) { return __float2half_rn(v); }
  __device__ static C max_finite() { return 65504.0f; }
};
template <> struct NativeTraits<float> {
  using U = uint32_t;
  using C = double;
  static constexpr int kBits = 32;
  __device__ static C to_c(float v) { return static_cast<double>(v); }
  __device__ static float from_c(C v) { return __double2float_rn(v); }
  __device__ static C max_finite() { return 3.4028234663852886e38; }
};

template <typename W>
__device__ __forceinline__ void native_neighbors(W w, W* lo, W* hi) {
  using Tr = NativeTraits<W>;
  using U = typename Tr::U;
  constexpr int n = Tr::kBits;
  const U u = *reinterpret_cast<const U*>(&w);
  const int64_t mag = static_cast<int64_t>(u & ((U(1) << (n - 1)) - 1));
  const int64_t o = (u >> (n - 1)) ? -mag : mag;
  auto from = [](int64_t v) -> W {
    U bits = v >= 0 ? static_cast<U>(v) : static_cast<U>((U(1) << (n - 1)) | static_cast<U>(-v));
    return *reinterpret_cast<W*>(&bits);
  };
  W a = from(o - 1), b = from(o + 1);
  *lo = isfinite(static_cast<float>(Tr::to_c(a))) ? a : w;
  *hi = isfinite(static_cast<float>(Tr::to_c(b))) ? b : w;
}

template <typename W>
__global__ void step_native_kernel(W* w, uint8_t* phase, const float* D, int64_t n, StepParams P) {
  using Tr = NativeTraits<W>;
  using C = typename Tr::C;
  const int64_t nchunks = (n + kEPT - 1) / kEPT;
  for (int64_t chunk = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; chunk < nchunks;
       chunk += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int ph[kEPT];
    load_phase(phase, chunk, P.pb, P.pcb, ph);
    int out_ph[kEPT];
#pragma unroll
    for (int i = 0; i < kEPT; ++i) {
      out_ph[i] = ph[i];
      const int64_t e = chunk * kEPT + i;
      if (e >= n) continue;
      const W wv = w[e];
      W lo, hi;
      native_neighbors(wv, &lo, &hi);
      const C wc = Tr::to_c(wv), loc = Tr::to_c(lo), hic = Tr::to_c(hi);
      C lam = wc;
      if (P.pb > 0) {
        const C c = static_cast<C>(static_cast<float>(ph[i]) / static_cast<float>(1 << (P.pb - 1)));
        lam = c < C(0) ? wc + c * (wc - loc) * C(0.5) : wc + c * (hic - wc) * C(0.5);
      }
      C t = lam * static_cast<C>(P.decay_d) - static_cast<C>(P.lr_d) * static_cast<C>(D[e]);
      const C big = Tr::max_finite();
      t = t < -big ? -big : (t > big ? big : t);
      const W w2 = Tr::from_c(t);
      W lo2, hi2;
      native_neighbors(w2, &lo2, &hi2);
      const C u = static_cast<C>(rounding_u(P.rp, static_cast<uint64_t>(e)));
      int code;
      const int move = project<C>(t, Tr::to_c(w2), Tr::to_c(lo2), Tr::to_c(hi2), u, P.pb, &code);
      w[e] = move > 0 ? hi2 : (move < 0 ? lo2 : w2);
      out_ph[i] = code;
    }
    store_phase(phase, chunk, P.pb, P.pcb, out_ph);
  }
}

// dequantization

template <typename O>
__device__ __forceinline__ O cast_out(float v);
template <> __device__ __forceinline__ float cast_out<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 cast_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ __half cast_out<__half>(float v) { return __float2half_rn(v); }

template <typename O>
__global__ void dequantize_kernel(Lattice L, const uint8_t* codes, O* out, int64_t nchunks) {
  __shared__ Tables T;
  load_tables(L, T);
  __syncthreads();
  for (int64_t chunk = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; chunk < nchunks;
       chunk += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t g = group_of(L, chunk * kEPT);
    const float s = old_scale(L, g);
    const float zero = group_zero(L, g);
    uint32_t raw[kEPT];
    load_codes(codes, chunk, L.cb, raw);
#pragma unroll
    for (int i = 0; i < kEPT; ++i)
      out[chunk * kEPT + i] = cast_out<O>(rank_level(L, T.sorted, raw_to_rank(L, T.rank, raw[i]), zero) * s);
  }
}

}  // namespace bl

// host side

namespace {

using at::Tensor;

bl::Lattice make_lattice(const Tensor& scales_in, const Tensor& scales_out, const std::optional<Tensor>& global_scale,
                         const std::optional<Tensor>& zeros, const Tensor& sorted, const Tensor& rank_of_code,
                         const Tensor& sorted_code, int64_t kind, int64_t bits, int64_t cb, int64_t qmin, int64_t qmax,
                         int64_t zero_point, int64_t scale_dt, int64_t group, int64_t rows, int64_t cols) {
  bl::Lattice L{};
  L.kind = static_cast<int>(kind);
  L.bits = static_cast<int>(bits);
  L.cb = static_cast<int>(cb);
  L.qmin = static_cast<int>(qmin);
  L.qmax = static_cast<int>(qmax);
  L.zero_point = static_cast<int>(zero_point);
  L.levels = static_cast<int>(sorted.numel());
  L.scale_dt = static_cast<int>(scale_dt);
  L.group = group;
  L.rows = rows;
  L.cols = cols;
  L.sorted = sorted.data_ptr<float>();
  L.rank_of_code = rank_of_code.data_ptr<int>();
  L.sorted_code = sorted_code.data_ptr<int>();
  L.scales_in = scales_in.data_ptr();
  L.scales_out = scales_out.data_ptr();
  L.global_scale = global_scale.has_value() ? global_scale->data_ptr<float>() : nullptr;
  L.zeros = zeros.has_value() ? reinterpret_cast<const __half*>(zeros->data_ptr<at::Half>()) : nullptr;
  TORCH_CHECK(L.kind != bl::kTable || L.levels <= bl::kMaxLevels, "table formats hold at most 256 levels");
  TORCH_CHECK(cols % bl::kEPT == 0, "last dimension must be a multiple of 8");
  if (group > 0) TORCH_CHECK(group % bl::kEPT == 0 && cols % group == 0, "group must divide cols and be a multiple of 8");
  return L;
}

bl::StepParams make_params(double lr, double decay, int64_t pb, int64_t pcb, int64_t rule, double kappa,
                           int64_t rounding, int64_t seed, int64_t step, int64_t strat_j, int64_t strat_k,
                           int64_t offset) {
  bl::StepParams P{};
  P.lr = static_cast<float>(lr);
  P.decay = static_cast<float>(decay);
  P.lr_d = lr;
  P.decay_d = decay;
  P.pb = static_cast<int>(pb);
  P.pcb = static_cast<int>(pcb);
  P.rule = static_cast<int>(rule);
  P.r_lo = static_cast<float>(std::exp(-kappa));
  P.r_hi = static_cast<float>(std::exp(kappa));
  P.rp.mode = static_cast<int>(rounding);
  P.rp.seed = static_cast<uint32_t>(seed & 0xFFFFFFFFll);
  P.rp.step = static_cast<uint32_t>(step & 0xFFFFFFFFll);
  P.rp.strat_j = static_cast<int>(strat_j);
  P.rp.strat_k = static_cast<int>(strat_k);
  P.rp.offset = static_cast<uint64_t>(offset);
  return P;
}

int grid_for(int64_t work, int threads) {
  const int64_t blocks = (work + threads - 1) / threads;
  return static_cast<int>(std::min<int64_t>(std::max<int64_t>(blocks, 1), 65535 * 8));
}

}  // namespace

Tensor bl_step_quantized(Tensor packed, std::optional<Tensor> phase, Tensor scales, std::optional<Tensor> global_scale,
                         std::optional<Tensor> zeros, Tensor D, Tensor sorted, Tensor rank_of_code, Tensor sorted_code,
                         int64_t kind, int64_t bits, int64_t cb, int64_t qmin, int64_t qmax, int64_t zero_point,
                         int64_t scale_dt, int64_t group, int64_t rows, int64_t cols, int64_t pb, int64_t pcb,
                         double lr, double decay, int64_t rounding, int64_t seed, int64_t step, int64_t strat_j,
                         int64_t strat_k, int64_t rule, double kappa, int64_t offset) {
  TORCH_CHECK(packed.is_cuda() && packed.is_contiguous(), "packed codes must be contiguous CUDA");
  TORCH_CHECK(D.scalar_type() == at::kFloat && D.is_contiguous() && D.numel() == rows * cols, "D: float32 [rows*cols]");
  const c10::cuda::CUDAGuard guard(packed.device());
  Tensor scales_out = scales.clone();
  bl::Lattice L = make_lattice(scales, scales_out, global_scale, zeros, sorted, rank_of_code, sorted_code, kind, bits,
                               cb, qmin, qmax, zero_point, scale_dt, group, rows, cols);
  bl::StepParams P = make_params(lr, decay, pb, pcb, rule, kappa, rounding, seed, step, strat_j, strat_k, offset);
  uint8_t* codes = static_cast<uint8_t*>(packed.data_ptr());
  uint8_t* ph = phase.has_value() && pb > 0 ? static_cast<uint8_t*>(phase->data_ptr()) : nullptr;
  const float* d = D.data_ptr<float>();
  const int64_t nchunks = rows * cols / bl::kEPT;
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int threads = 256;
  const bool pow2 = group > 0 && (group & (group - 1)) == 0;
  if (rule == bl::kFixed) {
    bl::step_fixed<<<grid_for(nchunks, threads), threads, 0, stream>>>(L, P, codes, ph, d, nchunks);
  } else if (pow2 && group <= 256) {
    const int blocks = static_cast<int>((nchunks + threads - 1) / threads);
    bl::step_small_groups<<<blocks, threads, 0, stream>>>(L, P, codes, ph, d, nchunks);
  } else {
    const int64_t gelems = group > 0 ? group : (group == -1 ? cols : rows * cols);
    const int64_t ngroups = rows * cols / gelems;
    bl::step_segments<<<static_cast<int>(ngroups), threads, 0, stream>>>(L, P, codes, ph, d, gelems);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scales_out;
}

void bl_step_native(Tensor w, std::optional<Tensor> phase, Tensor D, int64_t pb, int64_t pcb, double lr, double decay,
                    int64_t rounding, int64_t seed, int64_t step, int64_t strat_j, int64_t strat_k, int64_t offset) {
  TORCH_CHECK(w.is_cuda() && w.is_contiguous(), "weights must be contiguous CUDA");
  TORCH_CHECK(D.scalar_type() == at::kFloat && D.is_contiguous() && D.numel() == w.numel(), "D: float32, same numel");
  const c10::cuda::CUDAGuard guard(w.device());
  bl::StepParams P = make_params(lr, decay, pb, pcb, bl::kFixed, 0.0, rounding, seed, step, strat_j, strat_k, offset);
  uint8_t* ph = phase.has_value() && pb > 0 ? static_cast<uint8_t*>(phase->data_ptr()) : nullptr;
  const int64_t n = w.numel();
  constexpr int threads = 256;
  const int blocks = grid_for((n + bl::kEPT - 1) / bl::kEPT, threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (w.scalar_type() == at::kBFloat16) {
    bl::step_native_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(w.data_ptr<at::BFloat16>()), ph, D.data_ptr<float>(), n, P);
  } else if (w.scalar_type() == at::kHalf) {
    bl::step_native_kernel<__half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__half*>(w.data_ptr<at::Half>()), ph, D.data_ptr<float>(), n, P);
  } else if (w.scalar_type() == at::kFloat) {
    bl::step_native_kernel<float><<<blocks, threads, 0, stream>>>(w.data_ptr<float>(), ph, D.data_ptr<float>(), n, P);
  } else {
    TORCH_CHECK(false, "native step supports bf16, fp16, fp32");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

Tensor bl_dequantize(Tensor packed, Tensor scales, std::optional<Tensor> global_scale, std::optional<Tensor> zeros,
                     Tensor sorted, Tensor rank_of_code, Tensor sorted_code, int64_t kind, int64_t bits, int64_t cb,
                     int64_t qmin, int64_t qmax, int64_t zero_point, int64_t scale_dt, int64_t group, int64_t rows,
                     int64_t cols, int64_t out_dtype) {
  TORCH_CHECK(packed.is_cuda() && packed.is_contiguous(), "packed codes must be contiguous CUDA");
  const c10::cuda::CUDAGuard guard(packed.device());
  bl::Lattice L = make_lattice(scales, scales, global_scale, zeros, sorted, rank_of_code, sorted_code, kind, bits, cb,
                               qmin, qmax, zero_point, scale_dt, group, rows, cols);
  const auto dtype = out_dtype == 0 ? at::kFloat : (out_dtype == 1 ? at::kBFloat16 : at::kHalf);
  Tensor out = at::empty({rows, cols}, packed.options().dtype(dtype));
  const int64_t nchunks = rows * cols / bl::kEPT;
  constexpr int threads = 256;
  const int blocks = grid_for(nchunks, threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  const uint8_t* codes = static_cast<const uint8_t*>(packed.data_ptr());
  if (dtype == at::kFloat) {
    bl::dequantize_kernel<float><<<blocks, threads, 0, stream>>>(L, codes, out.data_ptr<float>(), nchunks);
  } else if (dtype == at::kBFloat16) {
    bl::dequantize_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        L, codes, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), nchunks);
  } else {
    bl::dequantize_kernel<__half><<<blocks, threads, 0, stream>>>(
        L, codes, reinterpret_cast<__half*>(out.data_ptr<at::Half>()), nchunks);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
