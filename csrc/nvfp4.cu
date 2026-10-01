// NVFP4 quantization kernels for the FP4 GEMM paths and caches (reference: bitlattice/fp4ref.py).
//
// One configurable pipeline per 16-element block:
//   [rotation]  randomized Hadamard transform over 128-element chunks (8 lanes x 16 values, warp shuffles)
//   scale       block amax / 6 (or / 4: four over six, choosing the lower MSE or MAE of the round-to-nearest codes)
//   codes       round to nearest | stochastic rounding | round to nearest with MS-EDEN scale correction
//   MS-EDEN     S = <x, x> / <x, x_rtn> per 128 elements; every block scale times S, stochastically rounded to E4M3
// Layouts: rows of a [rows, cols] tensor along cols ("row"), or rows of its transpose ("col", via a shared-memory
// tile); the input may be bf16 / fp16 / fp32 or an NVFP4 tensor given by its codes and scales (re-quantization of
// a transposed weight). The global scale is either given or computed by a first pass over the (rotated) input.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <optional>
#include <type_traits>

#include "common.cuh"

namespace {

constexpr int kRTN = 0, kSR = 1, kEDEN = 2;

__device__ __forceinline__ float e2m1_level(uint32_t m) {           // magnitude of code m in 0..7
  return m < 4 ? 0.5f * static_cast<float>(m) : (m == 4 ? 2.0f : m == 5 ? 3.0f : m == 6 ? 4.0f : 6.0f);
}

struct Opts {
  int mode;          // kRTN | kSR | kEDEN
  bool fos;          // four over six
  bool mae;          // four-over-six selection by mean absolute error (else squared error)
  bool rotate;       // randomized Hadamard over 128-element chunks
  uint32_t seed;     // stochastic draws
  uint32_t rseed;    // rotation signs (both operands of a GEMM use the same)
};

__device__ __forceinline__ uint32_t code_rtn(float v) {
  const float a = fabsf(v);
  uint32_t m;
  if (v >= 0.0f) m = (a > 0.25f) + (a > 0.75f) + (a > 1.25f) + (a > 1.75f) + (a > 2.5f) + (a > 3.5f) + (a > 5.0f);
  else m = (a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) + (a >= 1.75f) + (a >= 2.5f) + (a >= 3.5f) + (a >= 5.0f);
  return (v < 0.0f && m ? 8u : 0u) | m;
}

__device__ __forceinline__ uint32_t code_sr(float v, float u) {
  const float a = fminf(fabsf(v), 6.0f);
  uint32_t m = (a >= 0.5f) + (a >= 1.0f) + (a >= 1.5f) + (a >= 2.0f) + (a >= 3.0f) + (a >= 4.0f) + (a >= 6.0f);
  if (m < 7) m += u * (e2m1_level(m + 1) - e2m1_level(m)) < a - e2m1_level(m);
  return (v < 0.0f && m ? 8u : 0u) | m;
}

__device__ __forceinline__ float level(uint32_t code) {
  const float m = e2m1_level(code & 7u);
  return code & 8u ? -m : m;
}

__device__ __forceinline__ float e4m3_round(float v) {
  return static_cast<float>(__nv_fp8_e4m3(fminf(fmaxf(v, 0.001953125f), 448.0f)));
}

// stochastic rounding of a positive value to E4M3 (clamped to its range)
__device__ __forceinline__ uint8_t e4m3_sr(float v, float u) {
  v = fminf(fmaxf(v, 0.001953125f), 448.0f);
  __nv_fp8_e4m3 r(v);
  uint32_t bits = r.__x;
  __nv_fp8_e4m3 t;
  t.__x = bits;
  float lo = static_cast<float>(t);
  if (lo > v) {
    bits = bits > 1 ? bits - 1 : 1;
    t.__x = bits;
    lo = static_cast<float>(t);
  }
  const uint32_t hb = bits < 0x7E ? bits + 1 : 0x7E;
  t.__x = hb;
  const float hi = static_cast<float>(t);
  const float p = hi > lo ? (v - lo) / (hi - lo) : 0.0f;
  return static_cast<uint8_t>(u < p ? hb : bits);
}

// in-register 16-point FWHT, then 3 butterfly stages across the 8 lanes holding one 128-element chunk; the lanes of
// a chunk are 8 consecutive lanes of a warp
__device__ __forceinline__ void hadamard128(float v[16], int lane8) {
#pragma unroll
  for (int h = 1; h < 16; h <<= 1) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if ((i & h) == 0) {
        const float a = v[i], b = v[i + h];
        v[i] = a + b;
        v[i + h] = a - b;
      }
    }
  }
#pragma unroll
  for (int m = 1; m < 8; m <<= 1) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float p = __shfl_xor_sync(0xffffffffu, v[i], m);
      v[i] = (lane8 & m) ? p - v[i] : v[i] + p;
    }
  }
  constexpr float inv = 0.08838834764831845f;           // 1/sqrt(128)
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] *= inv;
}

__device__ __forceinline__ void rotate_block(float v[16], int lane8, uint32_t seed) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float u = bl::hash_uniform(static_cast<uint64_t>(lane8 * 16 + i), seed, 0x5151u);
    v[i] = u < 0.5f ? -v[i] : v[i];
  }
  hadamard128(v, lane8);
}

__device__ __forceinline__ float lane8_sum(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 1);
  x += __shfl_xor_sync(0xffffffffu, x, 2);
  x += __shfl_xor_sync(0xffffffffu, x, 4);
  return x;
}

// quantize one block: returns packed codes, writes the scale byte. ``elem0`` / ``block`` index the stochastic draws.
__device__ __forceinline__ uint64_t quantize16(const float v[16], float g, const Opts& o, uint64_t elem0,
                                               uint64_t block, int lane8, uint8_t* scale_out) {
  // one division per block; elements are scaled by the reciprocal (the reference does the same)
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
  float s = e4m3_round(amax / 6.0f / g) * g;
  float inv = s > 0.0f ? 1.0f / s : 0.0f;
  uint32_t c[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) c[i] = code_rtn(v[i] * inv);
  if (o.fos) {
    const float s4 = e4m3_round(amax / 4.0f / g) * g;
    const float inv4 = s4 > 0.0f ? 1.0f / s4 : 0.0f;
    float e6 = 0.0f, e4 = 0.0f;
    uint32_t c4[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      c4[i] = code_rtn(v[i] * inv4);
      const float d6 = level(c[i]) * s - v[i], d4 = level(c4[i]) * s4 - v[i];
      e6 += o.mae ? fabsf(d6) : d6 * d6;
      e4 += o.mae ? fabsf(d4) : d4 * d4;
    }
    if (e4 < e6) {
      s = s4;
      inv = inv4;
#pragma unroll
      for (int i = 0; i < 16; ++i) c[i] = c4[i];
    }
  }
  uint8_t s8 = __nv_fp8_e4m3(s / g).__x;
  if (o.mode == kSR) {
#pragma unroll
    for (int i = 0; i < 16; ++i) c[i] = code_sr(v[i] * inv, bl::hash_uniform(elem0 + i, o.seed, 0u));
  } else if (o.mode == kEDEN) {
    float xx = 0.0f, xr = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      xx += v[i] * v[i];
      xr += v[i] * (level(c[i]) * s);
    }
    xx = lane8_sum(xx);
    xr = lane8_sum(xr);
    const float S = xr != 0.0f ? xx / xr : 1.0f;
    s8 = e4m3_sr(s / g * S, bl::hash_uniform(block, o.seed ^ 0x9E3779B9u, 0u));
  }
  uint64_t out = 0;
#pragma unroll
  for (int i = 0; i < 16; ++i) out |= static_cast<uint64_t>(c[i]) << (4 * i);
  *scale_out = s8;
  return out;
}

template <typename T> __device__ __forceinline__ float tof(T v);
template <> __device__ __forceinline__ float tof<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float tof<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float tof<float>(float v) { return v; }
template <> __device__ __forceinline__ float tof<uint8_t>(uint8_t) { return 0.0f; }   // NVFP4 input: col layout only

template <typename T>
__device__ __forceinline__ void load_row16(const T* p, float v[16]) {
  constexpr int per = 16 / sizeof(T);
  const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
  for (int j = 0; j < 16 / per; ++j) {
    const uint4 raw = q[j];
    const T* e = reinterpret_cast<const T*>(&raw);
#pragma unroll
    for (int i = 0; i < per; ++i) v[j * per + i] = tof<T>(e[i]);
  }
}

__device__ __forceinline__ void atomic_max_pos(float* addr, float v) {
  atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
}

// ---------------------------------------------------------------- row layout: blocks along the last dimension
// thread b handles block b; 8 consecutive threads hold one 128-element chunk (cols % 128 == 0 when rotating or EDEN)

template <typename T, bool kAmax>
__global__ void row_kernel(const T* x, int64_t nblocks, int64_t cols, Opts o, float* amax, const float* gptr,
                           uint8_t* packed, uint8_t* scales) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool live = b < nblocks;
  const int lane8 = static_cast<int>(b & 7);
  float v[16];
  if (live) load_row16(x + b * 16, v);
  else {
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = 0.0f;
  }
  if (o.rotate) rotate_block(v, lane8, o.rseed);
  if (kAmax) {
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) m = fmaxf(m, fabsf(v[i]));
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, s));
    if ((threadIdx.x & 31) == 0) atomic_max_pos(amax, m);
    return;
  }
  const float g = *gptr;
  uint8_t s8;
  const uint64_t out = quantize16(v, g, o, static_cast<uint64_t>(b) * 16, static_cast<uint64_t>(b), lane8, &s8);
  if (live) {
    reinterpret_cast<uint64_t*>(packed)[b] = out;
    scales[b] = s8;
  }
}

// ---------------------------------------------------------------- column layout: blocks along the first dimension
// A CTA of 256 threads takes a tile of 128 rows x 32 columns of the input (coalesced loads into shared memory);
// thread t then owns column t / 8 and the 16 rows (t % 8) * 16 .. + 15, so 8 consecutive lanes hold one 128-element
// chunk of a row of the transpose. Output row = input column.

struct Fp4Input {                                     // NVFP4 tensor [rows, cols]: codes along cols
  const uint8_t* packed;
  const uint8_t* scales;
  const float* g;
};

template <typename T>
__device__ __forceinline__ float load_elem(const T* x, int64_t cols, int64_t r, int64_t c, Fp4Input) {
  return tof<T>(x[r * cols + c]);
}

template <>
__device__ __forceinline__ float load_elem<uint8_t>(const uint8_t*, int64_t cols, int64_t r, int64_t c, Fp4Input w) {
  const uint32_t code = (w.packed[r * (cols / 2) + c / 2] >> (4 * (c & 1))) & 0xFu;
  __nv_fp8_e4m3 s8;
  s8.__x = w.scales[r * (cols / 16) + c / 16];
  return level(code) * static_cast<float>(s8) * (*w.g);
}

template <typename T, bool kAmax>
__global__ void col_kernel(const T* x, int64_t rows, int64_t cols, Fp4Input w, Opts o, float* amax,
                           const float* gptr, uint8_t* packed, uint8_t* scales) {
  __shared__ float tile[128][33];
  const int t = threadIdx.x;
  const int64_t r0 = static_cast<int64_t>(blockIdx.y) * 128, c0 = static_cast<int64_t>(blockIdx.x) * 32;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int idx = i * 256 + t, rr = idx >> 5, cc = idx & 31;
    const int64_t r = r0 + rr, c = c0 + cc;
    tile[rr][cc] = (r < rows && c < cols) ? load_elem<T>(x, cols, r, c, w) : 0.0f;
  }
  __syncthreads();
  const int col = t >> 3, lane8 = t & 7;
  float v[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = tile[lane8 * 16 + i][col];
  if (o.rotate) rotate_block(v, lane8, o.rseed);
  const int64_t oc = c0 + col;                          // output row
  if (kAmax) {
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) m = fmaxf(m, fabsf(v[i]));
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, s));
    if ((t & 31) == 0) atomic_max_pos(amax, m);
    return;
  }
  const float g = *gptr;
  const int64_t ob = r0 / 16 + lane8;                    // block index inside the output row
  const int64_t nb = rows / 16;
  uint8_t s8;
  const uint64_t out = quantize16(v, g, o, static_cast<uint64_t>(oc * rows + ob * 16),
                                  static_cast<uint64_t>(oc * nb + ob), lane8, &s8);
  if (oc < cols) {
    reinterpret_cast<uint64_t*>(packed + oc * (rows / 2))[ob] = out;
    scales[oc * nb + ob] = s8;
  }
}

}  // namespace

// x: [rows, cols] bf16 / fp16 / fp32, or uint8 NVFP4 codes [rows, cols/2] with ``w_scales`` / ``w_global`` (col
// layout only). transpose: quantize rows of x^T. mode: 0 round to nearest, 1 stochastic, 2 MS-EDEN.
// global_scale: given, or computed as amax / (6 * (four_over_six ? 256 : 448)) of the (rotated) input.
// rotate_seed: seed of the rotation signs (-1: ``seed``).
std::tuple<at::Tensor, at::Tensor, at::Tensor> bl_nvfp4_quantize(at::Tensor x, bool transpose, int64_t mode,
                                                                 bool four_over_six, bool mae, bool rotate,
                                                                 int64_t seed, std::optional<at::Tensor> global_scale,
                                                                 std::optional<at::Tensor> w_scales,
                                                                 std::optional<at::Tensor> w_global,
                                                                 int64_t rotate_seed) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2, "x must be a contiguous 2-D CUDA tensor");
  const c10::cuda::CUDAGuard guard(x.device());
  const bool fp4_in = x.scalar_type() == at::kByte;
  TORCH_CHECK(!fp4_in || (transpose && w_scales.has_value() && w_global.has_value()),
              "an NVFP4 input needs transpose and its scales");
  const int64_t rows = x.size(0), cols = fp4_in ? x.size(1) * 2 : x.size(1);
  const int64_t out_rows = transpose ? cols : rows, out_cols = transpose ? rows : cols;
  const int64_t chunk = (rotate || mode == kEDEN) ? 128 : 16;
  TORCH_CHECK(out_cols % chunk == 0, "the quantized dimension must be a multiple of ", chunk);
  Opts o{static_cast<int>(mode), four_over_six, mae, rotate, static_cast<uint32_t>(seed),
         static_cast<uint32_t>(rotate_seed < 0 ? seed : rotate_seed)};
  auto opt = x.options();
  at::Tensor packed = at::empty({out_rows, out_cols / 2}, opt.dtype(at::kByte));
  at::Tensor scales = at::empty({out_rows, out_cols / 16}, opt.dtype(at::kByte));
  at::Tensor amax = at::zeros({1}, opt.dtype(at::kFloat));
  at::Tensor global;
  const bool given = global_scale.has_value();
  auto stream = at::cuda::getCurrentCUDAStream();
  Fp4Input w{nullptr, nullptr, nullptr};
  at::Tensor wg;
  if (fp4_in) {
    wg = w_global->to(at::kFloat).contiguous();
    w = Fp4Input{x.data_ptr<uint8_t>(), reinterpret_cast<const uint8_t*>(w_scales->data_ptr()), wg.data_ptr<float>()};
  }
  const float top = 6.0f * (four_over_six ? 256.0f : 448.0f);
  auto finish_global = [&]() {
    global = given ? global_scale->to(at::kFloat).reshape({1}).contiguous() : (amax / top).clamp_min(1e-12);
  };
  auto launch = [&](auto* ptr) {
    using T = std::remove_const_t<std::remove_pointer_t<decltype(ptr)>>;
    if (!transpose) {
      const int64_t nb = rows * cols / 16;
      const int blocks = static_cast<int>((nb + 255) / 256);
      if (!given) row_kernel<T, true><<<blocks, 256, 0, stream>>>(ptr, nb, cols, o, amax.data_ptr<float>(), nullptr,
                                                                  nullptr, nullptr);
      finish_global();
      row_kernel<T, false><<<blocks, 256, 0, stream>>>(ptr, nb, cols, o, nullptr, global.data_ptr<float>(),
                                                       packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>());
    } else {
      TORCH_CHECK(rows % 128 == 0, "transposed quantization needs rows % 128 == 0 (pad first)");
      dim3 grid(static_cast<unsigned>((cols + 31) / 32), static_cast<unsigned>(rows / 128));
      if (!given) col_kernel<T, true><<<grid, 256, 0, stream>>>(ptr, rows, cols, w, o, amax.data_ptr<float>(),
                                                                nullptr, nullptr, nullptr);
      finish_global();
      col_kernel<T, false><<<grid, 256, 0, stream>>>(ptr, rows, cols, w, o, nullptr, global.data_ptr<float>(),
                                                     packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>());
    }
  };
  if (fp4_in) launch(x.data_ptr<uint8_t>());
  else if (x.scalar_type() == at::kBFloat16) launch(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()));
  else if (x.scalar_type() == at::kHalf) launch(reinterpret_cast<const __half*>(x.data_ptr<at::Half>()));
  else {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be bf16, fp16, fp32 or NVFP4 codes");
    launch(x.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales.view(at::kFloat8_e4m3fn), global};
}
