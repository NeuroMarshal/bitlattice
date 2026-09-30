// Activation quantization to NVFP4: E2M1 codes (low nibble first), one E4M3 scale per 16 elements along the last
// dimension, relative to a per-tensor fp32 global scale (value = code * scale * global).
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <optional>
#include <type_traits>

#include "common.cuh"

namespace {

__device__ __forceinline__ uint32_t e2m1_code(float v) {
  // nearest E2M1 level {0, .5, 1, 1.5, 2, 3, 4, 6}; ties go to the lower value (library-wide rule), i.e. to the
  // smaller magnitude for positive and to the larger magnitude for negative inputs
  const float a = fabsf(v);
  uint32_t m;
  if (v >= 0.0f) {
    m = (a > 0.25f) + (a > 0.75f) + (a > 1.25f) + (a > 1.75f) + (a > 2.5f) + (a > 3.5f) + (a > 5.0f);
  } else {
    m = (a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) + (a >= 1.75f) + (a >= 2.5f) + (a >= 3.5f) + (a >= 5.0f);
  }
  return (v < 0.0f && m ? 8u : 0u) | m;
}

// stochastic rounding between the two E2M1 neighbours of v (|v| <= 6): E[decode(code)] = v
__device__ __forceinline__ uint32_t e2m1_code_sr(float v, float u) {
  constexpr float L[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float a = fminf(fabsf(v), 6.0f);
  uint32_t m = (a >= 0.5f) + (a >= 1.0f) + (a >= 1.5f) + (a >= 2.0f) + (a >= 3.0f) + (a >= 4.0f) + (a >= 6.0f);
  if (m < 7) {
    const float lo = L[m], hi = L[m + 1];
    m += u * (hi - lo) < a - lo;
  }
  return (v < 0.0f && m ? 8u : 0u) | m;
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }

template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }

// 16 consecutive elements with 16-byte vector loads (the block start is 32/64-byte aligned for 2/4-byte types)
template <typename T>
__device__ __forceinline__ void load16(const T* p, float v[16]) {
  constexpr int per = 16 / sizeof(T);               // elements per 16-byte load
  const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
  for (int j = 0; j < 16 / per; ++j) {
    const uint4 raw = q[j];
    const T* e = reinterpret_cast<const T*>(&raw);
#pragma unroll
    for (int i = 0; i < per; ++i) v[j * per + i] = to_f<T>(e[i]);
  }
}

// quantize 16 values against global scale g: returns the packed codes, writes the E4M3 scale byte
__device__ __forceinline__ uint64_t quantize_block(const float v[16], float g, int64_t seed, uint32_t step,
                                                   uint64_t idx0, uint8_t* scale_out) {
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
  __nv_fp8_e4m3 s8(fminf(fmaxf(amax / 6.0f / g, 0.001953125f), 448.0f));
  const float s = static_cast<float>(s8) * g;
  uint64_t out = 0;
  if (seed < 0) {
#pragma unroll
    for (int i = 0; i < 16; ++i) out |= static_cast<uint64_t>(e2m1_code(s > 0.0f ? v[i] / s : 0.0f)) << (4 * i);
  } else {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float u = bl::hash_uniform(idx0 + i, static_cast<uint32_t>(seed), step);
      out |= static_cast<uint64_t>(e2m1_code_sr(s > 0.0f ? v[i] / s : 0.0f, u)) << (4 * i);
    }
  }
  *scale_out = s8.__x;
  return out;
}

__device__ __forceinline__ float e2m1_value(uint32_t code) {
  constexpr float L[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float m = L[code & 7u];
  return code & 8u ? -m : m;
}

__device__ __forceinline__ void atomic_max_pos(float* addr, float v) {       // v >= 0: int order == float order
  atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
}

// x^T of a row-major [M, K] tensor, 16 consecutive rows of column k, optionally rotated: v <- v H
template <typename T>
__device__ __forceinline__ void load_column_block(const T* x, int64_t K, int64_t k, int64_t m0, const float* H,
                                                  float v[16]) {
  float raw[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) raw[i] = to_f<T>(x[(m0 + i) * K + k]);
  if (H == nullptr) {
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = raw[i];
    return;
  }
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    float a = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) a = fmaf(raw[i], __ldg(H + i * 16 + j), a);
    v[j] = a;
  }
}

// thread (k, mb) with k fastest: consecutive threads read consecutive addresses
template <typename T>
__global__ void amax_t(const T* x, int64_t M, int64_t K, const float* H, float* amax) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  float m = 0.0f;
  if (t < (M / 16) * K) {
    float v[16];
    load_column_block(x, K, t % K, (t / K) * 16, H, v);
#pragma unroll
    for (int i = 0; i < 16; ++i) m = fmaxf(m, fabsf(v[i]));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) atomic_max_pos(amax, m);
}

template <typename T>
__global__ void quantize_t(const T* x, int64_t M, int64_t K, const float* H, const float* amax, float* global,
                           uint8_t* packed, uint8_t* scales, int64_t seed, uint32_t step) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const float g = fmaxf(*amax / (6.0f * 448.0f), 1e-12f);
  if (t == 0) *global = g;
  if (t >= (M / 16) * K) return;
  const int64_t k = t % K, mb = t / K;
  float v[16];
  load_column_block(x, K, k, mb * 16, H, v);
  const uint64_t out = quantize_block(v, g, seed, step, static_cast<uint64_t>(k) * M + mb * 16,
                                      scales + k * (M / 16) + mb);
  reinterpret_cast<uint64_t*>(packed + k * (M / 2))[mb] = out;
}

// W^T from NVFP4 W [N, K] (codes packed along K, E4M3 scales per 16 along K, global gw): 16 rows of column k
__device__ __forceinline__ void load_w_column(const uint8_t* packed, const uint8_t* scales, float gw, int64_t K,
                                              int64_t k, int64_t n0, float v[16]) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int64_t n = n0 + i;
    const uint32_t code = (packed[n * (K / 2) + k / 2] >> (4 * (k & 1))) & 0xFu;
    __nv_fp8_e4m3 s8;
    s8.__x = scales[n * (K / 16) + k / 16];
    v[i] = e2m1_value(code) * static_cast<float>(s8) * gw;
  }
}

// W^T holds the same values as W, so W's global scale bounds every block of W^T as well: no amax pass is needed
__global__ void requantize_wt(const uint8_t* packed, const uint8_t* scales, const float* gw, int64_t N, int64_t K,
                              float* global, uint8_t* out_packed, uint8_t* out_scales) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const float g = *gw;
  if (t == 0) *global = g;
  if (t >= (N / 16) * K) return;
  const int64_t k = t % K, nb = t / K;
  float v[16];
  load_w_column(packed, scales, *gw, K, k, nb * 16, v);
  const uint64_t out = quantize_block(v, g, -1, 0, 0, out_scales + k * (N / 16) + nb);
  reinterpret_cast<uint64_t*>(out_packed + k * (N / 2))[nb] = out;
}

// one thread per 16-element block; seed < 0: round to nearest, else stochastic rounding with the library hash
// (element index, seed, step)
template <typename T>
__global__ void quantize_nvfp4(const T* x, int64_t nblocks, const float* global, uint8_t* packed, uint8_t* scales,
                               int64_t seed, uint32_t step) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= nblocks) return;
  float v[16];
  const T* p = x + b * 16;
  load16(p, v);
  const float amax = 0.0f;
  (void)amax;
  reinterpret_cast<uint64_t*>(packed)[b] = quantize_block(v, global[0], seed, step, static_cast<uint64_t>(b) * 16,
                                                          scales + b);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> bl_quantize_nvfp4(at::Tensor x, at::Tensor global_scale, int64_t seed,
                                                     int64_t step) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
  TORCH_CHECK(x.size(-1) % 16 == 0, "last dimension must be a multiple of 16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned");
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t cols = x.size(-1), rows = x.numel() / cols, nblocks = x.numel() / 16;
  at::Tensor packed = at::empty({rows, cols / 2}, x.options().dtype(at::kByte));
  at::Tensor scales = at::empty({rows, cols / 16}, x.options().dtype(at::kByte));
  constexpr int threads = 256;
  const int blocks = static_cast<int>((nblocks + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  const float* g = global_scale.data_ptr<float>();
  const uint32_t st = static_cast<uint32_t>(step);
  if (x.scalar_type() == at::kBFloat16) {
    quantize_nvfp4<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), nblocks, g, packed.data_ptr<uint8_t>(),
        scales.data_ptr<uint8_t>(), seed, st);
  } else if (x.scalar_type() == at::kHalf) {
    quantize_nvfp4<__half><<<blocks, threads, 0, stream>>>(reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                                                           nblocks, g, packed.data_ptr<uint8_t>(),
                                                           scales.data_ptr<uint8_t>(), seed, st);
  } else {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be bf16, fp16 or fp32");
    quantize_nvfp4<float><<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), nblocks, g,
                                                          packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), seed,
                                                          st);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales.view(at::kFloat8_e4m3fn)};
}

// x [M, K] (M % 16 == 0) -> NVFP4 of x^T (optionally rotated along M by ``hadamard`` [16, 16] fp32): packed
// [K, M/2], E4M3 scales [K, M/16], fp32 global scale [1] from the amax of the (rotated) tensor
std::tuple<at::Tensor, at::Tensor, at::Tensor> bl_quantize_nvfp4_t(at::Tensor x, std::optional<at::Tensor> hadamard,
                                                                   int64_t seed, int64_t step) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2, "x must be a contiguous 2-D CUDA tensor");
  TORCH_CHECK(x.size(0) % 16 == 0, "rows must be a multiple of 16");
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t M = x.size(0), K = x.size(1), n = (M / 16) * K;
  auto opt = x.options();
  at::Tensor packed = at::empty({K, M / 2}, opt.dtype(at::kByte));
  at::Tensor scales = at::empty({K, M / 16}, opt.dtype(at::kByte));
  at::Tensor amax = at::zeros({1}, opt.dtype(at::kFloat));
  at::Tensor global = at::empty({1}, opt.dtype(at::kFloat));
  const float* H = nullptr;
  if (hadamard.has_value()) {
    TORCH_CHECK(hadamard->scalar_type() == at::kFloat && hadamard->numel() == 256 && hadamard->is_contiguous(),
                "hadamard must be a contiguous fp32 [16, 16] tensor");
    H = hadamard->data_ptr<float>();
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto* ptr) {
    using T = std::remove_const_t<std::remove_pointer_t<decltype(ptr)>>;
    amax_t<T><<<blocks, threads, 0, stream>>>(ptr, M, K, H, amax.data_ptr<float>());
    quantize_t<T><<<blocks, threads, 0, stream>>>(ptr, M, K, H, amax.data_ptr<float>(), global.data_ptr<float>(),
                                                  packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), seed,
                                                  static_cast<uint32_t>(step));
  };
  if (x.scalar_type() == at::kBFloat16) {
    launch(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()));
  } else if (x.scalar_type() == at::kHalf) {
    launch(reinterpret_cast<const __half*>(x.data_ptr<at::Half>()));
  } else {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be bf16, fp16 or fp32");
    launch(x.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales.view(at::kFloat8_e4m3fn), global};
}

// NVFP4 W [N, K] -> NVFP4 of W^T ([K, N/2] codes, [K, N/16] scales, new global scale), N % 16 == 0
std::tuple<at::Tensor, at::Tensor, at::Tensor> bl_requantize_nvfp4_t(at::Tensor packed, at::Tensor scales,
                                                                     at::Tensor global_scale) {
  TORCH_CHECK(packed.is_cuda() && packed.is_contiguous() && scales.is_contiguous() && packed.dim() == 2,
              "packed and scales must be contiguous 2-D CUDA tensors");
  const c10::cuda::CUDAGuard guard(packed.device());
  const int64_t N = packed.size(0), K = packed.size(1) * 2, n = (N / 16) * K;
  TORCH_CHECK(N % 16 == 0 && K % 16 == 0, "both dimensions must be multiples of 16");
  auto opt = packed.options();
  at::Tensor out_packed = at::empty({K, N / 2}, opt);
  at::Tensor out_scales = at::empty({K, N / 16}, opt);
  at::Tensor global = at::empty({1}, opt.dtype(at::kFloat));
  at::Tensor gw = global_scale.to(at::kFloat).contiguous();
  const uint8_t* sc = reinterpret_cast<const uint8_t*>(scales.data_ptr());
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  requantize_wt<<<blocks, threads, 0, stream>>>(packed.data_ptr<uint8_t>(), sc, gw.data_ptr<float>(), N, K,
                                                global.data_ptr<float>(), out_packed.data_ptr<uint8_t>(),
                                                out_scales.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_packed, out_scales.view(at::kFloat8_e4m3fn), global};
}

namespace {

// y *= alpha[0] in place; 16-byte vector accesses, alpha read on the device (no host synchronization)
template <typename T>
__global__ void scale_inplace(T* y, int64_t n_vec, int64_t n, const float* alpha) {
  constexpr int per = 16 / sizeof(T);
  const float a = *alpha;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_vec) {
    uint4 raw = reinterpret_cast<uint4*>(y)[i];
    T* e = reinterpret_cast<T*>(&raw);
#pragma unroll
    for (int j = 0; j < per; ++j) e[j] = from_f<T>(to_f<T>(e[j]) * a);
    reinterpret_cast<uint4*>(y)[i] = raw;
  }
  const int64_t tail = n_vec * per + i;                 // leftover elements (n not a multiple of the vector)
  if (i < per && tail < n) y[tail] = from_f<T>(to_f<T>(y[tail]) * a);
}

}  // namespace

at::Tensor bl_scale_(at::Tensor y, at::Tensor alpha) {
  TORCH_CHECK(y.is_cuda() && y.is_contiguous(), "y must be contiguous CUDA");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0, "y must be 16-byte aligned");
  const c10::cuda::CUDAGuard guard(y.device());
  at::Tensor a = alpha.to(at::kFloat).contiguous();
  const int64_t n = y.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int threads = 256;
  auto launch = [&](auto* p) {
    using T = std::remove_pointer_t<decltype(p)>;
    const int64_t n_vec = n / (16 / sizeof(T));
    const int blocks = static_cast<int>(std::max<int64_t>(1, (n_vec + threads - 1) / threads));
    scale_inplace<T><<<blocks, threads, 0, stream>>>(p, n_vec, n, a.data_ptr<float>());
  };
  if (y.scalar_type() == at::kBFloat16) launch(reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()));
  else if (y.scalar_type() == at::kHalf) launch(reinterpret_cast<__half*>(y.data_ptr<at::Half>()));
  else { TORCH_CHECK(y.scalar_type() == at::kFloat, "y must be bf16, fp16 or fp32"); launch(y.data_ptr<float>()); }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
