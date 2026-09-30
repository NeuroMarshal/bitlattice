// Activation quantization to NVFP4: E2M1 codes (low nibble first), one E4M3 scale per 16 elements along the last
// dimension, relative to a per-tensor fp32 global scale (value = code * scale * global).
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

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

// one thread per 16-element block; seed < 0: round to nearest, else stochastic rounding with the library hash
// (element index, seed, step)
template <typename T>
__global__ void quantize_nvfp4(const T* x, int64_t nblocks, const float* global, uint8_t* packed, uint8_t* scales,
                               int64_t seed, uint32_t step) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= nblocks) return;
  float v[16];
  const T* p = x + b * 16;
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    v[i] = to_f<T>(p[i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  const float g = global[0];
  __nv_fp8_e4m3 s8(fminf(fmaxf(amax / 6.0f / g, 0.001953125f), 448.0f));
  const float s = static_cast<float>(s8) * g;
  uint64_t out = 0;
  if (seed < 0) {
#pragma unroll
    for (int i = 0; i < 16; ++i) out |= static_cast<uint64_t>(e2m1_code(s > 0.0f ? v[i] / s : 0.0f)) << (4 * i);
  } else {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float u = bl::hash_uniform(static_cast<uint64_t>(b) * 16 + i, static_cast<uint32_t>(seed), step);
      out |= static_cast<uint64_t>(e2m1_code_sr(s > 0.0f ? v[i] / s : 0.0f, u)) << (4 * i);
    }
  }
  reinterpret_cast<uint64_t*>(packed)[b] = out;
  scales[b] = s8.__x;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> bl_quantize_nvfp4(at::Tensor x, at::Tensor global_scale, int64_t seed,
                                                     int64_t step) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
  TORCH_CHECK(x.size(-1) % 16 == 0, "last dimension must be a multiple of 16");
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
