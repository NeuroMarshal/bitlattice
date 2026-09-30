// Fused factored (row/column second moment) direction.
//   X = G / max|G|;  r <- b2 r + (1-b2) mean_j X^2;  c <- b2 c + (1-b2) mean_i X^2  (bias corrected)
//   U_ij = X_ij / sqrt(r_i / mean(r) * c_j);  D = U / max(1, rms(U) / clip)
// Three passes over G: (1) row/column sums of G^2 and max|G|, (2) per-row sums of G^2 * b_j^2 for rms(U),
// (3) D = G * a_i * b_j. The epsilon of the reference (1e-12 added to sqrt(V)) is omitted.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cmath>
#include <optional>

namespace {

constexpr int kThreads = 256;
constexpr int kColsPerThread = 4;
constexpr int kTileRows = 32;

template <typename T> __device__ __forceinline__ void load4(const T* p, float v[4]);
template <> __device__ __forceinline__ void load4<float>(const float* p, float v[4]) {
  const float4 a = *reinterpret_cast<const float4*>(p);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
}
template <> __device__ __forceinline__ void load4<__nv_bfloat16>(const __nv_bfloat16* p, float v[4]) {
  const uint2 a = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&a.x);
  const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&a.y);
  v[0] = __low2float(lo); v[1] = __high2float(lo); v[2] = __low2float(hi); v[3] = __high2float(hi);
}

__device__ __forceinline__ void atomic_max_pos(float* addr, float v) {
  atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));  // valid for non-negative floats
}

// pass 1: row sums, column sums of G^2, max |G|
template <typename T>
__global__ void sq_sums(const T* G, int64_t rows, int64_t cols, float* row_sq, float* col_sq, float* amax) {
  __shared__ float srow[kTileRows];
  const int64_t c0 = (static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x) * kColsPerThread;
  const int64_t r0 = static_cast<int64_t>(blockIdx.y) * kTileRows;
  if (threadIdx.x < kTileRows) srow[threadIdx.x] = 0.0f;
  __syncthreads();
  float cacc[4] = {0.f, 0.f, 0.f, 0.f};
  float m = 0.0f;
  const bool active = c0 < cols;
  for (int i = 0; i < kTileRows && r0 + i < rows; ++i) {
    float part = 0.0f;
    if (active) {
      float v[4];
      load4<T>(G + (r0 + i) * cols + c0, v);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float s = v[j] * v[j];
        cacc[j] += s;
        part += s;
        m = fmaxf(m, fabsf(v[j]));
      }
    }
    for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
    if ((threadIdx.x & 31) == 0 && part != 0.0f) atomicAdd(&srow[i], part);
  }
  if (active) {
#pragma unroll
    for (int j = 0; j < 4; ++j) atomicAdd(&col_sq[c0 + j], cacc[j]);
  }
  for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  if ((threadIdx.x & 31) == 0) atomic_max_pos(amax, m);
  __syncthreads();
  if (threadIdx.x < kTileRows && r0 + threadIdx.x < rows) atomicAdd(&row_sq[r0 + threadIdx.x], srow[threadIdx.x]);
}

// pass 2: per-row sums of G^2 * b_j^2
template <typename T>
__global__ void weighted_sums(const T* G, int64_t rows, int64_t cols, const float* b, float* row_w) {
  __shared__ float srow[kTileRows];
  const int64_t c0 = (static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x) * kColsPerThread;
  const int64_t r0 = static_cast<int64_t>(blockIdx.y) * kTileRows;
  if (threadIdx.x < kTileRows) srow[threadIdx.x] = 0.0f;
  __syncthreads();
  const bool active = c0 < cols;
  float bb[4] = {0.f, 0.f, 0.f, 0.f};
  if (active) {
#pragma unroll
    for (int j = 0; j < 4; ++j) bb[j] = b[c0 + j] * b[c0 + j];
  }
  for (int i = 0; i < kTileRows && r0 + i < rows; ++i) {
    float part = 0.0f;
    if (active) {
      float v[4];
      load4<T>(G + (r0 + i) * cols + c0, v);
#pragma unroll
      for (int j = 0; j < 4; ++j) part += v[j] * v[j] * bb[j];
    }
    for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
    if ((threadIdx.x & 31) == 0 && part != 0.0f) atomicAdd(&srow[i], part);
  }
  __syncthreads();
  if (threadIdx.x < kTileRows && r0 + threadIdx.x < rows) atomicAdd(&row_w[r0 + threadIdx.x], srow[threadIdx.x]);
}

// pass 3: D = G * a_i * b_j
template <typename T>
__global__ void scale_out(const T* G, int64_t rows, int64_t cols, const float* a, const float* b, float* D) {
  const int64_t c0 = (static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x) * kColsPerThread;
  const int64_t r0 = static_cast<int64_t>(blockIdx.y) * kTileRows;
  if (c0 >= cols) return;
  const float4 bv = *reinterpret_cast<const float4*>(b + c0);
  for (int i = 0; i < kTileRows && r0 + i < rows; ++i) {
    float v[4];
    load4<T>(G + (r0 + i) * cols + c0, v);
    const float ai = a[r0 + i];
    *reinterpret_cast<float4*>(D + (r0 + i) * cols + c0) =
        make_float4(v[0] * ai * bv.x, v[1] * ai * bv.y, v[2] * ai * bv.z, v[3] * ai * bv.w);
  }
}

}  // namespace

at::Tensor bl_factored_direction(at::Tensor G, at::Tensor r, at::Tensor c, int64_t k, double beta2, double clip) {
  TORCH_CHECK(G.is_cuda(), "G must be CUDA");
  const c10::cuda::CUDAGuard guard(G.device());
  at::Tensor g2 = (G.dim() == 1 ? G.reshape({1, -1}) : G.reshape({G.size(0), -1}));
  if (g2.scalar_type() != at::kBFloat16 && g2.scalar_type() != at::kFloat) g2 = g2.to(at::kFloat);
  g2 = g2.contiguous();
  const int64_t rows = g2.size(0), cols = g2.size(1);
  TORCH_CHECK(cols % kColsPerThread == 0, "factored kernel needs cols % 4 == 0");
  auto f32 = g2.options().dtype(at::kFloat);
  at::Tensor row_sq = at::zeros({rows}, f32), col_sq = at::zeros({cols}, f32), amax = at::zeros({1}, f32);
  const dim3 grid(static_cast<unsigned>((cols / kColsPerThread + kThreads - 1) / kThreads),
                  static_cast<unsigned>((rows + kTileRows - 1) / kTileRows));
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool bf = g2.scalar_type() == at::kBFloat16;
  if (bf) {
    sq_sums<__nv_bfloat16><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(g2.data_ptr<at::BFloat16>()), rows, cols, row_sq.data_ptr<float>(),
        col_sq.data_ptr<float>(), amax.data_ptr<float>());
  } else {
    sq_sums<float><<<grid, kThreads, 0, stream>>>(g2.data_ptr<float>(), rows, cols, row_sq.data_ptr<float>(),
                                                  col_sq.data_ptr<float>(), amax.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // statistics on O(rows + cols) vectors
  at::Tensor pk = amax.clamp_min(1e-30);
  at::Tensor inv2 = 1.0 / (pk * pk);
  r.mul_(beta2).add_(row_sq * inv2 / static_cast<double>(cols), 1.0 - beta2);
  c.mul_(beta2).add_(col_sq * inv2 / static_cast<double>(rows), 1.0 - beta2);
  const double bc = 1.0 - std::pow(beta2, static_cast<double>(k));
  at::Tensor rr = r / bc, cc = c / bc;
  at::Tensor rbar = rr.mean().clamp_min(1e-12);
  at::Tensor a = at::where(rr > 0, at::rsqrt(rr / rbar), at::zeros_like(rr)) / pk;  // row factor incl. 1/max|G|
  at::Tensor b = at::where(cc > 0, at::rsqrt(cc), at::zeros_like(cc));
  at::Tensor row_w = at::zeros({rows}, f32);
  if (bf) {
    weighted_sums<__nv_bfloat16><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(g2.data_ptr<at::BFloat16>()), rows, cols, b.data_ptr<float>(),
        row_w.data_ptr<float>());
  } else {
    weighted_sums<float><<<grid, kThreads, 0, stream>>>(g2.data_ptr<float>(), rows, cols, b.data_ptr<float>(),
                                                        row_w.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  at::Tensor rms = ((a * a * row_w).sum() / static_cast<double>(rows * cols)).sqrt();
  a = a / at::clamp_min(rms / clip, 1.0);
  at::Tensor D = at::empty({rows, cols}, f32);
  if (bf) {
    scale_out<__nv_bfloat16><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(g2.data_ptr<at::BFloat16>()), rows, cols, a.data_ptr<float>(),
        b.data_ptr<float>(), D.data_ptr<float>());
  } else {
    scale_out<float><<<grid, kThreads, 0, stream>>>(g2.data_ptr<float>(), rows, cols, a.data_ptr<float>(),
                                                    b.data_ptr<float>(), D.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return D;
}
