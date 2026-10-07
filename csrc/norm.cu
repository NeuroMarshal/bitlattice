// RMSNorm and normalize-then-gate RMSNorm, forward and backward, in one pass over the data each. The per-element
// arithmetic -- float32 math and roundings to the input dtype at the same points -- is that of the reference
// implementations in bitlattice/nn/norm.py; only the order of the sums along a row (and of the weight gradient over
// rows) differs. Rows: one warp each. Weight gradient: a column kernel writes per-slice partial sums that are added
// in a fixed order, so results are deterministic.
//
//   plain   y = round(w * round(x * r))            (cast_first, Llama / Qwen convention)
//           y = round(x * r * (w + offset))        (offset convention, Gemma / Qwen3.5)
//   gated   y = round(round(w * round(x * r)) * silu(z))
//   r = rsqrt(mean(x^2) + eps)
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kWarps = 8;        // rows per block
constexpr int kCols = 256;       // columns per block of the weight-gradient kernel

template <typename T> __device__ __forceinline__ float f(T v);
template <> __device__ __forceinline__ float f<float>(float v) { return v; }
template <> __device__ __forceinline__ float f<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T to(float v);
template <> __device__ __forceinline__ float to<float>(float v) { return v; }
template <> __device__ __forceinline__ __half to<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 to<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

template <typename T> __device__ __forceinline__ float rnd(float v) { return f<T>(to<T>(v)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float sigmoid(float z) { return 1.0f / (1.0f + expf(-z)); }

template <typename T>
__global__ void norm_fwd(const T* __restrict__ x, const T* __restrict__ gate, const T* __restrict__ w,
                         T* __restrict__ y, float* __restrict__ rstd, int64_t R, int64_t D, float eps, float offset,
                         bool cast_first) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= R) return;
  const T* xr = x + row * D;
  float ss = 0.0f;
  for (int64_t c = lane; c < D; c += 32) {
    const float v = f(xr[c]);
    ss += v * v;
  }
  const float r = rsqrtf(warp_sum(ss) * (1.0f / static_cast<float>(D)) + eps);   // as ATen: sum times 1/D
  if (lane == 0) rstd[row] = r;
  T* yr = y + row * D;
  if (gate != nullptr) {
    const T* gr = gate + row * D;
    for (int64_t c = lane; c < D; c += 32) {
      const float b = rnd<T>(f(w[c]) * rnd<T>(f(xr[c]) * r));
      const float z = f(gr[c]);
      yr[c] = to<T>(b * (z / (1.0f + expf(-z))));
    }
  } else if (cast_first) {
    for (int64_t c = lane; c < D; c += 32) yr[c] = to<T>(f(w[c]) * rnd<T>(f(xr[c]) * r));
  } else {
    for (int64_t c = lane; c < D; c += 32) yr[c] = to<T>(f(xr[c]) * r * (f(w[c]) + offset));
  }
}

// input (and gate) gradients
template <typename T>
__global__ void norm_bwd_rows(const T* __restrict__ x, const T* __restrict__ gate, const T* __restrict__ w,
                              const float* __restrict__ rstd, const T* __restrict__ dy, T* __restrict__ dx,
                              T* __restrict__ dgate, int64_t R, int64_t D, float offset) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= R) return;
  const T* xr = x + row * D;
  const T* dr = dy + row * D;
  const T* gr = gate ? gate + row * D : nullptr;
  const float r = rstd[row];
  float s = 0.0f;                                   // sum(dn * n) over the row
  for (int64_t c = lane; c < D; c += 32) {
    const float n = f(xr[c]) * r;
    float dn;
    if (gr) {
      const float z = f(gr[c]);
      dn = f(dr[c]) * (z * sigmoid(z)) * f(w[c]);
    } else {
      dn = f(dr[c]) * (f(w[c]) + offset);
    }
    s += dn * n;
  }
  const float mean = warp_sum(s) * (1.0f / static_cast<float>(D));
  for (int64_t c = lane; c < D; c += 32) {
    const float n = f(xr[c]) * r;
    const float d = f(dr[c]);
    float dn;
    if (gr) {
      const float z = f(gr[c]);
      const float sg = sigmoid(z);
      dn = d * (z * sg) * f(w[c]);
      if (dgate) {
        const float b = rnd<T>(f(w[c]) * rnd<T>(n));
        dgate[row * D + c] = to<T>(d * b * sg * (1.0f + z * (1.0f - sg)));
      }
    } else {
      dn = d * (f(w[c]) + offset);
    }
    if (dx) dx[row * D + c] = to<T>(r * (dn - n * mean));
  }
}

// weight gradient: partial[s, c] = sum over the rows of slice s of the per-element term
template <typename T>
__global__ void norm_bwd_weight(const T* __restrict__ x, const T* __restrict__ gate, const float* __restrict__ rstd,
                                const T* __restrict__ dy, float* __restrict__ partial, int64_t R, int64_t D,
                                int64_t rows_per_slice) {
  const int64_t c = static_cast<int64_t>(blockIdx.x) * kCols + threadIdx.x;
  const int64_t s = blockIdx.y;
  if (c >= D) return;
  const int64_t r0 = s * rows_per_slice, r1 = min(R, r0 + rows_per_slice);
  float acc = 0.0f;
  for (int64_t row = r0; row < r1; ++row) {
    const float n = f(x[row * D + c]) * rstd[row];
    const float d = f(dy[row * D + c]);
    if (gate) {
      const float z = f(gate[row * D + c]);
      acc += d * (z * sigmoid(z)) * rnd<T>(n);
    } else {
      acc += d * n;
    }
  }
  partial[s * D + c] = acc;
}

void check(const at::Tensor& t, const at::Tensor& x, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == x.scalar_type(), name,
              ": expected a contiguous CUDA tensor of the input dtype");
}

}  // namespace

// x [R, D], gate [R, D] or none, w [D] -> (y [R, D], rstd [R] float32)
std::tuple<at::Tensor, at::Tensor> bl_rms_norm_fwd(at::Tensor x, std::optional<at::Tensor> gate, at::Tensor w,
                                                   double eps, double offset, bool cast_first) {
  check(x, x, "x");
  check(w, x, "w");
  if (gate) check(*gate, x, "gate");
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t R = x.size(0), D = x.size(1);
  auto y = at::empty_like(x);
  auto rstd = at::empty({R}, x.options().dtype(at::kFloat));
  if (R == 0) return {y, rstd};
  const dim3 grid((R + kWarps - 1) / kWarps);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rms_norm_fwd", [&] {
    using T = std::conditional_t<std::is_same_v<scalar_t, at::Half>, __half,
              std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>, __nv_bfloat16, float>>;
    norm_fwd<T><<<grid, kWarps * 32, 0, stream>>>(
        reinterpret_cast<const T*>(x.data_ptr()), gate ? reinterpret_cast<const T*>(gate->data_ptr()) : nullptr,
        reinterpret_cast<const T*>(w.data_ptr()), reinterpret_cast<T*>(y.data_ptr()), rstd.data_ptr<float>(), R, D,
        static_cast<float>(eps), static_cast<float>(offset), cast_first);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, rstd};
}

// -> (dx or empty, dgate or empty, dw float32 [D] or empty)
std::tuple<at::Tensor, at::Tensor, at::Tensor> bl_rms_norm_bwd(at::Tensor x, std::optional<at::Tensor> gate,
                                                               at::Tensor w, at::Tensor rstd, at::Tensor dy,
                                                               double offset, bool need_x, bool need_gate,
                                                               bool need_w) {
  check(x, x, "x");
  check(w, x, "w");
  check(dy, x, "dy");
  if (gate) check(*gate, x, "gate");
  TORCH_CHECK(rstd.is_cuda() && rstd.scalar_type() == at::kFloat && rstd.is_contiguous(), "rstd: float32 CUDA");
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t R = x.size(0), D = x.size(1);
  auto dx = need_x ? at::empty_like(x) : at::Tensor();
  auto dg = (need_gate && gate) ? at::empty_like(x) : at::Tensor();
  at::Tensor dw;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rms_norm_bwd", [&] {
    using T = std::conditional_t<std::is_same_v<scalar_t, at::Half>, __half,
              std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>, __nv_bfloat16, float>>;
    const T* xp = reinterpret_cast<const T*>(x.data_ptr());
    const T* gp = gate ? reinterpret_cast<const T*>(gate->data_ptr()) : nullptr;
    const T* dyp = reinterpret_cast<const T*>(dy.data_ptr());
    if (R > 0 && (need_x || dg.defined())) {
      norm_bwd_rows<T><<<dim3((R + kWarps - 1) / kWarps), kWarps * 32, 0, stream>>>(
          xp, gp, reinterpret_cast<const T*>(w.data_ptr()), rstd.data_ptr<float>(), dyp,
          need_x ? reinterpret_cast<T*>(dx.data_ptr()) : nullptr,
          dg.defined() ? reinterpret_cast<T*>(dg.data_ptr()) : nullptr, R, D, static_cast<float>(offset));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (need_w) {
      const int64_t slices = std::max<int64_t>(1, std::min<int64_t>(512, (R + 63) / 64));
      const int64_t per = (R + slices - 1) / slices;
      auto partial = at::empty({slices, D}, x.options().dtype(at::kFloat));
      if (R > 0) {
        norm_bwd_weight<T><<<dim3((D + kCols - 1) / kCols, slices), kCols, 0, stream>>>(
            xp, gp, rstd.data_ptr<float>(), dyp, partial.data_ptr<float>(), R, D, per);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        dw = partial.sum(0);
      } else {
        dw = at::zeros({D}, x.options().dtype(at::kFloat));
      }
    }
  });
  return {dx, dg, dw};
}
