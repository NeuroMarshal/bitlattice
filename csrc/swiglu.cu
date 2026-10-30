// SwiGLU forward/backward, rounding SiLU and each product to the activation dtype as PyTorch autograd does.
// Each thread handles eight elements with 16-byte loads.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <tuple>

namespace {

template <typename T> __device__ __forceinline__ float tof(T v);
template <> __device__ __forceinline__ float tof<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float tof<__half>(__half v) { return __half2float(v); }
template <typename T> __device__ __forceinline__ T fromf(float v);
template <> __device__ __forceinline__ __nv_bfloat16 fromf<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ __half fromf<__half>(float v) { return __float2half_rn(v); }

// ATen's silu: x / (1 + exp(-x)) in float
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

template <typename T>
__global__ void swiglu_fwd_kernel(const T* __restrict__ g, const T* __restrict__ u, T* __restrict__ a, int64_t n) {
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  if (i + 8 <= n) {
    const uint4 gv = *reinterpret_cast<const uint4*>(g + i), uv = *reinterpret_cast<const uint4*>(u + i);
    const T* ge = reinterpret_cast<const T*>(&gv);
    const T* ue = reinterpret_cast<const T*>(&uv);
    uint4 out;
    T* oe = reinterpret_cast<T*>(&out);
#pragma unroll
    for (int k = 0; k < 8; ++k) oe[k] = fromf<T>(tof<T>(fromf<T>(silu(tof<T>(ge[k])))) * tof<T>(ue[k]));
    *reinterpret_cast<uint4*>(a + i) = out;
  } else {
    for (int64_t k = i; k < n; ++k) a[k] = fromf<T>(tof<T>(fromf<T>(silu(tof<T>(g[k])))) * tof<T>(u[k]));
  }
}

template <typename T>
__device__ __forceinline__ void swiglu_bwd1(T g, T u, T da, T& dg, T& du) {
  const float gf = tof<T>(g), daf = tof<T>(da);
  const float s = tof<T>(fromf<T>(silu(gf)));
  du = fromf<T>(daf * s);
  const float ds = tof<T>(fromf<T>(daf * tof<T>(u)));
  const float sig = 1.0f / (1.0f + expf(-gf));
  dg = fromf<T>(ds * sig * (1.0f + gf * (1.0f - sig)));            // ATen's silu_backward order
}

template <typename T>
__global__ void swiglu_bwd_kernel(const T* __restrict__ g, const T* __restrict__ u, const T* __restrict__ da,
                                  T* __restrict__ dg, T* __restrict__ du, int64_t n) {
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  if (i + 8 <= n) {
    const uint4 gv = *reinterpret_cast<const uint4*>(g + i), uv = *reinterpret_cast<const uint4*>(u + i),
                dv = *reinterpret_cast<const uint4*>(da + i);
    const T* ge = reinterpret_cast<const T*>(&gv);
    const T* ue = reinterpret_cast<const T*>(&uv);
    const T* de = reinterpret_cast<const T*>(&dv);
    uint4 og, ou;
    T* oge = reinterpret_cast<T*>(&og);
    T* oue = reinterpret_cast<T*>(&ou);
#pragma unroll
    for (int k = 0; k < 8; ++k) swiglu_bwd1<T>(ge[k], ue[k], de[k], oge[k], oue[k]);
    *reinterpret_cast<uint4*>(dg + i) = og;
    *reinterpret_cast<uint4*>(du + i) = ou;
  } else {
    for (int64_t k = i; k < n; ++k) swiglu_bwd1<T>(g[k], u[k], da[k], dg[k], du[k]);
  }
}

void check(const at::Tensor& t, const at::Tensor& ref, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.sizes() == ref.sizes() && t.scalar_type() == ref.scalar_type(),
              name, ": a contiguous CUDA tensor of the shape and dtype of g");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}

template <typename F>
void dispatch(at::ScalarType st, F&& f) {
  if (st == at::kBFloat16) f(static_cast<__nv_bfloat16*>(nullptr));
  else if (st == at::kHalf) f(static_cast<__half*>(nullptr));
  else TORCH_CHECK(false, "swiglu: bf16 or fp16 tensors");
}

}  // namespace

at::Tensor bl_swiglu_fwd(at::Tensor g, at::Tensor u) {
  check(g, g, "g");
  check(u, g, "u");
  const c10::cuda::CUDAGuard guard(g.device());
  at::Tensor a = at::empty_like(g);
  const int64_t n = g.numel();
  if (n == 0) return a;
  const unsigned blocks = static_cast<unsigned>((n / 8 + 1 + 255) / 256);
  auto stream = at::cuda::getCurrentCUDAStream();
  dispatch(g.scalar_type(), [&](auto* tag) {
    using T = std::remove_pointer_t<decltype(tag)>;
    swiglu_fwd_kernel<T><<<blocks, 256, 0, stream>>>(reinterpret_cast<const T*>(g.data_ptr()),
                                                     reinterpret_cast<const T*>(u.data_ptr()),
                                                     reinterpret_cast<T*>(a.data_ptr()), n);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return a;
}

std::tuple<at::Tensor, at::Tensor> bl_swiglu_bwd(at::Tensor g, at::Tensor u, at::Tensor da) {
  check(g, g, "g");
  check(u, g, "u");
  check(da, g, "da");
  const c10::cuda::CUDAGuard guard(g.device());
  at::Tensor dg = at::empty_like(g), du = at::empty_like(g);
  const int64_t n = g.numel();
  if (n == 0) return {dg, du};
  const unsigned blocks = static_cast<unsigned>((n / 8 + 1 + 255) / 256);
  auto stream = at::cuda::getCurrentCUDAStream();
  dispatch(g.scalar_type(), [&](auto* tag) {
    using T = std::remove_pointer_t<decltype(tag)>;
    swiglu_bwd_kernel<T><<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const T*>(g.data_ptr()), reinterpret_cast<const T*>(u.data_ptr()),
        reinterpret_cast<const T*>(da.data_ptr()), reinterpret_cast<T*>(dg.data_ptr()),
        reinterpret_cast<T*>(du.data_ptr()), n);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dg, du};
}
