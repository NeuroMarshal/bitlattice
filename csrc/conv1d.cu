// Causal depthwise convolution along time with an optional SiLU, as in the short convolutions of linear-attention
// and state-space layers:   y[b, t, c] = act(bias[c] + sum_j w[c, j] * x[b, t - (W - 1) + j, c]),  x[t < 0] = 0.
// Channels-last memory ([B, T, C], C contiguous): a thread owns two channels and a run of kTT time steps; the
// backward recomputes the pre-activation, so only x is kept between forward and backward.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <optional>
#include <type_traits>

namespace {

constexpr int kTT = 16;         // time steps per thread
constexpr int kMaxW = 8;        // kernel width
constexpr int kTiles = 8;       // time tiles per block (block: 32 channel pairs x 8 tiles)

template <typename T> __device__ __forceinline__ float2 ld2(const T* p);
template <> __device__ __forceinline__ float2 ld2<__nv_bfloat16>(const __nv_bfloat16* p) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
}
template <> __device__ __forceinline__ float2 ld2<__half>(const __half* p) {
  return __half22float2(*reinterpret_cast<const __half2*>(p));
}
template <> __device__ __forceinline__ float2 ld2<float>(const float* p) { return *reinterpret_cast<const float2*>(p); }

template <typename T> __device__ __forceinline__ void st2(T* p, float a, float b);
template <> __device__ __forceinline__ void st2<__nv_bfloat16>(__nv_bfloat16* p, float a, float b) {
  *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
}
template <> __device__ __forceinline__ void st2<__half>(__half* p, float a, float b) {
  *reinterpret_cast<__half2*>(p) = __floats2half2_rn(a, b);
}
template <> __device__ __forceinline__ void st2<float>(float* p, float a, float b) {
  *reinterpret_cast<float2*>(p) = make_float2(a, b);
}

__device__ __forceinline__ float silu(float v) { return v / (1.0f + __expf(-v)); }
__device__ __forceinline__ float dsilu(float v) {      // d silu / d v
  const float s = 1.0f / (1.0f + __expf(-v));
  return s * (1.0f + v * (1.0f - s));
}

template <typename T, int W>
__global__ void conv_fwd(const T* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bias,
                         T* __restrict__ y, int64_t B, int64_t Tn, int64_t C, bool act) {
  const int64_t c = (static_cast<int64_t>(blockIdx.x) * 32 + (threadIdx.x & 31)) * 2;
  const int64_t t0 = (static_cast<int64_t>(blockIdx.y) * kTiles + (threadIdx.x >> 5)) * kTT;
  const int64_t b = blockIdx.z;
  if (c >= C || t0 >= Tn) return;
  float w0[W], w1[W];
#pragma unroll
  for (int j = 0; j < W; ++j) {
    w0[j] = w[c * W + j];
    w1[j] = w[(c + 1) * W + j];
  }
  const float b0 = bias ? bias[c] : 0.0f, b1 = bias ? bias[c + 1] : 0.0f;
  const T* xb = x + b * Tn * C + c;
  T* yb = y + b * Tn * C + c;
  float h0[W], h1[W];                                     // sliding window x[t - W + 1 .. t]
#pragma unroll
  for (int j = 0; j < W - 1; ++j) {
    const int64_t t = t0 - (W - 1) + j;
    const float2 v = t >= 0 ? ld2<T>(xb + t * C) : make_float2(0.0f, 0.0f);
    h0[j + 1] = v.x;
    h1[j + 1] = v.y;
  }
  for (int k = 0; k < kTT && t0 + k < Tn; ++k) {
#pragma unroll
    for (int j = 0; j < W - 1; ++j) {
      h0[j] = h0[j + 1];
      h1[j] = h1[j + 1];
    }
    const float2 v = ld2<T>(xb + (t0 + k) * C);
    h0[W - 1] = v.x;
    h1[W - 1] = v.y;
    float a0 = b0, a1 = b1;
#pragma unroll
    for (int j = 0; j < W; ++j) {
      a0 += w0[j] * h0[j];
      a1 += w1[j] * h1[j];
    }
    st2<T>(yb + (t0 + k) * C, act ? silu(a0) : a0, act ? silu(a1) : a1);
  }
}

template <typename T, int W>
__global__ void conv_bwd(const T* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bias,
                         const T* __restrict__ dy, T* __restrict__ dx, float* __restrict__ dw, float* __restrict__ db,
                         int64_t B, int64_t Tn, int64_t C, bool act) {
  __shared__ float red[kTiles][32][2 * (kMaxW + 1)];
  const int lane = threadIdx.x & 31, tile = threadIdx.x >> 5;
  const int64_t c = (static_cast<int64_t>(blockIdx.x) * 32 + lane) * 2;
  const int64_t t0 = (static_cast<int64_t>(blockIdx.y) * kTiles + tile) * kTT;
  const int64_t b = blockIdx.z;
  float gw0[W], gw1[W], gb0 = 0.0f, gb1 = 0.0f;
#pragma unroll
  for (int j = 0; j < W; ++j) gw0[j] = gw1[j] = 0.0f;
  if (c < C && t0 < Tn) {
    float w0[W], w1[W];
#pragma unroll
    for (int j = 0; j < W; ++j) {
      w0[j] = w[c * W + j];
      w1[j] = w[(c + 1) * W + j];
    }
    const float b0 = bias ? bias[c] : 0.0f, b1 = bias ? bias[c + 1] : 0.0f;
    const T* xb = x + b * Tn * C + c;
    const T* gb = dy + b * Tn * C + c;
    T* dxb = dx + b * Tn * C + c;
    // x for t0 - W + 1 .. t0 + kTT + W - 2 in registers (position k holds time t0 - W + 1 + k)
    constexpr int kX = kTT + 2 * W - 2;
    float x0[kX], x1[kX];
#pragma unroll
    for (int k = 0; k < kX; ++k) {
      const int64_t t = t0 - (W - 1) + k;
      const float2 v = (t >= 0 && t < Tn) ? ld2<T>(xb + t * C) : make_float2(0.0f, 0.0f);
      x0[k] = v.x;
      x1[k] = v.y;
    }
    // d pre for t0 .. t0 + kTT + W - 2 (the last W - 1 belong to the next tile, needed for dx of this one)
    constexpr int kD = kTT + W - 1;
    float d0[kD], d1[kD];
#pragma unroll
    for (int k = 0; k < kD; ++k) {
      const int64_t t = t0 + k;
      if (t < Tn) {
        const float2 g = ld2<T>(gb + t * C);
        if (act) {
          float p0 = b0, p1 = b1;                       // pre(t) = bias + sum_j w[j] x[t - W + 1 + j]
#pragma unroll
          for (int j = 0; j < W; ++j) {
            p0 += w0[j] * x0[k + j];
            p1 += w1[j] * x1[k + j];
          }
          d0[k] = g.x * dsilu(p0);
          d1[k] = g.y * dsilu(p1);
        } else {
          d0[k] = g.x;
          d1[k] = g.y;
        }
      } else {
        d0[k] = d1[k] = 0.0f;
      }
    }
#pragma unroll
    for (int k = 0; k < kTT; ++k) {
      if (t0 + k < Tn) {
        // dx[t] = sum_j w[j] * dpre[t + W - 1 - j]
        float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
        for (int j = 0; j < W; ++j) {
          a0 += w0[j] * d0[k + W - 1 - j];
          a1 += w1[j] * d1[k + W - 1 - j];
        }
        st2<T>(dxb + (t0 + k) * C, a0, a1);
        // dw[j] += dpre[t] * x[t - W + 1 + j]; db += dpre[t]
#pragma unroll
        for (int j = 0; j < W; ++j) {
          gw0[j] += d0[k] * x0[k + j];
          gw1[j] += d1[k] * x1[k + j];
        }
        gb0 += d0[k];
        gb1 += d1[k];
      }
    }
  }
  // reduce over the time tiles of the block, one atomic per channel and tap
#pragma unroll
  for (int j = 0; j < W; ++j) {
    red[tile][lane][2 * j] = gw0[j];
    red[tile][lane][2 * j + 1] = gw1[j];
  }
  red[tile][lane][2 * W] = gb0;
  red[tile][lane][2 * W + 1] = gb1;
  __syncthreads();
  if (tile == 0 && c < C) {
    for (int q = 0; q < 2 * W + 2; ++q) {
      float s = 0.0f;
#pragma unroll
      for (int i = 0; i < kTiles; ++i) s += red[i][lane][q];
      if (q < 2 * W) atomicAdd(dw + (c + (q & 1)) * W + q / 2, s);
      else if (db) atomicAdd(db + c + (q & 1), s);
    }
  }
}

template <typename F>
void with_width(int64_t W, F&& f) {
  switch (W) {
    case 2: f(std::integral_constant<int, 2>{}); break;
    case 3: f(std::integral_constant<int, 3>{}); break;
    case 4: f(std::integral_constant<int, 4>{}); break;
    case 5: f(std::integral_constant<int, 5>{}); break;
    case 6: f(std::integral_constant<int, 6>{}); break;
    case 7: f(std::integral_constant<int, 7>{}); break;
    case 8: f(std::integral_constant<int, 8>{}); break;
    default: TORCH_CHECK(false, "kernel width must be in [2, 8]");
  }
}

template <typename F>
void with_dtype(const at::Tensor& x, F&& f) {
  if (x.scalar_type() == at::kBFloat16) f(static_cast<__nv_bfloat16*>(nullptr));
  else if (x.scalar_type() == at::kHalf) f(static_cast<__half*>(nullptr));
  else {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be bf16, fp16 or fp32");
    f(static_cast<float*>(nullptr));
  }
}

void check(const at::Tensor& x, const at::Tensor& w) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 3 && x.is_contiguous(), "x: contiguous CUDA [B, T, C] (channels last)");
  TORCH_CHECK(x.size(2) % 2 == 0, "channels must be even");
  TORCH_CHECK(w.dim() == 2 && w.size(0) == x.size(2), "weight: [C, W]");
}

}  // namespace

// x [B, T, C] channels-last; weight [C, W]; returns y [B, T, C]
at::Tensor bl_causal_conv1d_fwd(at::Tensor x, at::Tensor weight, std::optional<at::Tensor> bias, bool silu_act) {
  check(x, weight);
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t B = x.size(0), Tn = x.size(1), C = x.size(2), W = weight.size(1);
  at::Tensor wf = weight.to(at::kFloat).contiguous();
  at::Tensor bf = bias.has_value() ? bias->to(at::kFloat).contiguous() : at::Tensor();
  at::Tensor y = at::empty_like(x);
  if (x.numel() == 0) return y;
  dim3 grid(static_cast<unsigned>((C / 2 + 31) / 32), static_cast<unsigned>((Tn + kTT * kTiles - 1) / (kTT * kTiles)),
            static_cast<unsigned>(B));
  auto stream = at::cuda::getCurrentCUDAStream();
  with_dtype(x, [&](auto* tag) {
    using T = std::remove_pointer_t<decltype(tag)>;
    with_width(W, [&](auto wc) {
      conv_fwd<T, decltype(wc)::value><<<grid, 32 * kTiles, 0, stream>>>(
          reinterpret_cast<const T*>(x.data_ptr()), wf.data_ptr<float>(), bias.has_value() ? bf.data_ptr<float>() : nullptr,
          reinterpret_cast<T*>(y.data_ptr()), B, Tn, C, silu_act);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// returns (dx [B, T, C], dweight fp32 [C, W], dbias fp32 [C] or empty)
std::tuple<at::Tensor, at::Tensor, at::Tensor> bl_causal_conv1d_bwd(at::Tensor x, at::Tensor weight,
                                                                    std::optional<at::Tensor> bias, at::Tensor dy,
                                                                    bool silu_act) {
  check(x, weight);
  TORCH_CHECK(dy.sizes() == x.sizes() && dy.is_contiguous() && dy.scalar_type() == x.scalar_type(), "dy like x");
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t B = x.size(0), Tn = x.size(1), C = x.size(2), W = weight.size(1);
  at::Tensor wf = weight.to(at::kFloat).contiguous();
  at::Tensor bf = bias.has_value() ? bias->to(at::kFloat).contiguous() : at::Tensor();
  at::Tensor dx = at::empty_like(x);
  at::Tensor dw = at::zeros({C, W}, x.options().dtype(at::kFloat));
  at::Tensor db = bias.has_value() ? at::zeros({C}, x.options().dtype(at::kFloat)) : at::Tensor();
  if (x.numel() == 0) return {dx, dw, db};
  dim3 grid(static_cast<unsigned>((C / 2 + 31) / 32), static_cast<unsigned>((Tn + kTT * kTiles - 1) / (kTT * kTiles)),
            static_cast<unsigned>(B));
  auto stream = at::cuda::getCurrentCUDAStream();
  with_dtype(x, [&](auto* tag) {
    using T = std::remove_pointer_t<decltype(tag)>;
    with_width(W, [&](auto wc) {
      conv_bwd<T, decltype(wc)::value><<<grid, 32 * kTiles, 0, stream>>>(
          reinterpret_cast<const T*>(x.data_ptr()), wf.data_ptr<float>(), bias.has_value() ? bf.data_ptr<float>() : nullptr,
          reinterpret_cast<const T*>(dy.data_ptr()), reinterpret_cast<T*>(dx.data_ptr()), dw.data_ptr<float>(),
          bias.has_value() ? db.data_ptr<float>() : nullptr, B, Tn, C, silu_act);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dw, db};
}
