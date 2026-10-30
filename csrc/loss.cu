// Vocabulary-tiled cross entropy. A frozen head forms its input gradient during the forward pass, retaining only
// that gradient. Logits and softmax weights stay in the GEMM dtype; online softmax and gradient sums use float32.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <optional>

namespace {

constexpr int kThreads = 256;

__device__ float max_nan(float a, float b) { return isnan(a) || isnan(b) ? NAN : fmaxf(a, b); }

template <bool kMax>
__device__ float reduce(float x, float* shared) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    const float other = __shfl_down_sync(0xffffffffu, x, d);
    x = kMax ? max_nan(x, other) : x + other;
  }
  if (lane == 0) shared[warp] = x;
  __syncthreads();
  if (warp == 0) {
    x = lane < kThreads / 32 ? shared[lane] : (kMax ? -INFINITY : 0.0f);
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
      const float other = __shfl_down_sync(0xffffffffu, x, d);
      x = kMax ? max_nan(x, other) : x + other;
    }
    if (lane == 0) shared[0] = x;
  }
  __syncthreads();
  return shared[0];
}

template <typename T, bool kGrad>
__global__ void online_softmax(const T* logits, const float* bias, const int64_t* labels, int64_t cols,
                               int64_t first, int64_t vocab, float* m, float* z, float* target,
                               T* p, float* rescale) {
  __shared__ float reductions[16];
  const int64_t row = blockIdx.x;
  const int64_t y = labels[row];
  CUDA_KERNEL_ASSERT(y >= 0 && y < vocab);
  float mx = -INFINITY;
  for (int64_t c = threadIdx.x; c < cols; c += kThreads) {
    const float v = static_cast<float>(logits[row * cols + c]) + (bias ? bias[first + c] : 0.0f);
    mx = max_nan(mx, v);
  }
  const float next = max_nan(m[row], reduce<true>(mx, reductions));
  const float alpha = expf(m[row] - next);
  float sum = 0.0f;
  for (int64_t c = threadIdx.x; c < cols; c += kThreads) {
    const float v = static_cast<float>(logits[row * cols + c]) + (bias ? bias[first + c] : 0.0f);
    const float e = expf(v - next);
    sum += e;
    if constexpr (kGrad) p[row * cols + c] = static_cast<T>(e);
  }
  sum = reduce<false>(sum, reductions + 8);
  if (threadIdx.x == 0) {
    z[row] = z[row] * alpha + sum;
    m[row] = next;
    if (y >= first && y < first + cols)
      target[row] = static_cast<float>(logits[row * cols + y - first]) + (bias ? bias[y] : 0.0f);
    if constexpr (kGrad) rescale[row] = alpha;
  }
}

template <typename T>
__global__ void add_weighted(float* acc, const T* weighted, const float* rescale, int64_t n, int64_t width) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) acc[i] = acc[i] * rescale[i / width] + static_cast<float>(weighted[i]);
}

template <typename T, bool kGrad>
__global__ void finish(const float* m, const float* z, const float* target, const float* acc, const T* weight,
                       const int64_t* labels, const float* valid, int64_t rows, int64_t width,
                       float* lse, float* tgt, T* grad) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < rows) {
    lse[i] = m[i] + logf(z[i]);
    tgt[i] = target[i];
  }
  if constexpr (kGrad) {
    if (i < rows * width) {
      const int64_t r = i / width, c = i % width;
      const float g = acc[i] / z[r] - static_cast<float>(weight[labels[r] * width + c]);
      grad[i] = static_cast<T>(g * valid[r]);
    }
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> bl_vocab_pass(at::Tensor h, at::Tensor w,
    std::optional<at::Tensor> bias, at::Tensor labels, at::Tensor valid, int64_t chunk, int64_t tile_bytes,
    bool want_grad) {
  TORCH_CHECK(h.is_cuda() && w.device() == h.device() && h.dim() == 2 && w.dim() == 2 &&
              h.is_contiguous() && w.is_contiguous() && h.size(1) == w.size(1) && h.scalar_type() == w.scalar_type(),
              "vocab_pass: contiguous matrices of the same dtype on one CUDA device");
  TORCH_CHECK(labels.device() == h.device() && labels.scalar_type() == at::kLong && labels.is_contiguous() &&
              labels.numel() == h.size(0) && valid.device() == h.device() && valid.scalar_type() == at::kFloat &&
              valid.is_contiguous() && valid.numel() == h.size(0), "vocab_pass: int64 labels and float32 validity");
  TORCH_CHECK(chunk > 0 && tile_bytes > 0 && w.size(0) > 0, "vocab_pass: positive chunk, tile budget and vocabulary");
  TORCH_CHECK(!bias || (bias->device() == h.device() && bias->numel() == w.size(0)), "vocab_pass: bias shape/device");
  const c10::cuda::CUDAGuard guard(h.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t n = h.size(0), d = h.size(1), vocab = w.size(0);
  auto fp32 = h.options().dtype(at::kFloat);
  auto lse = at::empty({n}, fp32), tgt = at::empty({n}, fp32);
  auto grad = at::empty(want_grad ? at::IntArrayRef({n, d}) : at::IntArrayRef({0}), h.options());
  at::Tensor b = bias ? bias->to(at::kFloat).contiguous() : at::Tensor();
  auto run = [&](auto dummy) {
    using T = decltype(dummy);
    for (int64_t s = 0; s < n; s += chunk) {
      const int64_t rows = std::min(chunk, n - s);
      auto hc = h.narrow(0, s, rows);
      auto m = at::full({rows}, -INFINITY, fp32), z = at::zeros({rows}, fp32), target = at::zeros({rows}, fp32);
      auto acc = want_grad ? at::zeros({rows, d}, fp32) : at::Tensor();
      auto rescale = want_grad ? at::empty({rows}, fp32) : at::Tensor();
      int64_t tile = std::max<int64_t>(128, std::min(vocab, tile_bytes / (4 * rows)));
      if (tile < vocab) tile -= tile % 128;
      for (int64_t v0 = 0; v0 < vocab; v0 += tile) {
        const int64_t cols = std::min(tile, vocab - v0);
        auto wt = w.narrow(0, v0, cols);
        auto logits = at::matmul(hc, wt.t());
        auto p = want_grad ? at::empty_like(logits) : at::Tensor();
        if (want_grad)
          online_softmax<T, true><<<rows, kThreads, 0, stream>>>(logits.data_ptr<T>(), b.defined() ? b.data_ptr<float>() : nullptr,
              labels.data_ptr<int64_t>() + s, cols, v0, vocab, m.data_ptr<float>(), z.data_ptr<float>(),
              target.data_ptr<float>(), p.data_ptr<T>(), rescale.data_ptr<float>());
        else
          online_softmax<T, false><<<rows, kThreads, 0, stream>>>(logits.data_ptr<T>(), b.defined() ? b.data_ptr<float>() : nullptr,
              labels.data_ptr<int64_t>() + s, cols, v0, vocab, m.data_ptr<float>(), z.data_ptr<float>(),
              target.data_ptr<float>(), nullptr, nullptr);
        if (want_grad) {
          auto weighted = at::matmul(p, wt);
          add_weighted<T><<<(rows * d + kThreads - 1) / kThreads, kThreads, 0, stream>>>(acc.data_ptr<float>(),
              weighted.data_ptr<T>(), rescale.data_ptr<float>(), rows * d, d);
        }
      }
      const int64_t count = want_grad ? rows * d : rows;
      if (want_grad)
        finish<T, true><<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(m.data_ptr<float>(), z.data_ptr<float>(),
            target.data_ptr<float>(), acc.data_ptr<float>(), w.data_ptr<T>(), labels.data_ptr<int64_t>() + s,
            valid.data_ptr<float>() + s, rows, d, lse.data_ptr<float>() + s, tgt.data_ptr<float>() + s, grad.data_ptr<T>() + s * d);
      else
        finish<T, false><<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(m.data_ptr<float>(), z.data_ptr<float>(),
            target.data_ptr<float>(), nullptr, nullptr, nullptr, nullptr, rows, d, lse.data_ptr<float>() + s, tgt.data_ptr<float>() + s, nullptr);
    }
  };
  if (h.scalar_type() == at::kBFloat16) run(at::BFloat16{});
  else if (h.scalar_type() == at::kHalf) run(at::Half{});
  else {
    TORCH_CHECK(h.scalar_type() == at::kFloat, "vocab_pass: bf16, fp16 or fp32");
    run(float{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {lse, tgt, grad};
}
