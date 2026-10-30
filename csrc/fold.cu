// Folding of concatenated NVFP4 operands to one global scale. Deferred weight-gradient operands of the micro-batches
// of a unit are concatenated along the token axis; each piece j has a power-of-two global scale g_j, so its E4M3
// block scales times g_j / max(g) (a power of two <= 1) give the same values under the largest global scale. One
// pass over the concatenated scales (128 x 4-tiled layout, 512 bytes per tile) does it for all pieces; products that
// fall below E4M3's normal range lose mantissa bits and are counted.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp8.h>

namespace {

constexpr float kMinNormal = 0.015625f;     // 2^-6

__global__ void fold_kernel(uint8_t* __restrict__ scales, int64_t n, int64_t ntiles_col, const int32_t* __restrict__ piece,
                            const float* __restrict__ g, int npieces, float* __restrict__ g_out,
                            unsigned long long* __restrict__ rounded) {
  float gmax = 0.0f;
  for (int j = 0; j < npieces; ++j) gmax = fmaxf(gmax, g[j]);
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i == 0) g_out[0] = gmax;
  unsigned r = 0;
  if (i < n) {
    const int64_t col_tile = (i / 512) % ntiles_col;
    const float f = g[piece[col_tile]] / gmax;
    if (f != 1.0f) {
      __nv_fp8_e4m3 s;
      s.__x = scales[i];
      const float v = static_cast<float>(s) * f;
      scales[i] = __nv_fp8_e4m3(v).__x;
      r = v > 0.0f && v < kMinNormal;
    }
  }
  r = __reduce_add_sync(0xffffffffu, r);
  if ((threadIdx.x & 31) == 0 && r) atomicAdd(rounded, static_cast<unsigned long long>(r));
}

}  // namespace

// scales: uint8 / e4m3 [rows_tiles * ntiles_col * 512] (folded in place); piece: int32 [ntiles_col], the piece of
// every column tile; g: float32 [npieces] power-of-two global scales; rounded: int64 [1] counter. Returns the common
// global scale [1].
at::Tensor bl_fp4_fold_scales(at::Tensor scales, at::Tensor piece, at::Tensor g, at::Tensor rounded) {
  TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.element_size() == 1, "scales: contiguous bytes");
  TORCH_CHECK(piece.scalar_type() == at::kInt && g.scalar_type() == at::kFloat && rounded.scalar_type() == at::kLong,
              "piece int32, g float32, rounded int64");
  const c10::cuda::CUDAGuard guard(scales.device());
  const int64_t ntiles_col = piece.numel();
  const int64_t n = scales.numel();
  TORCH_CHECK(ntiles_col > 0 && n % (512 * ntiles_col) == 0, "scales must be whole 128 x 4 tiles");
  at::Tensor g_out = at::empty({1}, g.options());
  const int threads = 256;
  const unsigned blocks = static_cast<unsigned>((n + threads - 1) / threads);
  fold_kernel<<<std::max(1u, blocks), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint8_t*>(scales.data_ptr()), n, ntiles_col, piece.data_ptr<int32_t>(),
      g.data_ptr<float>(), static_cast<int>(g.numel()), g_out.data_ptr<float>(),
      reinterpret_cast<unsigned long long*>(rounded.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return g_out;
}
