// One parameter's lattice update in one call: for every row tile, the update direction (Newton-Schulz on row
// bands, sign or raw gradient, RMS-clipped) and the fused lattice step of lattice.cu, the tile's new group scales
// written in place. The caller (an optimizer thread) hands over a whole parameter and the interpreter lock is not
// held while its tiles are issued.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <optional>

using at::Tensor;

Tensor bl_direction(Tensor G, int64_t kind, std::optional<Tensor> r, std::optional<Tensor> c, int64_t k, double beta2,
                    int64_t block, int64_t ns_steps, int64_t ns_dtype, int64_t sinkhorn_iters, double clip,
                    int64_t block_cols, int64_t ns_coeffs, int64_t ns_algo, std::optional<Tensor> row_state,
                    int64_t row_k, double row_beta, double row_eps);
Tensor bl_step_quantized(Tensor packed, std::optional<Tensor> phase, Tensor scales, std::optional<Tensor> global_scale,
                         std::optional<Tensor> zeros, Tensor D, Tensor sorted, Tensor rank_of_code, Tensor sorted_code,
                         int64_t kind, int64_t bits, int64_t cb, int64_t qmin, int64_t qmax, int64_t zero_point,
                         int64_t scale_dt, int64_t group, int64_t rows, int64_t cols, int64_t pb, int64_t pcb,
                         double lr, double decay, int64_t rounding, int64_t seed, int64_t step, int64_t strat_j,
                         int64_t strat_k, int64_t rule, double kappa, int64_t offset);

namespace {
std::optional<Tensor> rows_of(const std::optional<Tensor>& t, int64_t r0, int64_t n) {
  if (!t.has_value() || !t->defined() || t->dim() == 0 || t->numel() <= 1) return t;
  return t->narrow(0, r0, n);
}
}  // namespace

// packed / phase / scales / zeros: the parameter's storage, rows first (updated in place); G: its gradient
// [rows, cols]; direction = (kind, block, block_cols, ns_steps, ns_dtype, ns_coeffs, ns_algo) as in bl_direction;
// tile_rows: rows per tile (a multiple of the Newton-Schulz block rows).
void bl_lattice_update(Tensor packed, std::optional<Tensor> phase, Tensor scales, std::optional<Tensor> global_scale,
                       std::optional<Tensor> zeros, Tensor G, Tensor sorted, Tensor rank_of_code, Tensor sorted_code,
                       int64_t kind, int64_t bits, int64_t cb, int64_t qmin, int64_t qmax, int64_t zero_point,
                       int64_t scale_dt, int64_t group, int64_t rows, int64_t cols, int64_t pb, int64_t pcb, double lr,
                       double decay, int64_t rounding, int64_t seed, int64_t step, int64_t strat_j, int64_t strat_k,
                       int64_t rule, double kappa, int64_t dir_kind, int64_t block, int64_t block_cols,
                       int64_t ns_steps, int64_t ns_dtype, int64_t ns_coeffs, int64_t ns_algo, double clip,
                       int64_t tile_rows) {
  TORCH_CHECK(tile_rows > 0 && G.numel() == rows * cols, "lattice_update: G of rows x cols, positive tile_rows");
  TORCH_CHECK(scales.size(0) == rows, "lattice_update: scales must be rows first");
  const c10::cuda::CUDAGuard guard(packed.device());
  const Tensor G2 = G.reshape({rows, cols});
  for (int64_t r0 = 0; r0 < rows; r0 += tile_rows) {
    const int64_t n = std::min(tile_rows, rows - r0);
    Tensor D = bl_direction(G2.narrow(0, r0, n), dir_kind, std::nullopt, std::nullopt, 1, 0.95, block, ns_steps,
                            ns_dtype, 2, clip, block_cols, ns_coeffs, ns_algo, std::nullopt, 1, 0.95, 1e-8);
    Tensor sc = bl_step_quantized(packed.narrow(0, r0, n), rows_of(phase, r0, n), scales.narrow(0, r0, n),
                                  global_scale, rows_of(zeros, r0, n), D.reshape({-1}).contiguous(), sorted,
                                  rank_of_code, sorted_code, kind, bits, cb, qmin, qmax, zero_point, scale_dt, group, n,
                                  cols, pb, pcb, lr, decay, rounding, seed, step, strat_j, strat_k, rule, kappa,
                                  r0 * cols);
    scales.narrow(0, r0, n).copy_(sc);
  }
}
