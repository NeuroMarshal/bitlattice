// Operator registration: torch.ops.bitlattice.*
#include <torch/extension.h>
#include <torch/library.h>

#include <optional>

using at::Tensor;

Tensor bl_step_quantized(Tensor packed, std::optional<Tensor> phase, Tensor scales, std::optional<Tensor> global_scale,
                         std::optional<Tensor> zeros, Tensor D, Tensor sorted, Tensor rank_of_code, Tensor sorted_code,
                         int64_t kind, int64_t bits, int64_t cb, int64_t qmin, int64_t qmax, int64_t zero_point,
                         int64_t scale_dt, int64_t group, int64_t rows, int64_t cols, int64_t pb, int64_t pcb,
                         double lr, double decay, int64_t rounding, int64_t seed, int64_t step, int64_t strat_j,
                         int64_t strat_k, int64_t rule, double kappa);
void bl_step_native(Tensor w, std::optional<Tensor> phase, Tensor D, int64_t pb, int64_t pcb, double lr, double decay,
                    int64_t rounding, int64_t seed, int64_t step, int64_t strat_j, int64_t strat_k);
Tensor bl_dequantize(Tensor packed, Tensor scales, std::optional<Tensor> global_scale, std::optional<Tensor> zeros,
                     Tensor sorted, Tensor rank_of_code, Tensor sorted_code, int64_t kind, int64_t bits, int64_t cb,
                     int64_t qmin, int64_t qmax, int64_t zero_point, int64_t scale_dt, int64_t group, int64_t rows,
                     int64_t cols, int64_t out_dtype);
Tensor bl_direction(Tensor G, int64_t kind, std::optional<Tensor> r, std::optional<Tensor> c, int64_t k, double beta2,
                    int64_t block, int64_t ns_steps, int64_t ns_dtype, int64_t sinkhorn_iters, double clip);

TORCH_LIBRARY(bitlattice, m) {
  m.def(
      "step_quantized(Tensor(a!) packed, Tensor(b!)? phase, Tensor scales, Tensor? global_scale, Tensor? zeros, "
      "Tensor D, Tensor sorted, Tensor rank_of_code, Tensor sorted_code, int kind, int bits, int cb, int qmin, "
      "int qmax, int zero_point, int scale_dt, int group, int rows, int cols, int pb, int pcb, float lr, "
      "float decay, int rounding, int seed, int step, int strat_j, int strat_k, int rule, float kappa) -> Tensor");
  m.def(
      "step_native(Tensor(a!) w, Tensor(b!)? phase, Tensor D, int pb, int pcb, float lr, float decay, int rounding, "
      "int seed, int step, int strat_j, int strat_k) -> ()");
  m.def(
      "dequantize(Tensor packed, Tensor scales, Tensor? global_scale, Tensor? zeros, Tensor sorted, "
      "Tensor rank_of_code, Tensor sorted_code, int kind, int bits, int cb, int qmin, int qmax, int zero_point, "
      "int scale_dt, int group, int rows, int cols, int out_dtype) -> Tensor");
  m.def(
      "direction(Tensor G, int kind, Tensor(a!)? r, Tensor(b!)? c, int k, float beta2, int block, int ns_steps, "
      "int ns_dtype, int sinkhorn_iters, float clip) -> Tensor");
}

TORCH_LIBRARY_IMPL(bitlattice, CUDA, m) {
  m.impl("step_quantized", &bl_step_quantized);
  m.impl("step_native", &bl_step_native);
  m.impl("dequantize", &bl_dequantize);
}

TORCH_LIBRARY_IMPL(bitlattice, CompositeExplicitAutograd, m) { m.impl("direction", &bl_direction); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.doc() = "bitlattice native operators"; }
