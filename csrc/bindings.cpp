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
                         int64_t strat_k, int64_t rule, double kappa, int64_t offset);
void bl_step_native(Tensor w, std::optional<Tensor> phase, Tensor D, int64_t pb, int64_t pcb, double lr, double decay,
                    int64_t rounding, int64_t seed, int64_t step, int64_t strat_j, int64_t strat_k, int64_t offset);
Tensor bl_dequantize(Tensor packed, Tensor scales, std::optional<Tensor> global_scale, std::optional<Tensor> zeros,
                     Tensor sorted, Tensor rank_of_code, Tensor sorted_code, int64_t kind, int64_t bits, int64_t cb,
                     int64_t qmin, int64_t qmax, int64_t zero_point, int64_t scale_dt, int64_t group, int64_t rows,
                     int64_t cols, int64_t out_dtype);
Tensor bl_factored_direction(Tensor G, Tensor r, Tensor c, int64_t k, double beta2, double clip);
std::tuple<Tensor, Tensor> bl_factored_scales(Tensor G, Tensor r, Tensor c, int64_t k, double beta2, double clip);
Tensor bl_factored_apply(Tensor G, Tensor a, Tensor b, int64_t row0, int64_t nrows);
std::tuple<Tensor, Tensor> bl_quantize_nvfp4(Tensor x, Tensor global_scale, int64_t seed, int64_t step);
std::tuple<Tensor, Tensor, Tensor> bl_quantize_nvfp4_t(Tensor x, std::optional<Tensor> hadamard, int64_t seed,
                                                       int64_t step, std::optional<Tensor> global_scale);
std::tuple<Tensor, Tensor, Tensor> bl_requantize_nvfp4_t(Tensor packed, Tensor scales, Tensor global_scale);
Tensor bl_scale_(Tensor y, Tensor alpha);
std::tuple<Tensor, Tensor, Tensor> bl_nvfp4_quantize(Tensor x, bool transpose, int64_t mode, bool four_over_six,
                                                     bool mae, bool rotate, int64_t seed,
                                                     std::optional<Tensor> global_scale,
                                                     std::optional<Tensor> w_scales, std::optional<Tensor> w_global,
                                                     int64_t rotate_seed, bool blocked);
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> bl_nvfp4_quantize2(Tensor x, std::vector<int64_t> row,
                                                                             std::vector<int64_t> col, bool mae,
                                                                             std::optional<Tensor> global_scales,
                                                                             bool blocked,
                                                                             std::optional<Tensor> amax_out);
Tensor bl_direction(Tensor G, int64_t kind, std::optional<Tensor> r, std::optional<Tensor> c, int64_t k, double beta2,
                    int64_t block, int64_t ns_steps, int64_t ns_dtype, int64_t sinkhorn_iters, double clip,
                    int64_t block_cols, int64_t ns_coeffs, int64_t ns_algo, std::optional<Tensor> row_state,
                    int64_t row_k, double row_beta, double row_eps);

TORCH_LIBRARY(bitlattice, m) {
  m.def(
      "step_quantized(Tensor(a!) packed, Tensor(b!)? phase, Tensor scales, Tensor? global_scale, Tensor? zeros, "
      "Tensor D, Tensor sorted, Tensor rank_of_code, Tensor sorted_code, int kind, int bits, int cb, int qmin, "
      "int qmax, int zero_point, int scale_dt, int group, int rows, int cols, int pb, int pcb, float lr, "
      "float decay, int rounding, int seed, int step, int strat_j, int strat_k, int rule, float kappa, "
      "int offset=0) -> Tensor");
  m.def(
      "step_native(Tensor(a!) w, Tensor(b!)? phase, Tensor D, int pb, int pcb, float lr, float decay, int rounding, "
      "int seed, int step, int strat_j, int strat_k, int offset=0) -> ()");
  m.def(
      "dequantize(Tensor packed, Tensor scales, Tensor? global_scale, Tensor? zeros, Tensor sorted, "
      "Tensor rank_of_code, Tensor sorted_code, int kind, int bits, int cb, int qmin, int qmax, int zero_point, "
      "int scale_dt, int group, int rows, int cols, int out_dtype) -> Tensor");
  m.def(
      "direction(Tensor G, int kind, Tensor(a!)? r, Tensor(b!)? c, int k, float beta2, int block, int ns_steps, "
      "int ns_dtype, int sinkhorn_iters, float clip, int block_cols=0, int ns_coeffs=0, int ns_algo=0, "
      "Tensor(c!)? row_state=None, int row_k=1, float row_beta=0.95, float row_eps=1e-8) -> Tensor");
  m.def("factored_direction(Tensor G, Tensor(a!) r, Tensor(b!) c, int k, float beta2, float clip) -> Tensor");
  m.def("factored_scales(Tensor G, Tensor(a!) r, Tensor(b!) c, int k, float beta2, float clip) -> (Tensor, Tensor)");
  m.def("factored_apply(Tensor G, Tensor a, Tensor b, int row0, int nrows) -> Tensor");
  m.def("quantize_nvfp4(Tensor x, Tensor global_scale, int seed=-1, int step=0) -> (Tensor, Tensor)");
  m.def("quantize_nvfp4_t(Tensor x, Tensor? hadamard, int seed=-1, int step=0, Tensor? global_scale=None) -> "
        "(Tensor, Tensor, Tensor)");
  m.def("requantize_nvfp4_t(Tensor packed, Tensor scales, Tensor global_scale) -> (Tensor, Tensor, Tensor)");
  m.def("scale_(Tensor(a!) y, Tensor alpha) -> Tensor(a!)");
  m.def("nvfp4_quantize(Tensor x, bool transpose=False, int mode=0, bool four_over_six=False, bool mae=False, "
        "bool rotate=False, int seed=0, Tensor? global_scale=None, Tensor? w_scales=None, Tensor? w_global=None, "
        "int rotate_seed=-1, bool blocked=False) "
        "-> (Tensor, Tensor, Tensor)");
  m.def("nvfp4_quantize2(Tensor x, int[] row, int[] col, bool mae=False, Tensor? global_scales=None, "
        "bool blocked=False, Tensor(a!)? amax_out=None) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(bitlattice, CUDA, m) {
  m.impl("step_quantized", &bl_step_quantized);
  m.impl("step_native", &bl_step_native);
  m.impl("dequantize", &bl_dequantize);
  m.impl("factored_direction", &bl_factored_direction);
  m.impl("factored_scales", &bl_factored_scales);
  m.impl("factored_apply", &bl_factored_apply);
  m.impl("quantize_nvfp4", &bl_quantize_nvfp4);
  m.impl("quantize_nvfp4_t", &bl_quantize_nvfp4_t);
  m.impl("requantize_nvfp4_t", &bl_requantize_nvfp4_t);
  m.impl("scale_", &bl_scale_);
  m.impl("nvfp4_quantize", &bl_nvfp4_quantize);
  m.impl("nvfp4_quantize2", &bl_nvfp4_quantize2);
}

TORCH_LIBRARY_IMPL(bitlattice, CompositeExplicitAutograd, m) { m.impl("direction", &bl_direction); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.doc() = "bitlattice native operators"; }
