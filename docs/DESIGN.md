# Design

## Weight representation

A weight is viewed as a 2-D matrix `[rows, cols]` (`cols` = product of trailing dimensions). Its represented
values form an ordered set (lattice):

* table formats: `value = level[code] · scale_g` (codes ≤ 8 bits, levels from a lookup table);
* uniform formats: `value = (k − zero_g) · scale_g`, `k ∈ [qmin, qmax]`;
* native formats: all finite values of a torch floating dtype, ordered by bit pattern.

`scale_g` covers `group` consecutive elements of a row (or a whole row, or the whole tensor) and is stored in
fp32/fp16/bf16/e4m3/e8m0, optionally multiplied by a per-tensor fp32 global scale.

Optimizer state per element: a `b`-bit phase code `c ∈ [−2^(b−1), 2^(b−1))`, `ĉ = c / 2^(b−1)`. The latent
position is `λ = w + ĉ·(hi − w)/2` for `ĉ ≥ 0` and `λ = w + ĉ·(w − lo)/2` for `ĉ < 0`, where `lo < w < hi` are
the neighbouring lattice values. `ĉ = 0` is the represented value.

## Update

For a parameter with gradient `G`:

1. `m = momentum(G)` — none, fp32 EMA, or ternary codes with a per-row scale;
2. `U = direction(m)` — Newton–Schulz orthogonalization (optionally on `b×b` blocks), factored second moment
   (row/column EMAs), Sinkhorn, column normalization, sign or identity; `D = U / max(1, RMS(U)/clip)`;
3. `t = λ·(1 − lr·wd) − lr·D` (in units of the current group scale);
4. scale refit (table/uniform formats): with `a` the represented levels and `τ = a·(1 − lr·wd) − lr·D/s`,
   `r = Σaτ / Σa²`, clamped to `[e^−κ, e^κ]` with `κ = min(trust, trust_lr·lr)`, rounded to the scale dtype and
   accepted only if `Σ(r'a − τ)² ≤ Σ(a − τ)²` after rounding. Fitting to the represented values (not to `λ`) avoids
   the systematic shrinkage caused by the one-sided phase at the ends of the range;
5. projection: `t ← clamp(t)`, `w' = nearest(t)`, phase `ĉ' = (t − w')/half-width` rounded stochastically on the
   `2^(b−1)` sub-grid (`b = 0`: stochastic rounding between the two neighbours). `E[λ'] = t`.

Random thresholds come from a counter-based hash of `(seed + 7919·param_id, step, element index)`; stratified
rounding uses `frac((bitrev(step mod K) + h_i)/K)`.

## Kernels

* `step_quantized`: one kernel for all table/uniform formats; format parameters are passed at run time (uniform
  branches, memory-bound). Each thread owns 8 consecutive elements (whole packed bytes). Groups of ≤ 256 elements
  (power of two) reduce within a warp segment (butterfly, deterministic); per-row, per-tensor and large groups use
  one block per group. Fixed scales use a purely element-wise kernel.
* `step_native`: bf16/fp16 computed in fp32, fp32 in fp64; neighbours from the ordinal bit pattern.
* `dequantize`: packed codes → bf16/fp16/fp32.
* `factored_direction` / `factored_scales` / `factored_apply`: fused factored direction in three passes (row and
  column sums of squares into deterministic partial buffers, then the scaled direction); the scales/apply split
  lets large matrices be updated in row tiles with the same result as the untiled update.
* `direction`: ATen implementation of the other directions. Newton–Schulz runs batched over blocks with fused
  `baddbmm` iterations; options: coefficient schedule (`muon` fixed quintic, or `polar_express`: per-iteration
  minimax schedule of Amsel et al. 2025 with a 1.05 safety factor), block shape (square or `(rows, cols)`), dtype
  (bf16, fp16, fp32) and algorithm (`standard`, or `gram`: iterate on the Gram matrix `R = XXᵀ` and accumulate the
  polynomial, restarted after two steps, after Zhang, Amsel, Chen and Dao 2026 — mathematically the same result,
  fewer FLOPs for wide blocks; `auto` uses it for aspect ≥ 2 outside bf16).
* `quantize_nvfp4`: activations → E2M1 codes with E4M3 block scales (16 elements) and a per-tensor fp32 global
  scale; ties go to the lower value, as everywhere in the library.
* Compiled with `--fmad=false` so that kernels and the reference (`bitlattice/reference.py`) agree bit for bit
  except where a group sum order decides a scale step at a numerical tie.

## Integration with PyTorch

`LatticeTensor` is a wrapper subclass (`_make_wrapper_subclass`) holding `packed`, `scales`, `global_scale`,
`zeros`, and optional high-precision columns `hi_idx`/`hi_val`. `F.linear` and `F.embedding` have dedicated
autograd functions (no dense weight kept between forward and backward; embeddings dequantize only referenced
rows). All other operators dequantize and run the dense operator; autograd then yields a dense gradient of the
logical shape, which the optimizer consumes. `__tensor_flatten__`/`__tensor_unflatten__` are defined.

## Compute policies

`exact` dequantizes to the activation dtype and runs a dense GEMM. `fp4` (nvfp4 weights, compute capability ≥ 10)
quantizes activations to NVFP4 on the fly and runs the forward GEMM on FP4 tensor cores (`torch._scaled_mm` with
block scales in the 128×4 tiled layout); the backward stays exact. The policy is set globally, per layer
(`QuantConfig.compute`) or for a region (`bl.compute(...)`, process-wide so that the backward and checkpoint
recomputation on autograd threads see it).

## Memory

* Optimizer in backward (`fused_backward`): each parameter is updated when its gradient is complete and the gradient
  is freed; `Lattice.accumulate()` defers updates for gradient accumulation.
* Storage policies: optimizer state in pinned host memory, brought in on a side stream in the recorded update order
  (one stream and one order per device); tensors saved for backward in host memory (`save_on_cpu`).
* Output layer + cross-entropy (`bl.nn.linear_cross_entropy`): token chunks forward, keeping only the log-sum-exp;
  backward over vocabulary tiles × token chunks, so neither logits nor an fp32 copy of the head gradient exist.
* Token-chunked position-wise modules (`bl.nn.token_chunked_`): MLPs run on sequence slices, each checkpointed,
  so the intermediate activations of one slice exist at a time.
* Row-sparse embedding gradients (only referenced rows are materialized and updated).

## Multiple GPUs

`bl.parallel.infer_device_map` splits the model into contiguous runs of blocks (the largest `ModuleList`) over the
GPUs, minimizing the largest ratio of assigned bytes to the device budget; tied parameters stay together.
`bl.dispatch` moves the modules and installs pre-hooks that move inputs to each module's device, so forward,
autograd, generation and the optimizer work unchanged; updates stay local to each device. `bl.io.load(...,
device_map="auto")` sizes the split from the checkpoint headers and loads every tensor directly onto its GPU.

## Pipeline (one process per GPU)

`bl.parallel.init` + `bl.io.load(..., device_map="interleaved")`: every process loads only its blocks.
`bl.parallel.train_step` cuts the model into segments (maximal runs of embedding / blocks / head on one GPU). It runs
them block-major over the micro-batches, forward and backward, sending each micro-batch on as soon as it is done.
There is one NCCL process group per direction, because NCCL P2P streams are in order. A segment's gradients are
accumulated over the micro-batches (`Lattice.accumulate`) and applied by the fused update after the last one.

## Loading

`bl.io.load` builds the model with `accelerate.init_empty_weights(include_buffers=False)`, groups checkpoint
tensors by module, converts pre-quantized layouts losslessly and quantizes dense weights on the target device, one
module at a time.

## Planned

1. 2-D 16×16 weight blocks, so that `W` and `Wᵀ` share codes in `fp4_all`. Today the input-gradient GEMM uses a
   1×16 re-quantization of `Wᵀ`.
2. Fused dequantize-GEMM for formats without a hardware path (level table in shared memory); FP8 path for int4/int5.
3. Symmetric GEMM kernels for Newton–Schulz (`XXᵀ`, `R²` at half the FLOPs).
4. Pipeline:
   - a split backward (input gradient first, weight gradient in the bubbles);
   - a copy-engine transport without head-of-line blocking (~18 GB/s instead of ~5 GB/s).
5. Kernels for linear-attention layers of hybrid models.
6. GGUF loading.
