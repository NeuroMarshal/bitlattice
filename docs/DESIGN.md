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
* `direction`: ATen implementation (Newton–Schulz batched over blocks, factored, Sinkhorn, …).
* Compiled with `--fmad=false` so that kernels and the reference (`bitlattice/reference.py`) agree bit for bit
  except where a group sum order decides a scale step at a numerical tie.

## Integration with PyTorch

`LatticeTensor` is a wrapper subclass (`_make_wrapper_subclass`) holding `packed`, `scales`, `global_scale`,
`zeros`, and optional high-precision columns `hi_idx`/`hi_val`. `F.linear` and `F.embedding` have dedicated
autograd functions (no dense weight kept between forward and backward; embeddings dequantize only referenced
rows). All other operators dequantize and run the dense operator; autograd then yields a dense gradient of the
logical shape, which the optimizer consumes. `__tensor_flatten__`/`__tensor_unflatten__` are defined.

## Loading

`bl.io.load` builds the model with `accelerate.init_empty_weights(include_buffers=False)`, groups checkpoint
tensors by module, converts pre-quantized layouts losslessly and quantizes dense weights on the target device, one
module at a time.

## Planned

1. Fused factored-direction kernel (row/column statistics and on-the-fly `U` inside the step kernel).
2. Fused dequantize-GEMM and FP8/FP4 tensor-core paths; selection of the widest exact hardware block per format.
3. Optimizer state and activation checkpoints in host memory with prefetch (storage policies).
4. Row-sparse embedding gradients; vocabulary-tiled output head.
5. Pipeline parallelism with per-layer local updates.
