# Bitlattice

Training and fine-tuning of PyTorch models directly in low-bit and arbitrary number formats.

Weights stay in their storage format during training (NVFP4, MXFP4, INT2–INT8, FP8, ternary, binary, NF4, any
ExMy float, custom level tables, BF16/FP16/FP32). There is no full-precision master copy: each weight carries a
small phase (0–8 bits) that accumulates sub-step updates, and projections back to the format use unbiased
stochastic rounding. Optimizer state is 0–8 bits per weight instead of 64–96.

Bitlattice extends PyTorch rather than replacing it: a quantized weight is a `torch.Tensor` subclass, models keep
their `nn.Module` code, and the optimizer is a `torch.optim.Optimizer`.

## Install

Requirements: PyTorch ≥ 2.6 with CUDA, a CUDA toolkit matching the PyTorch build, a C++20 host compiler.

```bash
pip install -e . --no-build-isolation            # builds bitlattice._C for TORCH_CUDA_ARCH_LIST (default 8.9;12.0)
pip install -e ".[hf,test]" --no-build-isolation  # + transformers loading, tests
```

## Quick start

```python
import torch
import bitlattice as bl

# load a Hugging Face model straight into NVFP4, keeping 5% of input channels of every layer in BF16
model = bl.io.load("Qwen/Qwen3-0.6B", bl.quant.presets.default("nvfp4", keep=0.05), device="cuda")

# or quantize an existing model in place
bl.quantize_(model, format="int4", keep=0.1, method="mse")

opt = bl.optim.build(model, "muon", lr=3e-3, fused_backward=True, storage="auto")
for batch in data:
    loss = bl.nn.causal_lm_loss(model, batch["input_ids"])   # chunked head + cross-entropy, no full logits
    loss.backward()                                         # parameters are updated during backward
    opt.step()

bl.io.save(model, "out/")                           # native format, reload with bl.io.load("out/")
```

## Formats

`bl.formats.get(name)` accepts `nvfp4`, `mxfp4`, `fp4`, `nf4`, `fp6_e2m3`, `fp6_e3m2`, `fp8`/`fp8_e4m3`, `fp8_e5m2`,
`int1`…`int32`, `uint1`…`uint32` (with per-group zero point), `ternary`, `binary`, `bf16`, `fp16`, `fp32`, `fp64`,
and any `fp<N>_e<E>m<M>`. Custom formats: `bl.formats.table(name, values, group=..., scale=...)` and
`bl.formats.uniform(bits, group=..., symmetric=..., zero_point=...)`.

Codes use the hardware encoding of each format (E2M1 nibbles for fp4/nvfp4/mxfp4, E4M3 bytes for fp8, two's
complement for integers), so NVFP4/FP8 checkpoints load without re-encoding. Scales cover groups along the last
(input) dimension; scale storage is fp32, fp16, bf16, e4m3 (with a per-tensor global scale) or e8m0.

## Quantization

`bl.QuantConfig` fields:

| field | meaning |
|---|---|
| `format` | target format |
| `method` | `rtn` or `mse` (per-group clipping search) |
| `keep` | fraction 0–1 of input channels per layer stored in `keep_dtype` (chosen by saliency) |
| `saliency` | `weight` (column norm) or `activation` (with `bl.quant.calibrate`) |
| `group`, `scale` | override the scale group and scale dtype |
| `targets` | module types to quantize (default `nn.Linear`) |
| `rules` | ordered `Rule(pattern, action)`: glob or `re:` pattern → `"skip"` or field overrides |

Presets: `bl.quant.presets.default(fmt)` (dense head and embeddings), `aggressive(fmt)`,
`accuracy_critical(model, fmt)` (dense first/last blocks, head, embeddings, cross-attention; 8-bit output
projections; BF16 channel budget). Groups adapt to layer width where the format allows it; hardware layouts
(nvfp4, mxfp4) are kept strict and non-conforming layers stay dense.

## Optimizer

`bl.Lattice(params, lr, **options)`; every option can be set per parameter group.

| option | values |
|---|---|
| `direction` | `muon_block` (block-diagonal Newton–Schulz, `block`), `muon`, `factored`, `sinkhorn`, `colnorm`, `sign`, `raw` |
| `vector_direction` | direction for 1-D parameters |
| `phase_bits` | 0–8 |
| `rounding` | `sr`, `strat`, `rtn` |
| `scale_rule` | `fit` (refit group scales to the updated values), `fit_latent`, `fixed` |
| `momentum` | `none`, `fp32`, `ternary` |
| `block` | Newton–Schulz block: int (square), `(rows, cols)`, or `(rows, None)` for row bands over the full width |
| `ns_steps`, `ns_dtype` | iterations; `bf16`, `fp16`, `fp32` |
| `ns_coeffs` | `muon` (fixed quintic), `polar_express` (per-iteration schedule) |
| `ns_algo` | `auto`, `standard`, `gram` (Gram-matrix iteration, fewer FLOPs for wide blocks) |
| `weight_decay`, `clip`, `beta2`, `trust`, `trust_lr`, `strata`, `tile_elems` | see docstrings |

`fused_backward=True` applies each parameter update as soon as its gradient is accumulated and frees the gradient;
`with opt.accumulate(): loss.backward()` sums gradients over micro-batches instead, and the next backward outside
the context applies the update. Matrices larger than `tile_elems` are updated in row tiles.
`bl.optim.build(model, preset)` groups parameters automatically (embeddings and output heads use the factored
direction, vectors are exempt from weight decay).

Presets:

| preset | direction | phase | use |
|---|---|---|---|
| `muon` | block Newton–Schulz | 2 bits | default; 4–16-bit formats |
| `muon_fast` | Gram Newton–Schulz on 256-row bands (fp16, Polar Express) | 2 bits | large models: ~7× cheaper orthogonalization |
| `muon_precise` | block Newton–Schulz | 8 bits | ternary, binary, 2-bit formats |
| `light` | factored | 1 bit | minimal state |
| `balanced` | factored | 2 bits | |
| `precise` | factored | 8 bits | hard formats and tasks |
| `stateless` | Sinkhorn | 0 bits | no optimizer state; formats of ≥ 8 bits |

## Memory

| tool | effect |
|---|---|
| `build(..., storage=...)` | `gpu`; `state_in_ram` (optimizer state in pinned host memory, prefetched per layer); `state_and_acts_in_ram` (also run the forward in `bl.storage.activations_in_ram()`); `auto` |
| `bl.nn.causal_lm_loss`, `bl.nn.linear_cross_entropy` | output layer + cross-entropy in chunks: no logits, no fp32 head gradient |
| `bl.nn.token_chunked_(model, "*mlp", chunk=2048)` | MLPs run on checkpointed sequence slices (long contexts) |
| `model.gradient_checkpointing_enable()` | Hugging Face checkpointing, compatible with all of the above |
| `bl.plan(model, opt)` | weights, optimizer state (GPU / host), gradient peak, weights per device |

## Several GPUs

```python
model = bl.io.load("Qwen/Qwen3-8B", bl.quant.presets.default("nvfp4"), device_map="auto",
                   max_memory={0: "13GiB", 1: "11GiB"})       # contiguous layer split, loaded directly per GPU
# or for a model in memory:  bl.dispatch(model, bl.infer_device_map(model, {0: "12GiB", 1: "12GiB"}))
```

Each GPU holds a contiguous run of layers; activations cross the split points, and every parameter is updated on
its own device during backward. Works with generation, checkpointing and all storage policies.

## Compute

`bl.set_compute(policy)` (or `QuantConfig(compute=...)` per layer, or `with bl.compute(...):`) selects how nvfp4 layers
compute (compute capability ≥ 10.0; other layers and devices fall back to `exact`):

| policy | GEMMs on FP4 tensor cores | linear layer 8192², 8192 tokens, fwd+bwd on RTX 5070 Ti |
|---|---|---:|
| `exact` (default) | none (dequantize to BF16) | 35.3 ms (dense BF16: 34.5 ms) |
| `fp4` | forward | 27.3 ms |
| `fp4_all` | forward, input gradient, weight gradient (stochastic rounding of gradients, Hadamard-rotated weight-gradient operands, layer input saved in FP4) | 16.0 ms |

## Checkpoints

`bl.io.load(path_or_hub_id, quant=None, device=..., dtype=...)` streams safetensors shards module by module.
Pre-quantized layouts are converted losslessly: NVFP4 (compressed-tensors, ModelOpt), FP8 (per-channel,
per-tensor, 128×128 blocks), compressed-tensors int4, GPTQ (without act-order; act-order loads densely), AWQ.
`bl.io.save` writes the native format; `bl.io.export_compressed_tensors` writes NVFP4 layers in the
compressed-tensors layout (experimental).

## Discrete activations

`bl.nn.DiscreteAct(kind, mode)` with kinds `sign`, `step`, `ternary`, `int<k>` and modes `expect` (stochastic
forward, derivative of the expectation), `expect_det`, `ste`, `true`.

## Status

Version 0.1 (development). Implemented: formats, CUDA step, dequantization and fused factored-direction kernels,
tensor subclass, quantization API, optimizer (fused backward, row tiles, sparse embeddings, gradient accumulation),
storage policies, chunked output layer, token-chunked modules, layer split over GPUs, NVFP4 forward on FP4 tensor
cores, checkpoint IO for dense and pre-quantized layouts. Planned: FP4 backward GEMMs, fused dequantize-GEMM for
other formats, pipeline schedule, GGUF. See `docs/DESIGN.md`.

## Tests

```bash
pytest            # kernels against the reference implementation, tensor subclass, quantization, optimizer, IO
python bench/bench_kernels.py                      # kernel timings
python bench/profile_context.py --model <id>       # peak memory and step time versus context length
python bench/optimizer_ablation.py --variant ...   # optimizer options on a small byte-level LM
```

## License

MIT
