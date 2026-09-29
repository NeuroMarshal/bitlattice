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

opt = bl.optim.build(model, "muon", lr=3e-3)       # presets: muon, muon_precise, light, balanced, precise, stateless
for batch in data:
    loss = model(**batch).loss
    loss.backward()
    opt.step()
    opt.zero_grad()

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
| `weight_decay`, `clip`, `beta2`, `ns_steps`, `ns_dtype`, `trust`, `trust_lr`, `strata` | see docstrings |

`fused_backward=True` applies each parameter update as soon as its gradient is accumulated and frees the gradient.
`bl.optim.build(model, preset)` groups parameters automatically (embeddings and output heads use the factored
direction, vectors are exempt from weight decay).

Presets:

| preset | direction | phase | use |
|---|---|---|---|
| `muon` | block Newton–Schulz | 2 bits | default; 4–16-bit formats |
| `muon_precise` | block Newton–Schulz | 8 bits | ternary, binary, 2-bit formats |
| `light` | factored | 1 bit | minimal state |
| `balanced` | factored | 2 bits | |
| `precise` | factored | 8 bits | hard formats and tasks |
| `stateless` | Sinkhorn | 0 bits | no optimizer state; formats of ≥ 8 bits |

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

Version 0.1 (development). Implemented: formats, CUDA step and dequantization kernels, tensor subclass,
quantization API, optimizer, checkpoint IO. Not yet: fused factored-direction kernel, fused low-bit GEMM paths
(FP8/FP4 tensor cores), optimizer state offload to host memory, pipeline parallelism. See `docs/DESIGN.md`.

## Tests

```bash
pytest            # kernels against the reference implementation, tensor subclass, quantization, optimizer, IO
python bench/bench_kernels.py
```

## License

MIT
