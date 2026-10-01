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
pip install -e . --no-build-isolation            # builds bitlattice._C for the visible GPUs (or TORCH_CUDA_ARCH_LIST)
pip install -e ".[hf,test]" --no-build-isolation  # + transformers loading, tests
```

## Quick start

```python
import torch
import bitlattice as bl

# load a Hugging Face model straight into NVFP4, keeping 5% of input channels of every layer in BF16
model = bl.io.load("Qwen/Qwen3-0.6B", bl.quant.presets.default("nvfp4", keep=0.05), device="cuda")

# or quantize layers of an existing model in place
bl.surgery.quantize_(model, "*", bl.QuantConfig(format="int4", keep=0.1, method="mse"))

opt = bl.optim.Lattice(model, "muon", lr=3e-3, fused_backward=True)
runner = bl.Runner(model, opt, micro=2)
for batch in data:
    loss = runner.step(batch["input_ids"])   # micro-batches, chunked head + cross-entropy, update during backward

bl.io.save(model, "out/")                    # native format, reload with bl.io.load("out/")
```

Every change to a model is an explicit call; nothing converts or compresses on its own (`docs/API.md`):

```python
S = bl.surgery
S.convert_(model, "*norm", "rms_norm", skip_unfit=True)            # lean RMSNorm where it reproduces the original
S.convert_(model, "*.mlp", "token_chunks", chunk=2048)              # MLPs over slices of the sequence
S.convert_(model, "re:layers\.\d+$", "checkpoint", saved="fp8")   # recompute layers, recomputed saves in FP8
S.set_compute_(model, "*", "fp4_all", skip_unfit=True)              # FP4 tensor cores where the layer fits
print(bl.explain(model))                                            # format, compute, conversions, memory per layer
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
`bl.optim.Lattice(model, preset)` groups parameters automatically (embeddings and output heads use the factored
direction, vectors are exempt from weight decay); options given next to the preset override it.

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
| `Lattice(..., state=...)` | optimizer state: `gpu`; `ram` (pinned host memory, prefetched per layer); `auto` |
| `Runner(..., memory=bl.Memory(activations="ram"))` | tensors kept for backward live in pinned host memory |
| `bl.nn.causal_lm_loss`, `bl.nn.linear_cross_entropy` | output layer + cross-entropy in chunks: no logits, no fp32 head gradient |
| `convert_(model, sel, "checkpoint", saved=None \| "fp8" \| "nvfp4")` | recompute the selected modules in backward; tensors saved during recomputation optionally packed |
| `convert_(model, "*.mlp", "token_chunks", chunk=2048)` | MLPs run on checkpointed sequence slices (long contexts) |
| `convert_(model, "*.linear_attn", "gated_delta_net", seq_chunk=4096)` | Gated DeltaNet without copies of q/k per value head; recurrence over sequence pieces |
| `convert_(model, "*.self_attn", "attention_chunks", chunk=4096)` | causal attention over query pieces: the layer keeps its input, K and V; each piece recomputed in backward |
| `bl.plan(model, opt)` | weights, optimizer state (GPU / host), gradient peak, weights per device |

## Several GPUs

**Pipeline, one process per GPU** (fastest; for decoder-only Hugging Face models):

```python
# torchrun --nproc-per-node 2 train.py      (every process runs this script on the same batches)
rank, world = bl.parallel.init()
model = bl.io.load("Qwen/Qwen3-8B", bl.quant.presets.default("nvfp4"), device_map="interleaved")
bl.surgery.convert_(model, "re:layers\.\d+$", "checkpoint")
opt = bl.optim.Lattice(model, "muon_fast", lr=2e-5, fused_backward=True)
runner = bl.Runner(model, opt, micro=4)        # with a process group: the pipeline below
for batch in data:
    loss = runner.step(batch)                  # forward + backward + update
```

Blocks alternate between the GPUs, and each process loads only its own. Every block processes all micro-batches
before the next one, forward and backward, so one GPU works on block i of micro-batch m while the other works on
block i+1 of micro-batch m−1. Each block's gradient is accumulated over the micro-batches and applied at once:
one block of gradients in memory, the same update as the full batch.

Qwen3-8B NVFP4, 8 × 1024 tokens per step, two RTX 5070 Ti:

| layout | tokens/s |
|---|---:|
| one GPU | 1 086 |
| contiguous split, one process | 818 |
| pipeline, two processes, 4 micro-batches | 1 729 |
| same with `fp4_all` compute | 1 953 |

**Single process** (any model, simplest):

```python
model = bl.io.load("Qwen/Qwen3-8B", bl.quant.presets.default("nvfp4"), device_map="auto",
                   max_memory={0: "13GiB", 1: "11GiB"})       # contiguous layer split, loaded directly per GPU
# or for a model in memory:  bl.dispatch(model, bl.infer_device_map(model, {0: "12GiB", 1: "12GiB"}))
```

Each GPU holds a contiguous run of layers, and the GPUs work one after the other. Use it to fit a model, not for
speed. Works with generation, checkpointing and every memory placement.

## Compute

How a quantized layer computes is a property of the layer: `bl.surgery.set_compute_(model, selector, compute)` with
a preset name (`bl.COMPUTE_PRESETS`) or a `bl.Compute(gemm=..., backward=..., nvfp4=bl.backend.Fp4Recipe(...),
kernels=...)`; `with bl.compute(...):` overrides it temporarily. A setting the layer cannot honour (no FP4 tensor
cores, dimensions that do not fit) is an error; `skip_unfit=True` sets it where possible and logs the rest. Layers
without a setting compute exactly, on the fastest kernels that give the same result.

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

Version 0.1.0. Implemented:
- formats; CUDA kernels for the step, dequantization, fused factored direction and NVFP4 activation quantization;
- tensor subclass and quantization API;
- optimizer with presets: fused backward, row tiles, sparse embeddings, gradient accumulation, Newton–Schulz
  variants;
- optimizer state and activation placement; chunked output layer; module conversions (operating table);
- layer split and one-process-per-GPU pipeline;
- FP4 training path (`fp4`, `fp4_all`);
- checkpoint IO for dense and pre-quantized layouts.

Verified by fine-tuning Qwen3-0.6B and Qwen3-8B checkpoints. Planned: fused dequantize-GEMM for formats without a
hardware path, kernels for linear-attention layers of hybrid models, GGUF. See `docs/DESIGN.md`.

## Tests

```bash
pytest            # kernels against the reference implementation, tensor subclass, quantization, optimizer, IO
python bench/bench_kernels.py                      # kernel timings
python bench/profile_context.py --model <id>       # peak memory and step time versus context length
python bench/optimizer_ablation.py --variant ...   # optimizer options on a small byte-level LM
```

## License

MIT
