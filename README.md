# Bitlattice

Training and fine-tuning of PyTorch models with the weights kept in a low-bit format the whole time.

## Why

Low-bit training usually keeps a full-precision master copy of every weight next to the optimizer's moments and
only casts to the low-bit format for the forward pass. The copy and the moments take 12 to 16 bytes per weight,
far more than the 4-bit weight itself, and they decide how large a model fits on a GPU.

Bitlattice has no master copy. Weights stay in their storage format during training (NVFP4, MXFP4, INT2 to INT8,
FP8, ternary, binary, NF4, any ExMy float, custom level tables, BF16/FP16/FP32). Each weight carries a small
phase of 0 to 8 bits that accumulates updates smaller than one step of the format, and projections back to the
format use unbiased stochastic rounding. Optimizer state is 0 to 8 bits per weight instead of 64 to 96.

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
| `keep` | fraction from 0 to 1 of input channels per layer stored in `keep_dtype` (chosen by saliency) |
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
| `direction` | `muon_block` (block-diagonal Newton-Schulz, `block`), `muon`, `factored`, `sinkhorn`, `colnorm`, `sign`, `raw` |
| `vector_direction` | direction for 1-D parameters |
| `phase_bits` | 0 to 8 |
| `rounding` | `sr`, `strat`, `rtn` |
| `scale_rule` | `fit` (refit group scales to the updated values), `fit_latent`, `fixed` |
| `momentum` | `none`, `fp32`, `bf16`, `int8`, `fp8`, `nvfp4`, `ternary` |
| `normalize` | `none`, `row` (NorMuon: each row of the direction divided by a running RMS of the row) |
| `block` | Newton-Schulz block: int (square), `(rows, cols)`, or `(rows, None)` for row bands over the full width |
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
| `muon` | block Newton-Schulz | 2 bits | default; 4- to 16-bit formats |
| `muon_fast` | Gram Newton-Schulz on 256-row bands (fp16, Polar Express) | 2 bits | large models: ~7× cheaper orthogonalization |
| `normuon`, `normuon_fast` | as `muon` and `muon_fast`, rows normalized (NorMuon) | 2 bits | evens out step sizes between neurons |
| `muon_precise` | block Newton-Schulz | 8 bits | ternary, binary, 2-bit formats |
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
block i+1 of micro-batch m-1. Each block's gradient is accumulated over the micro-batches and applied at once:
one block of gradients in memory, the same update as the full batch.

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

Version 0.2.0. Implemented:
- formats; CUDA kernels for the optimizer step, dequantization, NVFP4 / MXFP8 quantization (RTN, SR, MS-EDEN, four
  over six, Hadamard rotations), RMSNorm, causal convolution, FP8 packing;
- tensor subclass and quantization API; the operating table (`bl.surgery`) with reversible module conversions;
- optimizer with presets (Muon variants, NorMuon, low-bit momentum), fused backward, host-memory state;
- FP4 training recipes (NVIDIA, four over six, Quartet II) with deferred weight gradients; MXFP8 path;
- long context: chunked loss, token / attention / linear-attention pieces, recomputation with compressed saves,
  weights streamed from host memory, frozen embedding and output layer in host memory;
- `bl.Kernels`: every automatic implementation choice can be switched off; `bl.explain`;
- layer split, one-process-per-GPU pipeline (`bl.Runner`);
- checkpoint IO for dense and pre-quantized layouts; quantized KV cache.

See `docs/API.md` and `docs/DESIGN.md`.

## Tests

```bash
pytest            # kernels against the reference implementation, tensor subclass, quantization, optimizer, IO,
                  # training a small model end to end
python bench/bench_kernels.py                      # kernel timings
python bench/bench_fp4_layer.py                    # one NVFP4 linear layer per compute policy and FP4 recipe
python bench/optimizer_ablation.py --variant ...   # optimizer options on a small byte-level LM
```

The tests build small models from random weights and download nothing. Tests of layer splits and the pipeline
use two GPUs when there are two and run on one otherwise; tests of FP4 and MXFP8 GEMMs need GPUs that have them
(compute capability 10.0 or 12.0) and are skipped elsewhere.

## License

MIT

This is a personal project that I build for my own use, with a great deal of help from AI. Much of the code and
nearly all of the documentation were written with AI, so the documentation may contain mistakes; the code and its
tests are the reference.
