# Bitlattice

> This document was written with AI and may contain mistakes. The code and
> its tests are the reference.

Bitlattice is an extension of PyTorch for training and fine-tuning models whose weights stay in a low-bit format
the whole time. It works on ordinary `nn.Module` models and loads Hugging Face safetensors checkpoints, including
ones already quantized with NVIDIA ModelOpt, compressed-tensors, GPTQ or AWQ, and trains all of their weights in
place.

## Why

The goal is a full fine-tune of a large language model on consumer GPUs: every weight trained, not a low-rank
adapter on top of frozen weights. In short, a model that runs on your GPUs for inference can be trained on them too.

Low-bit training usually keeps a full-precision master copy of every weight next to the optimizer's moments and only
casts to the low-bit format for the forward pass. The copy and the moments take 12 to 16 bytes per weight, far more
than a 4-bit weight, and they, not the weights, decide how large a model fits.

Bitlattice has no master copy. Weights stay in their storage format during training, and the optimizer keeps 0 to 8
bits of state per weight instead of 64 to 96. With that state in host memory, or with the `stateless` preset that
keeps none (for formats of 8 bits and more), and with each gradient freed as soon as its update is applied, the GPU
holds little more than for inference: the weights in their format and the activations of the step, which can be
recomputed or moved to host memory as well. That was enough for a full fine-tune of Qwen3-8B with NVFP4 weights on
one 16 GB RTX 5070 Ti: the optimizer state took 1.9 GiB, where AdamW would need about 98 GiB for fp32 master weights
and moments.

Bitlattice also trains Qwen3.8-27B, a hybrid model of the Qwen3.5 family (48 Gated DeltaNet and 16 attention
blocks), stably in mixed precision with 8192-token sequences on two RTX 5070 Ti with 16 GB each. The weights stay in
NVFP4 (a ModelOpt checkpoint, 18.3 GiB over the two GPUs), the matrix multiplications run on FP4 tensor cores with
the `quartet2_fast` recipe, the embeddings and the output head stay in BF16 and frozen, and the optimizer state
lives in host memory. In that setup steps of 2 × 8192 tokens ran at about 950 tokens/s with 13.5 and 12.9 GiB in use
on the two GPUs, steps of 8 × 1024 tokens at about 1 100 tokens/s, and contexts of 16k and 32k tokens trained as
well. In a short run the held-out perplexity fell from 3.70 to 3.49.

These figures come from development runs on that server; the tests do not reproduce them.

## Status

Bitlattice is a research prototype at version 0.2.1.

- Stability is not guaranteed. APIs, file formats and numerical results may change between versions, and there are
  bugs nobody has found yet.
- It was developed and tested on a server with two RTX 5070 Ti (compute capability 12.0) and on an RTX 4060
  Laptop GPU (8.9). Other GPUs, including data-center Blackwell, Hopper and older generations, have not been
  tested, and neither has more than one machine.
- Most of the work went into NVFP4: its kernels, the FP4 training recipes and FP4 tensor cores of consumer
  Blackwell cards. The other formats work, but with fewer measurements behind them.
- Training across GPUs was tested with decoder-only Hugging Face models (Qwen3 and the Qwen3.5 family).

## How it works

- A quantized weight is `LatticeTensor`, a `torch.Tensor` subclass holding packed codes, group scales
  and an optional per-tensor scale. Linear and embedding layers have their own autograd functions, so no dense copy
  of the weight is kept between forward and backward; any other operation dequantizes and runs the dense operator.
- Each weight carries a phase of 0 to 8 bits: its position between the two neighbouring values the
  format can represent. Updates smaller than one step of the format accumulate in the phase, and the projection
  back to the format rounds stochastically, so on average a weight lands where the update put it. Group scales are
  refit to the updated values within a trust region.
- The update direction comes from orthogonalization (Newton-Schulz, as in Muon), a factored second
  moment (as in Adafactor), Sinkhorn normalization, column normalization or the sign, with optional low-bit
  momentum.
- By default a layer dequantizes to BF16 and computes exactly. On request it runs its GEMMs on FP4
  tensor cores, with one of several FP4 training recipes, or on FP8 tensor cores through MXFP8 for any weight
  format.
- Optimizer state and activations can live in host memory, layers can be recomputed in backward with
  their saved tensors compressed, and the loss, MLPs, attention and linear attention can run over pieces of the
  sequence.
- A model can be split over GPUs layer by layer, or trained in a pipeline with one process per
  GPU.

`docs/DESIGN.md` gives the formulas and the kernels, and `docs/API.md` the principles of the interface.

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

On one Linux host, `bl.parallel.init(transport="host")` selects copy-engine transfers through shared pinned memory.
`bl.parallel.share_head_(model)` can distribute the loss of a frozen, bias-free output head across the processes.
Both are collective choices. See [Execution](docs/API.md#execution) for their requirements.

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

## Tests

```bash
pytest
```

The suite has 696 cases in 26 files. They check the CUDA kernels against reference implementations in Python
(lattice steps across formats and phase widths, NVFP4 and MX quantization, RMSNorm, causal convolution), the
tensor subclass inside ordinary PyTorch code, quantization rules and presets, the optimizer training in quantized
and native formats, checkpoint loading for dense and pre-quantized layouts, layer conversions, the chunked loss,
compressed and offloaded activations, the quantized KV cache, layer splits, the pipeline, and a small language
model trained from scratch in a low-bit format.

The tests build small models from random weights and download nothing. Layer splits and the pipeline use two GPUs
when there are two and run on one otherwise. Tests of FP4 and MXFP8 GEMMs need GPUs that have them (compute
capability 10.0 or 12.0) and are skipped elsewhere: on the RTX 4060 Laptop GPU 637 cases pass and 59 are skipped.

The benchmarks in `bench/` time the kernels and one NVFP4 linear layer and compare optimizer options on a small
byte-level language model:

```bash
python bench/bench_kernels.py                      # kernel timings
python bench/bench_fp4_layer.py                    # one NVFP4 linear layer per compute policy and FP4 recipe
python bench/optimizer_ablation.py --variant ...   # optimizer options on a small byte-level LM
```

## Research it builds on

- Muon (Jordan et al., 2024): orthogonalized updates with Newton-Schulz iterations; its quintic coefficients are
  one of the schedules.
- Polar Express (Amsel et al., 2025): the per-iteration minimax Newton-Schulz schedule.
- Newton-Schulz on the Gram matrix (Zhang, Amsel, Chen and Dao, 2026): the `gram` algorithm for wide blocks.
- NorMuon (arXiv 2510.05491): row normalization of the orthogonalized direction.
- arXiv 2509.23106: Muon keeps working with 8-bit momentum, which the low-bit momentum options rely on.
- Adafactor (Shazeer and Stern, 2018): the factored second moment.
- NVIDIA's NVFP4 training recipe: round-to-nearest activations, stochastically rounded gradients and random
  Hadamard rotations of the weight-gradient operands; the `nvidia` recipe.
- Four over six: the choice between two NVFP4 block scales; the `4over6` recipe and format suffix.
- Quartet II: MS-EDEN rounding of gradients with 128-point rotations; the `quartet2` recipes.
- OCP Microscaling (MX) formats: MXFP8, MXFP6, MXFP4 and the MXFP8 compute path.
- QLoRA (Dettmers et al., 2023): the NF4 format.
- GPTQ and AWQ, compressed-tensors and NVIDIA ModelOpt: checkpoint layouts that load without loss.
- COAT (arXiv 2410.19313): FP8 compression of activations saved for backward.
- Mini-Sequence Transformer (arXiv 2407.15892), MEMO (arXiv 2407.12117), Seq1F1B (arXiv 2406.03488) and FPDT
  (arXiv 2408.16978): running layers over pieces of the sequence and keeping fewer activations for long contexts.
- Gated DeltaNet (Yang et al., 2024): the linear-attention layers of the Qwen3.5 family, run on the kernels of
  flash-linear-attention.

## License

MIT
