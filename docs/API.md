# Bitlattice API

> This document was written with AI and may contain mistakes. The code and
> its tests are the reference.

Principles every part of the library follows.

## A model stays a model

Bitlattice works on ordinary `torch.nn.Module` models. Whatever it changes is visible: converted layers are
Bitlattice modules in `print(model)`, quantized weights are `LatticeTensor` parameters, and `bl.explain(model)`
lists, per layer, the format, how it computes, conversions, whether it trains and its memory.

## Two worlds: plain defaults and explicit requirements

- **Not specified → predictable defaults.** Nothing changes the model or its behaviour in a way you could not
  predict: the model is not converted, computation is lossless, nothing is compressed, state stays on the GPU.
  Within that, the runtime uses the fastest kernels that compute the same results (bit for bit or with more
  accurate rounding: fused GEMM epilogues, hardware conversions, the output layer's gradient formed in forward,
  ...): they only make training faster. Each of them is a field of `bl.Kernels` and can be switched off (below).
  Whatever changes the model or loses accuracy is switched on by you. The runtime logs its choices (logger
  `bitlattice`) and `bl.explain` shows them. Settings without a sensible default are required arguments.
- **Specified → required.** A setting you give explicitly is a requirement: if it cannot be met (no FP4 tensor
  cores, a dimension that does not fit a kernel), you get an error that says why, when configuring if it can be
  known then and otherwise at first use. There are no silent fallbacks.

## One way to do one thing

There are no global switches. How a layer computes is a property of that layer (`bl.surgery.set_compute_`); a
temporary override is an explicit context in your code. Settings are immutable data objects with visible fields:
`Format`, `Compute`, `Memory`, optimizer presets; a string is only the name of a preset and expands to the object.
Each resource has one owner: the optimizer places its state (`Lattice(..., state="gpu" | "ram" | "auto")`), the
runner places activations (`Memory`), a `checkpoint` conversion decides what a module recomputes and how its saves
are stored.

## Automatic choices, each of which can be switched off

`bl.Kernels` lists what the runtime picks on its own. Every choice computes the same result as the plain path (bit
for bit, or with sums in another order) and is on by default:

| field | what it does |
|---|---|
| `two_level_gemm` | NVFP4 GEMMs apply the per-tensor scales in the GEMM epilogue |
| `shared_operands` | an input read by several FP4 layers (q / k / v, gate / up) is quantized once |
| `operand_cache` | quantized weight operands are kept per weight version |
| `fused_norm` | CUDA kernels for the lean RMSNorm / gated RMSNorm conversions |
| `fused_conv` | CUDA causal convolution in the `gated_delta_net` conversion |
| `fused_pack` | CUDA FP8 packing of compressed saved tensors |
| `grad_in_forward` | a frozen output layer forms its input gradient in the forward pass of the loss |
| `native_update` | the optimizer updates a large quantized matrix (every row tile) in one native call |
| `fused_fold` | deferred weight-gradient operands are brought to one global scale by one CUDA pass |
| `fused_ce` | vocabulary-tiled loss and frozen-head input gradients use native softmax reductions |

```python
with bl.kernels(fused_norm=False):                        # a region; other fields keep their current value
    ...
runner = bl.Runner(model, opt, kernels=bl.Kernels.reference())     # every step on the plain paths
bl.surgery.set_compute_(model, "*.mlp", bl.Compute("nvfp4", kernels="reference"))   # one layer's GEMMs
```

The innermost setting applies (a layer's own, then a region's, then a runner's). `bl.kernels_available()` and the
first line of `bl.explain(model)` tell which fast paths exist on the machine.

## Where memory lives

Every placement is chosen explicitly; nothing moves on its own.

| what | where it is chosen | options |
|---|---|---|
| optimizer state | `Lattice(..., state=)` | `gpu`, `ram` (pinned, prefetched per layer), `auto` |
| activations kept for backward | `Runner(..., memory=Memory(activations=))` | `gpu`, `ram` |
| what a module recomputes, how its saves are kept | `convert_(..., "checkpoint", saved=)` | `None`, `fp8`, `nvfp4` |
| a module's weights | `convert_(..., "checkpoint", weights=)` | `gpu`, `ram` (streamed in around its uses) |
| a frozen embedding table / output layer | `convert_(..., "host_embedding")`, `convert_(..., "host_linear")` | pinned host memory |

Host memory is page-locked at exactly the requested size (`bl.hostmem`).

## Execution

`Lattice(..., overlap=True)` issues optimizer updates from a worker thread on a side stream. `step()` and `wait()`
order subsequent work after those updates; `synchronize()` also waits for transfers to finish on the host.

For processes on one Linux host, `bl.parallel.init(transport="host")` exchanges activations through shared pinned
memory using copy engines. The default remains NCCL. Every process must choose the same transport.

`bl.parallel.share_head_(model)` distributes a frozen, bias-free output head's loss across the processes. Call it
collectively after freezing the head. A trainable final norm keeps receiving all micro-batch gradients on its owner.
Shared heads do not support `extra_loss` or a head stored through `host_linear`.

## The operating table

`bl.surgery` holds every edit of a model. Edits work in place (names end in `_`), take a **selector**, are
reversible and return a report of what changed.

A selector is: a module name, a glob over module names (`"*.mlp"`), a regular expression (`"re:.*layers\.[0-3]"`),
a module class, a module object, a list of selectors, or a function `(name, module) -> bool`.

| edit | inverse |
|---|---|
| `quantize_(model, selector, format)` | `dequantize_` |
| `convert_(model, selector, conversion, **options)` | `revert_` |
| `set_compute_(model, selector, compute)` | `set_compute_(..., "auto")` |
| `freeze_(model, selector)` | `unfreeze_` |
| `resize_`, `select_`, `assign_`, `update_` (parameters) | none |
| `drop_`, `insert_`, `replace_` (modules) | none |

A conversion replaces a layer by a Bitlattice module that adopts the same parameter objects under the same names, so
nothing is copied or lost and the state dict and the optimizer are unaffected. It is checked against the original
on a random input before it is applied. `bl.surgery.conversions()` lists them.

A selector that selects nothing is an error: it was asked for. `skip_unfit=True` (where offered) turns modules
the edit cannot serve into log lines instead of an error; every other mismatch raises.

## Fused decoder blocks

For supported Qwen3.5-family decoder blocks, `fused_block` combines activation recomputation and a manual backward.
Its `mlp_chunk` bounds the number of MLP rows computed at once; `seq_chunk` splits the delta-rule recurrence into
pieces, carrying both state and its gradient across boundaries. It must be a multiple of 64. These options preserve
the sequence's context. `graphs=True` records the block's kernels per input shape:

```python
S.convert_(model, r"re:.*layers\.\d+$", "fused_block", graphs=True, mlp_chunk=1024, seq_chunk=2048,
           skip_unfit=True)
```

`quartet2_bound` is an experimental compute preset based on `quartet2_fast`. It removes the amax pass when making
the rotated weight transpose: `|H₁₂₈ W| ≤ √128 max|W|` supplies an upper bound on the global scale. Local block scales
still use the current values. E4M3 scale rounding and underflow can differ, so this is an explicit approximation:

```python
S.set_compute_(model, "*.layers.*", "quartet2_bound", skip_unfit=True)
```

## Layers of the API

1. **Presets**: a few lines: `bl.io.load`, compute presets, `bl.optim.Lattice(model, preset)`, `bl.Runner`.
2. **Components**: `Format`, `LatticeTensor`, `Compute`, `Lattice` (optimizer), `Memory`, `Runner`.
3. **Primitives**: kernels in `bl.ops`, reference implementations in `bl.reference`, functions in `bl.nn`
   (`rms_norm`, `causal_conv1d`, `linear_cross_entropy`, ...).

## Example

```python
import bitlattice as bl

model = bl.io.load("Qwen/Qwen3.5-27B", bl.quant.presets.default("nvfp4"), device="cuda")

S = bl.surgery
S.convert_(model, "*.linear_attn", "gated_delta_net", seq_chunk=4096)
S.convert_(model, "*norm", "rms_norm", skip_unfit=True)        # the norms this conversion fits; others are logged
S.convert_(model, "*.mlp", "token_chunks", chunk=2048)
S.convert_(model, "*.self_attn", "attention_chunks", chunk=4096)  # attention over query pieces
S.convert_(model, "re:layers\.\d+$", "checkpoint", weights="ram")  # recomputed in backward, weights streamed
S.set_compute_(model, "*.layers.*", "quartet2")              # NVFP4 forward and backward, Quartet II rounding
S.set_compute_(model, "*.layers.0.*", "mxfp8_all")            # a sensitive layer on FP8
S.freeze_(model, ["*embed_tokens", "lm_head"])
S.convert_(model, "*embed_tokens", "host_embedding")           # frozen tables in host memory
S.convert_(model, "lm_head", "host_linear")

opt = bl.optim.Lattice(model, "normuon_fast", lr=1e-5, state="ram", fused_backward=True)
runner = bl.Runner(model, opt, micro=8, memory=bl.Memory(activations="ram"))

print(bl.explain(model))
for batch in data:
    runner.step(batch)
```
