# Bitlattice API

Principles every part of the library follows.

## A model stays a model

Bitlattice works on ordinary `torch.nn.Module` models. Whatever it changes is visible: converted layers are
Bitlattice modules in `print(model)`, quantized weights are `LatticeTensor` parameters, and `bl.explain(model)`
lists, per layer, the format, how it computes, conversions, whether it trains and its memory.

## Two worlds: plain defaults and explicit requirements

- **Not specified → predictable defaults.** Nothing changes the model or its behaviour in a way you could not
  predict: the model is not converted, computation is lossless, nothing is compressed, state stays on the GPU.
  Within that, the runtime uses the fastest kernels that compute the same results (bit for bit or with more
  accurate rounding — fused GEMM epilogues, hardware conversions, the output layer's gradient formed in forward,
  ...): they only make training faster. `Compute(kernels="reference")` selects the plain reference kernels instead.
  Whatever changes the model or loses accuracy is switched on by you. The runtime logs its choices (logger
  `bitlattice`) and `bl.explain` shows them. Settings without a sensible default are required arguments.
- **Specified → required.** A setting you give explicitly is a requirement: if it cannot be met (no FP4 tensor
  cores, a dimension that does not fit a kernel), you get an error that says why — when configuring if it can be
  known then, otherwise at first use. There are no silent fallbacks.

## One way to do one thing

There are no global switches. How a layer computes is a property of that layer (`bl.surgery.set_compute_`); a
temporary override is an explicit context in your code. Settings are immutable data objects with visible fields —
`Format`, `Compute`, `Memory`, optimizer presets; a string is only the name of a preset and expands to the object.
Each resource has one owner: the optimizer places its state (`Lattice(..., state="gpu" | "ram" | "auto")`), the
runner places activations (`Memory`), a `checkpoint` conversion decides what a module recomputes and how its saves
are stored.

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
| `resize_`, `select_`, `assign_`, `update_` (parameters) | — |
| `drop_`, `insert_`, `replace_` (modules) | — |

A conversion replaces a layer by a Bitlattice module that adopts the same parameter objects under the same names —
nothing is copied or lost, the state dict and the optimizer are unaffected — and is checked against the original on
a random input before it is applied. `bl.surgery.conversions()` lists them.

A selector that selects nothing is an error: it was asked for. `skip_unfit=True` (where offered) turns modules
the edit cannot serve into log lines instead of an error; every other mismatch raises.

## Layers of the API

1. **Presets** — a few lines: `bl.io.load`, compute presets, `bl.optim.Lattice(model, preset)`, `bl.Runner`.
2. **Components** — `Format`, `LatticeTensor`, `Compute`, `Lattice` (optimizer), `Memory`, `Runner`.
3. **Primitives** — kernels in `bl.ops`, reference implementations in `bl.reference`, functions in `bl.nn`
   (`rms_norm`, `causal_conv1d`, `linear_cross_entropy`, ...).

## Example

```python
import bitlattice as bl

model = bl.io.load("Qwen/Qwen3.5-27B", bl.quant.presets.default("nvfp4"), device="cuda")

S = bl.surgery
S.convert_(model, "*.linear_attn", "gated_delta_net", seq_chunk=4096)
S.convert_(model, "*norm", "rms_norm", skip_unfit=True)        # the norms this conversion fits; others are logged
S.convert_(model, "*.mlp", "token_chunks", chunk=2048)
S.convert_(model, "re:layers\.\d+$", "checkpoint", saved="fp8")  # recomputed in backward, saves packed to FP8
S.set_compute_(model, "*.layers.*", "quartet2")              # NVFP4 forward and backward, Quartet II rounding
S.set_compute_(model, "*.layers.0.*", "mxfp8_all")            # a sensitive layer on FP8
S.freeze_(model, ["*embed_tokens", "lm_head"])

opt = bl.optim.Lattice(model, "normuon_fast", lr=1e-5, state="ram", fused_backward=True)
runner = bl.Runner(model, opt, micro=8, memory=bl.Memory(activations="ram"))

print(bl.explain(model))
for batch in data:
    runner.step(batch)
```
