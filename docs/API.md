# Bitlattice API

Principles every part of the library follows.

## A model stays a model

Bitlattice works on ordinary `torch.nn.Module` models. Whatever it changes is visible: converted layers are
Bitlattice modules in `print(model)`, quantized weights are `LatticeTensor` parameters, and `bl.explain(model)`
lists, per layer, the format, how it computes, conversions, whether it trains and its memory.

## Two worlds: automatic and explicit

- **Not specified → chosen automatically.** Every setting has an automatic value: the runtime picks the best route
  the hardware and shapes allow (for example FP4 tensor cores for NVFP4 weights, MXFP8 for other formats, exact
  bf16 otherwise) and logs the choice once per layer (logger `bitlattice`). `bl.explain` shows the choice and why.
- **Specified → required.** A setting you give explicitly is a requirement: if it cannot be met (no FP4 tensor
  cores, a dimension that does not fit a kernel), you get an error that says why — when configuring if it can be
  known then, otherwise at first use. There are no silent fallbacks.

Write the essentials and let the runtime fill in the rest, or pin everything down and get every mismatch reported.

## One way to do one thing

There are no global switches. How a layer computes is a property of that layer (`bl.surgery.set_compute_`); a
temporary override is an explicit context in your code. Settings are immutable data objects with visible fields —
`Format`, `Compute`, `Memory`, optimizer presets; a string is only the name of a preset and expands to the object.

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

## Layers of the API

1. **Presets** — a few lines: `bl.load`, conversion and compute presets, `bl.optim.Lattice(model, preset=...)`,
   `bl.Runner`.
2. **Components** — `Format`, `LatticeTensor`, `Compute`, `Lattice` (optimizer), `Memory`, `Runner`.
3. **Primitives** — kernels in `bl.ops`, reference implementations in `bl.reference`, functions in `bl.nn`
   (`rms_norm`, `causal_conv1d`, `linear_cross_entropy`, ...).

## Example

```python
import bitlattice as bl

model = bl.load("Qwen3.8-27B-NVFP4")

S = bl.surgery
S.convert_(model, "*.linear_attn", "gated_delta_net", seq_chunk=4096)
S.convert_(model, "*norm", "rms_norm")
S.convert_(model, "*.mlp", "token_chunks", chunk=2048)
S.set_compute_(model, "*.layers.*", "quartet2")          # NVFP4 forward and backward, Quartet II rounding
S.set_compute_(model, "*.layers.0.*", "mxfp8_all")        # a sensitive layer on FP8
S.freeze_(model, ["*embed_tokens*", "lm_head"])

opt = bl.optim.Lattice(model, preset="normuon_fast", lr=1e-5)
runner = bl.Runner(model, opt, memory=bl.Memory(state="ram", activations="ram", saved="fp8"), micro=8)

print(bl.explain(model))
for batch in data:
    runner.step(batch)
```
