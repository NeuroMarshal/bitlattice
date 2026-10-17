"""Editing loaded models with a small set of general operations.

Parameter level (dense and lattice alike; a lattice parameter that is rewritten is re-encoded in its own format by
rounding to nearest, so untouched entries keep their values up to that re-quantization):

    values(p)                        dense float32 copy of a parameter
    assign_(model, name, t)          new value of any shape
    update_(model, name, fn)         value <- fn(value)
    resize_(model, name, size, dim)  grow or shrink along one dimension; new slices from an init rule
    select_(model, name, index, dim) keep / reorder slices (pruning channels, permutations)
    set_parameter(model, name, t)    swap the Parameter object itself (tied references follow)

Module level (``target`` is a selector, see :mod:`bitlattice.select`):

    quantize_(model, target, fmt)    weights of the selected layers into a lattice format    / dequantize_
    convert_(model, target, to)      layers into Bitlattice modules (same parameter objects)  / revert_
    set_compute_(model, target, c)   how the selected layers compute
    freeze_(model, target)           no gradients for their parameters                       / unfreeze_
    drop_, replace_, insert_         remove, swap or add modules
    sync_shapes_                     module attributes (in_features, num_embeddings) from the actual weights

Parameters on the meta device (owned by another pipeline process) are edited in shape only. Task-level edits are
compositions of these, e.g. :func:`extend_vocab`.
"""
from __future__ import annotations

from collections.abc import Callable

import torch

from .nn.convert import Conversion, conversions, convert_, register_conversion, revert_
from .select import Selector, select
from .tensor import LatticeTensor

__all__ = [
    "Conversion",
    "assign_",
    "conversions",
    "convert_",
    "dequantize_",
    "drop_",
    "extend_vocab",
    "freeze_",
    "insert_",
    "mean_of_pieces",
    "quantize_",
    "register_conversion",
    "replace_",
    "resize_",
    "revert_",
    "select_",
    "set_compute_",
    "set_parameter",
    "sparse_embeddings_",
    "sync_shapes_",
    "train_rows_",
    "unfreeze_",
    "update_",
    "values",
]

Init = str | torch.Tensor | Callable[[torch.Tensor, int, int], torch.Tensor]


# parameter level

def values(p: torch.Tensor) -> torch.Tensor:
    """Dense float32 copy of a (dense or lattice) tensor."""
    return p.dequantize(torch.float32) if isinstance(p, LatticeTensor) else p.detach().float().clone()


def set_parameter(model: torch.nn.Module, name: str, value: torch.Tensor, requires_grad: bool | None = None) -> None:
    """Replace the Parameter ``name`` by ``value`` (dense or LatticeTensor); every module holding the same Parameter
    object (tied weights) gets the new one."""
    old = model.get_parameter(name)
    rg = old.requires_grad if requires_grad is None else requires_grad
    new = torch.nn.Parameter(value, requires_grad=rg and value.is_floating_point())
    for mod in model.modules():
        for k, p in list(mod._parameters.items()):
            if p is old:
                mod._parameters[k] = new


def _encode_like(old: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    if isinstance(old, LatticeTensor):
        new = LatticeTensor.from_float(dense.to(old.device), old.fmt, method="rtn", dtype=old.dtype)
        new.compute = old.compute
        return new
    return dense.to(old.device, old.dtype)


def assign_(model: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    """Give parameter ``name`` a new value of any shape, stored like the old one (same format / dtype / device)."""
    old = model.get_parameter(name)
    if old.device.type == "meta":
        set_parameter(model, name, torch.empty(value.shape, dtype=old.dtype, device="meta"))
        return
    if not isinstance(old, LatticeTensor) and value.shape == old.shape:
        with torch.no_grad():
            old.copy_(value.to(old.dtype))
        return
    set_parameter(model, name, _encode_like(old, value))


def update_(model: torch.nn.Module, name: str, fn: Callable[[torch.Tensor], torch.Tensor | None]) -> None:
    """value <- fn(value) on a dense float32 copy (fn may edit in place and return None)."""
    p = model.get_parameter(name)
    if p.device.type == "meta":
        return
    v = values(p)
    out = fn(v)
    assign_(model, name, v if out is None else out)


def _init_slices(init: Init, v: torch.Tensor, n: int, dim: int) -> torch.Tensor:
    shape = list(v.shape)
    shape[dim] = n
    if callable(init) and not isinstance(init, str):
        return init(v, n, dim)
    if isinstance(init, torch.Tensor):
        return init.to(v.device, v.dtype).expand(shape)
    if init == "zeros":
        return v.new_zeros(shape)
    if init == "mean":
        return v.mean(dim, keepdim=True).expand(shape).clone()
    if init == "normal":
        return torch.randn(shape, device=v.device) * v.std()
    raise ValueError(f"unknown init {init!r}")


def resize_(model: torch.nn.Module, name: str, size: int, dim: int = 0, init: Init = "mean") -> None:
    """Grow (new slices from ``init``: 'mean', 'zeros', 'normal', a tensor, or fn(old, n_new, dim)) or shrink
    (keep the leading slices) parameter ``name`` along ``dim``."""
    p = model.get_parameter(name)
    cur = p.shape[dim]
    if size == cur:
        return
    if p.device.type == "meta":
        shape = list(p.shape)
        shape[dim] = size
        set_parameter(model, name, torch.empty(shape, dtype=p.dtype, device="meta"))
        return
    v = values(p)
    if size < cur:
        assign_(model, name, v.narrow(dim, 0, size).contiguous())
    else:
        assign_(model, name, torch.cat([v, _init_slices(init, v, size - cur, dim)], dim))


def select_(model: torch.nn.Module, name: str, index: torch.Tensor, dim: int = 0) -> None:
    """Keep the slices ``index`` of parameter ``name`` along ``dim``, in that order."""
    p = model.get_parameter(name)
    if p.device.type == "meta":
        shape = list(p.shape)
        shape[dim] = len(index)
        set_parameter(model, name, torch.empty(shape, dtype=p.dtype, device="meta"))
        return
    v = values(p)
    assign_(model, name, v.index_select(dim, torch.as_tensor(index, device=v.device)).contiguous())


def train_rows_(model: torch.nn.Module, name: str, rows) -> None:
    """Train only ``rows`` of parameter ``name`` (None: all rows again). The optimizer updates just those rows, and
    ``bitlattice.nn.linear_cross_entropy`` computes the output-head gradient for those rows only, so a large
    vocabulary matrix never gets a dense gradient. The parameter must stay trainable (requires_grad)."""
    p = model.get_parameter(name)
    if rows is None:
        if hasattr(p, "_bl_rows"):
            del p._bl_rows
        return
    p._bl_rows = torch.as_tensor(rows, dtype=torch.long, device=p.device if p.device.type != "meta" else "cpu")
    p.requires_grad_(True)


# module level

def sparse_embeddings_(model: torch.nn.Module) -> int:
    """Row-sparse gradients for every nn.Embedding (only the looked-up rows get a gradient); returns the count."""
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.sparse = True
            n += 1
    return n


def sync_shapes_(module: torch.nn.Module) -> None:
    """Set the size attributes of standard modules from their current weights."""
    for m in module.modules():
        w = getattr(m, "weight", None)
        if w is None:
            continue
        if isinstance(m, torch.nn.Linear):
            m.out_features, m.in_features = w.shape[0], w.shape[1]
        elif isinstance(m, torch.nn.Embedding):
            m.num_embeddings, m.embedding_dim = w.shape[0], w.shape[1]
        elif isinstance(m, (torch.nn.LayerNorm,)) and w.dim() == 1:
            m.normalized_shape = (w.shape[0],)


def _chosen(model: torch.nn.Module, target: Selector, outermost: bool = True) -> list:
    """The selected (name, module) pairs; a selector that selects nothing is an error (it was asked for)."""
    chosen = select(model, target, outermost=outermost)
    if not chosen:
        raise ValueError(f"no module matches {target!r}")
    return chosen


def drop_(model: torch.nn.Module, target: Selector) -> list[str]:
    """Remove the selected submodules; entries of a ModuleList become ``Identity``. Returns the removed names. The
    caller makes sure nothing calls a removed child."""
    names = [n for n, _ in _chosen(model, target) if n]                      # never the model itself
    for n in names:
        parent, _, attr = n.rpartition(".")
        mod = model.get_submodule(parent) if parent else model
        if isinstance(mod, torch.nn.ModuleList):
            mod[int(attr)] = torch.nn.Identity()
        else:
            delattr(mod, attr)
    return names


def freeze_(model: torch.nn.Module, target: Selector) -> list[str]:
    """No gradients for the parameters of the selected modules. Returns the parameter names."""
    return _requires_grad(model, target, False)


def unfreeze_(model: torch.nn.Module, target: Selector) -> list[str]:
    """Gradients again for the (floating-point) parameters of the selected modules. Returns the parameter names."""
    return _requires_grad(model, target, True)


def _requires_grad(model: torch.nn.Module, target: Selector, on: bool) -> list[str]:
    done = []
    for name, mod in _chosen(model, target):
        for pname, p in mod.named_parameters():
            if p.is_floating_point() or isinstance(p, LatticeTensor):
                p.requires_grad_(on)
                done.append(f"{name}.{pname}")
    return done


def quantize_(model: torch.nn.Module, target: Selector, fmt="nvfp4", *, calibration: dict | None = None,
              measure: bool = False, **options):
    """Weights of the selected layers (and of layers inside them) into a lattice format. ``fmt``: a format name or
    Format, or a :class:`bitlattice.QuantConfig`; ``options`` override QuantConfig fields (method, keep, ...);
    ``calibration``: :func:`bitlattice.quant.calibrate` statistics (``saliency='activation'``); ``measure``: record
    each weight's quantization error. Returns the report (quantized layers with bits per weight, skipped layers)."""
    from dataclasses import replace

    from .quant.api import quantize_layers_
    from .quant.config import QuantConfig
    config = fmt if isinstance(fmt, QuantConfig) else QuantConfig(format=fmt)
    if options:
        config = replace(config, **options)
    return quantize_layers_(model, [n for n, _ in _chosen(model, target)], config, calibration, measure)


def dequantize_(model: torch.nn.Module, target: Selector, dtype: torch.dtype | None = None) -> list[str]:
    """Lattice weights of the selected layers back to dense tensors. Returns the parameter names."""
    done = []
    for name, mod in _chosen(model, target):
        for sub_name, sub in mod.named_modules():
            for pname, p in list(sub.named_parameters(recurse=False)):
                if isinstance(p, LatticeTensor):
                    setattr(sub, pname, torch.nn.Parameter(p.dequantize(dtype or p.dtype),
                                                           requires_grad=p.requires_grad))
                    done.append(".".join(x for x in (name, sub_name, pname) if x))
    return done


def set_compute_(model: torch.nn.Module, target: Selector, compute, *, skip_unfit: bool = False) -> list[str]:
    """How the lattice layers among the selected modules compute: a :class:`bitlattice.Compute`, a preset name
    (``bitlattice.COMPUTE_PRESETS``) or 'auto' (back to the default, exact). A layer that cannot honour the setting
    raises ValueError naming it and why; with ``skip_unfit`` such layers keep their setting and are logged. Returns
    the names of the weights set."""
    from .compute_policy import assign_compute_
    return assign_compute_(model, _chosen(model, target, outermost=False), compute, skip_unfit)


def replace_(model: torch.nn.Module, name: str, new: torch.nn.Module) -> torch.nn.Module:
    """Put ``new`` in place of submodule ``name`` (moved to the old module's device); returns the old module."""
    parent, _, attr = name.rpartition(".")
    mod = model.get_submodule(parent) if parent else model
    old = mod.get_submodule(attr)
    dev = next((p.device for p in old.parameters() if p.device.type != "meta"), None)
    if dev is not None:
        new = new.to(dev)
    if isinstance(mod, torch.nn.ModuleList):
        mod[int(attr)] = new
    else:
        setattr(mod, attr, new)
    return old


def insert_(model: torch.nn.Module, list_name: str, index: int, new: torch.nn.Module) -> None:
    """Insert ``new`` into the ModuleList ``list_name`` at ``index`` (e.g. an extra decoder block)."""
    ml = model.get_submodule(list_name)
    if not isinstance(ml, torch.nn.ModuleList):
        raise TypeError(f"{list_name} is not a ModuleList")
    ml.insert(index, new)


# compositions

def mean_of_pieces(pieces: list[list[int]]) -> Callable[[torch.Tensor], None]:
    """Row update for :func:`update_`: row i of the new block becomes the mean of the existing rows ``pieces[i]``.
    Returns fn(value, rows) editing ``value`` in place."""
    def fn(v: torch.Tensor, rows: list[int]):
        for r, pc in zip(rows, pieces):
            v[r] = v[torch.tensor(pc, device=v.device)].mean(0) if pc else v.mean(0)
    return fn


def extend_vocab(model: torch.nn.Module, tokenizer, tokens: list[str], *, special: bool = False,
                 pad_to: int = 64) -> list[int]:
    """Add ``tokens`` to the tokenizer and to the input / output embeddings; returns their ids. Composition of
    resize_ (rows rounded up to ``pad_to``) and update_ (each new row = mean of the rows of the pieces the old
    tokenizer split the token into, so a new token starts from what its string meant)."""
    pieces = [tokenizer(t, add_special_tokens=False).input_ids for t in tokens]
    tokenizer.add_tokens(tokens, special_tokens=special)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    names = {n for n, p in model.named_parameters(remove_duplicate=False)
             if p is model.get_input_embeddings().weight
             or (model.get_output_embeddings() is not None and p is model.get_output_embeddings().weight)}
    names = {n for n in names if n.rpartition(".")[0] in dict(model.named_modules())}
    seen = set()
    fill = mean_of_pieces(pieces)
    for name in sorted(names):
        p = model.get_parameter(name)
        if id(p) in seen:
            continue
        seen.add(id(p))
        need = max(ids) + 1
        if need > p.shape[0]:
            resize_(model, name, -(-need // pad_to) * pad_to, 0)
        update_(model, name, lambda v: fill(v, ids))
    sync_shapes_(model)
    n = model.get_input_embeddings().weight.shape[0]
    for cfg in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if cfg is not None and hasattr(cfg, "vocab_size"):
            cfg.vocab_size = n
    return ids
