"""Editing loaded models: new tokens, removed or replaced submodules, parameters swapped in place.

Works on dense and lattice weights alike. Lattice weights that change shape are re-quantized in their own format
(round to nearest; existing rows keep their values up to that re-quantization).
"""
from __future__ import annotations

import fnmatch
import re

import torch

from .tensor import LatticeTensor

__all__ = ["extend_vocab", "resize_rows_", "drop_modules", "replace_module", "set_parameter"]


def _match(pattern: str):
    if pattern.startswith("re:"):
        rx = re.compile(pattern[3:])
        return lambda n: rx.search(n) is not None
    return lambda n: fnmatch.fnmatchcase(n, pattern)


def set_parameter(model: torch.nn.Module, name: str, value: torch.Tensor, requires_grad: bool | None = None) -> None:
    """Replace parameter ``name`` (dotted path) by ``value`` (dense or LatticeTensor), keeping it trainable unless
    told otherwise. Tied parameters referring to the same object are all replaced."""
    old = model.get_parameter(name)
    rg = old.requires_grad if requires_grad is None else requires_grad
    new = torch.nn.Parameter(value, requires_grad=rg and value.is_floating_point())
    for mod in model.modules():
        for k, p in list(mod._parameters.items()):
            if p is old:
                mod._parameters[k] = new


def _rows_like(weight: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Append dense ``rows`` [n, cols] to ``weight`` (dense or lattice) and return the new tensor."""
    if isinstance(weight, LatticeTensor):
        dense = torch.cat([weight.dequantize(torch.float32), rows.float().to(weight.device)], 0)
        new = LatticeTensor.from_float(dense, weight.fmt, method="rtn", dtype=weight.dtype)
        new.compute = weight.compute
        return new
    return torch.cat([weight.detach(), rows.to(weight.device, weight.dtype)], 0)


def resize_rows_(model: torch.nn.Module, name: str, n_rows: int, init: torch.Tensor | None = None) -> None:
    """Grow the first dimension of parameter ``name`` to ``n_rows``; new rows from ``init`` [n_new, cols] (default:
    the mean of the existing rows)."""
    w = model.get_parameter(name)
    n_new = n_rows - w.shape[0]
    if n_new <= 0:
        return
    if init is None:
        base = w.dequantize(torch.float32) if isinstance(w, LatticeTensor) else w.detach().float()
        init = base.mean(0, keepdim=True).expand(n_new, -1)
    set_parameter(model, name, _rows_like(w, init))


def _param_name(model, param) -> str:
    return next(n for n, p in model.named_parameters(remove_duplicate=False) if p is param)


def extend_vocab(model: torch.nn.Module, tokenizer, tokens: list[str], *, special: bool = False,
                 pad_to: int = 64) -> list[int]:
    """Add ``tokens`` to ``tokenizer`` and to the input and output embeddings of ``model``; returns their ids.

    Each new row starts as the mean of the rows of the pieces the old tokenizer split the token string into (for
    the output head likewise), so the model begins from what the string meant before. Rows unused by the old
    tokenizer are reused before the matrices grow; the new size is rounded up to ``pad_to``."""
    pieces = [tokenizer(t, add_special_tokens=False).input_ids for t in tokens]
    tokenizer.add_tokens(tokens, special_tokens=special)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    need = max(ids) + 1
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    params = [emb.weight] + ([head.weight] if head is not None and head.weight is not emb.weight else [])
    for p in params:
        name = _param_name(model, p)
        rows = p.shape[0]
        if need > rows:
            size = -(-need // pad_to) * pad_to
            resize_rows_(model, name, size)
        p = model.get_parameter(name)
        dense = p.dequantize(torch.float32) if isinstance(p, LatticeTensor) else p.detach().float()
        init = torch.stack([dense[torch.tensor(pc, device=dense.device)].mean(0) if pc else dense.mean(0)
                            for pc in pieces])
        dense[torch.tensor(ids, device=dense.device)] = init
        if isinstance(p, LatticeTensor):
            new = LatticeTensor.from_float(dense, p.fmt, method="rtn", dtype=p.dtype)
            new.compute = p.compute
            set_parameter(model, name, new)
        else:
            with torch.no_grad():
                p.copy_(dense.to(p.dtype))
    n = model.get_input_embeddings().weight.shape[0]
    if hasattr(emb, "num_embeddings"):
        emb.num_embeddings = n
    if head is not None and hasattr(head, "out_features"):
        head.out_features = head.weight.shape[0]
    for cfg in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if cfg is not None and hasattr(cfg, "vocab_size"):
            cfg.vocab_size = n
    return ids


def drop_modules(model: torch.nn.Module, pattern: str) -> list[str]:
    """Remove every submodule whose name matches ``pattern`` (glob or ``re:`` regex); returns the removed names.
    The parent must not call a removed child any more (e.g. drop a vision tower from a text-only run)."""
    match = _match(pattern)
    names = [n for n, _ in model.named_modules() if n and match(n)]
    names = [n for n in names if not any(n.startswith(o + ".") for o in names if o != n)]
    for n in names:
        parent, _, attr = n.rpartition(".")
        mod = model.get_submodule(parent) if parent else model
        if isinstance(mod, torch.nn.ModuleList):
            mod[int(attr)] = torch.nn.Identity()
        else:
            delattr(mod, attr)
    return names


def replace_module(model: torch.nn.Module, name: str, new: torch.nn.Module) -> torch.nn.Module:
    """Put ``new`` in place of submodule ``name`` (moved to the old module's device); returns the old module."""
    parent, _, attr = name.rpartition(".")
    mod = model.get_submodule(parent) if parent else model
    old = mod.get_submodule(attr) if attr else None
    dev = next((p.device for p in old.parameters()), None) if old is not None else None
    if dev is not None:
        new = new.to(dev)
    if isinstance(mod, torch.nn.ModuleList):
        mod[int(attr)] = new
    else:
        setattr(mod, attr, new)
    return old
