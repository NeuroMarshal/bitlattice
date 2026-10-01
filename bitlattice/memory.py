"""Compressed activations: tensors saved for backward stored in a smaller format.

Inside :func:`compress_saved` every tensor an autograd function saves for its backward pass goes through a pack
hook: large 16-bit floating-point activations are stored as

    'fp8'    E4M3 values with one power-of-two scale per 32 consecutive values (as MXFP8; 1.03 bytes per value)
    'nvfp4'  NVFP4 with four-over-six block scales (0.56 bytes per value; lossier)

and expanded again when backward needs them. Everything else is kept as is: fp32 tensors (log-sum-exp of attention,
cumulative decays of linear-attention layers, normalization statistics -- errors in them are amplified),
parameters and views of parameters, integer and 8-bit tensors (already quantized operands), small tensors.

This is activation compression for the nonlinear parts of a layer (norm inputs, gating, attention and recurrence
operands), as in COAT (arXiv 2410.19313); the inputs of lattice linear layers under the FP4 / MXFP8 policies are
already saved quantized. With activation checkpointing the hooks act during the recomputation, so they lower the
per-layer peak that limits the context length; without it, they shrink all saved activations.

    with bl.memory.compress_saved("fp8"):
        loss = model(x).loss
    loss.backward()

    bl.memory.compress_saved_(model, "*layers.*")      # every decoder layer's forward runs inside the hooks
"""
from __future__ import annotations

import fnmatch
import re
import weakref
from dataclasses import dataclass

import torch

from . import _ops

__all__ = ["SavedSpec", "compress_saved", "compress_saved_", "stats"]

_GROUP = 32


@dataclass(frozen=True)
class SavedSpec:
    """``fmt`` 'fp8' | 'nvfp4' | 'none'; tensors with fewer than ``min_numel`` values are kept as they are."""

    fmt: str = "fp8"
    min_numel: int = 1 << 16

    def __post_init__(self):
        if self.fmt not in ("fp8", "nvfp4", "none"):
            raise ValueError("fmt must be 'fp8', 'nvfp4' or 'none'")


_STATS = {"packed": 0, "kept": 0, "bytes_in": 0, "bytes_out": 0}


def stats(reset: bool = False) -> dict:
    """Counts of saved tensors packed / kept and their bytes before and after packing (since the last reset)."""
    out = dict(_STATS)
    if reset:
        for k in _STATS:
            _STATS[k] = 0
    return out


def _is_param(t: torch.Tensor) -> bool:
    base = t._base if t._base is not None else t
    return isinstance(base, torch.nn.Parameter) or (base.is_leaf and base.requires_grad)


class _Packed:
    __slots__ = ("kind", "data", "scale", "glob", "shape", "dtype", "rows", "cols")


def _pack_fp8(t: torch.Tensor) -> _Packed:
    p = _Packed()
    p.kind, p.shape, p.dtype = "fp8", t.shape, t.dtype
    g = t.reshape(-1, _GROUP).float()
    e = torch.ceil(torch.log2(g.abs().amax(1).clamp_min(2.0 ** -126) / 448.0)).clamp_(-127, 127)
    p.data = (g * torch.exp2(-e)[:, None]).to(torch.float8_e4m3fn)
    p.scale = e.to(torch.int8)
    return p


def _pack_nvfp4(t: torch.Tensor) -> _Packed:
    p = _Packed()
    p.kind, p.shape, p.dtype = "nvfp4", t.shape, t.dtype
    x2 = t.reshape(-1, t.shape[-1])
    p.rows, p.cols = x2.shape
    p.data, p.scale, p.glob = torch.ops.bitlattice.nvfp4_quantize(x2.contiguous(), False, 0, True, False, False, 0)
    return p


def _unpack(p) -> torch.Tensor:
    if not isinstance(p, _Packed):
        return p
    if p.kind == "fp8":
        g = p.data.float() * torch.exp2(p.scale.float())[:, None]
        return g.to(p.dtype).view(p.shape)
    from .fp4ref import E2M1
    codes = p.data.to(torch.int32)
    codes = torch.stack([codes & 15, codes >> 4], -1).reshape(p.rows, p.cols)
    lv = E2M1.to(codes.device)[codes & 7] * torch.where(codes & 8 > 0, -1.0, 1.0)
    s = p.scale.float() * p.glob.float()
    return (lv.view(p.rows, -1, 16) * s.view(p.rows, -1, 1)).to(p.dtype).view(p.shape)


def _packer(spec: SavedSpec):
    seen: dict = {}

    def pack(t: torch.Tensor):
        ok = (spec.fmt != "none" and type(t) is torch.Tensor and t.dtype in (torch.bfloat16, torch.float16)
              and t.numel() >= spec.min_numel and t.is_cuda and not _is_param(t))
        if ok and spec.fmt == "fp8":
            ok = t.numel() % _GROUP == 0 and t.shape[-1] % _GROUP == 0
        if ok and spec.fmt == "nvfp4":
            ok = _ops.HAVE_EXT and t.shape[-1] % 16 == 0
        if not ok:
            _STATS["kept"] += 1
            return t
        key = (id(t), t._version)
        hit = seen.get(key)
        if hit is not None and hit[0]() is t:          # the same tensor saved by several functions: pack once
            return hit[1]
        p = _pack_fp8(t.contiguous()) if spec.fmt == "fp8" else _pack_nvfp4(t)
        _STATS["packed"] += 1
        _STATS["bytes_in"] += t.numel() * t.element_size()
        _STATS["bytes_out"] += sum(x.numel() * x.element_size() for x in (p.data, p.scale) if x is not None)
        seen[key] = (weakref.ref(t), p)
        return p

    return pack


def compress_saved(spec="fp8"):
    """Context in which tensors saved for backward are packed as described in the module docstring. ``spec``: a
    :class:`SavedSpec` or a format name."""
    spec = spec if isinstance(spec, SavedSpec) else SavedSpec(spec)
    return torch.autograd.graph.saved_tensors_hooks(_packer(spec), _unpack)


def compress_saved_(model: torch.nn.Module, pattern: str = "*layers.*", spec="fp8") -> list[str]:
    """Run the forward of every module whose name matches ``pattern`` (glob on the full name with no further dots
    after the match, or ``re:`` regex) inside :func:`compress_saved`; the forward of a checkpointed layer is also
    its recomputation, so the hooks act there too. Returns the wrapped module names."""
    spec = spec if isinstance(spec, SavedSpec) else SavedSpec(spec)
    if pattern.startswith("re:"):
        rx = re.compile(pattern[3:])
        match = lambda n: rx.fullmatch(n) is not None  # noqa: E731
    else:
        match = lambda n: fnmatch.fnmatchcase(n, pattern) and "." not in n[len(pattern.rstrip("*")):]  # noqa: E731
    done = []
    for name, mod in model.named_modules():
        if not name or not match(name) or getattr(mod, "_bl_compressed", None) is not None:
            continue
        inner = mod.forward

        def forward(*args, _inner=inner, _mod=mod, **kwargs):
            with compress_saved(_mod._bl_compressed):
                return _inner(*args, **kwargs)

        mod.forward = forward
        mod._bl_compressed = spec
        done.append(name)
    return done
