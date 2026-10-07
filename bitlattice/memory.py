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

    bl.surgery.convert_(model, "*.mlp", "compressed_saves", fmt="fp8")   # inside chosen layers (also in
                                                                          # checkpoint recomputation)
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass

import torch

from . import _ops

__all__ = ["SavedSpec", "compress_saved", "stats"]

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
    __slots__ = ("__weakref__", "cols", "data", "dtype", "glob", "kind", "rows", "scale", "shape")


_PIECE = 1 << 22            # values per piece when packing / unpacking: float32 temporaries of 16 MiB


def _fused() -> bool:
    from .kernel_choice import active
    return _ops.HAVE_EXT and active().fused_pack


def _pack_fp8(t: torch.Tensor) -> _Packed:
    p = _Packed()
    p.kind, p.shape, p.dtype = "fp8", t.shape, t.dtype
    if _fused():                                         # one read, one write, no float32 temporaries
        p.data, p.scale = torch.ops.bitlattice.mx_pack(t.contiguous())
        return p
    g = t.reshape(-1, _GROUP)
    p.data = torch.empty(g.shape, dtype=torch.float8_e4m3fn, device=t.device)
    p.scale = torch.empty(g.shape[0], dtype=torch.uint8, device=t.device)
    for s in range(0, g.shape[0], _PIECE // _GROUP):             # pieces: bounded float32 temporaries
        gf = g[s:s + _PIECE // _GROUP].float()
        e = torch.ceil(torch.log2(gf.abs().amax(1).clamp_min(2.0 ** -126) / 448.0)).clamp_(-127, 127)
        p.data[s:s + _PIECE // _GROUP] = (gf * torch.exp2(-e)[:, None]).to(torch.float8_e4m3fn)
        p.scale[s:s + _PIECE // _GROUP] = (e + 127).to(torch.uint8)
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
        if _fused():
            return torch.ops.bitlattice.mx_unpack(p.data, p.scale, p.dtype).view(p.shape)
        out = torch.empty(p.data.shape, dtype=p.dtype, device=p.data.device)
        for s in range(0, out.shape[0], _PIECE // _GROUP):
            sl = slice(s, s + _PIECE // _GROUP)
            scale = torch.exp2(p.scale[sl].view(torch.uint8).float() - 127)
            out[sl] = (p.data[sl].float() * scale[:, None]).to(p.dtype)
        return out.view(p.shape)
    from .fp4ref import E2M1
    out = torch.empty(p.rows, p.cols, dtype=p.dtype, device=p.data.device)
    table, g = E2M1.to(p.data.device), p.glob.float()
    step = max(1, _PIECE // p.cols)                     # row pieces: bounded int32 / float32 temporaries
    for r in range(0, p.rows, step):
        codes = p.data[r:r + step].to(torch.int32)
        n = codes.shape[0]
        codes = torch.stack([codes & 15, codes >> 4], -1).reshape(n, p.cols)
        lv = table[codes & 7] * torch.where(codes & 8 > 0, -1.0, 1.0)
        s = p.scale[r:r + step].float() * g
        out[r:r + step] = (lv.view(n, -1, 16) * s.view(n, -1, 1)).reshape(n, p.cols).to(p.dtype)
    return out.view(p.shape)


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
        if hit is not None and hit[0]() is t and hit[1]() is not None:
            return hit[1]()                            # the same tensor saved by several functions: pack once
        p = _pack_fp8(t.contiguous()) if spec.fmt == "fp8" else _pack_nvfp4(t)
        _STATS["packed"] += 1
        _STATS["bytes_in"] += t.numel() * t.element_size()
        _STATS["bytes_out"] += sum(x.numel() * x.element_size() for x in (p.data, p.scale) if x is not None)
        # weak references only: a packed tensor lives as long as the graph that saved it, not as long as this context
        if len(seen) >= 1024:
            for k in [k for k, (tr, pr) in seen.items() if tr() is None or pr() is None]:
                del seen[k]
        seen[key] = (weakref.ref(t), weakref.ref(p))
        return p

    return pack


def compress_saved(spec="fp8"):
    """Context in which tensors saved for backward are packed as described in the module docstring. ``spec``: a
    :class:`SavedSpec` or a format name."""
    spec = spec if isinstance(spec, SavedSpec) else SavedSpec(spec)
    return torch.autograd.graph.saved_tensors_hooks(_packer(spec), _unpack)
