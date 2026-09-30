"""Bindings to the native operators with reference fallbacks.

The CUDA path requires a contiguous CUDA tensor whose last dimension is a multiple of 8 and whose scale groups
are multiples of 8; otherwise the reference implementation is used.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch

from . import formats as F
from . import reference as R

try:
    from . import _C  # noqa: F401  (registers torch.ops.bitlattice)
    HAVE_EXT = True
except ImportError:  # pragma: no cover - source checkout without a build
    HAVE_EXT = False

_SCALE_DT = {"fp32": 0, "fp16": 1, "bf16": 2, "e4m3": 3, "e8m0": 4}
_ROUNDING = {"sr": 0, "strat": 1, "rtn": 2}
_RULE = {"fixed": 0, "fit": 1, "fit_latent": 2}
_DIRECTION = {"raw": 0, "sign": 1, "colnorm": 2, "sinkhorn": 3, "factored": 4, "muon": 5, "muon_block": 5}


@lru_cache(maxsize=None)
def _empty(device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(0, dtype=dtype, device=device)


def lattice_args(fmt, device) -> tuple:
    """Positional kernel arguments describing a quantized format."""
    dev = str(device)
    if fmt.kind == "table":
        t = fmt.tables(dev)
        tables = (t["sorted"], t["rank"], t["sorted_code"])
        qmin, qmax, zp = 0, 0, 0
    else:
        tables = (_empty(dev, torch.float32), _empty(dev, torch.int32), _empty(dev, torch.int32))
        qmin, qmax, zp = fmt.qmin, fmt.qmax, int(fmt.zero_point)
    kind = 0 if fmt.kind == "table" else 1
    return tables, (kind, fmt.bits, fmt.container_bits, qmin, qmax, zp, _SCALE_DT[fmt.scale.dtype], fmt.scale.group)


def cuda_ok(fmt, rows: int, cols: int, device) -> bool:
    if not HAVE_EXT or torch.device(device).type != "cuda" or cols % 8:
        return False
    g = fmt.scale.group if fmt.kind != "native" else 1
    return not (g > 0 and (g % 8 or cols % g))


def dequantize(packed, scales, global_scale, zeros, fmt, rows: int, cols: int, dtype=torch.bfloat16):
    """Dense [rows, cols] tensor of the represented values."""
    if cuda_ok(fmt, rows, cols, packed.device) and dtype in (torch.float32, torch.bfloat16, torch.float16):
        tables, args = lattice_args(fmt, packed.device)
        out_dt = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}[dtype]
        return torch.ops.bitlattice.dequantize(packed, scales, global_scale, zeros, *tables, *args, rows, cols, out_dt)
    return R.dequantize(R.Quantized(packed, scales, global_scale, zeros, rows, cols), fmt, dtype)


def strat_args(rounding: str, step: int, strata: int) -> tuple[int, int]:
    if rounding != "strat":
        return 0, 1
    bits = max(1, (strata - 1).bit_length())
    k = 1 << bits
    return R.bitrev(step % k, bits), k


def kappa(trust: float, trust_lr: float, lr: float) -> float:
    return trust if trust_lr <= 0 else min(trust, trust_lr * lr)


def step_quantized(packed, phase, scales, global_scale, zeros, D, fmt, rows: int, cols: int, *, lr: float,
                   weight_decay: float, phase_bits: int, phase_cb: int, rounding: str, seed: int, step: int,
                   strata: int, scale_rule: str, trust: float, trust_lr: float) -> torch.Tensor:
    """In-place update of ``packed`` (and ``phase``); returns the new scales tensor."""
    tables, args = lattice_args(fmt, packed.device)
    j, k = strat_args(rounding, step, strata)
    return torch.ops.bitlattice.step_quantized(
        packed, phase if phase_bits > 0 else None, scales, global_scale, zeros, D.reshape(-1).float().contiguous(),
        *tables, *args, rows, cols, phase_bits, phase_cb, float(lr), float(1.0 - lr * weight_decay),
        _ROUNDING[rounding], int(seed) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, j, k, _RULE[scale_rule],
        float(kappa(trust, trust_lr, lr)))


def step_native(w, phase, D, *, lr: float, weight_decay: float, phase_bits: int, phase_cb: int, rounding: str,
                seed: int, step: int, strata: int) -> None:
    j, k = strat_args(rounding, step, strata)
    torch.ops.bitlattice.step_native(w, phase if phase_bits > 0 else None, D.reshape(-1).float().contiguous(),
                                     phase_bits, phase_cb, float(lr), float(1.0 - lr * weight_decay),
                                     _ROUNDING[rounding], int(seed) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, j, k)


def direction(G, kind: str, r=None, c=None, k: int = 1, beta2: float = 0.95, block: int = 0, ns_steps: int = 5,
              ns_dtype: str = "bf16", sinkhorn_iters: int = 2, clip: float = 1.0) -> torch.Tensor:
    """Update direction as float32 [rows, cols], RMS clipped to ``clip``."""
    if kind == "factored" and HAVE_EXT and G.is_cuda and (G.numel() // max(1, G.shape[0] if G.dim() > 1 else 1)) % 4 == 0:
        return torch.ops.bitlattice.factored_direction(G, r, c, int(k), float(beta2), float(clip))
    if HAVE_EXT:
        return torch.ops.bitlattice.direction(G, _DIRECTION[kind], r, c, int(k), float(beta2),
                                              int(block if kind == "muon_block" else 0), int(ns_steps),
                                              1 if ns_dtype == "bf16" else 0, int(sinkhorn_iters), float(clip))
    if kind in ("muon", "muon_block"):
        U = R.muon_direction(G, block if kind == "muon_block" else 0, ns_steps,
                             torch.bfloat16 if ns_dtype == "bf16" else torch.float32)
    elif kind == "factored":
        st = {"r": r, "c": c, "k": k - 1}
        U = R.factored_direction(G, st, beta2)
    elif kind == "sinkhorn":
        U = R.sinkhorn_direction(G, sinkhorn_iters)
    elif kind == "sign":
        U = R.view2d(G).float().sign()
    elif kind == "colnorm":
        g = R.view2d(G).float()
        U = g / (g.square().mean(0, keepdim=True).sqrt() + 1e-30)
    else:
        U = R.view2d(G).float()
    return R.clip_rms(U, clip)


def phase_container(phase_bits: int) -> int:
    return 0 if phase_bits == 0 else 1 if phase_bits == 1 else 2 if phase_bits == 2 else 4 if phase_bits <= 4 else 8


def phase_numel(n: int) -> int:
    return math.ceil(n / 8) * 8
