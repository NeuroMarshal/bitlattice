"""Compute policy for layers with lattice weights.

    'exact'  dequantize to the activation dtype and use a dense GEMM (default; every device)
    'fp4'    nvfp4 weights: activations are quantized to NVFP4 and the forward GEMM runs on FP4 tensor cores
             (block-scaled MMA, compute capability >= 10.0); the backward stays exact (dense), i.e. the activation
             quantizer is passed straight through. Layers or devices that do not qualify use 'exact'.
"""
from __future__ import annotations

import contextlib
import threading
from functools import lru_cache

import torch

__all__ = ["POLICIES", "set_compute", "get_compute", "compute", "fp4_supported", "to_blocked", "nvfp4_linear"]

POLICIES = ("exact", "fp4")
_state = threading.local()
_default = ["exact"]


def set_compute(policy: str) -> None:
    if policy not in POLICIES:
        raise ValueError(f"compute policy must be one of {POLICIES}")
    _default[0] = policy


def get_compute(tensor=None) -> str:
    """Effective policy: an active ``compute(...)`` context, else the tensor's own setting, else the default."""
    ctx = getattr(_state, "policy", None)
    if ctx is not None:
        return ctx
    own = getattr(tensor, "compute", None)
    return own or _default[0]


@contextlib.contextmanager
def compute(policy: str):
    if policy not in POLICIES:
        raise ValueError(f"compute policy must be one of {POLICIES}")
    prev = getattr(_state, "policy", None)
    _state.policy = policy
    try:
        yield
    finally:
        _state.policy = prev


@lru_cache(maxsize=None)
def fp4_supported(device_index: int) -> bool:
    if not hasattr(torch, "float4_e2m1fn_x2"):
        return False
    return torch.cuda.get_device_capability(device_index) >= (10, 0)


def to_blocked(s: torch.Tensor) -> torch.Tensor:
    """[rows, cols] block scales -> the 128x4-tiled layout expected by block-scaled GEMMs."""
    rows, cols = s.shape
    nr, nc = -(-rows // 128), -(-cols // 4)
    if nr * 128 != rows or nc * 4 != cols:
        p = torch.zeros(nr * 128, nc * 4, dtype=s.dtype, device=s.device)
        p[:rows, :cols] = s
        s = p
    b = s.view(nr, 128, nc, 4).permute(0, 2, 1, 3)
    return b.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def qualifies_fp4(x: torch.Tensor, w) -> bool:
    return (get_compute(w) == "fp4" and x.is_cuda and w.fmt.name == "nvfp4" and w.hi_idx is None and
            w.dim() == 2 and w.shape[1] % 32 == 0 and w.shape[0] % 16 == 0 and fp4_supported(x.device.index or 0))


def nvfp4_linear(x: torch.Tensor, w, bias: torch.Tensor | None = None) -> torch.Tensor:
    """y = Q(x) W^T (+ bias) with x quantized to NVFP4 on the fly; W must be an nvfp4 LatticeTensor."""
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    pad = (-M) % 16
    if pad:
        x2 = torch.nn.functional.pad(x2, (0, 0, 0, pad))
    x2 = x2.contiguous()
    gx = (x2.abs().amax().float() / (6.0 * 448.0)).clamp_min(1e-12).reshape(1)
    packed, scales = torch.ops.bitlattice.quantize_nvfp4(x2, gx)
    fp4 = torch.float4_e2m1fn_x2
    y = torch._scaled_mm(packed.view(fp4), w.packed.view(fp4).t(), scale_a=to_blocked(scales),
                         scale_b=to_blocked(w.scales), out_dtype=torch.float32)
    y = y[:M].mul_(gx * w.global_scale.float())
    if bias is not None:
        y += bias.float()
    return y.to(x.dtype).reshape(*x.shape[:-1], w.shape[0])
