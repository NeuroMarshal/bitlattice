"""Compute policy for layers with lattice weights.

    'exact'    dequantize to the activation dtype and use a dense GEMM (default; every device)
    'fp4'      nvfp4 weights: activations are quantized to NVFP4 and the forward GEMM runs on FP4 tensor cores
               (block-scaled MMA, compute capability >= 10.0); the backward stays exact (dense), i.e. the
               activation quantizer is passed straight through.
    'fp4_all'  forward, input-gradient and weight-gradient GEMMs on FP4 tensor cores: output gradients are
               quantized with stochastic rounding, the weight-gradient operands are rotated along the token axis
               by a random-sign 16x16 Hadamard transform, the transposed weight is re-quantized (1x16 blocks) for
               the input gradient; the layer input is saved for backward in its transposed FP4 form (0.56 instead
               of 2 bytes per element)
Layers or devices that do not qualify use 'exact'.
"""
from __future__ import annotations

import contextlib
import itertools
from functools import lru_cache
from typing import NamedTuple

import torch

__all__ = ["POLICIES", "set_compute", "get_compute", "compute", "fp4_supported", "to_blocked", "nvfp4_linear",
           "fp4_policy", "quantize_fp4", "fp4_mm", "rht_last", "Fp4Operand"]

POLICIES = ("exact", "fp4", "fp4_all")
# process-wide rather than thread-local: the backward pass (and checkpoint recomputation) runs on autograd worker
# threads and must see the policy of the forward pass
_default = ["exact"]
_context: list[str] = []


def set_compute(policy: str) -> None:
    if policy not in POLICIES:
        raise ValueError(f"compute policy must be one of {POLICIES}")
    _default[0] = policy


def get_compute(tensor=None) -> str:
    """Effective policy: an active ``compute(...)`` context, else the tensor's own setting, else the default."""
    if _context:
        return _context[-1]
    own = getattr(tensor, "compute", None)
    return own or _default[0]


@contextlib.contextmanager
def compute(policy: str):
    if policy not in POLICIES:
        raise ValueError(f"compute policy must be one of {POLICIES}")
    _context.append(policy)
    try:
        yield
    finally:
        _context.pop()


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


def fp4_policy(x: torch.Tensor, w) -> str:
    """'fp4_all', 'fp4' or 'exact': the policy that applies to this layer call."""
    pol = get_compute(w)
    if pol == "exact" or not (x.is_cuda and w.fmt.name == "nvfp4" and w.hi_idx is None and w.dim() == 2
                              and w.shape[1] % 32 == 0 and w.shape[0] % 16 == 0
                              and fp4_supported(x.device.index or 0)):
        return "exact"
    if pol == "fp4_all" and w.shape[0] % 32 == 0:
        return "fp4_all"
    return "fp4"


def qualifies_fp4(x: torch.Tensor, w) -> bool:
    return fp4_policy(x, w) != "exact"


class Fp4Operand(NamedTuple):
    """A [rows, cols] matrix in NVFP4: packed E2M1 codes, E4M3 block scales along cols, fp32 global scale."""

    packed: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor


_sr_counter = itertools.count(1)


def quantize_fp4(x2: torch.Tensor, stochastic: bool = False) -> Fp4Operand:
    """NVFP4 quantization along the last dimension of a contiguous 2-D tensor (round to nearest, or stochastic
    rounding with a fresh seed per call)."""
    g = (x2.abs().amax().float() / (6.0 * 448.0)).clamp_min(1e-12).reshape(1)
    seed = next(_sr_counter) & 0x7FFFFFFF if stochastic else -1
    packed, scales = torch.ops.bitlattice.quantize_nvfp4(x2, g, seed, 0)
    return Fp4Operand(packed, scales, g)


def fp4_mm(a: Fp4Operand, b: Fp4Operand) -> torch.Tensor:
    """A B^T in fp32 for NVFP4 operands A [M, K], B [N, K] (K: multiple of 32)."""
    fp4 = torch.float4_e2m1fn_x2
    y = torch._scaled_mm(a.packed.view(fp4), b.packed.view(fp4).t(), scale_a=to_blocked(a.scales),
                         scale_b=to_blocked(b.scales), out_dtype=torch.float32)
    return y.mul_(a.global_scale * b.global_scale.float())


@lru_cache(maxsize=None)
def _hadamard16(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Orthogonal 16x16 matrix diag(s) H / 4 with a fixed random sign vector s."""
    h = torch.ones(1, 1)
    for _ in range(4):
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    s = torch.randint(0, 2, (16,), generator=torch.Generator().manual_seed(0x5EED)).float() * 2 - 1
    return (s[:, None] * h / 4.0).to(device, dtype)


def rht_last(x: torch.Tensor) -> torch.Tensor:
    """Random Hadamard transform over consecutive 16-element blocks of the last dimension."""
    r, c = x.shape
    return (x.view(r, c // 16, 16) @ _hadamard16(x.device, x.dtype)).view(r, c)


def _pad_rows(x2: torch.Tensor, multiple: int) -> torch.Tensor:
    pad = (-x2.shape[0]) % multiple
    return torch.nn.functional.pad(x2, (0, 0, 0, pad)) if pad else x2


def nvfp4_linear(x: torch.Tensor, w, bias: torch.Tensor | None = None) -> torch.Tensor:
    """y = Q(x) W^T (+ bias) with x quantized to NVFP4 on the fly; W must be an nvfp4 LatticeTensor."""
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    xq = quantize_fp4(_pad_rows(x2, 16).contiguous())
    y = fp4_mm(xq, Fp4Operand(w.packed, w.scales, w.global_scale.float()))[:M]
    if bias is not None:
        y += bias.float()
    return y.to(x.dtype).reshape(*x.shape[:-1], w.shape[0])


def fp4_forward_saving(x: torch.Tensor, w, bias):
    """Forward of 'fp4_all': returns (y, saved), where saved holds x^T (rotated, NVFP4) for the weight gradient."""
    K = x.shape[-1]
    x2 = _pad_rows(x.reshape(-1, K), 32).contiguous()
    M = x.numel() // K
    y = fp4_mm(quantize_fp4(x2), Fp4Operand(w.packed, w.scales, w.global_scale.float()))[:M]
    if bias is not None:
        y += bias.float()
    xt = Fp4Operand(*torch.ops.bitlattice.quantize_nvfp4_t(x2, _hadamard16(x2.device, torch.float32)))
    return y.to(x.dtype).reshape(*x.shape[:-1], w.shape[0]), xt


def fp4_backward(gy: torch.Tensor, w, xt: Fp4Operand, need_x: bool, need_w: bool):
    """Input and weight gradients of y = x W^T on FP4 tensor cores."""
    N = gy.shape[-1]
    g2 = gy.reshape(-1, N)
    M = g2.shape[0]
    g2p = _pad_rows(g2, 32).contiguous()
    gx = gw = None
    if need_x:
        wt = Fp4Operand(*torch.ops.bitlattice.requantize_nvfp4_t(w.packed, w.scales, w.global_scale))  # [K, N]
        gx = fp4_mm(quantize_fp4(g2p, stochastic=True), wt)[:M]
    if need_w:
        seed = next(_sr_counter) & 0x7FFFFFFF                                          # g^T rotated, [N, Mp]
        gt = Fp4Operand(*torch.ops.bitlattice.quantize_nvfp4_t(g2p, _hadamard16(g2p.device, torch.float32), seed))
        gw = fp4_mm(gt, xt)
    return gx, gw
