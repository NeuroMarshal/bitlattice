"""Reference (PyTorch) NVFP4 quantizers used to test the CUDA kernels: round to nearest, stochastic rounding, and
MS-EDEN, with optional four-over-six block scale selection and a randomized Hadamard rotation over 128-element
blocks.

Conventions (shared with csrc/nvfp4.cu):
    value = e2m1(code) * e4m3(scale) * global              blocks of 16 along the last dimension
    four over six: the tensor scale uses 256 instead of 448 (256 * 6/4 = 384 is exact in E4M3), every block is
                   quantized with its maximum mapped to 6 and to 4 and keeps the one with the lower error
                   ('mse' or 'mae'); the choice uses round-to-nearest codes
    rotation:      x <- x H D over consecutive 128-element blocks, H the normalized Hadamard matrix, D random signs
                   (one sign vector per call, from ``seed``); both GEMM operands use the same signs
    ms-eden:       codes by round to nearest; per 128 elements S = <x, x> / <x, x_rtn>; every block scale of the 128
                   is multiplied by S and stochastically rounded to E4M3 (unbiased in expectation)
"""
from __future__ import annotations

import torch

from . import reference as R

__all__ = ["hadamard_signs", "rht128", "quantize", "dequantize", "E2M1", "uniform_u"]

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E4M3_MAX = 448.0


def hadamard_signs(seed: int, n: int = 128) -> torch.Tensor:
    """Random +-1 vector of the rotation (counter-based: identical to the CUDA kernels)."""
    u = R.hash_uniform(n, seed & 0xFFFFFFFF, 0x5151, "cpu")
    return torch.where(u < 0.5, -1.0, 1.0)


def _hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n ** 0.5


def rht128(x: torch.Tensor, seed: int, inverse: bool = False) -> torch.Tensor:
    """Rotate consecutive 128-element blocks of the last dimension: y = (x * signs) @ H (inverse: (y @ H) * signs)."""
    *lead, n = x.shape
    if n % 128:
        raise ValueError("last dimension must be a multiple of 128")
    H = _hadamard(128).to(x.device)
    d = hadamard_signs(seed).to(x.device)
    xb = x.float().reshape(*lead, n // 128, 128)
    y = (xb @ H) * d if inverse else (xb * d) @ H
    return y.reshape(*lead, n)


def uniform_u(n: int, seed: int, device) -> torch.Tensor:
    return R.hash_uniform(n, seed & 0xFFFFFFFF, 0, device)


def _e4m3(v: torch.Tensor) -> torch.Tensor:
    return v.clamp(2.0 ** -9, E4M3_MAX).to(torch.float8_e4m3fn).float()


def _nearest(v: torch.Tensor) -> torch.Tensor:
    lv = E2M1.to(v.device)
    a = v.abs().clamp(max=6.0)
    mid = (lv[1:] + lv[:-1]) / 2
    # ties to the lower value: positive ties go down in magnitude, negative ties go up in magnitude
    idx_pos = torch.bucketize(a, mid, right=False)                 # a == mid -> lower index
    idx_neg = torch.bucketize(a, mid, right=True)                  # a == mid -> higher index (more negative)
    idx = torch.where(v < 0, idx_neg, idx_pos)
    return torch.where(v < 0, -lv[idx], lv[idx])


def _stochastic(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    lv = E2M1.to(v.device)
    a = v.abs().clamp(max=6.0)
    lo_i = (torch.bucketize(a, lv, right=True) - 1).clamp(0, 7)
    lo, hi = lv[lo_i], lv[(lo_i + 1).clamp(max=7)]
    up = (lo_i < 7) & (u * (hi - lo) < a - lo)
    m = torch.where(up, hi, lo)
    return torch.where(v < 0, -m, m)


def _sr_e4m3(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Stochastic rounding of positive values to E4M3 (clamped to its range)."""
    v = v.clamp(2.0 ** -9, E4M3_MAX)
    lo = v.to(torch.float8_e4m3fn).float()
    lo = torch.where(lo > v, _prev_e4m3(lo), lo)
    hi = _next_e4m3(lo)
    p = torch.where(hi > lo, (v - lo) / (hi - lo), torch.zeros_like(v))
    return torch.where(u < p, hi, lo)


def _next_e4m3(x: torch.Tensor) -> torch.Tensor:
    bits = x.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32)
    nb = (bits + 1).clamp(max=0x7E)
    return nb.to(torch.uint8).view(torch.float8_e4m3fn).float()


def _prev_e4m3(x: torch.Tensor) -> torch.Tensor:
    bits = x.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32)
    nb = (bits - 1).clamp(min=1)
    return nb.to(torch.uint8).view(torch.float8_e4m3fn).float()


def quantize(x: torch.Tensor, mode: str = "rtn", four_over_six: bool = False, select: str = "mse",
             seed: int = 0, rotate: bool = False, global_scale: torch.Tensor | None = None,
             rotate_seed: int | None = None):
    """Returns (values [rows, cols] as signed E2M1 levels, block scales [rows, cols/16] float (E4M3 values),
    global scale). Dequantized tensor = values * scales * global."""
    x2 = x.float().reshape(-1, x.shape[-1])
    if rotate:
        x2 = rht128(x2, seed if rotate_seed is None else rotate_seed)
    rows, cols = x2.shape
    b = x2.view(rows, cols // 16, 16)
    top = 256.0 if four_over_six else E4M3_MAX
    g = (global_scale.float().reshape(()) if global_scale is not None else
         (x2.abs().amax() / (6.0 * top)).clamp_min(1e-12))
    amax = b.abs().amax(-1)
    def inv(sc):                                   # one reciprocal per block, as the kernels do
        t = sc * g
        return torch.where(t > 0, 1.0 / torch.where(t > 0, t, torch.ones_like(t)), torch.zeros_like(t))

    s6 = _e4m3(amax / 6.0 / g)
    vals6 = _nearest(b * inv(s6)[..., None])
    s, vals = s6, vals6
    if four_over_six:
        s4 = _e4m3(amax / 4.0 / g)
        vals4 = _nearest(b * inv(s4)[..., None])
        e6 = vals6 * (s6 * g)[..., None] - b
        e4 = vals4 * (s4 * g)[..., None] - b
        err6 = e6.square().sum(-1) if select == "mse" else e6.abs().sum(-1)
        err4 = e4.square().sum(-1) if select == "mse" else e4.abs().sum(-1)
        pick4 = err4 < err6
        s = torch.where(pick4, s4, s6)
        vals = torch.where(pick4[..., None], vals4, vals6)
    idx = torch.arange(rows * cols, device=x.device).view(rows, cols // 16, 16)
    if mode == "sr":
        u = uniform_u(rows * cols, seed, x.device).view(rows, cols // 16, 16)
        vals = _stochastic(b * inv(s)[..., None], u)
    elif mode == "eden":
        xr = vals * (s * g)[..., None]                              # round-to-nearest dequantized
        bb = b.reshape(rows, cols // 128, 128)
        rr = xr.reshape(rows, cols // 128, 128)
        num, den = (bb * bb).sum(-1), (bb * rr).sum(-1)
        S = torch.where(den != 0, num / torch.where(den != 0, den, torch.ones_like(den)), torch.ones_like(den))
        S = S.repeat_interleave(8, -1)                              # one factor per 128 -> every 16-block
        u = uniform_u(rows * (cols // 16), seed ^ 0x9E3779B9, x.device).view(rows, cols // 16)
        s = _sr_e4m3(s * S, u)
    elif mode != "rtn":
        raise ValueError(mode)
    del idx
    return vals.reshape(rows, cols), s, g


def dequantize(vals, scales, g):
    rows, cols = vals.shape
    return (vals.view(rows, cols // 16, 16) * (scales * g)[..., None]).reshape(rows, cols)
