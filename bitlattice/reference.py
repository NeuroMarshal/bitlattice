"""Reference implementations in plain PyTorch.

These functions define the semantics that the CUDA kernels implement (same counter-based RNG, same tie rules,
same scale snapping). They are used by the test-suite and as a fallback on devices without the extension.
All weights are processed as a 2-D view ``[rows, cols]`` (``cols`` = flattened trailing dimensions).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

from . import formats as F
from .packing import pack, sign_extend, unpack

M32 = 0xFFFFFFFF


# counter-based uniform numbers

def hash_uniform(n: int, seed: int, step: int, device, offset: int = 0) -> torch.Tensor:
    """u in [0, 1) indexed by (element index, seed, step). Identical to ``bl_hash_uniform`` in csrc/common.cuh."""
    x = torch.arange(offset, offset + n, dtype=torch.int64, device=device) & M32
    key = ((seed & M32) * 0x9E3779B1 + (step & M32) * 0x85EBCA77) & M32
    x = (x * 0x27D4EB2F + key) & M32
    x = ((x ^ (x >> 16)) * 0x7FEB352D) & M32
    x = ((x ^ (x >> 15)) * 0x846CA68B) & M32
    x = x ^ (x >> 16)
    return (x >> 8).to(torch.float32) * (2.0 ** -24)


def bitrev(v: int, bits: int) -> int:
    r = 0
    for _ in range(bits):
        r = (r << 1) | (v & 1)
        v >>= 1
    return r


def rounding_u(n: int, seed: int, step: int, device, mode: str, strata: int = 16) -> torch.Tensor:
    if mode == "sr":
        return hash_uniform(n, seed, step, device)
    if mode == "strat":
        bits = max(1, (strata - 1).bit_length())
        k = 1 << bits
        j = bitrev(step % k, bits)
        return torch.frac((j + hash_uniform(n, seed, 0, device)) / k)
    if mode == "rtn":
        return torch.full((n,), 0.5, device=device)
    raise ValueError(mode)


# scales

def decode_scales(scales: torch.Tensor, spec: F.ScaleSpec) -> torch.Tensor:
    if spec.dtype == "e8m0":
        return torch.exp2(scales.to(torch.float32) - 127.0)
    return scales.to(torch.float32)


def encode_scales(s: torch.Tensor, spec: F.ScaleSpec, global_scale: torch.Tensor | None = None) -> torch.Tensor:
    """Round positive scales (already divided by the global scale) into the storage dtype."""
    s = s.to(torch.float32)
    if spec.dtype == "fp32":
        return s.clamp_min(1e-30)
    if spec.dtype == "fp16":
        return s.clamp(2.0 ** -24, 65504.0).to(torch.float16)
    if spec.dtype == "bf16":
        return s.clamp_min(1e-30).to(torch.bfloat16)
    if spec.dtype == "e4m3":
        return s.clamp(F.E4M3_MIN, F.E4M3_MAX).to(torch.float8_e4m3fn)
    if spec.dtype == "e8m0":
        return (torch.round(torch.log2(s.clamp_min(2.0 ** -126))) + 127.0).clamp(0, 254).to(torch.uint8)
    raise ValueError(spec.dtype)


def effective_scales(scales, spec: F.ScaleSpec, global_scale) -> torch.Tensor:
    s = decode_scales(scales, spec)
    return s * global_scale.to(torch.float32) if global_scale is not None else s


def expand_groups(t: torch.Tensor, rows: int, cols: int, spec: F.ScaleSpec) -> torch.Tensor:
    """[R, G] per-group values -> [rows, cols] per-element values."""
    if spec.group == 0:
        return t.reshape(1, 1).expand(rows, cols)
    if spec.group == -1:
        return t.reshape(rows, 1).expand(rows, cols)
    return t.reshape(rows, -1, 1).expand(rows, cols // spec.group, spec.group).reshape(rows, cols)


def group_sum(x: torch.Tensor, spec: F.ScaleSpec) -> torch.Tensor:
    """[rows, cols] -> per-group sums shaped like the scale tensor."""
    rows, cols = x.shape
    if spec.group == 0:
        return x.sum().reshape(1, 1)
    if spec.group == -1:
        return x.sum(1, keepdim=True)
    return x.reshape(rows, cols // spec.group, spec.group).sum(-1)


# codes <-> ranks <-> normalized levels

def num_levels(fmt) -> int:
    return len(fmt.levels) if fmt.kind == "table" else fmt.qmax - fmt.qmin + 1


def raw_to_rank(fmt, raw: torch.Tensor) -> torch.Tensor:
    if fmt.kind == "table":
        return fmt.tables(str(raw.device))["rank"][raw.long()].long().clamp_min(0)
    k = sign_extend(raw, fmt.bits) if not fmt.zero_point else raw.long()
    return k - fmt.qmin


def rank_to_raw(fmt, rank: torch.Tensor) -> torch.Tensor:
    if fmt.kind == "table":
        return fmt.tables(str(rank.device))["sorted_code"][rank.long()].long()
    k = rank.long() + fmt.qmin
    return k & ((1 << fmt.bits) - 1) if not fmt.zero_point else k


def rank_level(fmt, rank: torch.Tensor, zero: torch.Tensor | None = None) -> torch.Tensor:
    """Normalized value (units of the scale) of each rank."""
    if fmt.kind == "table":
        return fmt.tables(str(rank.device))["sorted"][rank.long()]
    v = (rank.long() + fmt.qmin).to(torch.float32)
    return v - zero if zero is not None else v


def nearest_rank(fmt, x: torch.Tensor, zero: torch.Tensor | None = None) -> torch.Tensor:
    """Rank of the nearest level; ties go to the lower level (count of midpoints strictly below x)."""
    if fmt.kind == "table":
        lv = fmt.tables(str(x.device))["sorted"]
        mid = (lv[1:] + lv[:-1]) * 0.5
        return torch.searchsorted(mid, x.contiguous(), right=False)
    xk = x + zero if zero is not None else x
    return torch.clamp(torch.round(xk), fmt.qmin, fmt.qmax).long() - fmt.qmin


# quantize / dequantize

class Quantized(NamedTuple):
    packed: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor | None
    zeros: torch.Tensor | None
    rows: int
    cols: int


def view2d(w: torch.Tensor) -> torch.Tensor:
    return w.reshape(1, -1) if w.dim() == 1 else w.reshape(w.shape[0], -1)


def _group_view(w2: torch.Tensor, spec: F.ScaleSpec) -> torch.Tensor:
    rows, cols = w2.shape
    if spec.group == 0:
        return w2.reshape(1, 1, -1)
    if spec.group == -1:
        return w2.reshape(rows, 1, cols)
    return w2.reshape(rows, cols // spec.group, spec.group)


def quantize(w: torch.Tensor, fmt, method: str = "rtn", grid: int = 16, shrink: float = 0.5,
             global_scale: torch.Tensor | None = None) -> Quantized:
    """Quantize a dense tensor. ``method``: 'rtn' (absmax/absmean/minmax scale) or 'mse' (per-group search of a
    clipping factor in [shrink, 1] minimising the squared error, ``grid`` candidates). ``global_scale``: use this
    per-tensor scale (formats that have one) instead of deriving it from the tensor, e.g. to append to storage
    quantized earlier; group scales are clamped to its range."""
    fmt = F.get(fmt)
    if fmt.kind == "native":
        raise ValueError("native formats are not quantized")
    spec = fmt.scale
    w2 = view2d(w).to(torch.float32)
    rows, cols = w2.shape
    spec.groups_per_row(cols)
    wg = _group_view(w2, spec)
    zero = None
    if getattr(fmt, "zero_point", False):
        mn, mx = wg.amin(-1).clamp_max(0.0), wg.amax(-1).clamp_min(0.0)
        s = ((mx - mn) / (fmt.qmax - fmt.qmin)).clamp_min(1e-12)
        zero = torch.clamp(torch.round(fmt.qmin - mn / s), fmt.qmin, fmt.qmax)
    elif fmt.init == "absmean":
        s = wg.abs().mean(-1).clamp_min(1e-12) / (fmt.max_abs if fmt.max_abs > 1 else 1.0)
    else:
        s = (wg.abs().amax(-1) / fmt.max_abs).clamp_min(1e-12)
    # four over six: a group may map its maximum to the second-highest level instead of the highest (the top gap
    # of an FP4 grid, 4 -> 6, is the widest); the per-tensor scale then leaves room for that larger group scale
    expand = []
    if fmt.kind == "table" and (method == "4over6" or getattr(fmt, "init", "") == "4over6" or method == "mse"):
        pos = sorted(v for v in fmt.levels if v > 0)
        if len(pos) >= 2:
            expand = [pos[-1] / pos[-2]]
    four6 = method == "4over6" or getattr(fmt, "init", "") == "4over6"
    gscale = None
    if spec.global_scale:
        # four over six: the largest power of two p with p * (top / second level) exactly representable (256 for
        # E2M1 / E4M3), so the block holding the tensor maximum can still choose either scale
        top = F.E4M3_MAX
        if four6 and expand:
            top = 2.0 ** math.floor(math.log2(F.E4M3_MAX / expand[0]))
        gscale = (global_scale.to(w2.device, torch.float32).reshape(1) if global_scale is not None else
                  (w2.abs().amax() / (fmt.max_abs * top)).clamp_min(1e-12).reshape(1))
    pow2 = spec.dtype == "e8m0"            # power-of-two scales round up, so the group maximum is never clipped

    def up(t):
        return torch.exp2(torch.ceil(torch.log2(t.clamp_min(2.0 ** -126)))) if pow2 else t

    if method in ("mse", "4over6") and zero is None or (four6 and zero is None):
        cands = ([1.0] + expand) if (four6 and method != "mse") else torch.linspace(1.0, shrink, grid).tolist() + expand
        best_err, best_s = None, None
        for a in cands:
            sq = effective_scales(encode_scales(up(s * a) / (gscale if gscale is not None else 1.0), spec), spec,
                                  gscale)
            se = expand_groups(sq, rows, cols, spec)
            r = nearest_rank(fmt, w2 / se)
            err = group_sum((rank_level(fmt, r) * se - w2) ** 2, spec)
            if best_err is None:
                best_err, best_s = err, up(s * a)
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_s = torch.where(better, up(s * a), best_s)
        s = best_s
    elif method not in ("rtn", "mse", "4over6"):
        raise ValueError(method)
    else:
        s = up(s)
    scales = encode_scales(s / (gscale if gscale is not None else 1.0), spec)
    se = expand_groups(effective_scales(scales, spec, gscale), rows, cols, spec)
    ze = expand_groups(zero, rows, cols, spec) if zero is not None else None
    rank = nearest_rank(fmt, w2 / se, ze)
    packed = pack(rank_to_raw(fmt, rank), fmt.container_bits)
    zeros = zero.to(torch.float16) if zero is not None else None
    return Quantized(packed, scales.reshape(spec.shape(rows, cols)), gscale, zeros, rows, cols)


def dequantize(q: Quantized, fmt, dtype=torch.float32) -> torch.Tensor:
    fmt = F.get(fmt)
    spec = fmt.scale
    raw = unpack(q.packed, fmt.container_bits, q.cols)
    rank = raw_to_rank(fmt, raw)
    ze = expand_groups(q.zeros.to(torch.float32), q.rows, q.cols, spec) if q.zeros is not None else None
    se = expand_groups(effective_scales(q.scales, spec, q.global_scale), q.rows, q.cols, spec)
    return (rank_level(fmt, rank, ze) * se).to(dtype)


# projection with a b-bit phase (shared by quantized and native formats)

def _project(t, w, lo, hi, u, bits: int):
    """t: target, w: nearest represented value, lo/hi: its neighbours (all same units).
    Returns (move: -1/0/+1 relative to w, phase code)."""
    if bits == 0:
        up = t >= w
        a = torch.where(up, w, lo)
        b = torch.where(up, hi, w)
        gap = b - a
        p = torch.where(gap > 0, (t - a) / torch.where(gap > 0, gap, torch.ones_like(gap)), torch.zeros_like(t))
        take_b = u < p
        move = torch.where(up, take_b.long(), -(~take_b).long())
        return move, None
    half = float(1 << (bits - 1))
    left = t < w
    width = torch.where(left, (w - lo) * 0.5, (hi - w) * 0.5)
    c = torch.where(width > 0, (t - w) / torch.where(width > 0, width, torch.ones_like(width)),
                    torch.zeros_like(t)).clamp(-1.0, 1.0)
    q = c * half
    fl = torch.floor(q)
    code = fl + (u < (q - fl)).to(q.dtype)
    to_upper = code >= half
    can = hi > w
    move = (to_upper & can).long()
    code = torch.where(to_upper, torch.where(can, torch.full_like(code, -half), torch.full_like(code, half - 1)), code)
    return move, code.to(torch.int8)


def latent(w, lo, hi, phase, bits: int):
    if bits == 0 or phase is None:
        return w
    c = (phase.to(torch.float32) / float(1 << (bits - 1))).to(w.dtype)
    return torch.where(c < 0, w + c * (w - lo) * 0.5, w + c * (hi - w) * 0.5)


def lattice_step_quantized(q: Quantized, fmt, phase: torch.Tensor | None, D: torch.Tensor, lr: float,
                           weight_decay: float = 0.0, phase_bits: int = 2, rounding: str = "sr", seed: int = 0,
                           step: int = 1, scale_rule: str = "fit", trust: float = 0.01, trust_lr: float = 1.0,
                           strata: int = 16):
    """One lattice update of a quantized tensor. ``phase``: int8 [rows, cols] (unpacked) or None for 0 bits.
    Returns (packed, scales, phase)."""
    fmt = F.get(fmt)
    spec = fmt.scale
    rows, cols = q.rows, q.cols
    rank = raw_to_rank(fmt, unpack(q.packed, fmt.container_bits, cols))
    L = num_levels(fmt)
    ze = expand_groups(q.zeros.to(torch.float32), rows, cols, spec) if q.zeros is not None else None
    a = rank_level(fmt, rank, ze)
    lo = rank_level(fmt, (rank - 1).clamp_min(0), ze)
    hi = rank_level(fmt, (rank + 1).clamp_max(L - 1), ze)
    s_old = effective_scales(q.scales, spec, q.global_scale).reshape(spec.shape(rows, cols))
    se = expand_groups(s_old, rows, cols, spec)
    d = D.reshape(rows, cols).to(torch.float32) / se
    decay = 1.0 - lr * weight_decay
    lam = latent(a, lo, hi, phase, phase_bits)
    tau = lam * decay - lr * d
    scales = q.scales
    s_new = s_old
    if scale_rule in ("fit", "fit_latent"):
        tf = a * decay - lr * d if scale_rule == "fit" else tau
        saa, sat, stt = group_sum(a * a, spec), group_sum(a * tf, spec), group_sum(tf * tf, spec)
        r = torch.where(saa > 0, sat / torch.where(saa > 0, saa, torch.ones_like(saa)), torch.ones_like(saa))
        kappa = trust if trust_lr <= 0 else min(trust, trust_lr * lr)
        r = r.clamp(math.exp(-kappa), math.exp(kappa))
        g = q.global_scale.to(torch.float32) if q.global_scale is not None else None
        enc = encode_scales(s_old * r / (g if g is not None else 1.0), spec)
        cand = effective_scales(enc, spec, q.global_scale).reshape(s_old.shape)
        rr = cand / s_old
        before = stt - 2 * sat + saa
        after = rr * rr * saa - 2 * rr * sat + stt
        ok = (after <= before) & torch.isfinite(cand) & (cand > 0)
        s_new = torch.where(ok, cand, s_old)
        scales = torch.where(ok.reshape(q.scales.shape), enc.reshape(q.scales.shape), q.scales)
    elif scale_rule != "fixed":
        raise ValueError(scale_rule)
    t = tau * se / expand_groups(s_new, rows, cols, spec)
    t = torch.minimum(torch.maximum(t, rank_level(fmt, torch.zeros_like(rank), ze)),
                      rank_level(fmt, torch.full_like(rank, L - 1), ze))
    r2 = nearest_rank(fmt, t, ze)
    w2 = rank_level(fmt, r2, ze)
    lo2 = rank_level(fmt, (r2 - 1).clamp_min(0), ze)
    hi2 = rank_level(fmt, (r2 + 1).clamp_max(L - 1), ze)
    u = rounding_u(rows * cols, seed, step, t.device, rounding, strata).reshape(rows, cols)
    move, code = _project(t, w2, lo2, hi2, u, phase_bits)
    r2 = (r2 + move).clamp(0, L - 1)
    packed = pack(rank_to_raw(fmt, r2), fmt.container_bits)
    return packed, scales, code


# native floating formats: neighbours by bit pattern

_INT_OF = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32,
           torch.float64: torch.int64}


def _ordinal(w: torch.Tensor) -> torch.Tensor:
    n = torch.finfo(w.dtype).bits
    u = w.contiguous().view(_INT_OF[w.dtype]).to(torch.int64) & ((1 << n) - 1) if n < 64 else \
        w.contiguous().view(torch.int64)
    if n == 64:
        return torch.where(u < 0, -(u & ((1 << 63) - 1)), u)
    sign = u >> (n - 1)
    mag = u & ((1 << (n - 1)) - 1)
    return torch.where(sign == 1, -mag, mag)


def _from_ordinal(o: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n = torch.finfo(dtype).bits
    if n == 64:
        u = torch.where(o < 0, (-o) | (-(1 << 63)), o)
        return u.view(torch.float64)
    u = torch.where(o >= 0, o, (1 << (n - 1)) | (-o))
    u = torch.where(u >= (1 << (n - 1)), u - (1 << n), u)
    return u.to(_INT_OF[dtype]).view(dtype)


def native_neighbors(w: torch.Tensor):
    o = _ordinal(w)
    lo = _from_ordinal(o - 1, w.dtype)
    hi = _from_ordinal(o + 1, w.dtype)
    lo = torch.where(torch.isfinite(lo), lo, w)
    hi = torch.where(torch.isfinite(hi), hi, w)
    return lo, hi


def lattice_step_native(w: torch.Tensor, phase: torch.Tensor | None, D: torch.Tensor, lr: float,
                        weight_decay: float = 0.0, phase_bits: int = 2, rounding: str = "sr", seed: int = 0,
                        step: int = 1, strata: int = 16):
    """One lattice update of a native floating tensor. Returns (new weights in w.dtype, phase)."""
    dt = w.dtype
    ct = torch.float32 if torch.finfo(dt).bits <= 16 else torch.float64   # compute type, as in the kernel
    lo, hi = native_neighbors(w)
    wf, lof, hif = w.to(ct), lo.to(ct), hi.to(ct)
    lam = latent(wf, lof, hif, phase, phase_bits)
    decay = torch.tensor(1.0 - lr * weight_decay, dtype=ct, device=w.device)
    lr_t = torch.tensor(lr, dtype=ct, device=w.device)
    t = lam * decay - lr_t * D.to(torch.float32).to(ct).reshape(w.shape)
    big = torch.finfo(dt).max
    t = t.clamp(-big, big)
    w2 = t.to(dt)
    lo2, hi2 = native_neighbors(w2)
    u = rounding_u(w.numel(), seed, step, w.device, rounding, strata).reshape(w.shape).to(ct)
    move, code = _project(t, w2.to(ct), lo2.to(ct), hi2.to(ct), u, phase_bits)
    out = torch.where(move > 0, hi2, torch.where(move < 0, lo2, w2))
    return out, code


# directions

_PE = ((8.28721201814563, -23.595886519098837, 17.300387312530933),
       (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
       (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
       (3.3184196573706015, -2.488488024314874, 0.51004894012372),
       (2.300652019954817, -1.6689039845747493, 0.4188073119525673))


def ns_coefficients(kind: str, i: int) -> tuple[float, float, float]:
    """(a, b, c) of iteration ``i``: 'muon' (fixed quintic) or 'polar_express' (Amsel et al. 2025, safety 1.05)."""
    if kind == "muon":
        return 3.4445, -4.7750, 2.0315
    a, b, c = _PE[min(i, 4)]
    return a / 1.05, b / 1.05 ** 3, c / 1.05 ** 5


def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7, coeffs: str = "muon",
                  algo: str = "standard") -> torch.Tensor:
    """Polar factor approximation. ``algo='gram'`` iterates on the Gram matrix (restart after 2 steps)."""
    dtype = g.dtype
    x = (g.float() / (g.float().norm(dim=(-2, -1), keepdim=True) + eps)).to(dtype)
    tr = x.shape[-2] > x.shape[-1]
    if tr:
        x = x.mT
    gram = algo == "gram" or (algo == "auto" and x.shape[-1] >= 2 * x.shape[-2] and dtype != torch.bfloat16)
    if not gram:
        for i in range(steps):
            a, b, c = ns_coefficients(coeffs, i)
            s = x @ x.mT
            x = a * x + (b * s + c * s @ s) @ x
        return x.mT if tr else x
    eye = torch.eye(x.shape[-2], device=x.device, dtype=x.dtype)
    R = x @ x.mT
    Q = None
    for i in range(steps):
        a, b, c = ns_coefficients(coeffs, i)
        if i == 2:
            x = Q @ x
            R = x @ x.mT
        P = a * eye + b * R + c * R @ R
        Q = P if i in (0, 2) else Q @ P
        if i + 1 not in (steps, 2):
            R = P @ R @ P
    x = Q @ x
    return x.mT if tr else x


def muon_direction(G: torch.Tensor, block=0, steps: int = 5, dtype=torch.float32, coeffs: str = "muon",
                   algo: str = "standard") -> torch.Tensor:
    """Block-diagonal orthogonalized direction; ``block`` is an int (square) or (rows, cols); cols = -1: bands of
    ``rows`` rows over the full width."""
    g = view2d(G).to(dtype)
    m, n = g.shape
    br, bc = (block, block) if isinstance(block, int) else block
    bc = n if bc is None or bc < 0 else br if bc == 0 else bc
    if br <= 0 or (m <= br and n <= bc):
        return (newton_schulz(g, steps, coeffs=coeffs, algo=algo) * math.sqrt(max(m, n))).float()
    X = torch.nn.functional.pad(g, (0, (-n) % bc, 0, (-m) % br))
    M, N = X.shape
    X = X.view(M // br, br, N // bc, bc).permute(0, 2, 1, 3).reshape(-1, br, bc)
    Y = newton_schulz(X, steps, coeffs=coeffs, algo=algo) * math.sqrt(max(br, bc))
    Y = Y.view(M // br, N // bc, br, bc).permute(0, 2, 1, 3).reshape(M, N)
    return Y[:m, :n].float()


def factored_direction(G: torch.Tensor, state: dict, beta2: float = 0.95, eps: float = 1e-12) -> torch.Tensor:
    X = view2d(G).to(torch.float32)
    pk = X.abs().amax()
    X = X / torch.where(pk > 0, pk, torch.ones_like(pk))
    m, n = X.shape
    if "r" not in state:
        state["r"] = torch.zeros(m, device=X.device)
        state["c"] = torch.zeros(n, device=X.device)
        state["k"] = 0
    state["k"] += 1
    state["r"].mul_(beta2).add_(X.square().mean(1), alpha=1 - beta2)
    state["c"].mul_(beta2).add_(X.square().mean(0), alpha=1 - beta2)
    bc = 1 - beta2 ** state["k"]
    r, c = state["r"] / bc, state["c"] / bc
    V = (r / r.mean().clamp_min(eps))[:, None] * c[None, :]
    return X / (V.sqrt() + eps)


def sinkhorn_direction(G: torch.Tensor, iters: int = 2) -> torch.Tensor:
    U = view2d(G).to(torch.float32)
    for _ in range(iters):
        U = U / (U.square().mean(1, keepdim=True).sqrt() + 1e-30)
        U = U / (U.square().mean(0, keepdim=True).sqrt() + 1e-30)
    return U


def clip_rms(U: torch.Tensor, clip: float = 1.0) -> torch.Tensor:
    rho = U.square().mean().sqrt()
    return U / torch.clamp(rho / clip, min=1.0)


def row_normalize(U: torch.Tensor, v: torch.Tensor, k: int, beta: float = 0.95, eps: float = 1e-8,
                  clip: float = 1.0) -> torch.Tensor:
    """Neuron-wise normalization (NorMuon): v <- beta v + (1 - beta) mean_cols(U^2) (in place), U / (sqrt(v / (1 -
    beta^k)) + eps) per row, then the RMS clip."""
    v.mul_(beta).add_(U.square().mean(1), alpha=1 - beta)
    rowf = 1.0 / ((v / (1 - beta ** max(k, 1))).sqrt() + eps)
    return clip_rms(U * rowf[:, None], clip)
