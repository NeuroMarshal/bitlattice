"""Lossless conversion of pre-quantized checkpoint layouts into LatticeTensor parts (and back for NVFP4).

Layouts:
    ct_nvfp4   compressed-tensors 'nvfp4-pack-quantized': weight_packed u8 [out, in/2], weight_scale e4m3
               [out, in/16], weight_global_scale f32 (dequant: e2m1 * weight_scale / weight_global_scale)
    mo_nvfp4   ModelOpt NVFP4: weight u8 [out, in/2], weight_scale e4m3 [out, in/16], weight_scale_2 f32
               (dequant: e2m1 * weight_scale * weight_scale_2)
    fp8        weight e4m3 [out, in] with weight_scale f32 [out, 1] or [1] (compressed-tensors 'float-quantized')
    fp8_block  weight e4m3 [out, in] with weight_scale_inv f32 [out/B, in/B] (dequant: w * scale_inv)
    ct_int4    compressed-tensors 'pack-quantized': weight_packed i32 [out, in/8] (offset-8 nibbles),
               weight_scale [out, in/group] (+ optional weight_zero_point)
    gptq       qweight i32 [in/8, out], qzeros i32 [in/group, out/8], scales f16 [in/group, out], g_idx;
               v1 stores zero-1
    awq        qweight i32 [in, out/8], qzeros i32 [in/group, out/8], scales f16 [in/group, out];
               nibble order [0, 2, 4, 6, 1, 3, 5, 7]
E2M1 nibbles are packed low-first, as in bitlattice.
"""
from __future__ import annotations

import torch

from .. import formats as F
from ..tensor import LatticeTensor

__all__ = [
    "from_awq",
    "from_ct_int4",
    "from_ct_nvfp4",
    "from_fp8",
    "from_fp8_block",
    "from_gptq",
    "from_mo_nvfp4",
    "to_ct_nvfp4",
    "unpack_int32_nibbles",
]

AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def _nvfp4(packed, scale_e4m3, global_mult, shape, dtype, device):
    return LatticeTensor(packed.to(device, torch.uint8).contiguous(),
                         scale_e4m3.to(device).view(torch.float8_e4m3fn).contiguous(),
                         global_mult.to(device, torch.float32).reshape(1), None, F.get("nvfp4"), shape, dtype)


def from_ct_nvfp4(t: dict, dtype=torch.bfloat16, device="cuda") -> LatticeTensor:
    out, half = t["weight_packed"].shape
    return _nvfp4(t["weight_packed"], t["weight_scale"], 1.0 / t["weight_global_scale"].float(), (out, half * 2),
                  dtype, device)


def from_mo_nvfp4(t: dict, dtype=torch.bfloat16, device="cuda") -> LatticeTensor:
    out, half = t["weight"].shape
    return _nvfp4(t["weight"], t["weight_scale"], t["weight_scale_2"].float(), (out, half * 2), dtype, device)


def from_fp8(t: dict, dtype=torch.bfloat16, device="cuda") -> LatticeTensor:
    w = t["weight"]
    out, inp = w.shape
    s = t.get("weight_scale")
    s = torch.ones(1, device=w.device) if s is None else s.float()
    group = 0 if s.numel() == 1 else -1
    fmt = F.get("fp8_e4m3").with_scale(group=group, dtype="fp32")
    scales = s.reshape(1, 1) if group == 0 else s.reshape(out, 1)
    return LatticeTensor(w.view(torch.uint8).to(device).contiguous(), scales.to(device).contiguous(), None, None, fmt,
                         (out, inp), dtype)


def from_fp8_block(t: dict, dtype=torch.bfloat16, device="cuda", block: tuple[int, int] = (128, 128)) -> LatticeTensor:
    w = t["weight"]
    out, inp = w.shape
    br, bc = block
    s = t["weight_scale_inv"].float()
    if inp % bc:
        raise ValueError(f"input features {inp} not a multiple of the block {bc}")
    scales = s.repeat_interleave(br, dim=0)[:out]                           # [out, in/bc]
    fmt = F.get("fp8_e4m3").with_scale(group=bc, dtype="fp32")
    return LatticeTensor(w.view(torch.uint8).to(device).contiguous(), scales.to(device).contiguous(), None, None, fmt,
                         (out, inp), dtype)


def unpack_int32_nibbles(x: torch.Tensor, dim: int) -> torch.Tensor:
    """int32 words holding 8 nibbles (low first) -> unsigned values, 8x along ``dim``."""
    shifts = torch.arange(0, 32, 4, device=x.device, dtype=torch.int32)
    v = (x.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF
    v = v.movedim(-1, dim + 1 if dim >= 0 else dim)
    shape = list(x.shape)
    shape[dim] *= 8
    return v.reshape(shape)


def _pack_nibbles(u: torch.Tensor) -> torch.Tensor:
    """[out, in] unsigned nibbles -> uint8 [out, in/2], low nibble first."""
    u = u.to(torch.int32)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).to(torch.uint8)


def from_ct_int4(t: dict, group: int, dtype=torch.bfloat16, device="cuda") -> LatticeTensor:
    wp = t["weight_packed"]
    out = wp.shape[0]
    u = unpack_int32_nibbles(wp.to(device), dim=1)                         # [out, in] unsigned
    inp = int(t["weight_shape"][1]) if "weight_shape" in t else u.shape[1]
    u = u[:, :inp]
    s = t["weight_scale"].to(device)
    zp = t.get("weight_zero_point")
    if zp is None:
        codes = (u ^ 8)                                                     # offset-8 -> two's complement nibble
        fmt = F.uniform(4, group=group, scale="fp16" if s.dtype == torch.float16 else "bf16")
        zeros = None
    else:
        codes = u
        fmt = F.uniform(4, group=group, scale="fp16" if s.dtype == torch.float16 else "bf16", zero_point=True)
        z = zp.to(device)
        zeros = (unpack_int32_nibbles(z, dim=0)[:out] if z.dtype == torch.int32 else z).to(torch.float16)
    return LatticeTensor(_pack_nibbles(codes).contiguous(), s.reshape(out, -1).contiguous(), None,
                         None if zeros is None else zeros.reshape(out, -1).contiguous(), fmt, (out, inp), dtype)


def from_gptq(t: dict, group: int, dtype=torch.bfloat16, device="cuda", v1_zero_offset: bool = True) -> LatticeTensor:
    qw = t["qweight"].to(device)
    inp, out = qw.shape[0] * 8, qw.shape[1]
    g_idx = t.get("g_idx")
    if g_idx is not None:
        natural = torch.arange(inp, device=g_idx.device, dtype=g_idx.dtype) // group
        if not torch.equal(g_idx.to(natural.device), natural):
            raise NotImplementedError("act-order GPTQ (permuted groups) is loaded densely")
    u = unpack_int32_nibbles(qw, dim=0).t().contiguous()                   # [out, in]
    z = unpack_int32_nibbles(t["qzeros"].to(device), dim=1)[:, :out]        # [in/group, out]
    if v1_zero_offset:
        z = (z + 1) & 0xF
    fmt = F.uniform(4, group=group, scale="fp16", zero_point=True)
    return LatticeTensor(_pack_nibbles(u).contiguous(), t["scales"].to(device).t().contiguous().to(torch.float16), None,
                         z.t().contiguous().to(torch.float16), fmt, (out, inp), dtype)


def _awq_unpack(x: torch.Tensor) -> torch.Tensor:
    v = unpack_int32_nibbles(x, dim=1)                                     # natural slot order
    rows, cols = v.shape
    v = v.reshape(rows, cols // 8, 8)
    rev = torch.tensor([AWQ_ORDER.index(c) for c in range(8)], device=x.device)
    return v[..., rev].reshape(rows, cols)


def from_awq(t: dict, group: int, dtype=torch.bfloat16, device="cuda") -> LatticeTensor:
    qw = t["qweight"].to(device)
    inp = qw.shape[0]
    u = _awq_unpack(qw).t().contiguous()                                    # [out, in]
    out = u.shape[0]
    z = _awq_unpack(t["qzeros"].to(device))[:, :out]                         # [in/group, out]
    fmt = F.uniform(4, group=group, scale="fp16", zero_point=True)
    return LatticeTensor(_pack_nibbles(u).contiguous(), t["scales"].to(device).t().contiguous().to(torch.float16), None,
                         z.t().contiguous().to(torch.float16), fmt, (out, inp), dtype)


def to_ct_nvfp4(w: LatticeTensor) -> dict[str, torch.Tensor]:
    """compressed-tensors NVFP4 tensors of an nvfp4 LatticeTensor without high-precision columns."""
    if w.fmt.name != "nvfp4" or w.hi_idx is not None:
        raise ValueError("only plain nvfp4 tensors are exported to compressed-tensors")
    return {"weight_packed": w.packed.contiguous(), "weight_scale": w.scales.contiguous(),
            "weight_global_scale": (1.0 / w.global_scale.float()).reshape(1)}
