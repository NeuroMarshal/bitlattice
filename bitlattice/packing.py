"""Bit packing of integer codes along the last dimension (little-endian inside each byte)."""
from __future__ import annotations

import torch

__all__ = ["container_dtype", "pack", "packed_cols", "sign_extend", "unpack"]


def container_dtype(container_bits: int) -> torch.dtype:
    return {16: torch.int16, 32: torch.int32, 64: torch.int64}.get(container_bits, torch.uint8)


def packed_cols(cols: int, container_bits: int) -> int:
    if container_bits >= 8:
        return cols
    per_byte = 8 // container_bits
    if cols % per_byte:
        raise ValueError(f"last dimension {cols} is not a multiple of {per_byte} ({container_bits}-bit codes)")
    return cols // per_byte


def pack(codes: torch.Tensor, container_bits: int) -> torch.Tensor:
    """Pack raw code bit patterns (any integer dtype) into bytes (or int16/int32 containers)."""
    if container_bits == 8:
        return (codes.to(torch.int32) & 0xFF).to(torch.uint8)
    if container_bits in (16, 32, 64):
        return codes.to(container_dtype(container_bits))
    per_byte = 8 // container_bits
    mask = (1 << container_bits) - 1
    c = (codes.to(torch.int32) & mask).reshape(*codes.shape[:-1], packed_cols(codes.shape[-1], container_bits),
                                                per_byte)
    shifts = torch.arange(per_byte, device=codes.device, dtype=torch.int32) * container_bits
    return (c << shifts).sum(-1, dtype=torch.int32).to(torch.uint8)


def unpack(packed: torch.Tensor, container_bits: int, cols: int) -> torch.Tensor:
    """Inverse of :func:`pack`; returns raw (unsigned) bit patterns as int32."""
    if container_bits == 8:
        return packed.to(torch.int32) & 0xFF
    if container_bits == 64:
        return packed.to(torch.int64)
    if container_bits in (16, 32):
        return packed.to(torch.int32)
    per_byte = 8 // container_bits
    mask = (1 << container_bits) - 1
    shifts = torch.arange(per_byte, device=packed.device, dtype=torch.int32) * container_bits
    c = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & mask
    return c.reshape(*packed.shape[:-1], packed.shape[-1] * per_byte)[..., :cols]


def sign_extend(raw: torch.Tensor, bits: int) -> torch.Tensor:
    """Interpret the low ``bits`` bits of ``raw`` as a two's complement integer."""
    if bits >= 64:
        return raw.to(torch.int64)
    if bits >= 32 and raw.dtype != torch.int64:
        return raw.to(torch.int64)
    raw = raw.to(torch.int64) & ((1 << bits) - 1)
    return torch.where(raw >= (1 << (bits - 1)), raw - (1 << bits), raw)
