"""Number formats as ordered lattices.

Every format exposes the representable values of a weight as an ordered set. Three families:

* ``TableFormat``   -- codes of at most 8 bits decoded through a lookup table (ExMy floats, NF4, binary, custom
                       tables), multiplied by a per-group scale (and optionally a per-tensor global scale).
* ``UniformFormat`` -- integer grid ``(k - zero) * scale`` with ``k`` in ``[qmin, qmax]``; 1..32-bit codes.
* ``NativeFormat``  -- torch floating dtypes (bf16, fp16, fp32, fp64); neighbours are adjacent bit patterns.

Codes are stored in the hardware encoding of the format (E2M1 nibbles for fp4/nvfp4/mxfp4, E4M3 bytes for fp8,
two's complement for integers), packed little-endian along the last dimension. Scales cover ``group`` consecutive
elements of the last dimension: ``group > 0`` -- fixed group, ``group == -1`` -- one scale per row,
``group == 0`` -- one scale per tensor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import torch

__all__ = [
    "Format", "TableFormat", "UniformFormat", "NativeFormat", "ScaleSpec",
    "get", "register", "available", "exmy_values", "table", "uniform", "native",
]

SCALE_DTYPES = ("fp32", "fp16", "bf16", "e4m3", "e8m0")
E4M3_MAX = 448.0
E4M3_MIN_NORMAL = 2.0 ** -6
E4M3_MIN = 2.0 ** -9


@dataclass(frozen=True)
class ScaleSpec:
    """Group scale layout: ``group`` elements per scale along the last dim, stored in ``dtype``."""

    group: int = 64
    dtype: str = "fp16"
    global_scale: bool = False

    def __post_init__(self):
        if self.dtype not in SCALE_DTYPES:
            raise ValueError(f"scale dtype must be one of {SCALE_DTYPES}, got {self.dtype!r}")
        if self.group < -1:
            raise ValueError("group must be > 0, -1 (per row) or 0 (per tensor)")

    @property
    def bits(self) -> int:
        return 32 if self.dtype == "fp32" else 16 if self.dtype in ("fp16", "bf16") else 8

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16,
                "e4m3": torch.float8_e4m3fn, "e8m0": torch.uint8}[self.dtype]

    def groups_per_row(self, cols: int) -> int:
        if self.group == -1 or self.group == 0:
            return 1
        if cols % self.group:
            raise ValueError(f"last dimension {cols} is not a multiple of the scale group {self.group}")
        return cols // self.group

    def group_size(self, rows: int, cols: int) -> int:
        return cols if self.group == -1 else rows * cols if self.group == 0 else self.group

    def shape(self, rows: int, cols: int) -> tuple[int, int]:
        return (1, 1) if self.group == 0 else (rows, self.groups_per_row(cols))


class Format:
    """Base class. ``bits`` is the logical code width, ``container_bits`` the storage width per code."""

    name: str
    bits: int
    kind: str

    @property
    def container_bits(self) -> int:
        b = self.bits
        return 1 if b == 1 else 2 if b == 2 else 4 if b <= 4 else 8 if b <= 8 else 16 if b <= 16 else 32

    @property
    def scale(self) -> ScaleSpec | None:
        return None

    def storage_bits(self, rows: int, cols: int) -> int:
        """Logical storage (codes + scales + global scale) in bits."""
        bits = rows * cols * self.bits
        sc = self.scale
        if sc is not None:
            r, g = sc.shape(rows, cols)
            bits += r * g * sc.bits + (32 if sc.global_scale else 0)
            if getattr(self, "zero_point", False):
                bits += r * g * 16
        return bits

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"


@dataclass(frozen=True, repr=False)
class TableFormat(Format):
    """Codes of ``bits`` <= 8 bits; ``values[code]`` is the level (NaN marks an unused code)."""

    name: str
    bits: int
    values: tuple[float, ...]
    scale_spec: ScaleSpec = field(default_factory=ScaleSpec)
    init: str = "absmax"          # absmax | absmean
    kind: str = "table"

    def __post_init__(self):
        if not 1 <= self.bits <= 8:
            raise ValueError("table formats hold 1..8-bit codes")
        if len(self.values) != 1 << self.bits:
            raise ValueError(f"{self.name}: need {1 << self.bits} code values, got {len(self.values)}")
        if len(self.levels) < 2:
            raise ValueError(f"{self.name}: fewer than two distinct levels")

    @property
    def scale(self) -> ScaleSpec:
        return self.scale_spec

    @property
    def levels(self) -> tuple[float, ...]:
        return tuple(sorted({v for v in self.values if not math.isnan(v)}))

    @property
    def max_abs(self) -> float:
        return max(abs(v) for v in self.levels)

    def with_scale(self, **kw) -> "TableFormat":
        spec = ScaleSpec(**{**self.scale_spec.__dict__, **kw})
        return TableFormat(self.name, self.bits, self.values, spec, self.init)

    @lru_cache(maxsize=None)
    def tables(self, device: str = "cpu") -> dict[str, torch.Tensor]:
        """Lookup tables used by kernels: decode, rank_of_code, sorted values, canonical code per rank."""
        levels = self.levels
        rank = {v: i for i, v in enumerate(levels)}
        canonical: dict[float, int] = {}
        for code, v in enumerate(self.values):
            if not math.isnan(v) and v not in canonical:
                canonical[v] = code
        if 0.0 in canonical:                       # prefer +0 as the canonical zero code
            canonical[0.0] = next(c for c, v in enumerate(self.values)
                                  if v == 0.0 and not math.copysign(1.0, v) < 0)
        dev = torch.device(device)
        return {
            "decode": torch.tensor(self.values, dtype=torch.float32, device=dev),
            "rank": torch.tensor([rank.get(v, -1) if not math.isnan(v) else -1 for v in self.values],
                                 dtype=torch.int32, device=dev),
            "sorted": torch.tensor(levels, dtype=torch.float32, device=dev),
            "sorted_code": torch.tensor([canonical[v] for v in levels], dtype=torch.int32, device=dev),
        }


@dataclass(frozen=True, repr=False)
class UniformFormat(Format):
    """Integer grid: value = (k - zero) * scale, k in [qmin, qmax]. Codes are k in two's complement."""

    name: str
    bits: int
    qmin: int
    qmax: int
    scale_spec: ScaleSpec = field(default_factory=ScaleSpec)
    zero_point: bool = False
    init: str = "absmax"
    kind: str = "uniform"

    def __post_init__(self):
        if not 1 <= self.bits <= 32:
            raise ValueError("uniform formats hold 1..32-bit codes")
        lo, hi = -(1 << (self.bits - 1)), (1 << (self.bits - 1)) - 1
        if self.zero_point:
            lo, hi = 0, (1 << self.bits) - 1
        if not (lo <= self.qmin < self.qmax <= hi):
            raise ValueError(f"{self.name}: [{self.qmin}, {self.qmax}] does not fit {self.bits} bits")

    @property
    def scale(self) -> ScaleSpec:
        return self.scale_spec

    @property
    def max_abs(self) -> float:
        return float(max(abs(self.qmin), abs(self.qmax)))

    @property
    def levels(self) -> tuple[float, ...]:
        return tuple(float(k) for k in range(self.qmin, self.qmax + 1))

    def with_scale(self, **kw) -> "UniformFormat":
        spec = ScaleSpec(**{**self.scale_spec.__dict__, **kw})
        return UniformFormat(self.name, self.bits, self.qmin, self.qmax, spec, self.zero_point, self.init)


@dataclass(frozen=True, repr=False)
class NativeFormat(Format):
    """Weights stored as a torch floating dtype; the lattice is the set of finite values of that dtype."""

    name: str
    dtype: torch.dtype
    kind: str = "native"

    @property
    def bits(self) -> int:
        return torch.finfo(self.dtype).bits


# ----------------------------------------------------------------------------------------------------------------
# constructors

def exmy_values(e: int, m: int, bias: int | None = None, special: str = "none") -> tuple[float, ...]:
    """Decode table of a 1+e+m bit sign/exponent/mantissa float with subnormals.

    special: 'none' (all codes finite), 'fn' (all-ones exponent+mantissa is NaN, as float8_e4m3fn),
    'ieee' (all-ones exponent is inf/NaN, excluded).
    """
    bias = (1 << (e - 1)) - 1 if bias is None else bias
    out = []
    for code in range(1 << (1 + e + m)):
        s = -1.0 if code >> (e + m) else 1.0
        ex = (code >> m) & ((1 << e) - 1)
        mant = code & ((1 << m) - 1)
        if special == "ieee" and ex == (1 << e) - 1:
            out.append(float("nan"))
            continue
        if special == "fn" and ex == (1 << e) - 1 and mant == (1 << m) - 1:
            out.append(float("nan"))
            continue
        if ex == 0:
            v = mant / (1 << m) * 2.0 ** (1 - bias)
        else:
            v = (1 + mant / (1 << m)) * 2.0 ** (ex - bias)
        out.append(s * v)
    return tuple(out)


NF4_VALUES = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
              -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
              0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
              0.7229568362236023, 1.0)


def table(name: str, values, bits: int | None = None, group: int = 64, scale: str = "fp16",
          global_scale: bool = False, init: str = "absmax") -> TableFormat:
    """Custom table format. ``values`` lists the level of every code (pad with NaN for unused codes)."""
    values = tuple(float(v) for v in values)
    bits = bits or max(1, math.ceil(math.log2(len(values))))
    values = values + (float("nan"),) * ((1 << bits) - len(values))
    return TableFormat(name, bits, values, ScaleSpec(group, scale, global_scale), init)


def uniform(bits: int, group: int = 64, scale: str = "fp16", symmetric: bool = False, zero_point: bool = False,
            name: str | None = None, init: str = "absmax") -> UniformFormat:
    """Integer format of ``bits`` bits. ``symmetric`` drops the most negative code; ``zero_point`` makes codes
    unsigned with a per-group zero (GPTQ/AWQ style)."""
    if zero_point:
        qmin, qmax = 0, (1 << bits) - 1
    else:
        qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        if symmetric:
            qmin = -qmax
    return UniformFormat(name or f"int{bits}", bits, qmin, qmax, ScaleSpec(group, scale), zero_point, init)


def native(dtype: torch.dtype) -> NativeFormat:
    return NativeFormat(str(dtype).replace("torch.", ""), dtype)


_E2M1 = exmy_values(2, 1)
_REGISTRY: dict[str, object] = {}


def register(name: str, fmt_or_factory) -> None:
    """Register a format (or a zero-argument factory) under ``name``."""
    _REGISTRY[name.lower()] = fmt_or_factory


def available() -> list[str]:
    return sorted(_REGISTRY)


def get(spec) -> Format:
    """Resolve a format from a name ('nvfp4', 'int4', 'fp6_e3m2', 'bf16', ...), a torch dtype or a Format."""
    if isinstance(spec, Format):
        return spec
    if isinstance(spec, torch.dtype):
        if not spec.is_floating_point:
            raise ValueError(f"{spec} is not a floating dtype; use uniform(bits) for integers")
        return native(spec)
    key = str(spec).lower()
    if key in _REGISTRY:
        f = _REGISTRY[key]
        return f() if callable(f) else f
    if key.startswith("int") and key[3:].isdigit():
        return uniform(int(key[3:]))
    if key.startswith("uint") and key[4:].isdigit():
        return uniform(int(key[4:]), zero_point=True, name=key)
    if key.startswith("fp") and "_e" in key:                   # fp6_e3m2, fp5_e2m2, ...
        e, m = key.split("_e", 1)[1].split("m")
        return TableFormat(key, 1 + int(e) + int(m), exmy_values(int(e), int(m)), ScaleSpec(32, "fp16"))
    raise KeyError(f"unknown format {spec!r}; known: {', '.join(available())}")


for _name, _fmt in {
    "fp4": TableFormat("fp4", 4, _E2M1, ScaleSpec(16, "fp32")),
    "e2m1": TableFormat("e2m1", 4, _E2M1, ScaleSpec(16, "fp32")),
    "nvfp4": TableFormat("nvfp4", 4, _E2M1, ScaleSpec(16, "e4m3", global_scale=True)),
    "mxfp4": TableFormat("mxfp4", 4, _E2M1, ScaleSpec(32, "e8m0")),
    "nf4": TableFormat("nf4", 4, NF4_VALUES, ScaleSpec(64, "fp16")),
    "fp6_e2m3": TableFormat("fp6_e2m3", 6, exmy_values(2, 3), ScaleSpec(32, "e8m0")),
    "fp6_e3m2": TableFormat("fp6_e3m2", 6, exmy_values(3, 2), ScaleSpec(32, "e8m0")),
    "fp8": TableFormat("fp8", 8, exmy_values(4, 3, special="fn"), ScaleSpec(-1, "fp32")),
    "fp8_e4m3": TableFormat("fp8_e4m3", 8, exmy_values(4, 3, special="fn"), ScaleSpec(-1, "fp32")),
    "fp8_e5m2": TableFormat("fp8_e5m2", 8, exmy_values(5, 2, special="ieee"), ScaleSpec(-1, "fp32")),
    "binary": TableFormat("binary", 1, (-1.0, 1.0), ScaleSpec(128, "fp16"), init="absmean"),
    "ternary": UniformFormat("ternary", 2, -1, 1, ScaleSpec(128, "fp16"), init="absmean"),
    "int8": uniform(8, group=-1, scale="fp32", symmetric=True),
    "int4": uniform(4, group=64, scale="fp16"),
    "int3": uniform(3, group=64, scale="fp16"),
    "int2": uniform(2, group=64, scale="fp16"),
    "bf16": native(torch.bfloat16),
    "fp16": native(torch.float16),
    "fp32": native(torch.float32),
    "fp64": native(torch.float64),
}.items():
    register(_name, _fmt)
