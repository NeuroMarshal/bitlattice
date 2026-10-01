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
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache

import torch

__all__ = [
    "Format", "TableFormat", "UniformFormat", "NativeFormat", "ScaleSpec",
    "get", "register", "available", "exmy_values", "table", "uniform", "native", "fit_group",
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
    strict: bool = False          # group fixed by a hardware layout (nvfp4, mxfp4); never adapted per layer

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
        return 1 if b == 1 else 2 if b == 2 else 4 if b <= 4 else 8 if b <= 8 else 16 if b <= 16 else 32 if b <= 32 \
            else 64

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
        if not 1 <= self.bits <= 64:
            raise ValueError("uniform formats hold 1..64-bit codes")
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
    'ieee' (all-ones exponent is inf/NaN, excluded), 'fnuz' (no negative zero: that code is NaN; bias + 1, as the
    *_fnuz float8 types).
    """
    if bias is None:
        bias = (1 << (e - 1)) - (0 if special == "fnuz" else 1)
    out = []
    for code in range(1 << (1 + e + m)):
        if special == "fnuz" and code == 1 << (e + m):
            out.append(float("nan"))
            continue
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


def normal_float_values(bits: int) -> tuple[float, ...]:
    """NF-k levels (QLoRA): quantiles of N(0, 1) with an exact zero, normalized to [-1, 1]; 2**bits codes."""
    if bits == 4:
        return NF4_VALUES
    n = 1 << bits
    normal = torch.distributions.Normal(0.0, 1.0)
    offset = 0.9677                                                   # as in the NF4 construction
    pos = normal.icdf(torch.linspace(offset, 0.5, n // 2 + 1)[:-1]).tolist()
    neg = (-normal.icdf(torch.linspace(offset, 0.5, n // 2)[:-1])).tolist()
    vals = sorted(set([v / pos[0] for v in pos] + [0.0] + [v / pos[0] for v in neg]))
    vals = vals[:n] + [float("nan")] * (n - len(vals))
    return tuple(vals)


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


def fit_group(fmt: Format, cols: int) -> Format | None:
    """Format with its scale group adapted to a layer of ``cols`` input features: the largest divisor of ``cols``
    that is a multiple of 8 and not above the nominal group, else one scale per row. None if impossible (strict
    hardware groups or ``cols`` not a multiple of 8)."""
    sc = fmt.scale
    if sc is None or sc.group <= 0 or cols % sc.group == 0:
        return fmt
    if sc.strict or cols % 8:
        return None
    g = next((g for g in range(sc.group - sc.group % 8, 7, -8) if cols % g == 0), -1)
    return fmt.with_scale(group=g)


def native(dtype: torch.dtype) -> NativeFormat:
    return NativeFormat(str(dtype).replace("torch.", ""), dtype)


_E2M1 = exmy_values(2, 1)
_REGISTRY: dict[str, object] = {}


def register(name: str, fmt_or_factory) -> None:
    """Register a format (or a zero-argument factory) under ``name``."""
    _REGISTRY[name.lower()] = fmt_or_factory


def available() -> list[str]:
    return sorted(_REGISTRY)


_SUFFIX = re.compile(r"_(g\d+|grow|gt|s(?:fp32|fp16|bf16|e4m3|e8m0)|gs|sym|asym|4o6)$")


def get(spec) -> Format:
    """Resolve a format.

    Names: a registered format ('nvfp4', 'mxfp8', 'q4_0', ...), or a family with parameters --
    ``intN`` / ``uintN`` (N = 1..64; uint has a per-group zero point), ``fpN_eXmY`` / ``eXmY`` (any sign/exponent/
    mantissa split up to 8 bits), ``nfN`` (normal float) -- followed by optional suffixes:
    ``_g<G>`` group size, ``_grow`` one scale per row, ``_gt`` one per tensor, ``_s<dtype>`` scale dtype
    (fp32, fp16, bf16, e4m3, e8m0), ``_gs`` per-tensor global scale, ``_sym`` / ``_asym`` integer range,
    ``_4o6`` four-over-six scale choice (blocks map their maximum to the top or the second level).
    Also torch floating dtypes and Format objects."""
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
    base, mods = key, []
    while True:
        m = _SUFFIX.search(base)
        if not m:
            break
        mods.insert(0, m.group(1))
        base = base[:m.start()]
    if not mods:
        fmt = _family(base)
        if fmt is None:
            raise KeyError(f"unknown format {spec!r}; known: {', '.join(available())}")
        return fmt
    fmt = get(base) if base in _REGISTRY else _family(base, "sym" in mods)
    if fmt is None:
        raise KeyError(f"unknown format {spec!r}")
    scale_kw = {}
    for mod in mods:
        if mod.startswith("g") and mod[1:].isdigit():
            scale_kw["group"] = int(mod[1:])
        elif mod == "grow":
            scale_kw["group"] = -1
        elif mod == "gt":
            scale_kw["group"] = 0
        elif mod.startswith("s") and mod != "sym":
            scale_kw["dtype"] = mod[1:]
        elif mod == "gs":
            scale_kw["global_scale"] = True
    if scale_kw:
        scale_kw.setdefault("strict", False)
        fmt = fmt.with_scale(**scale_kw)
    if "4o6" in mods:
        fmt = replace(fmt, name=fmt.name + "_4o6", init="4over6")
    if scale_kw or "4o6" in mods:
        fmt = replace(fmt, name=key)
    return fmt


def _family(key: str, symmetric: bool = False) -> Format | None:
    m = re.fullmatch(r"(u?)int(\d+)", key)
    if m:
        bits = int(m.group(2))
        if not 1 <= bits <= 64:
            return None
        group, scale = (64, "fp16") if bits <= 8 else (-1, "fp32")
        return uniform(bits, group=group, scale=scale, zero_point=bool(m.group(1)), symmetric=symmetric, name=key)
    m = re.fullmatch(r"(?:fp(\d+)_)?e(\d+)m(\d+)(fn|fnuz)?", key)
    if m:
        e, mm = int(m.group(2)), int(m.group(3))
        bits = 1 + e + mm
        if m.group(1) and int(m.group(1)) != bits or not 2 <= bits <= 8:
            return None
        special = m.group(4) or "none"
        return TableFormat(key, bits, exmy_values(e, mm, special=special), ScaleSpec(32, "fp16"))
    m = re.fullmatch(r"nf(\d)", key)
    if m and 2 <= int(m.group(1)) <= 8:
        return TableFormat(key, int(m.group(1)), normal_float_values(int(m.group(1))), ScaleSpec(64, "fp16"))
    return None


for _name, _fmt in {
    "fp4": TableFormat("fp4", 4, _E2M1, ScaleSpec(16, "fp32")),
    "e2m1": TableFormat("e2m1", 4, _E2M1, ScaleSpec(16, "fp32")),
    "nvfp4": TableFormat("nvfp4", 4, _E2M1, ScaleSpec(16, "e4m3", global_scale=True, strict=True)),
    "mxfp4": TableFormat("mxfp4", 4, _E2M1, ScaleSpec(32, "e8m0", strict=True)),
    "nf4": TableFormat("nf4", 4, NF4_VALUES, ScaleSpec(64, "fp16")),
    "fp6_e2m3": TableFormat("fp6_e2m3", 6, exmy_values(2, 3), ScaleSpec(32, "e8m0")),
    "fp6_e3m2": TableFormat("fp6_e3m2", 6, exmy_values(3, 2), ScaleSpec(32, "e8m0")),
    "fp8": TableFormat("fp8", 8, exmy_values(4, 3, special="fn"), ScaleSpec(-1, "fp32")),
    "fp8_e4m3": TableFormat("fp8_e4m3", 8, exmy_values(4, 3, special="fn"), ScaleSpec(-1, "fp32")),
    "fp8_e5m2": TableFormat("fp8_e5m2", 8, exmy_values(5, 2, special="ieee"), ScaleSpec(-1, "fp32")),
    # OCP microscaling (MX) formats: 32-element blocks with a power-of-two (E8M0) scale
    "mxfp8": TableFormat("mxfp8", 8, exmy_values(4, 3, special="fn"), ScaleSpec(32, "e8m0", strict=True)),
    "mxfp8_e5m2": TableFormat("mxfp8_e5m2", 8, exmy_values(5, 2, special="ieee"), ScaleSpec(32, "e8m0", strict=True)),
    "mxfp6": TableFormat("mxfp6", 6, exmy_values(2, 3), ScaleSpec(32, "e8m0", strict=True)),
    "mxfp6_e3m2": TableFormat("mxfp6_e3m2", 6, exmy_values(3, 2), ScaleSpec(32, "e8m0", strict=True)),
    "mxint8": uniform(8, group=32, scale="e8m0", symmetric=True, name="mxint8"),
    "mxint4": uniform(4, group=32, scale="e8m0", symmetric=True, name="mxint4"),
    # NVFP4 with the four-over-six block scale choice (same storage and hardware layout as nvfp4)
    "nvfp4_4o6": TableFormat("nvfp4_4o6", 4, _E2M1, ScaleSpec(16, "e4m3", global_scale=True, strict=True),
                             init="4over6"),
    "fp8_e4m3fnuz": TableFormat("fp8_e4m3fnuz", 8, exmy_values(4, 3, special="fnuz"), ScaleSpec(-1, "fp32")),
    "fp8_e5m2fnuz": TableFormat("fp8_e5m2fnuz", 8, exmy_values(5, 2, special="fnuz"), ScaleSpec(-1, "fp32")),
    "fp8_e3m4": TableFormat("fp8_e3m4", 8, exmy_values(3, 4), ScaleSpec(-1, "fp32")),
    # block formats in the style of llama.cpp (value = scale * q (+ min); 32-element blocks, fp16 scales)
    "q4_0": uniform(4, group=32, scale="fp16", name="q4_0"),
    "q4_1": uniform(4, group=32, scale="fp16", zero_point=True, name="q4_1"),
    "q5_0": uniform(5, group=32, scale="fp16", name="q5_0"),
    "q5_1": uniform(5, group=32, scale="fp16", zero_point=True, name="q5_1"),
    "q8_0": uniform(8, group=32, scale="fp16", name="q8_0"),
    "q2_k": uniform(2, group=16, scale="fp16", zero_point=True, name="q2_k"),
    "q3_k": uniform(3, group=16, scale="fp16", name="q3_k"),
    "q6_k": uniform(6, group=16, scale="fp16", name="q6_k"),
    "nf3": TableFormat("nf3", 3, normal_float_values(3), ScaleSpec(64, "fp16")),
    "nf2": TableFormat("nf2", 2, normal_float_values(2), ScaleSpec(64, "fp16")),
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
