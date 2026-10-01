"""GEMM back ends of lattice linear layers: NVFP4 on FP4 tensor cores and MXFP8 on FP8 tensor cores, with their
operand quantization; :mod:`bitlattice.compute` decides which one a layer uses.

NVFP4 routes ('fp4': forward only, 'fp4_all': forward, input- and weight-gradient GEMMs) quantize the operands
according to an :class:`Fp4Recipe`: 'nvidia' (the NVFP4 training recipe: round-to-nearest activations,
stochastically rounded gradients, a 16-point Hadamard rotation of the weight-gradient operands, the transposed
weight re-quantized for the input gradient; the layer input saved as transposed NVFP4, 0.56 bytes per value),
'4over6' (four-over-six activation block scales), 'quartet2' and variants (four-over-six forward; MS-EDEN, an
unbiased estimator with a lower error than stochastic rounding, on 128-point randomized Hadamard rotations of both
backward GEMMs). Any NVFP4-layout weight format qualifies (:func:`is_nvfp4`).

MXFP8 routes ('mxfp8', 'mxfp8_all') take any lattice format: the weight is dequantized and re-quantized to MXFP8
once per weight version, activations and gradients are quantized to MXFP8 (E4M3, power-of-two scales per 32).
"""
from __future__ import annotations

import contextlib
import itertools
import logging
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import torch

from . import formats as F

log = logging.getLogger("bitlattice")

__all__ = ["Fp4Recipe", "FP4_RECIPES", "Fp4Operand", "Mx8Operand", "is_nvfp4", "fp4_supported", "mxfp8_supported",
           "fp4_unfit", "mx_unfit", "to_blocked", "quantize_fp4", "fp4_mm", "mx_mm", "rht_last", "nvfp4_linear",
           "defer_wgrad", "flush_wgrad", "pending_wgrad", "fold_rounded", "clear_shared", "drop_cached"]


@dataclass(frozen=True)
class Fp4Recipe:
    """Quantization of the GEMM operands under the FP4 policies.

    activations      'rtn' | '4over6'   forward input: round to nearest, with four-over-six block scales or not
    gradients        'sr' | 'eden'      output gradient in backward: stochastic rounding, or MS-EDEN (rounding to
                                        nearest plus stochastically rounded, bias-correcting block scales; needs
                                        ``rotation=128``)
    rotation         16 | 128           randomized Hadamard block along the token axis of the weight-gradient GEMM
    rotate_dgrad     bool               also rotate (128) the output-feature axis of the input-gradient GEMM (layers
                                        whose output features are not a multiple of 128 skip it)
    backward_4over6  bool               four-over-six block scales for the backward operands too
    select           'mse' | 'mae'      four-over-six criterion
    dgrad_weight     'requantize' | 'cached'   with ``rotate_dgrad``: the rotated W^T is re-quantized in every
                                        backward (MS-EDEN with MS-EDEN gradients), or quantized once per weight
                                        version (round to nearest, one rotation per version; as the transposed
                                        weight of the NVIDIA recipe)
    scale_source     'current' | 'delayed'   per-tensor scales from an amax pass over the operand (exact), or from
                                        the amax the previous call of the same layer saw (times ``scale_margin``;
                                        no extra pass; values beyond the range saturate). Applies to the one-read
                                        kernels of the 128-rotation recipes; the first call of a layer is exact
    scale_margin     float               headroom of delayed scales
    """

    activations: str = "rtn"
    gradients: str = "sr"
    rotation: int = 16
    rotate_dgrad: bool = False
    backward_4over6: bool = False
    select: str = "mse"
    dgrad_weight: str = "requantize"
    scale_source: str = "current"
    scale_margin: float = 2.0

    def __post_init__(self):
        if self.activations not in ("rtn", "4over6"):
            raise ValueError("activations must be 'rtn' or '4over6'")
        if self.gradients not in ("sr", "eden"):
            raise ValueError("gradients must be 'sr' or 'eden'")
        if self.rotation not in (16, 128):
            raise ValueError("rotation must be 16 or 128")
        if self.gradients == "eden" and self.rotation != 128:
            raise ValueError("MS-EDEN gradients need rotation=128")
        if self.select not in ("mse", "mae"):
            raise ValueError("select must be 'mse' or 'mae'")
        if self.dgrad_weight not in ("requantize", "cached"):
            raise ValueError("dgrad_weight must be 'requantize' or 'cached'")
        if self.scale_source not in ("current", "delayed"):
            raise ValueError("scale_source must be 'current' or 'delayed'")
        if not self.scale_margin >= 1.0:
            raise ValueError("scale_margin must be >= 1")

    @property
    def legacy(self) -> bool:
        """True for the NVIDIA recipe (kernels of version 0.1, identical numbers)."""
        return self == Fp4Recipe()


FP4_RECIPES: dict[str, Fp4Recipe] = {
    "nvidia": Fp4Recipe(),
    "4over6": Fp4Recipe(activations="4over6"),
    "quartet2": Fp4Recipe(activations="4over6", gradients="eden", rotation=128, rotate_dgrad=True),
    "quartet2_4o6": Fp4Recipe(activations="4over6", gradients="eden", rotation=128, rotate_dgrad=True,
                              backward_4over6=True),
    "quartet2_fast": Fp4Recipe(activations="4over6", gradients="eden", rotation=128, rotate_dgrad=True,
                               dgrad_weight="cached"),
    "quartet2_delayed": Fp4Recipe(activations="4over6", gradients="eden", rotation=128, rotate_dgrad=True,
                                  dgrad_weight="cached", scale_source="delayed"),
}
def as_recipe(r) -> Fp4Recipe:
    """A recipe from its name in ``FP4_RECIPES`` (or the recipe itself)."""
    if isinstance(r, Fp4Recipe):
        return r
    if r not in FP4_RECIPES:
        raise ValueError(f"unknown FP4 recipe {r!r}; known: {sorted(FP4_RECIPES)}")
    return FP4_RECIPES[r]


def is_nvfp4(fmt) -> bool:
    """NVFP4 storage layout: E2M1 values, E4M3 scales per 16 elements, a per-tensor global scale."""
    sc = getattr(fmt, "scale", None)
    return (getattr(fmt, "kind", None) == "table" and fmt.bits == 4 and sc is not None and sc.group == 16
            and sc.dtype == "e4m3" and sc.global_scale and fmt.levels == _NVFP4_LEVELS)


_NVFP4_LEVELS = F.get("nvfp4").levels


@lru_cache(maxsize=None)
def mxfp8_supported(device_index: int) -> bool:
    """MXFP8 block-scaled GEMMs (E8M0 scales per 32) through PyTorch on this device (probed once)."""
    if not hasattr(torch, "float8_e8m0fnu"):
        return False
    try:
        dev = torch.device("cuda", device_index)
        a = torch.zeros(128, 128, device=dev).to(torch.float8_e4m3fn)
        s = torch.full((128 * 4,), 127, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
        torch._scaled_mm(a, a.t(), scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
        return True
    except (RuntimeError, NotImplementedError) as e:
        log.info("cuda:%d: no MXFP8 GEMM through PyTorch (%s)", device_index, str(e).splitlines()[0][:200])
        return False


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


def fp4_unfit(w, device_index: int | None, backward: bool) -> str | None:
    """Why the lattice weight ``w`` cannot use the NVFP4 GEMMs (None: it can)."""
    if not is_nvfp4(w.fmt):
        return f"format {w.fmt.name} is not NVFP4-layout"
    if w.hi_idx is not None:
        return "it keeps high-precision columns"
    if w.dim() != 2:
        return "the weight is not a matrix"
    if w.shape[1] % 32:
        return f"in_features {w.shape[1]} is not a multiple of 32"
    if w.shape[0] % (32 if backward else 16):
        return f"out_features {w.shape[0]} is not a multiple of {32 if backward else 16}"
    if device_index is None:
        return "the weight is not on a CUDA device"
    if not fp4_supported(device_index):
        return f"GPU {device_index} has no FP4 tensor cores reachable from this PyTorch"
    return None


def mx_unfit(w, device_index: int | None) -> str | None:
    """Why the lattice weight ``w`` cannot use the MXFP8 GEMMs (None: it can)."""
    if w.dim() != 2:
        return "the weight is not a matrix"
    if w.shape[0] % 128 or w.shape[1] % 128:
        return f"shape {tuple(w.shape)} is not made of 128 x 128 tiles"
    if device_index is None:
        return "the weight is not on a CUDA device"
    if not mxfp8_supported(device_index):
        return f"GPU {device_index} has no MXFP8 GEMMs reachable from this PyTorch"
    return None


class Fp4Operand(NamedTuple):
    """A [rows, cols] matrix in NVFP4: packed E2M1 codes, E4M3 block scales along cols, fp32 global scale.
    ``blocked``: the scales are already in the 128x4-tiled layout of the block-scaled GEMM."""

    packed: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor
    blocked: bool = False
    rotate_seed: int = -1         # seed of the 128-point rotation along cols (-1: none or the 16-point legacy one)


_sr_counter = itertools.count(1)


def quantize_fp4(x2: torch.Tensor, stochastic: bool = False) -> Fp4Operand:
    """NVFP4 quantization along the last dimension of a contiguous 2-D tensor (round to nearest, or stochastic
    rounding with a fresh seed per call)."""
    amax = torch.linalg.vector_norm(x2, float("inf"), dtype=torch.float32)       # no |x| temporary
    g = (amax / (6.0 * 448.0)).clamp_min(1e-12).reshape(1)
    seed = next(_sr_counter) & 0x7FFFFFFF if stochastic else -1
    packed, scales = torch.ops.bitlattice.quantize_nvfp4(x2, g, seed, 0)
    return Fp4Operand(packed, scales, g)


@lru_cache(maxsize=None)
def _two_level(device_index: int) -> bool:
    """Whether ``torch.nn.functional.scaled_mm`` takes NVFP4 block scales plus a per-tensor scale (applied inside the
    GEMM epilogue); probed once per device."""
    Fn = torch.nn.functional
    if not hasattr(Fn, "scaled_mm") or not hasattr(Fn, "ScalingType"):
        return False
    try:
        dev = torch.device("cuda", device_index)
        a = torch.zeros(128, 32, dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
        s = to_blocked(torch.ones(128, 4, device=dev).to(torch.float8_e4m3fn))
        one = torch.ones((), device=dev)
        _scaled_mm_2l(a, a, s, s, one, one, torch.bfloat16)
        return True
    except (RuntimeError, NotImplementedError, TypeError) as e:   # older torch / device: GEMM + scale kernel
        log.info("cuda:%d: no two-level NVFP4 GEMM (%s); the global scale is applied after the GEMM",
                 device_index, str(e).splitlines()[0][:200])
        return False


def _scaled_mm_2l(a, b, sa, sb, ga, gb, out_dtype):
    Fn = torch.nn.functional
    ST, SW = Fn.ScalingType, Fn.SwizzleType
    rec = [ST.BlockWise1x16, ST.TensorWise]
    sw = [SW.SWIZZLE_32_4_4, SW.NO_SWIZZLE]
    return Fn.scaled_mm(a, b.t(), [sa, ga], rec, [sb, gb], rec, swizzle_a=sw, swizzle_b=sw, output_dtype=out_dtype)


_REFERENCE = [0]          # > 0 inside a layer computing with reference kernels (see reference_kernels)


@contextlib.contextmanager
def reference_kernels(on: bool = True):
    """The plain kernels for the GEMM operands and epilogues inside the block (Compute(kernels='reference'))."""
    _REFERENCE[0] += int(on)
    try:
        yield
    finally:
        _REFERENCE[0] -= int(on)


def fp4_mm(a: Fp4Operand, b: Fp4Operand, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """A B^T for NVFP4 operands A [M, K], B [N, K] (K: multiple of 32), fp32 accumulation, ``out_dtype`` output.
    The global scales are applied in the GEMM epilogue where supported, else by an in-place scale kernel."""
    fp4 = torch.float4_e2m1fn_x2
    sa = a.scales if a.blocked else to_blocked(a.scales)
    sb = b.scales if b.blocked else to_blocked(b.scales)
    if not _REFERENCE[0] and _two_level(a.packed.device.index or 0):
        return _scaled_mm_2l(a.packed.view(fp4), b.packed.view(fp4), sa, sb, a.global_scale.float().reshape(()),
                             b.global_scale.float().reshape(()), out_dtype)
    y = torch._scaled_mm(a.packed.view(fp4), b.packed.view(fp4).t(), scale_a=sa, scale_b=sb, out_dtype=out_dtype)
    return torch.ops.bitlattice.scale_(y, a.global_scale * b.global_scale.float())


# ---- per-weight operands, cached while the weight is unchanged ------------------------------------------------
# A weight is used by the forward, the recomputed forward and the backward of every micro-batch before the optimizer
# changes it (which bumps ``_bl_version``); its GEMM-layout scales and its transposed re-quantization are built once
# per version. The cache is bounded (LRU by bytes) and holds only weak references to the weights.

_OPERANDS: "OrderedDict" = OrderedDict()
_OPERAND_BYTES = [0]
OPERAND_CACHE_BYTES = 768 << 20


def _cached(w, kind: str, make):
    key = (id(w), kind)
    ver = getattr(w, "_bl_version", 0)
    hit = _OPERANDS.get(key)
    if hit is not None and hit[0] == ver and hit[1]() is w:
        _OPERANDS.move_to_end(key)
        return hit[2]
    if hit is not None:
        _OPERAND_BYTES[0] -= hit[3]
        del _OPERANDS[key]
    val = make()
    size = sum(t.numel() * t.element_size() for t in val if isinstance(t, torch.Tensor))
    _OPERANDS[key] = (ver, weakref.ref(w), val, size)
    _OPERAND_BYTES[0] += size
    while _OPERAND_BYTES[0] > OPERAND_CACHE_BYTES and len(_OPERANDS) > 1:
        _, old = _OPERANDS.popitem(last=False)
        _OPERAND_BYTES[0] -= old[3]
    return val


def drop_cached(w) -> None:
    """Forget the cached operands of ``w`` (called when the optimizer changes it: old versions are never used again,
    and at long context their memory matters)."""
    for key in [k for k in _OPERANDS if k[0] == id(w)]:
        _OPERAND_BYTES[0] -= _OPERANDS.pop(key)[3]


def weight_operand(w) -> Fp4Operand:
    """W [N, K] as a GEMM operand (scales in the tiled layout)."""
    return _cached(w, "w", lambda: Fp4Operand(w.packed, to_blocked(w.scales), w.global_scale.float(), True))


def weight_t_operand(w) -> Fp4Operand:
    """W^T [K, N] re-quantized along N from the NVFP4 codes (for the input gradient)."""
    def make():
        packed, scales, g = torch.ops.bitlattice.requantize_nvfp4_t(w.packed, w.scales, w.global_scale)
        return Fp4Operand(packed, to_blocked(scales), g, True)
    return _cached(w, "wt", make)


def weight_t_rotated_operand(w, four_over_six: bool = False) -> Fp4Operand:
    """W^T [K, N] rotated along N (128-blocks) and re-quantized once per weight version (round to nearest); the
    rotation seed (``rotate_seed`` of the result) is fixed per version."""
    ver = getattr(w, "_bl_version", 0)
    seed = (id(w) * 2654435761 + ver * 40503 + 1) & 0x7FFFFFFF

    def make():
        packed, scales, g = torch.ops.bitlattice.nvfp4_quantize(w.packed, True, _RTN, four_over_six, False, True, 0,
                                                                None, w.scales, w.global_scale, seed, True)
        return Fp4Operand(packed, scales, g, True, seed)
    return _cached(w, "wt128_4o6" if four_over_six else "wt128", make)


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


# ---- recipe-driven quantization of GEMM operands ----------------------------------------------------------------

_RTN, _SR, _EDEN = 0, 1, 2


def _seed() -> int:
    return next(_sr_counter) & 0x7FFFFFFF


def _q(x2: torch.Tensor, *, transpose: bool = False, mode: int = _RTN, four_over_six: bool = False,
       recipe: Fp4Recipe | None = None, rotate_seed: int = -1, seed: int = 0, w=None, pow2: bool = False
       ) -> Fp4Operand:
    """NVFP4 operand from the general kernel: rows of x2 (or of x2^T), optionally rotated (128) with
    ``rotate_seed``; ``w``: an NVFP4 weight given by its codes (transposed re-quantization); ``pow2``: global scale
    rounded up to a power of two."""
    mae = recipe is not None and recipe.select == "mae"
    rot = rotate_seed >= 0
    if w is not None:
        packed, scales, g = torch.ops.bitlattice.nvfp4_quantize(w.packed, True, mode, four_over_six, mae, rot, seed,
                                                                None, w.scales, w.global_scale,
                                                                rotate_seed if rot else -1, True, pow2)
    else:
        packed, scales, g = torch.ops.bitlattice.nvfp4_quantize(x2, transpose, mode, four_over_six, mae, rot, seed,
                                                                None, None, None, rotate_seed if rot else -1, True,
                                                                pow2)
    return Fp4Operand(packed, scales, g, True, rotate_seed)


def _side(on: bool, mode: int = _RTN, four_over_six: bool = False, rotate_seed: int = -1, seed: int = 0,
          pow2: bool = False) -> list:
    return [int(on), mode, int(four_over_six), int(rotate_seed >= 0), seed, rotate_seed, int(pow2)]


def _q2(x2: torch.Tensor, row: list, col: list, recipe: Fp4Recipe, site: dict | None = None
        ) -> tuple[Fp4Operand | None, Fp4Operand | None]:
    """Row and column (x2^T) operands of x2 [M, K] (both multiples of 128) from one read of x2 per pass. ``site``:
    the delayed-scaling state of this operand (amax of the previous call, sides seen)."""
    given = seen = None
    if recipe.scale_source == "delayed" and site is not None:
        on = (bool(row[0]), bool(col[0]))
        hist, known = site.get("amax"), site.get("known", (False, False))
        if hist is not None and all(known[i] for i in range(2) if on[i]):
            f = [recipe.scale_margin / (6.0 * (256.0 if side[2] else 448.0)) for side in (row, col)]
            given = torch.stack([hist[0] * f[0], hist[1] * f[1]]).clamp_min(1e-30)    # device ops only, no sync
            pow2 = torch.tensor([bool(row[6]), bool(col[6])], device=x2.device) if len(row) > 6 else None
            if pow2 is not None and (row[6] or col[6]):          # concatenated operands need power-of-two scales
                given = torch.where(pow2, torch.exp2(torch.ceil(torch.log2(given))), given)
        seen = torch.zeros(2, device=x2.device)
    rp, rs, rg, cp, cs, cg = torch.ops.bitlattice.nvfp4_quantize2(x2, row, col, recipe.select == "mae", given, True,
                                                                  seen)
    if seen is not None:
        hist = site.get("amax")
        site["amax"] = seen if hist is None else torch.stack([seen[i] if on[i] else hist[i] for i in range(2)])
        known = site.get("known", (False, False))
        site["known"] = (known[0] or on[0], known[1] or on[1])
    return (Fp4Operand(rp, rs, rg, True, row[5] if row[3] else -1) if row[0] else None,
            Fp4Operand(cp, cs, cg, True, col[5] if col[3] else -1) if col[0] else None)


def _act_operand(x2: torch.Tensor, recipe: Fp4Recipe) -> Fp4Operand:
    if recipe.activations == "4over6":
        return _q(x2, four_over_six=True, recipe=recipe)
    return quantize_fp4(x2)


def _pad_to(recipe: Fp4Recipe) -> int:
    return 128 if recipe.rotation == 128 else 32


def nvfp4_linear(x: torch.Tensor, w, bias: torch.Tensor | None = None, recipe: Fp4Recipe = Fp4Recipe()
                 ) -> torch.Tensor:
    """y = Q(x) W^T (+ bias) with x quantized to NVFP4 on the fly; W must be an NVFP4 LatticeTensor."""
    K = x.shape[-1]
    M = x.numel() // K

    def make():
        return _act_operand(_pad_rows(x.reshape(-1, K), 16).contiguous(), recipe)
    xq = _shared(x, recipe, "fwd", make)
    y = fp4_mm(xq, weight_operand(w), x.dtype)[:M]
    if bias is not None:
        y += bias.to(y.dtype)
    return y.reshape(*x.shape[:-1], w.shape[0])


# ---- one quantization per input shared by its consumers -------------------------------------------------------
# Projections reading the same tensor (q / k / v, gate / up) would quantize it once each and save one copy of x^T
# each for backward. The last few inputs are remembered by identity (weak reference + version counter + recipe), so
# the consumers share the operands (and the saved x^T: less activation memory as well). Sharing one random rotation
# and one stochastic draw between consumers keeps every gradient estimate unbiased.

_SHARED: "OrderedDict" = OrderedDict()
_SHARED_MAX = 4


def _shared(x: torch.Tensor, recipe: Fp4Recipe, kind: str, make):
    if _REFERENCE[0]:
        return make()
    key = (id(x), x._version, recipe, kind, tuple(x.shape))
    hit = _SHARED.get(key)
    if hit is not None and hit[0]() is x:
        _SHARED.move_to_end(key)
        return hit[1]
    val = make()
    # the entry dies with the input (no operand outlives its consumers' forward passes)
    _SHARED[key] = (weakref.ref(x, lambda _, k=key: _SHARED.pop(k, None)), val)
    while len(_SHARED) > _SHARED_MAX:
        _SHARED.popitem(last=False)
    return val


def clear_shared() -> None:
    """Drop the remembered input operands (they also expire after a few other inputs)."""
    _SHARED.clear()


def fp4_forward_saving(x: torch.Tensor, w, bias, recipe: Fp4Recipe = Fp4Recipe()):
    """Forward of 'fp4_all': returns (y, saved), where saved holds x^T (rotated, NVFP4) for the weight gradient."""
    K = x.shape[-1]
    M = x.numel() // K
    xq, xt = _shared(x, recipe, "fwd+t", lambda: _input_operands(x, recipe, _site(w, "x")))
    y = fp4_mm(xq, weight_operand(w), x.dtype)[:M]
    if bias is not None:
        y += bias.to(y.dtype)
    return y.reshape(*x.shape[:-1], w.shape[0]), xt


def _site(w, role: str) -> dict:
    """Per-weight state of an operand site (delayed scaling)."""
    sites = w.__dict__.setdefault("_bl_sites", {})
    return sites.setdefault(role, {})


def _input_operands(x: torch.Tensor, recipe: Fp4Recipe, site: dict | None = None) -> tuple[Fp4Operand, Fp4Operand]:
    """Forward operand of x and its (rotated) transposed operand for the weight gradient."""
    K = x.shape[-1]
    x2 = _pad_rows(x.reshape(-1, K), _pad_to(recipe)).contiguous()
    if recipe.rotation == 128 and K % 128 == 0:
        # one read of x for both
        mode = _EDEN if recipe.gradients == "eden" else _RTN
        fos = recipe.backward_4over6 or (mode == _RTN and recipe.activations == "4over6")
        # the weight-gradient operands get power-of-two global scales: micro-batches can then be concatenated
        # exactly into one GEMM (deferred weight gradients)
        return _q2(x2, _side(True, _RTN, recipe.activations == "4over6"),
                   _side(True, mode, fos, _seed(), _seed(), pow2=True), recipe, site)
    xq = _act_operand(x2, recipe)
    if recipe.rotation == 16:
        # |x H/4| <= 4 max|x| for the orthogonal 16x16 rotation: the row-wise global scale bounds the rotated one
        bound = xq.global_scale * (4.0 * (256.0 / 448.0 if recipe.activations == "4over6" else 1.0))
        xt = Fp4Operand(*torch.ops.bitlattice.quantize_nvfp4_t(x2, _hadamard16(x2.device, torch.float32), -1, 0,
                                                               bound))
    else:
        mode = _EDEN if recipe.gradients == "eden" else _RTN
        fos = recipe.backward_4over6 or (mode == _RTN and recipe.activations == "4over6")
        xt = _q(x2, transpose=True, mode=mode, four_over_six=fos, recipe=recipe, rotate_seed=_seed(), seed=_seed(),
                pow2=True)
    return xq, xt


def fp4_backward(gy: torch.Tensor, w, xt: Fp4Operand, need_x: bool, need_w: bool,
                 recipe: Fp4Recipe = Fp4Recipe()):
    """Input and weight gradients of y = x W^T on FP4 tensor cores."""
    N = gy.shape[-1]
    g2 = gy.reshape(-1, N)
    M = g2.shape[0]
    g2p = _pad_rows(g2, _pad_to(recipe)).contiguous()
    gx = gw = None
    gq = None
    b46 = recipe.backward_4over6
    rotated_w = need_w and xt.rotate_seed >= 0
    if N % 128 == 0 and (rotated_w or not need_w) and (recipe.rotate_dgrad or not need_x) and not recipe.legacy:
        # one read of g for the input-gradient operand (rows, rotated along N) and the weight-gradient operand
        # (columns, rotated along the tokens with the rotation of the saved x^T)
        mode = _EDEN if recipe.gradients == "eden" else _SR
        wt = None
        if need_x and recipe.dgrad_weight == "cached":
            wt = weight_t_rotated_operand(w, b46)
        r = wt.rotate_seed if wt is not None else _seed()
        gq, gt = _q2(g2p, _side(need_x, mode, b46, r, _seed()),
                     _side(need_w, mode, b46, xt.rotate_seed if need_w else -1, _seed(), pow2=True), recipe,
                     _site(w, "g"))
        if need_x:
            if wt is None:
                wt = _q(None, mode=_EDEN if mode == _EDEN else _RTN, four_over_six=b46, recipe=recipe, rotate_seed=r,
                        seed=_seed(), w=w)
            gx = fp4_mm(gq, wt, gy.dtype)[:M]
        if need_w:
            gw = _weight_grad(w, gt, xt)
        return gx, gw
    if need_x:
        if recipe.rotate_dgrad and N % 128 == 0:
            mode = _EDEN if recipe.gradients == "eden" else _SR
            wt = weight_t_rotated_operand(w, b46) if recipe.dgrad_weight == "cached" else None
            r = wt.rotate_seed if wt is not None else _seed()
            gq = _q(g2p, mode=mode, four_over_six=b46, recipe=recipe, rotate_seed=r, seed=_seed())
            # W^T: MS-EDEN with MS-EDEN gradients (both factors unbiased), else round to nearest
            if wt is None:
                wt = _q(None, mode=_EDEN if mode == _EDEN else _RTN, four_over_six=b46, recipe=recipe,
                        rotate_seed=r, seed=_seed(), w=w)
            gx = fp4_mm(gq, wt, gy.dtype)[:M]
            gq = None                                     # rotated: its global scale does not bound g^T
        else:
            gq = _q(g2p, mode=_SR, four_over_six=True, recipe=recipe, seed=_seed()) if b46 else \
                quantize_fp4(g2p, stochastic=True)
            gx = fp4_mm(gq, weight_t_operand(w), gy.dtype)[:M]
    if need_w:
        if xt.rotate_seed >= 0:                                                        # g^T rotated (128), [N, Mp]
            mode = _EDEN if recipe.gradients == "eden" else _SR
            gt = _q(g2p, transpose=True, mode=mode, four_over_six=b46, recipe=recipe, rotate_seed=xt.rotate_seed,
                    seed=_seed(), pow2=True)
        else:
            seed = _seed()                                                             # g^T rotated (16), [N, Mp]
            bound = gq.global_scale * (4.0 * (256.0 / 448.0 if b46 else 1.0)) if gq is not None else None
            gt = Fp4Operand(*torch.ops.bitlattice.quantize_nvfp4_t(g2p, _hadamard16(g2p.device, torch.float32), seed,
                                                                   0, bound))
            return gx, fp4_mm(gt, xt, w.dtype)                  # not power-of-two scaled: never deferred
        gw = _weight_grad(w, gt, xt)
    return gx, gw


# ---- deferred weight gradients ------------------------------------------------------------------------------------
# With micro-batches, every backward of a layer adds its weight gradient to .grad (a full read-modify-write of the
# gradient per micro-batch) from a GEMM over that micro-batch's tokens only. Inside ``defer_wgrad()`` the FP4 path of
# the 128-rotation recipes keeps the quantized operands (Eᵀ, xᵀ: power-of-two global scales) instead; ``flush_wgrad``
# folds their scales to a common global scale (exact: powers of two shift E4M3 exponents), concatenates them along the
# token axis and runs one GEMM over all micro-batches. Deferred work can also be scheduled into pipeline bubbles
# (the weight-gradient half of a split backward).

_defer_wgrad = [0]
_PENDING: "OrderedDict" = OrderedDict()             # id(param) -> (weakref(param), [(gt, xt), ...])


@contextlib.contextmanager
def defer_wgrad():
    """Keep the weight-gradient operands of FP4 layers instead of computing their gradients (see flush_wgrad)."""
    _defer_wgrad[0] += 1
    try:
        yield
    finally:
        _defer_wgrad[0] -= 1


def _weight_grad(w, gt: Fp4Operand, xt: Fp4Operand):
    if not _defer_wgrad[0]:
        return fp4_mm(gt, xt, w.dtype)
    entry = _PENDING.get(id(w))
    if entry is None or entry[0]() is not w:
        entry = _PENDING[id(w)] = (weakref.ref(w), [])
    entry[1].append((gt, xt))
    return None


_FOLD_ROUNDED: dict = {}     # device -> blocks whose folded scale left E4M3's normal range (GPU counter, no sync)
_E4M3_MIN_NORMAL = 2.0 ** -6


def _concat(ops: list) -> Fp4Operand:
    """Operands [rows, K_j] with power-of-two global scales and blocked scales -> one operand [rows, sum K_j] with the
    largest global scale; the others' block scales are multiplied by g_j / g_max = 2^-k. That is exact while the
    products stay in E4M3's normal range; smaller products are rounded (micro-batch gradients differing by more than
    ~2^4 in magnitude) and counted -- :func:`fold_rounded` -- instead of passing silently."""
    if len(ops) == 1:
        return ops[0]
    g = torch.stack([o.global_scale.reshape(()) for o in ops]).amax()
    packed = torch.cat([o.packed for o in ops], 1)
    rows = ops[0].packed.shape[0]
    nr = -(-rows // 128)
    tiles = []
    dev = packed.device
    count = _FOLD_ROUNDED.setdefault(dev, torch.zeros((), dtype=torch.int64, device=dev))
    for o in ops:
        f = (o.global_scale.reshape(()) / g)                       # a power of two <= 1
        sf = o.scales.view(torch.float8_e4m3fn).float() * f
        count += ((sf > 0) & (sf < _E4M3_MIN_NORMAL)).sum()
        tiles.append(sf.to(torch.float8_e4m3fn).view(nr, -1, 512))
    return Fp4Operand(packed, torch.cat(tiles, 1).reshape(-1), g.reshape(1), True)


def fold_rounded(reset: bool = False) -> int:
    """Number of NVFP4 blocks whose scale was rounded when deferred weight-gradient operands of micro-batches with
    different magnitudes were folded into one GEMM (0: the deferred gradients were exact). Synchronizes."""
    n = int(sum(int(c) for c in _FOLD_ROUNDED.values()))
    if reset:
        for c in _FOLD_ROUNDED.values():
            c.zero_()
    return n


def pending_wgrad() -> list:
    """Parameters with deferred weight gradients."""
    return [ref() for ref, _ in _PENDING.values() if ref() is not None]


def flush_wgrad(params=None, optimizer=None) -> int:
    """Compute deferred weight gradients (all, or of ``params``) with one GEMM per parameter, add them to ``.grad``
    and, for a fused-backward optimizer, apply its update. Returns the number of parameters flushed."""
    keys = list(_PENDING) if params is None else [id(p) for p in params if id(p) in _PENDING]
    n = 0
    for k in keys:
        ref, items = _PENDING.pop(k)
        p = ref()
        if p is None or not items:
            continue
        gw = fp4_mm(_concat([gt for gt, _ in items]), _concat([xt for _, xt in items]), p.dtype)
        p.grad = gw if p.grad is None else p.grad.add_(gw)
        if optimizer is not None and getattr(optimizer, "fused_backward", False):
            optimizer._on_grad(p)
        n += 1
    return n


# ---- MXFP8 path: any weight format on FP8 tensor cores ------------------------------------------------------------

class Mx8Operand(NamedTuple):
    """A [rows, cols] matrix in MXFP8: E4M3 codes, E8M0 scales per 32 along cols in the GEMM's tiled layout."""

    codes: torch.Tensor
    scales: torch.Tensor


def _mx2(x2: torch.Tensor, row: bool, col: bool):
    rc, rs, cc, cs = torch.ops.bitlattice.mx_quantize2(x2, row, col, False, True)
    return Mx8Operand(rc, rs) if row else None, Mx8Operand(cc, cs) if col else None


def mx_mm(a: Mx8Operand, b: Mx8Operand, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """A B^T for MXFP8 operands A [M, K], B [N, K]."""
    return torch._scaled_mm(a.codes, b.codes.t(), scale_a=a.scales, scale_b=b.scales, out_dtype=out_dtype)


def mx_weight_operands(w) -> tuple[Mx8Operand, Mx8Operand]:
    """W [N, K] along K and W^T [K, N] along N in MXFP8, from the dequantized weight, once per weight version."""
    def make():
        dense = w.dequantize(torch.bfloat16).reshape(w.shape[0], -1).contiguous()
        wq, wt = _mx2(dense, True, True)
        return (wq.codes, wq.scales, wt.codes, wt.scales)          # flat: the cache counts tensor bytes
    c = _cached(w, "mx8", make)
    return Mx8Operand(c[0], c[1]), Mx8Operand(c[2], c[3])


def mx_linear(x: torch.Tensor, w, bias=None) -> torch.Tensor:
    K = x.shape[-1]
    M = x.numel() // K
    xq, _ = _shared(x, None, "mx8", lambda: _mx2(_pad_rows(x.reshape(-1, K), 128).contiguous(), True, False))
    wq, _ = mx_weight_operands(w)
    y = mx_mm(xq, wq, x.dtype)[:M]
    if bias is not None:
        y += bias.to(y.dtype)
    return y.reshape(*x.shape[:-1], w.shape[0])


def mx_forward_saving(x: torch.Tensor, w, bias=None):
    """Forward of 'mxfp8_all': (y, x^T in MXFP8 for the weight gradient)."""
    K = x.shape[-1]
    M = x.numel() // K
    xq, xt = _shared(x, None, "mx8+t", lambda: _mx2(_pad_rows(x.reshape(-1, K), 128).contiguous(), True, True))
    wq, _ = mx_weight_operands(w)
    y = mx_mm(xq, wq, x.dtype)[:M]
    if bias is not None:
        y += bias.to(y.dtype)
    return y.reshape(*x.shape[:-1], w.shape[0]), xt


def mx_backward(gy: torch.Tensor, w, xt: Mx8Operand | None, need_x: bool, need_w: bool):
    N = gy.shape[-1]
    g2 = gy.reshape(-1, N)
    M = g2.shape[0]
    gq, gt = _mx2(_pad_rows(g2, 128).contiguous(), need_x, need_w)
    gx = gw = None
    if need_x:
        _, wt = mx_weight_operands(w)
        gx = mx_mm(gq, wt, gy.dtype)[:M]
    if need_w:
        gw = mx_mm(gt, xt, w.dtype)
    return gx, gw