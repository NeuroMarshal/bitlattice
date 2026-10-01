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

How the FP4 policies quantize the operands is a recipe (:class:`Fp4Recipe`): 'nvidia' (the default; the NVFP4
training recipe described above), '4over6' (four-over-six block scales for the activations), 'quartet2'
(four-over-six forward; MS-EDEN, an unbiased estimator with lower error than stochastic rounding, on 128-point
randomized Hadamard rotations of the inner dimension of both backward GEMMs). Set it with :func:`set_fp4_recipe`,
the :func:`fp4_recipe` context, or per weight (``w.fp4_recipe``). Any NVFP4-layout format qualifies
(:func:`is_nvfp4`: E2M1 values, E4M3 scales per 16, a global scale), e.g. 'nvfp4' and 'nvfp4_4o6'.
"""
from __future__ import annotations

import contextlib
import itertools
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import torch

from . import formats as F

__all__ = ["POLICIES", "set_compute", "get_compute", "compute", "fp4_supported", "to_blocked", "nvfp4_linear",
           "fp4_policy", "quantize_fp4", "fp4_mm", "rht_last", "Fp4Operand", "Fp4Recipe", "FP4_RECIPES",
           "set_fp4_recipe", "get_fp4_recipe", "fp4_recipe", "is_nvfp4", "defer_wgrad", "flush_wgrad",
           "pending_wgrad", "clear_shared"]

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
    """Compute policy inside the block. With activation checkpointing the forward is recomputed during backward:
    run the backward inside the block too (or use :func:`set_compute`); the same holds for :func:`fp4_recipe`."""
    if policy not in POLICIES:
        raise ValueError(f"compute policy must be one of {POLICIES}")
    _context.append(policy)
    try:
        yield
    finally:
        _context.pop()


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
_recipe_default = [FP4_RECIPES["nvidia"]]
_recipe_context: list[Fp4Recipe] = []


def _as_recipe(r) -> Fp4Recipe:
    if isinstance(r, Fp4Recipe):
        return r
    if r not in FP4_RECIPES:
        raise ValueError(f"unknown FP4 recipe {r!r}; known: {sorted(FP4_RECIPES)}")
    return FP4_RECIPES[r]


def set_fp4_recipe(recipe) -> None:
    """Process-wide default recipe (a name from ``FP4_RECIPES`` or an :class:`Fp4Recipe`)."""
    _recipe_default[0] = _as_recipe(recipe)


def get_fp4_recipe(tensor=None) -> Fp4Recipe:
    """Effective recipe: an active ``fp4_recipe(...)`` context, else the tensor's own, else the default."""
    if _recipe_context:
        return _recipe_context[-1]
    own = getattr(tensor, "fp4_recipe", None)
    return _as_recipe(own) if own is not None else _recipe_default[0]


@contextlib.contextmanager
def fp4_recipe(recipe):
    _recipe_context.append(_as_recipe(recipe))
    try:
        yield
    finally:
        _recipe_context.pop()


def is_nvfp4(fmt) -> bool:
    """NVFP4 storage layout: E2M1 values, E4M3 scales per 16 elements, a per-tensor global scale."""
    sc = getattr(fmt, "scale", None)
    return (getattr(fmt, "kind", None) == "table" and fmt.bits == 4 and sc is not None and sc.group == 16
            and sc.dtype == "e4m3" and sc.global_scale and fmt.levels == _NVFP4_LEVELS)


_NVFP4_LEVELS = F.get("nvfp4").levels


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
    if pol == "exact" or not (x.is_cuda and is_nvfp4(w.fmt) and w.hi_idx is None and w.dim() == 2
                              and w.shape[1] % 32 == 0 and w.shape[0] % 16 == 0
                              and fp4_supported(x.device.index or 0)):
        return "exact"
    if pol == "fp4_all" and w.shape[0] % 32 == 0:
        return "fp4_all"
    return "fp4"


def qualifies_fp4(x: torch.Tensor, w) -> bool:
    return fp4_policy(x, w) != "exact"


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
    except Exception:                                    # older torch / unsupported device: GEMM + scale kernel
        return False


def _scaled_mm_2l(a, b, sa, sb, ga, gb, out_dtype):
    Fn = torch.nn.functional
    ST, SW = Fn.ScalingType, Fn.SwizzleType
    rec = [ST.BlockWise1x16, ST.TensorWise]
    sw = [SW.SWIZZLE_32_4_4, SW.NO_SWIZZLE]
    return Fn.scaled_mm(a, b.t(), [sa, ga], rec, [sb, gb], rec, swizzle_a=sw, swizzle_b=sw, output_dtype=out_dtype)


def fp4_mm(a: Fp4Operand, b: Fp4Operand, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """A B^T for NVFP4 operands A [M, K], B [N, K] (K: multiple of 32), fp32 accumulation, ``out_dtype`` output.
    The global scales are applied in the GEMM epilogue where supported, else by an in-place scale kernel."""
    fp4 = torch.float4_e2m1fn_x2
    sa = a.scales if a.blocked else to_blocked(a.scales)
    sb = b.scales if b.blocked else to_blocked(b.scales)
    if _two_level(a.packed.device.index or 0):
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


def nvfp4_linear(x: torch.Tensor, w, bias: torch.Tensor | None = None, recipe: Fp4Recipe | None = None
                 ) -> torch.Tensor:
    """y = Q(x) W^T (+ bias) with x quantized to NVFP4 on the fly; W must be an NVFP4 LatticeTensor."""
    recipe = recipe or get_fp4_recipe(w)
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

SHARE_INPUTS = True
_SHARED: "OrderedDict" = OrderedDict()
_SHARED_MAX = 4


def _shared(x: torch.Tensor, recipe: Fp4Recipe, kind: str, make):
    if not SHARE_INPUTS:
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


def fp4_forward_saving(x: torch.Tensor, w, bias, recipe: Fp4Recipe | None = None):
    """Forward of 'fp4_all': returns (y, saved), where saved holds x^T (rotated, NVFP4) for the weight gradient."""
    recipe = recipe or get_fp4_recipe(w)
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
                 recipe: Fp4Recipe | None = None):
    """Input and weight gradients of y = x W^T on FP4 tensor cores."""
    recipe = recipe or get_fp4_recipe(w)
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


def _concat(ops: list) -> Fp4Operand:
    """Operands [rows, K_j] with power-of-two global scales and blocked scales -> one operand [rows, sum K_j]."""
    if len(ops) == 1:
        return ops[0]
    g = torch.stack([o.global_scale.reshape(()) for o in ops]).amax()
    packed = torch.cat([o.packed for o in ops], 1)
    rows = ops[0].packed.shape[0]
    nr = -(-rows // 128)
    tiles = []
    for o in ops:
        f = (o.global_scale.reshape(()) / g)                       # a power of two <= 1: exact in E4M3 (normal range)
        s = (o.scales.view(torch.float8_e4m3fn).float() * f).to(torch.float8_e4m3fn)
        tiles.append(s.view(nr, -1, 512))
    return Fp4Operand(packed, torch.cat(tiles, 1).reshape(-1), g.reshape(1), True)


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
