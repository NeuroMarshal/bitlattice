"""How layers with lattice weights compute.

A lattice weight carries a :class:`Compute` (``weight.compute``); ``None`` is the plain default, exact computation:
the weight is dequantized and the GEMMs run in the activations' dtype. Anything else is asked for explicitly with
``bitlattice.surgery.set_compute_`` and is a requirement: a layer that cannot honour it is reported with the reason
when the setting is made, and a call that cannot (e.g. an input on another device) raises instead of falling back.

    bl.surgery.set_compute_(model, "*.layers.*", "quartet2")        # a preset name
    bl.surgery.set_compute_(model, "*.mlp", bl.Compute("mxfp8"))    # or an object
    with bl.compute("exact"):                                        # a temporary override for every layer
        loss = evaluate(model)

With activation checkpointing the forward pass is recomputed during backward: keep the backward inside a temporary
override as well.
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field

import torch

from .backend import FP4_RECIPES, Fp4Recipe, as_recipe, fp4_unfit, mx_unfit
from .kernel_choice import Kernels, as_kernels

__all__ = ["PRESETS", "Compute", "as_compute", "assign_compute_", "compute", "route"]

log = logging.getLogger("bitlattice")

GEMMS = ("exact", "nvfp4", "mxfp8")


@dataclass(frozen=True)
class Compute:
    """How ``y = x Wᵀ`` and its gradients are computed.

    gemm      'exact'  dequantized weight, GEMMs in the activations' dtype (lossless w.r.t. the stored weight)
              'nvfp4'  activations quantized to NVFP4 on FP4 tensor cores (NVFP4-layout weights)
              'mxfp8'  activations and the weight (re-quantized once per version) in MXFP8 on FP8 tensor cores
    backward  with 'nvfp4' / 'mxfp8': the input- and weight-gradient GEMMs on the same tensor cores (gradients
              quantized as well); False keeps the backward exact
    nvfp4     operand quantization of the NVFP4 GEMMs (an :class:`Fp4Recipe` or its name)
    kernels   the layer's automatic kernel choices (a :class:`bitlattice.Kernels`, or 'reference': all off); 'auto'
              (None): the setting in effect (``bitlattice.kernels``, a runner's ``kernels``), by default all on
    """

    gemm: str = "exact"
    backward: bool = True
    nvfp4: Fp4Recipe = field(default_factory=Fp4Recipe)
    kernels: Kernels | None = None

    def __post_init__(self):
        if self.gemm not in GEMMS:
            raise ValueError(f"gemm must be one of {GEMMS}")
        object.__setattr__(self, "kernels", as_kernels(self.kernels))
        if not isinstance(self.nvfp4, Fp4Recipe):
            object.__setattr__(self, "nvfp4", as_recipe(self.nvfp4))

    @property
    def route(self) -> str:
        """'exact', 'fp4', 'fp4_all', 'mxfp8' or 'mxfp8_all'."""
        if self.gemm == "exact":
            return "exact"
        base = "fp4" if self.gemm == "nvfp4" else "mxfp8"
        return base + "_all" if self.backward else base

    def describe(self) -> str:
        if self.gemm == "exact":
            return "exact"
        extra = ""
        if self.gemm == "nvfp4":
            name = next((k for k, r in FP4_RECIPES.items() if r == self.nvfp4), None)
            extra = f", recipe {name}" if name else f", {self.nvfp4}"
        ref = f", kernels {self.kernels.describe()}" if self.kernels is not None else ""
        return f"{self.gemm}{'' if self.backward else ' forward only'}{extra}{ref}"


PRESETS: dict[str, Compute] = {
    "exact": Compute(),
    "fp4": Compute("nvfp4", backward=False),
    "fp4_all": Compute("nvfp4"),
    **{name: Compute("nvfp4", nvfp4=r) for name, r in FP4_RECIPES.items() if name != "nvidia"},
    "mxfp8": Compute("mxfp8", backward=False),
    "mxfp8_all": Compute("mxfp8"),
}


def as_compute(c) -> Compute | None:
    """A Compute from a preset name (``'auto'`` / None: the default, returned as None)."""
    if c is None or c == "auto":
        return None
    if isinstance(c, Compute):
        return c
    if c not in PRESETS:
        raise ValueError(f"unknown compute {c!r}; presets: {sorted(PRESETS)}")
    return PRESETS[c]


# A temporary override applies to every layer. Process-wide rather than thread-local: the backward pass (and
# checkpoint recomputation) runs on autograd worker threads and must see the setting of the forward pass.
_scope: list[Compute] = []


@contextlib.contextmanager
def compute(c):
    """Every lattice layer computes as ``c`` (a :class:`Compute` or preset name) inside the block."""
    _scope.append(as_compute(c) or Compute())
    try:
        yield
    finally:
        _scope.pop()


def _device_index(t: torch.Tensor) -> int | None:
    return t.device.index if t.is_cuda else None


def _unfit(c: Compute, w, device_index: int | None) -> str | None:
    if c.gemm == "nvfp4":
        return fp4_unfit(w, device_index, c.backward)
    if c.gemm == "mxfp8":
        return mx_unfit(w, device_index)
    return None


def route(x: torch.Tensor, w) -> tuple[str, Compute]:
    """(route, Compute) of a call of the lattice layer with weight ``w`` on input ``x``. A requirement that cannot
    be met raises RuntimeError."""
    c = _scope[-1] if _scope else getattr(w, "compute", None)
    if c is None or c.gemm == "exact":
        return "exact", c or Compute()
    why = _unfit(c, w, _device_index(x) if x.device == w.device else None)
    if why is not None:
        name = getattr(w, "_bl_name", None) or f"weight {tuple(w.shape)}"
        raise RuntimeError(f"{name}: cannot compute as {c.describe()}: {why}")
    return c.route, c


def assign_compute_(model: torch.nn.Module, chosen: list[tuple[str, torch.nn.Module]], c,
                    skip_unfit: bool = False) -> list[str]:
    """Set ``c`` on every lattice weight inside the ``chosen`` (name, module) pairs. A weight that cannot honour it
    raises ValueError naming it (nothing is changed), or with ``skip_unfit`` keeps its setting and is logged with
    the reason. Returns the names of the weights set."""
    from .tensor import LatticeTensor
    comp = as_compute(c)
    targets, seen = [], set()
    for name, mod in chosen:
        for pname, p in mod.named_parameters():
            if not isinstance(p, LatticeTensor) or id(p) in seen:
                continue
            seen.add(id(p))
            full = ".".join(x for x in (name, pname) if x)
            why = _unfit(comp, p, p.device.index if p.is_cuda else None) if comp is not None else None
            if why is not None:
                if not skip_unfit:
                    raise ValueError(f"{full}: cannot compute as {comp.describe()}: {why}")
                log.info("%s keeps its compute setting (%s): %s", full, comp.describe(), why)
                continue
            targets.append((full, p))
    for full, p in targets:
        p.compute = comp
        p._bl_name = full
    return [full for full, _ in targets]
