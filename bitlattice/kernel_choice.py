"""Automatic implementation choices, each of which can be switched off.

Bitlattice picks some implementations on its own: they compute the same result as the plain implementation (bit
for bit, or with sums in another order) and only take less time or memory. :class:`Kernels` lists them; all are on by
default, and any can be turned off -- for a region of code, for a runner's steps, or for one layer:

    with bl.kernels(fused_norm=False):                 # a region (fields not given keep their current value)
        loss = runner.step(batch)
    runner = bl.Runner(model, opt, kernels=bl.Kernels.reference())                  # every step: the plain paths
    bl.surgery.set_compute_(model, "*.mlp", bl.Compute("nvfp4", kernels="reference"))   # one layer's GEMMs

The innermost setting applies: a layer's own ``Compute.kernels`` inside a region, a region inside a runner's step.
``bl.kernels_available()`` tells which fast paths exist on this machine.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, fields, replace

__all__ = ["Kernels", "active", "kernels", "kernels_available"]


@dataclass(frozen=True)
class Kernels:
    """Which automatic implementations may be used (True: used where available).

    two_level_gemm   NVFP4 GEMMs apply the per-tensor scales in the GEMM epilogue (else a scaling kernel follows)
    shared_operands  an input read by several FP4 layers (q / k / v, gate / up) is quantized once for all of them
    operand_cache    quantized weight operands are kept per weight version instead of rebuilt at every call
    fused_norm       CUDA kernels for the lean RMSNorm / gated RMSNorm conversions (else ATen code over row chunks)
    fused_conv       CUDA causal depthwise convolution in the gated_delta_net conversion (else conv1d)
    fused_pack       CUDA FP8 packing of compressed saved tensors (else ATen code over pieces)
    grad_in_forward  a frozen output layer forms its input gradient in the forward pass of the vocabulary-wide loss
                     (one pass over the vocabulary, nothing recomputed); else the gradient is formed in backward
    native_update    the optimizer's update of a large quantized matrix (direction and lattice step of every row
                     tile) in one native call; else a Python loop over the tiles issues the same operations
    fused_fold       deferred weight-gradient operands of the micro-batches are brought to one global scale by one
                     CUDA pass over their block scales; else per micro-batch with ATen operations
    """

    two_level_gemm: bool = True
    shared_operands: bool = True
    operand_cache: bool = True
    fused_norm: bool = True
    fused_conv: bool = True
    fused_pack: bool = True
    grad_in_forward: bool = True
    native_update: bool = True
    fused_fold: bool = True

    @classmethod
    def reference(cls) -> Kernels:
        """Every automatic choice off: the plain implementations."""
        return cls(**{f.name: False for f in fields(cls)})

    def describe(self) -> str:
        off = [f.name for f in fields(self) if not getattr(self, f.name)]
        if not off:
            return "auto"
        if len(off) == len(fields(self)):
            return "reference"
        return "auto except " + ", ".join(off)


def as_kernels(k) -> Kernels | None:
    """A Kernels from 'auto' (None: inherit), 'reference' or a Kernels."""
    if k is None or k == "auto":
        return None
    if k == "reference":
        return Kernels.reference()
    if isinstance(k, Kernels):
        return k
    raise ValueError(f"kernels must be 'auto', 'reference' or a bitlattice.Kernels, not {k!r}")


# Process-wide rather than thread-local: the backward pass runs on autograd worker threads and must see the settings
# of the forward pass that created it.
_stack: list[Kernels] = []


def active() -> Kernels:
    """The setting in effect."""
    return _stack[-1] if _stack else _DEFAULT


_DEFAULT = Kernels()


@contextlib.contextmanager
def kernels(k=None, **changes):
    """Use ``k`` (a :class:`Kernels`, 'auto' or 'reference'; default: the current setting) with ``changes`` applied,
    inside the block."""
    base = as_kernels(k) if k is not None else active()
    if k == "auto":
        base = _DEFAULT
    unknown = set(changes) - {f.name for f in fields(Kernels)}
    if unknown:
        raise TypeError(f"unknown kernel choices {sorted(unknown)}; known: {[f.name for f in fields(Kernels)]}")
    _stack.append(replace(base, **changes))
    try:
        yield _stack[-1]
    finally:
        _stack.pop()


def kernels_available(device_index: int = 0) -> dict[str, bool]:
    """Which fast paths exist here (an unavailable path is not used, whatever the setting)."""
    import torch

    from . import _ops, backend
    cuda = torch.cuda.is_available()
    return {"extension": _ops.HAVE_EXT,
            "fp4_tensor_cores": cuda and backend.fp4_supported(device_index),
            "two_level_gemm": cuda and backend.fp4_supported(device_index) and backend._two_level(device_index),
            "mxfp8_tensor_cores": cuda and backend.mxfp8_supported(device_index),
            "fused_norm": _ops.HAVE_EXT, "fused_conv": _ops.HAVE_EXT, "fused_pack": _ops.HAVE_EXT,
            "native_update": _ops.HAVE_EXT, "fused_fold": _ops.HAVE_EXT}
