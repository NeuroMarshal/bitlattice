"""Bitlattice: training and fine-tuning of PyTorch models directly in low-bit and arbitrary number formats.

Principles of the API: docs/API.md. Edits of a model are in :mod:`bitlattice.surgery` (selectors: :mod:`bitlattice.
select`); how lattice layers compute: :class:`Compute`; automatic kernel choices and how to switch them off:
:class:`Kernels`; what was done: :func:`explain`.
"""
from . import backend, data, formats, hostmem, io, kvcache, memory, nn, optim, parallel, pipeline, quant, select, \
    storage, surgery
from ._ops import HAVE_EXT
from .compute_policy import PRESETS as COMPUTE_PRESETS
from .compute_policy import Compute, compute
from .explain import explain
from .formats import Format, NativeFormat, TableFormat, UniformFormat
from .kernel_choice import Kernels, kernels, kernels_available
from .optim import Lattice
from .parallel import dispatch, infer_device_map
from .quant import QuantConfig, Rule
from .runner import Memory, Runner
from .storage import plan
from .tensor import LatticeTensor, is_lattice

__version__ = "0.2.0rc1"

__all__ = ["backend", "data", "formats", "io", "kvcache", "memory", "nn", "optim", "parallel", "pipeline", "quant",
           "select", "storage", "surgery", "HAVE_EXT", "Compute", "COMPUTE_PRESETS", "compute", "explain", "Format",
           "NativeFormat", "TableFormat", "UniformFormat", "Lattice", "dispatch", "infer_device_map", "QuantConfig",
           "Rule", "plan", "LatticeTensor", "is_lattice", "Memory", "Runner", "Kernels", "kernels", "kernels_available",
           "hostmem"]
