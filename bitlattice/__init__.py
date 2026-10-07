"""Bitlattice: training and fine-tuning of PyTorch models directly in low-bit and arbitrary number formats.

Principles of the API: docs/API.md. Edits of a model are in :mod:`bitlattice.surgery` (selectors: :mod:`bitlattice.
select`); how lattice layers compute: :class:`Compute`; automatic kernel choices and how to switch them off:
:class:`Kernels`; what was done: :func:`explain`.
"""
from . import (
           backend,
           data,
           formats,
           hostmem,
           io,
           kvcache,
           memory,
           nn,
           optim,
           parallel,
           pipeline,
           quant,
           select,
           storage,
           surgery,
)
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

__version__ = "0.2.0"

__all__ = [
           "COMPUTE_PRESETS",
           "HAVE_EXT",
           "Compute",
           "Format",
           "Kernels",
           "Lattice",
           "LatticeTensor",
           "Memory",
           "NativeFormat",
           "QuantConfig",
           "Rule",
           "Runner",
           "TableFormat",
           "UniformFormat",
           "backend",
           "compute",
           "data",
           "dispatch",
           "explain",
           "formats",
           "hostmem",
           "infer_device_map",
           "io",
           "is_lattice",
           "kernels",
           "kernels_available",
           "kvcache",
           "memory",
           "nn",
           "optim",
           "parallel",
           "pipeline",
           "plan",
           "quant",
           "select",
           "storage",
           "surgery",
]
