"""Bitlattice: training and fine-tuning of PyTorch models directly in low-bit and arbitrary number formats."""
from . import backend, data, formats, io, kvcache, memory, nn, optim, parallel, pipeline, quant, storage, surgery
from .backend import compute, fp4_recipe, set_compute, set_fp4_recipe
from ._ops import HAVE_EXT
from .formats import Format, NativeFormat, TableFormat, UniformFormat
from .optim import Lattice
from .parallel import dispatch, infer_device_map
from .quant import QuantConfig, Rule, dequantize_, quantize_
from .storage import plan
from .tensor import LatticeTensor, is_lattice

__version__ = "0.1.0"

__all__ = ["backend", "data", "kvcache", "memory", "pipeline", "compute", "set_compute", "fp4_recipe", "set_fp4_recipe", "formats", "io", "nn", "optim", "parallel", "quant", "storage", "surgery", "plan",
           "dispatch", "infer_device_map", "Lattice", "LatticeTensor", "is_lattice", "QuantConfig", "Rule",
           "quantize_", "dequantize_", "Format", "TableFormat", "UniformFormat", "NativeFormat", "HAVE_EXT"]
