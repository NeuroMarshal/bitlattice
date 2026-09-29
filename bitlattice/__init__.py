"""Bitlattice: training and fine-tuning of PyTorch models directly in low-bit and arbitrary number formats."""
from . import formats, nn, optim, quant
from ._ops import HAVE_EXT
from .formats import Format, NativeFormat, TableFormat, UniformFormat
from .optim import Lattice
from .quant import QuantConfig, Rule, dequantize_, quantize_
from .tensor import LatticeTensor, is_lattice

__version__ = "0.1.0.dev0"

__all__ = ["formats", "nn", "optim", "quant", "Lattice", "LatticeTensor", "is_lattice", "QuantConfig", "Rule",
           "quantize_", "dequantize_", "Format", "TableFormat", "UniformFormat", "NativeFormat", "HAVE_EXT"]
