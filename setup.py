import glob
import os
import shutil

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# TORCH_CUDA_ARCH_LIST unset: build for the GPUs visible at build time (e.g. "8.6;8.9;12.0" for a portable build)

cxx = ["-O3", "-std=c++20", "-DGLOG_USE_GLOG_EXPORT"]
nvcc = ["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-lineinfo", "--fmad=false", "-DGLOG_USE_GLOG_EXPORT"]
host = os.environ.get("CUDAHOSTCXX") or next((c for c in ("g++-15", "g++-14", "g++-13") if shutil.which(c)), None)
if host:
    nvcc += ["-ccbin", host]

setup(
    ext_modules=[
        CUDAExtension(
            "bitlattice._C",
            sources=sorted(glob.glob("csrc/*.cpp") + glob.glob("csrc/*.cu")),
            include_dirs=[os.path.abspath("csrc")],
            extra_compile_args={"cxx": cxx, "nvcc": nvcc},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
