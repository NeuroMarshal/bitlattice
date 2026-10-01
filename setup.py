import glob
import os
import shutil

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# TORCH_CUDA_ARCH_LIST unset: build for the GPUs visible at build time, with the architecture-specific target
# (e.g. 12.0a) from compute capability 9.0 on, which enables instructions such as the hardware FP4 conversion; set it
# explicitly for a portable build (e.g. "8.6;8.9;12.0")
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    try:
        import torch
        caps = sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
        if caps:
            os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(f"{a}.{b}{'a' if a >= 9 else ''}" for a, b in caps)
    except Exception:  # no driver at build time: torch's default list
        pass

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
