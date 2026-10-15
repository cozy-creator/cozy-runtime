"""Builds `cozy_int8` (the `torch.ops.cozy.int8_*` library) for one card, `COZY_KERNEL_SM`.

Hopper and datacenter Blackwell take the TMA GEMM (wgmma, tcgen05) for their arch-specific
target; every other card the mma.sync GEMM. CUTLASS is the `cutlass` tree beside this file.
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sm = int(os.environ["COZY_KERNEL_SM"])
cutlass = os.path.abspath("cutlass")
tma = sm in (90, 100)
arch = f"{sm}a" if tma else str(sm)
nvcc = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-DNDEBUG",
    f"-DCOZY_SM={sm}",
    # The GEMM waits on its rotate kernel by grid dependency (programmatic launch).
    "-DCUTLASS_ENABLE_GDC_FOR_SM90",
    "-DCUTLASS_ENABLE_GDC_FOR_SM100",
    f"-gencode=arch=compute_{arch},code=sm_{arch}",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-Xcompiler=-fvisibility=hidden,-fno-gnu-unique",
]
# Its CUTLASS instantiations may match another library's in this process (comfy-kitchen's
# serves until this builds): hidden, never unique, bound to itself, nothing of it is shared.
isolated = ["-fvisibility=hidden", "-fno-gnu-unique"]
setup(
    name="cozy-int8",
    version="1",
    ext_modules=[
        CUDAExtension(
            "cozy_int8",
            ["op.cpp", "rotate.cu", "gemm_tma.cu" if tma else "gemm_sm80.cu"],
            include_dirs=[f"{cutlass}/include", f"{cutlass}/tools/util/include"],
            extra_compile_args={"cxx": ["-O3", "-std=c++20", *isolated], "nvcc": nvcc},
            extra_link_args=["-Wl,-Bsymbolic"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
