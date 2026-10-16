"""Builds `cozy_int8` (the `torch.ops.cozy.int8_*` and `fp8_*` library) for one card,
`COZY_KERNEL_SM`, by its row (`COZY_KERNEL_ROW`: the MMA family, then its formats).

int8: wgmma and tcgen05 rows with int8 take the TMA GEMM, every other card the mma.sync GEMM.
fp8: wgmma, tcgen05 and block-scaled mma.sync (fp4 rows) take CUTLASS's TMA GEMM, other fp8
cards cuBLASLt's. Those build for the card's arch-specific target. CUTLASS is the `cutlass`
tree beside this file.
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sm = int(os.environ["COZY_KERNEL_SM"])
family, *formats = os.environ["COZY_KERNEL_ROW"].split()
cutlass = os.path.abspath("cutlass")
tma = family in ("wgmma", "tcgen05") and "int8" in formats
fp8 = "fp8" in formats and (family != "mma.sync" or "fp4" in formats)
arch = f"{sm}a" if tma or fp8 else str(sm)
defines = [
    f"-DCOZY_{family.replace('.', '_').upper()}",
    *(["-DCOZY_TMA"] if tma else []),
    *(["-DCOZY_FP8_GEMM"] if fp8 else []),
]
nvcc = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-DNDEBUG",
    *defines,
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
            ["op.cpp", "rotate.cu", "gemm_tma.cu" if tma else "gemm_sm80.cu"]
            + (["gemm_fp8.cu"] if fp8 else []),
            include_dirs=[f"{cutlass}/include", f"{cutlass}/tools/util/include"],
            extra_compile_args={"cxx": ["-O3", "-std=c++20", *defines, *isolated], "nvcc": nvcc},
            extra_link_args=["-Wl,-Bsymbolic"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
