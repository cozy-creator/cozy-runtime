"""Cozy's prebuilt CUDA kernels: one wheel per OS and CPU, every GPU target inside.

Each module is nanobind over the CUDA runtime (linked static) on Python's stable ABI, so neither
torch nor the Python version is a build input. A module carries a cubin for each target in
`ARCHS` and PTX that the driver compiles for a card none names. Runtime verifies every kernel on
the device before it serves; this package holds no policy.
"""

#: The CUDA toolkit the wheel is built with, and the oldest driver that runs it, per
#: `sys.platform`. cozy-platform's `gpu` holds the same floor for TensorD's startup check.
CUDA = "13.0"
DRIVER_FLOOR = {"linux": "580.65.06", "win32": "580.88"}

#: The tensor-core families, by Runtime's `arch` names, in the index order `_gemm.gemm` takes.
FAMILIES = ("mma.sync", "wgmma", "tcgen05")

#: Each module's cubin targets and PTX, as the CUDA toolkit names them (`cuobjdump`).
ARCHS: dict[str, dict[str, tuple[str, ...]]] = {
    "_gemm": {
        "cubin": (
            "sm_80",
            "sm_86",
            "sm_89",
            "sm_90",
            "sm_90a",
            "sm_100",
            "sm_100a",
            "sm_103",
            "sm_103a",
            "sm_120",
            "sm_120a",
            "sm_121",
            "sm_121a",
        ),
        "ptx": ("compute_80",),
    },
    # FlashAttention-3's forward runs on Hopper only (wgmma, TMA).
    "_fa3": {"cubin": ("sm_90a",), "ptx": ()},
}

#: The cards `_gemm.gemm` has a cubin for, by format and family; quantize and gemv run on every
#: target above.
GEMM: dict[str, dict[str, tuple[int, ...]]] = {
    "int8": {"mma.sync": (80, 86, 89, 120, 121), "wgmma": (90,), "tcgen05": (100,)},
    "fp8": {"mma.sync": (120, 121), "wgmma": (90,), "tcgen05": (100, 103)},
}
