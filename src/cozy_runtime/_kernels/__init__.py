"""Cozy's prebuilt CUDA kernels (`kernels/`), carried by cozy-runtime's native wheels: one per OS
and CPU, every GPU target inside. The pure wheel carries none (`ABSENT` says why).

Kernel modules use nanobind; the execution owner uses Rust/PyO3. Both use the static CUDA runtime
and Python's stable ABI. A module
carries a cubin for each target in `ARCHS` and PTX that the driver compiles for a card none names.
Runtime verifies every kernel on the device before it serves (`internal.prebuilt`); package
authors reach `_norm` through `author.kernels`.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

try:
    import cozy_runtime._kernels._gemm as _gemm
    import cozy_runtime._kernels._norm as _norm

    ABSENT = ""
except ModuleNotFoundError:  # the pure wheel: an OS or CPU no native wheel serves
    ABSENT = "this cozy-runtime wheel carries no kernels (the pure wheel)"
except ImportError as broken:
    ABSENT = f"this cozy-runtime wheel's kernels do not load ({broken})"
try:
    import cozy_runtime._kernels._fa3 as _fa3

    NO_FA3 = ""
except ModuleNotFoundError:  # Linux aarch64 (cr-200), Windows (cr-201) and the pure wheel
    NO_FA3 = "this cozy-runtime wheel carries no FlashAttention-3"
except ImportError as broken:
    NO_FA3 = f"this cozy-runtime wheel's FlashAttention-3 does not load ({broken})"
try:
    import cozy_runtime._kernels._sage3 as _sage3

    NO_SAGE3 = ""
except ModuleNotFoundError:  # Linux aarch64 (cr-200), Windows (cr-201) and the pure wheel
    NO_SAGE3 = "this cozy-runtime wheel carries no SageAttention3"
except ImportError as broken:
    NO_SAGE3 = f"this cozy-runtime wheel's SageAttention3 does not load ({broken})"

#: The CUDA toolkit the kernels are built with, and the oldest driver that runs it, per
#: `sys.platform`. cozy's `platform::gpu` holds the same floor for TensorD's startup check.
CUDA = "13.0"
DRIVER_FLOOR = {"linux": "580.65.06", "win32": "580.88"}

#: The tensor-core families, by Runtime's `arch` names, in the index order `_gemm.gemm` takes.
FAMILIES = ("mma.sync", "wgmma", "tcgen05")

#: Each module's cubin targets and PTX, as the CUDA toolkit names them (`cuobjdump`).
ARCHS: dict[str, dict[str, tuple[str, ...]]] = {
    "_execution": {
        "cubin": ("sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_103", "sm_120", "sm_121"),
        "ptx": (),
    },
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
    # SageAttention3's forward runs on RTX 50 and DGX Spark only (block-scaled FP4 mma.sync, TMA).
    "_sage3": {"cubin": ("sm_120a", "sm_121a"), "ptx": ()},
    "_norm": {
        "cubin": ("sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_103", "sm_120", "sm_121"),
        "ptx": ("compute_80",),
    },
}

#: The GEMMs that are CUTLASS's TMA kernels, by format and family: a Windows wheel lacks them
#: (`_gemm.TMA`, cr-201), and with them every one of `ARCHS`' arch-specific (`a`) cubins.
TMA_GEMM: dict[str, dict[str, tuple[int, ...]]] = {
    "int8": {"wgmma": (90,), "tcgen05": (100,)},
    "fp8": {"mma.sync": (120, 121), "wgmma": (90,), "tcgen05": (100, 103)},
}
_TMA = not ABSENT and _gemm.TMA
#: The cards `_gemm.gemm` has a cubin for, by format and family; quantize and gemv run on every
#: target above.
GEMM: dict[str, dict[str, tuple[int, ...]]] = {
    "int8": {"mma.sync": (80, 86, 89, 120, 121), **(TMA_GEMM["int8"] if _TMA else {})},
    "fp8": dict(TMA_GEMM["fp8"]) if _TMA else {},
}


def rms_rope_split_half(
    q: Any,
    k: Any,
    freqs: Any,
    q_scale: Any,
    k_scale: Any,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[Any, Any]:
    """Return paired out-of-place RMS-normalized split-half RoPE tensors."""
    import torch

    tensors = (q, k, freqs, q_scale, k_scale)
    if not all(isinstance(value, torch.Tensor) and value.is_cuda for value in tensors):
        raise ValueError("RMSNorm+RoPE inputs must all be CUDA tensors")
    if any(value.device != q.device for value in tensors[1:]):
        raise ValueError("RMSNorm+RoPE inputs must occupy one CUDA device")
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    stream = torch.cuda.current_stream(q.device).cuda_stream
    _norm.rms_rope_split_half(
        q,
        k,
        freqs,
        q_scale,
        k_scale,
        q_out,
        k_out,
        epsilon,
        stream,
        rot_dim,
    )
    return q_out, k_out


# The modules are `internal.prebuilt`'s to call; `rms_rope_split_half` is `author.kernels`'.
__all__ = ["_fa3", "_gemm", "_norm", "_sage3", "rms_rope_split_half"]
