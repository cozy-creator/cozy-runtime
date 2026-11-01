"""The oldest torch and Triton the Runtime serves on, as minimums, never exact releases.

`TORCH` (2.10): `torch._C._accelerator_setAllocatorSettings`, which configures the CUDA allocator
at every executor's start (`accel.initialize`); below it an executor refuses to start
(`torch_unsupported`). `GRAPHS` (torch 2.13): the cuBLAS and cuBLASLt workspace sizes
`graphs.domain` holds its graphs to; below it CUDA graphs are off, so native denoise and LM decode
run eager. `TRITON_SETS` (Triton 3.8): `triton.runtime.build._get_cache_manager`, the launcher's
cache key (`triton_kernels`); below it every Triton set is off, prebuilt or compiled, and the eager
paths serve. A capability off is warned once per process and observed on every run it serves
(`capability.off`).
"""

from __future__ import annotations

import sys
from contextlib import suppress
from importlib import metadata
from typing import Any

from packaging.version import InvalidVersion, Version

TORCH = (2, 10)
GRAPHS = (2, 13)
TRITON_SETS = (3, 8)

#: Each capability this process serves without: the torch and Triton it found, and why.
OFF: dict[str, dict[str, str]] = {}
_NAMES = {"cuda_graphs": "CUDA graphs", "triton_sets": "Triton sets"}


def older(name: str, version: str, floor: tuple[int, int]) -> str:
    """Why `name` `version` is below `floor`, else ""; an unreadable version is below."""
    try:
        release = Version(version).release
    except InvalidVersion:
        release = (0,)
    return f"{name} {version} is older than {floor[0]}.{floor[1]}" if release < floor else ""


def triton() -> str:
    """The installed Triton's release, read from its distribution without importing it."""
    for distribution in ("triton", "triton-windows"):
        with suppress(metadata.PackageNotFoundError):
            return metadata.version(distribution)
    return ""


def off(capability: str, reason: str, torch: str, triton: str) -> None:
    """`capability` off in this process: warned once, then on every run's record."""
    if capability in OFF:
        return
    OFF[capability] = {"torch": torch, "triton": triton, "reason": reason}
    found = f"torch {torch}, Triton {triton or 'absent'}"
    print(f"[cozy] warning: {_NAMES[capability]} off: {reason} ({found})", file=sys.stderr)


def start(torch: Any, kind: str) -> str:
    """An executor's start on `torch`: why it cannot serve (older than `TORCH`), else "", with
    each capability a CUDA executor's torch or Triton turns off recorded (`off`)."""
    version, found = str(torch.__version__), triton()
    if refused := older("torch", version, TORCH):
        return f"{refused}, the Runtime's floor"
    if kind == "cuda":
        if reason := older("torch", version, GRAPHS):
            off("cuda_graphs", reason, version, found)
        if found and (reason := older("Triton", found, TRITON_SETS)):
            off("triton_sets", reason, version, found)
    return ""
