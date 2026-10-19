"""Build tools the Runtime uses when present and never bundles; costs and text are cozy's
(`crates/platform/src/tools.rs` in cozy-creator/cozy-creator).

A missing one never fails a run: the build that needs it does not start, its fallback serves,
and every run of the process carries one warning for the tool. Codes: `kernel_prebuilt_missing`,
`kernel_verify_failed`, `triton_no_compiler` (`extension_needs_toolchain`: no package recipe
builds here yet).
"""

from __future__ import annotations

import shutil

from cozy_runtime.internal import config

COSTS = {
    "nvcc": "cards without a prebuilt kernel use the slower SDPA/kitchen path",
    "cc": "Triton kernels and package extensions can't build; those ops use slower fallbacks",
    "ninja": "kernel builds compile one file at a time",
}
#: tool -> the code of the first build its absence stopped in this process
_MISSING: dict[str, str] = {}


def warning(tool: str) -> str:
    return f"{tool} not found: {COSTS[tool]}; see `cozy doctor`"


def missing(tool: str, code: str) -> None:
    _MISSING.setdefault(tool, code)


def warnings() -> list[tuple[str, dict[str, str | bool | int | float | None]]]:
    """A run's rows: one per missing tool."""
    return [(warning(tool), {"code": code, "tool": tool}) for tool, code in _MISSING.items()]


def triton() -> bool:
    """Whether Triton finds a C compiler here, as it looks (`CC`, else gcc or clang)."""
    cc = config.inherited_environment().get("CC")
    if shutil.which(cc) if cc else shutil.which("gcc") or shutil.which("clang"):
        return True
    missing("cc", "triton_no_compiler")
    return False
