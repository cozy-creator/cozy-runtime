"""The Runtime-owned operations an attempt may call: `Start.call_interfaces` rows marked
`builtin`, bound from the Runtime's own code."""

from __future__ import annotations

from cozy_runtime.author import ConformanceError

MODULE = "cozy_runtime.derive.operations"
EXPORTS = ("quantize",)


def validate_binding(module: str, export: str) -> None:
    if module != MODULE or export not in EXPORTS:
        raise ConformanceError("unknown Runtime operation", code="builtin_operation")
