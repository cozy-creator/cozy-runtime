"""The Runtime-owned operations an attempt may call: `Start.call_interfaces` rows marked
`builtin`, bound from the Runtime's own code."""

from __future__ import annotations

from cozy_runtime.author._errors import ConformanceError

#: Each module's operations: quantize, and the attention bench (xs-171).
EXPORTS = {
    "cozy_runtime.derive.operations": ("quantize",),
    "cozy_runtime.bench": ("attention_inventory", "attention_speed", "attention_quality"),
}


def validate_binding(module: str, export: str) -> None:
    if export not in EXPORTS.get(module, ()):
        raise ConformanceError("unknown Runtime operation", code="builtin_operation")
