"""The Runtime-owned operations an attempt may call: `Start.call_interfaces` rows marked
`builtin`, bound from the Runtime's own code."""

from __future__ import annotations

from cozy_runtime.author._errors import ConformanceError

#: Each module's operations: quantize, and the release gate's exercisers (`cozy_runtime.bench`, unstable): the attention
#: bench (xs-171), the GEMM bench (xs-177), the kernels suite and the varena and execution suites' Runtime cells (xs-174).
EXPORTS = {
    "cozy_runtime.derive.operations": ("quantize",),
    "cozy_runtime.bench": (
        "attention_inventory",
        "attention_speed",
        "attention_quality",
        "gemm_speed",
        "gemm_tiles",
        "kernels_inventory",
        "kernel_check",
        "kernel_ready",
        "child_wire",
        "varena_core",
        "offload_in_executor",
        "quantize_reuse",
        "overlay_replay",
        "lora_overlay_refill",
    ),
}


def validate_binding(module: str, export: str) -> None:
    if export not in EXPORTS.get(module, ()):
        raise ConformanceError("unknown Runtime operation", code="builtin_operation")
