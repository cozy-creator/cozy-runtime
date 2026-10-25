"""`python -m cozy_runtime.cli.conformance`: the conformance suite on this process's first card.

A hand-run proof, not data: run on a real card of a new tensor-core family before a rule names
its kernels (xs-065). It checks every decoder and native leaf against its independent oracle
(`probe.conform`) and the wheel's prebuilt kernels against their float64 laws (`verify`),
prints one JSON report and exits 1 on any failure. The Runtime reads nothing it prints.
"""

from __future__ import annotations

import json

import msgspec
import torch

from cozy_runtime import __version__
from cozy_runtime.internal import arch, attention, int8_linear, prebuilt
from cozy_runtime.internal.encoding import launch_providers, measure_device
from cozy_runtime.internal.probe import UnmeasuredEvidence, conform

DTYPES = ("bfloat16", "float16", "float32")


def main() -> int:
    device = measure_device(torch, 0)
    report = conform(torch, launch_providers(), device, DTYPES, release=__version__)
    failed = [
        f"{one.implementation}@{one.output_dtype}"
        for one in report.observations
        if not one.evidence.passed and not isinstance(one.evidence, UnmeasuredEvidence)
    ]
    kernels: dict[str, str] = {}
    if device.kind == "cuda":
        op = int8_linear.native(torch, 0)
        kernels["prebuilt"] = (
            f"{int8_linear.library(0).family}: serves"
            if op is not None
            else "; ".join(f"{one.op}: {one.reason}" for one in int8_linear.unavailable())
            or "absent"
        )
        if arch.row(device.sm).family == arch.WGMMA:
            cases = attention.BY_NAME["flash-attn3"].cases()
            fa3, why = prebuilt.fa3(torch, 0, device.sm, cases)
            kernels["fa3"] = "serves" if fa3 is not None else why
            if fa3 is None:
                failed.append("prebuilt-fa3")
        # A family the wheel does not carry serves torch's kernels by design; a kernel that
        # fails its self-check here is the failure.
        if "failed verification" in kernels["prebuilt"]:
            failed.append("prebuilt")
    print(
        json.dumps(
            {
                "device": device.document(),
                "row": arch.row(device.sm).name if device.kind == "cuda" else device.kind,
                "suite_ms": report.suite_ms,
                "observations": msgspec.to_builtins(report.observations),
                "kernels": kernels,
                "failed": failed,
            }
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
