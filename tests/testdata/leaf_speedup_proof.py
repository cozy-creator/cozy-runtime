"""The leaf speedup measured on THIS card, read through the real resolution (cr-135).

Run as a child process by `tests/test_leaf_speedup_record.py`, which skips it below sm89 or
without a CUDA device. Nothing is recorded or faked: the qualification suite runs on the card,
the rowwise record it mints resolves one bf16 linear at the probe's own shape, and the worker's
renderer prints the line. Prints one JSON line.
"""

from __future__ import annotations

import dataclasses
import json

import msgspec
import torch

from cozy_runtime.internal import probe
from cozy_runtime.internal.encoding import (
    SPEC_ROWWISE,
    Encoded,
    RolePart,
    launch_providers,
    measure_device,
    measure_runtime,
)
from cozy_runtime.internal.fill import dtype_name
from cozy_runtime.internal.planfacts import PlanFacts
from cozy_runtime.internal.resolution import ConstructionFacts, Variant, resolve
from cozy_runtime.internal.worker.session import _Delivery, _delivery_observation


@dataclasses.dataclass(frozen=True, slots=True)
class Row:
    key: str
    name: str
    component: str
    dtype: str
    shape: tuple[int, ...]
    encoded: Encoded


def main() -> None:
    device = measure_device(torch, 0)
    runtime = measure_runtime(torch, "test")
    providers = launch_providers()
    qualification = probe.qualified(
        torch, providers, device, ["bfloat16"], release="test", runtime=runtime
    )
    encoded = Encoded(
        encoding=SPEC_ROWWISE,
        alias="fp8-rowwise/1",
        parts=(
            RolePart("data", "f8_e4m3fn", (2048, 2048), 2048 * 2048),
            RolePart("scale", "f32", (2048,), 2048 * 4),
        ),
    )
    row = Row("dit.to_q.weight", "to_q.weight", "dit", "bf16", (2048, 2048), encoded)
    plan = resolve(
        variants=[
            Variant(
                name="fp8",
                store="store",
                snapshot="sha256:aa",
                snapshots={"dit": "sha256:aa"},
                reference=True,
            )
        ],
        rows_for={"fp8": [row]},
        providers=providers,
        capabilities=qualification.capabilities,
        device=device,
        runtime=runtime,
        construction=ConstructionFacts(release="test", model_class="pkg:Dit", components=("dit",)),
        facts=PlanFacts(),
        objective="latency",
        encoded_leaves="accept",
        dtype_name=dtype_name,
    )
    wire = plan.wire_document()
    print(
        json.dumps(
            {
                "sm": device.sm,
                "qualification": qualification.document(),
                "leaf_records": sorted(
                    r.leaf_speedup_x
                    for r in qualification.capabilities.records
                    if r.delivery_route == "encoded_gemm"
                ),
                "wire": wire,
                "line": _delivery_observation(msgspec.convert(wire, _Delivery)),
                "confession": plan.confession,
            }
        )
    )


if __name__ == "__main__":
    main()
