"""A plan that serves tensors decoded because this device REFUSED their leaf says so, once per
(component, lane, leaf, reason). A decode by design (a geometry the format rules out, a package
that declines leaves) says nothing.

Real measurements only: the shipped sm86 table (RTX 3090) and the recorded sm89 probe, whose
fp8 leaf failed its float16 GEMM check and raised for float32.
"""

from __future__ import annotations

import json

import msgspec

from cozy_runtime.internal import probe, qualification
from cozy_runtime.internal.encoding import (
    SPEC_INT8_CONVROT256,
    SPEC_ROWWISE,
    SPEC_ROWWISE_KEEPDIM,
    Capabilities,
    DeviceFacts,
    Encoded,
    RolePart,
    aliases,
    launch_providers,
)
from cozy_runtime.internal.fill import PlanRow, dtype_name
from cozy_runtime.internal.planfacts import PlanFacts
from cozy_runtime.internal.probe import QualificationResult
from cozy_runtime.internal.resolution import (
    ConstructionFacts,
    RefusedLeaves,
    ResolvedModelPlan,
    Variant,
    resolve,
)
from test_leaf_speedup_record import rekeyed

FP8_LEAF = "cozy.fp8-rowwise.native-leaf/1"


def row(key: str, spec: str, dtype: str, shape: tuple[int, int]) -> PlanRow:
    data = "i8" if spec == SPEC_INT8_CONVROT256 else "f8_e4m3fn"
    parts = (
        RolePart("data", data, shape, shape[0] * shape[1]),
        RolePart("scale", "f32", (shape[0], 1), 4 * shape[0]),
    )
    component, _, name = key.partition(".")
    encoded = Encoded(spec, aliases()[spec], parts)
    return PlanRow(key, name, component, dtype, shape, 2 * shape[0] * shape[1], encoded)


def plan(
    rows: list[PlanRow], capabilities: Capabilities, device: DeviceFacts, consent: str = "accept"
) -> ResolvedModelPlan:
    return resolve(
        variants=[Variant("lane", "store", "sha256:" + "1" * 64, {}, True)],
        rows_for={"lane": rows},
        providers=launch_providers(),
        capabilities=capabilities,
        device=device,
        runtime=rekeyed().runtime,
        construction=ConstructionFacts("t/1", "t:Model", ("te", "unet")),
        facts=PlanFacts(),
        encoded_leaves=consent,
        dtype_name=dtype_name,
    )


def test_a_leaf_below_its_floor_warns_and_a_geometry_or_a_declined_leaf_does_not() -> None:
    path = next(qualification.SHIPPED.glob("cuda.sm86-*.json"))
    table = msgspec.convert(json.loads(path.read_bytes()), QualificationResult)
    providers = launch_providers()
    cells = {s: providers[s] for s in (SPEC_ROWWISE, SPEC_ROWWISE_KEEPDIM, SPEC_INT8_CONVROT256)}
    admitted = qualification.qualified(
        object(), cells, table.device, ("bfloat16",), release="t/1", runtime=table.runtime
    )
    rows = [
        row("unet.a.weight", SPEC_ROWWISE, "bf16", (64, 256)),
        row("unet.b.weight", SPEC_ROWWISE, "bf16", (128, 256)),
        row("te.c.weight", SPEC_INT8_CONVROT256, "bf16", (64, 256)),
        # The int8 GEMM needs N % 8 == 0: decoded by the format's rule, not refused.
        row("te.d.weight", SPEC_INT8_CONVROT256, "bf16", (60, 256)),
    ]
    served = plan(rows, admitted.capabilities, table.device)
    assert {t.key: t.route for t in served.tensors} == {
        "unet.a.weight": "decoded_float",
        "unet.b.weight": "decoded_float",
        "te.c.weight": "encoded_gemm",
        "te.d.weight": "decoded_float",
    }
    assert served.refused_leaves() == (
        RefusedLeaves(
            "unet",
            "fp8-rowwise/1",
            FP8_LEAF,
            "its kernel needs cuda.sm89+ and this device is cuda.sm86",
            2,
        ),
    )
    assert plan(rows, admitted.capabilities, table.device, "refuse").refused_leaves() == ()


def test_a_leaf_that_failed_or_raised_in_qualification_warns_once_per_reason() -> None:
    result = rekeyed()
    rows = [
        row("unet.a.weight", SPEC_ROWWISE, "f16", (64, 256)),
        row("unet.b.weight", SPEC_ROWWISE, "f32", (64, 256)),
        row("unet.c.weight", SPEC_ROWWISE, "f32", (128, 256)),
        row("unet.d.weight", SPEC_ROWWISE, "bf16", (64, 256)),
    ]
    served = plan(rows, probe.admission_table(result), result.device)
    assert served.by_key()["unet.d.weight"].route == "encoded_gemm"
    assert [(g.reason, g.tensors) for g in served.refused_leaves()] == [
        (
            "builtins.ValueError: Only bf16 and fp16 high precision output types are "
            "supported for row-wise scaling.",
            2,
        ),
        (
            "its GEMM measured 0.9 relative error against the 0.3 bound (finite True, out float16)",
            1,
        ),
    ]
