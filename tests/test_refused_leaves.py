"""A plan that serves tensors decoded because this device REFUSED their leaf says so, once per
(component, lane, leaf, reason). A decode by design (a geometry the format rules out, a package
that declines leaves) says nothing. A run names the kernel each component's leaves served
(`gemm.selected`) and each fallback below its lane (`gemm.fallback`).

Real measurements only: the shipped sm86 table (RTX 3090) and the recorded sm89 probe, whose
fp8 leaf failed its float16 GEMM check and raised for float32.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest

from cozy_runtime.author._services import Attempt
from cozy_runtime.author.kernels import INT8_CONVROT
from cozy_runtime.derive.microscale import encode_int8_convrot256
from cozy_runtime.internal import int8_linear, probe, qualification
from cozy_runtime.internal.encoding import (
    SPEC_INT8_CONVROT256,
    SPEC_NVFP4,
    SPEC_ROWWISE,
    SPEC_ROWWISE_KEEPDIM,
    Capabilities,
    DeviceFacts,
    Encoded,
    Int8ConvRotNativeLeaf,
    Nvfp4NativeLeaf,
    RolePart,
    aliases,
    launch_providers,
)
from cozy_runtime.internal.encoding.leaves import NVFP4
from cozy_runtime.internal.encoding.leaves import sites as leaf_sites
from cozy_runtime.internal.executor import Executor
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
from test_int8_convrot import weight
from test_leaf_speedup_record import rekeyed
from test_nvfp4 import roles

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
            "bfloat16",
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


def test_a_run_names_each_components_gemm_kernel_and_each_fallback(tmp_path: Path) -> None:
    """Real leaves, counted as the plane counts them when they install; on the CPU each provider
    serves its eager kernel, read once for the run. The refused fp8 leaves of the sm86 plan serve
    bf16, and an 8-bit op library that failed for good is the reason int8 leaves serve torch's GEMM.
    A leaf's call records nothing: the forward is 0.26.4's."""
    torch = pytest.importorskip("torch")
    encoded = encode_int8_convrot256("w", weight(64, 512, 5))
    int8_roles = {
        "data": torch.frombuffer(bytearray(encoded.payload), dtype=torch.int8).reshape(64, 512),
        "scale": torch.frombuffer(bytearray(encoded.companions[0].raw), dtype=torch.float32),
    }
    stand_in = probe._StandIn(512, 64)
    int8 = Int8ConvRotNativeLeaf(SPEC_INT8_CONVROT256)
    leaves = {
        key: (int8.leaf(torch, int8_roles, stand_in, torch.bfloat16), None, ())
        for key in ("transformer.a.weight", "transformer.b.weight", "text_encoder.c.weight")
    }
    nvfp4 = Nvfp4NativeLeaf(SPEC_NVFP4).leaf(
        torch, roles(torch, weight(256, 512, 6), "cpu"), probe._StandIn(512, 256), torch.bfloat16
    )
    leaves["transformer.d.weight"] = (nvfp4, None, ())
    before = set(vars(nvfp4))
    nvfp4(torch.randn(40, 512, dtype=torch.bfloat16))
    assert set(vars(nvfp4)) == before  # nothing written per call

    table = msgspec.convert(
        json.loads(next(qualification.SHIPPED.glob("cuda.sm86-*.json")).read_bytes()),
        QualificationResult,
    )
    providers = launch_providers()
    cells = {s: providers[s] for s in (SPEC_ROWWISE, SPEC_ROWWISE_KEEPDIM)}
    admitted = qualification.qualified(
        object(), cells, table.device, ("bfloat16",), release="t/1", runtime=table.runtime
    )
    rows = [row(f"unet.{n}.weight", SPEC_ROWWISE, "bf16", (64, 256)) for n in "ab"]
    executor = Executor(None, tmp_path)  # type: ignore[arg-type]
    executor.backend = SimpleNamespace(gemm_sites=leaf_sites(leaves))
    executor.models = {"klein": object()}
    executor.refused_leaves = plan(rows, admitted.capabilities, table.device).refused_leaves()

    def emitted() -> list[tuple[object, ...]]:
        record = Attempt("gemm", tmp_path, 0)
        Executor._emit_gemm(record, executor)
        rows = record.ring.rows()
        confessions = [row for row in rows if row["kind"] == "confession"]
        warnings = [row for row in rows if row["kind"] == "log"]
        # Each fallback is also one warning line, its facts as fields.
        assert [(w["name"], w["value"], w["fields"]) for w in warnings] == [
            ("a GEMM served below its lane", "warn", {**c["fields"], "kernel": c["value"]})  # type: ignore[dict-item]
            for c in confessions
            if c["name"] == "gemm.fallback"
        ]
        return [
            (row["name"], row["value"], *sorted(row["fields"].items()))  # type: ignore[union-attr]
            for row in confessions
        ]

    def int8_row(component: str, sites: int) -> tuple[object, ...]:
        return ("gemm.selected", int8_linear.EAGER, ("component", f"klein/{component}"),
                ("format", "int8"), ("leaf", INT8_CONVROT), ("sites", sites))  # fmt: skip

    def int8_fallback(component: str, sites: int) -> tuple[object, ...]:
        return ("gemm.fallback", int8_linear.EAGER, ("component", f"klein/{component}"),
                ("reason", "RuntimeError: no nvcc"), ("served", int8_linear.EAGER),
                ("sites", sites), ("wanted", int8_linear.OP))  # fmt: skip

    nvfp4_row = ("gemm.selected", "torch-decode", ("component", "klein/transformer"),
                 ("format", "nvfp4"), ("leaf", NVFP4), ("sites", 1))  # fmt: skip
    refused = (
        "gemm.fallback",
        "torch.linear",
        ("component", "klein/unet"),
        ("reason", "its kernel needs cuda.sm89+ and this device is cuda.sm86"),
        ("served", "bf16"),
        ("sites", 2),
        ("wanted", "fp8"),
    )
    assert emitted() == [
        int8_row("text_encoder", 1),
        int8_row("transformer", 2),
        nvfp4_row,
        refused,
    ]

    int8_linear._NATIVE[0] = int8_linear._Build(final="RuntimeError: no nvcc")
    try:
        assert emitted() == [
            int8_row("text_encoder", 1),
            int8_fallback("text_encoder", 1),
            int8_row("transformer", 2),
            int8_fallback("transformer", 2),
            nvfp4_row,
            refused,
        ]
    finally:
        del int8_linear._NATIVE[0]
