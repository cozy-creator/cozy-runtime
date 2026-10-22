"""A plan that serves tensors decoded because this card does not serve their leaf (its row
lacks the leaf's tensor cores, `selection.admit`) says so, once per (component, lane, leaf,
reason). A decode by design (a geometry the format rules out, a package that declines leaves)
says nothing. A run names the kernel each component's leaves served (`gemm.selected`) and each
fallback below its lane (`gemm.fallback`).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cozy_runtime import _kernels
from cozy_runtime.author._services import Attempt
from cozy_runtime.author.kernels import INT8_CONVROT, ROWWISE_FP8
from cozy_runtime.derive.microscale import encode_int8_convrot256
from cozy_runtime.internal import fp8_linear, int8_linear, prebuilt, probe
from cozy_runtime.internal.encoding import (
    DeviceFacts,
    Encoded,
    Int8ConvRotNativeLeaf,
    Nvfp4NativeLeaf,
    RolePart,
    RuntimeIdentity,
    launch_providers,
    params_of,
)
from cozy_runtime.internal.encoding.leaves import NVFP4
from cozy_runtime.internal.encoding.leaves import sites as leaf_sites
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.fill import PlanRow, dtype_name
from cozy_runtime.internal.resolution import (
    ConstructionFacts,
    RefusedLeaves,
    ResolvedModelPlan,
    Variant,
    resolve,
)
from test_int8_convrot import weight
from test_nvfp4 import roles

FP8_LEAF = "cozy.fp8-rowwise.native-leaf/1"
INT8_LEAF = "cozy.int8-convrot256.native-leaf/1"
RTX_3090 = DeviceFacts("cuda", "NVIDIA GeForce RTX 3090", 86, "580.95.05/api13000", "350W/N/A")
B300 = DeviceFacts("cuda", "NVIDIA B300", 103, "580.95.05/api13000", "1100W/N/A")


PARAMS: dict[str, dict[str, object]] = {
    "fp8.rowwise": {},
    "int8.convrot": {"rotation": "kron_regular_h4", "rotation_size": 256},
}


def row(key: str, scheme: str, dtype: str, shape: tuple[int, int]) -> PlanRow:
    weight = "i8" if scheme == "int8.convrot" else "f8_e4m3fn"
    parts = (
        RolePart("weight", weight, shape, shape[0] * shape[1]),
        RolePart("scale", "f32", (shape[0],), 4 * shape[0]),
    )
    component, _, name = key.partition(".")
    encoded = Encoded(scheme, params_of(PARAMS[scheme]), parts)
    return PlanRow(key, name, component, dtype, shape, 2 * shape[0] * shape[1], encoded)


def plan(rows: list[PlanRow], device: DeviceFacts, consent: str = "accept") -> ResolvedModelPlan:
    return resolve(
        variant=Variant("lane", "store", "sha256:" + "1" * 64),
        rows=rows,
        providers=launch_providers(),
        device=device,
        runtime=RuntimeIdentity("t/1", "2.14.0"),
        construction=ConstructionFacts("t/1", "t:Model", ("te", "unet")),
        encoded_leaves=consent,
        dtype_name=dtype_name,
    )


def test_a_leaf_below_its_floor_warns_and_a_geometry_or_a_declined_leaf_does_not() -> None:
    rows = [
        row("unet.a.weight", "fp8.rowwise", "bf16", (64, 256)),
        row("unet.b.weight", "fp8.rowwise", "bf16", (128, 256)),
        row("te.c.weight", "int8.convrot", "bf16", (64, 256)),
        # The int8 GEMM needs N % 8 == 0: decoded by the format's rule, not refused.
        row("te.d.weight", "int8.convrot", "bf16", (60, 256)),
    ]
    served = plan(rows, RTX_3090)
    assert {t.key: t.route for t in served.tensors} == {
        "unet.a.weight": "decoded_float",
        "unet.b.weight": "decoded_float",
        "te.c.weight": "encoded_gemm",
        "te.d.weight": "decoded_float",
    }
    assert served.refused_leaves() == (
        RefusedLeaves(
            "unet",
            "fp8.rowwise",
            FP8_LEAF,
            f"{FP8_LEAF} needs cuda.sm89+ and this device is cuda.sm86",
            "bfloat16",
            2,
        ),
    )
    assert plan(rows, RTX_3090, "refuse").refused_leaves() == ()


def test_a_card_whose_row_lacks_the_lanes_tensor_cores_decodes_and_says_why() -> None:
    """B300 (sm103) has fp8 and fp4 tensor cores and no int8: its int8 leaves decode, by rule."""
    rows = [
        row("unet.a.weight", "int8.convrot", "bf16", (64, 256)),
        row("unet.b.weight", "int8.convrot", "f16", (64, 256)),
        row("unet.c.weight", "fp8.rowwise", "bf16", (64, 256)),
    ]
    served = plan(rows, B300)
    assert {t.key: t.route for t in served.tensors} == {
        "unet.a.weight": "decoded_float",
        "unet.b.weight": "decoded_float",
        "unet.c.weight": "encoded_gemm",
    }
    why = f"cuda.sm103 (Blackwell Ultra) has no int8 tensor cores for {INT8_LEAF}"
    assert [(g.leaf, g.reason, g.dtype, g.tensors) for g in served.refused_leaves()] == [
        (INT8_LEAF, why, "bfloat16", 1),
        (INT8_LEAF, why, "float16", 1),
    ]


def test_a_run_names_each_components_gemm_kernel_and_each_fallback(tmp_path: Path) -> None:
    """Real leaves, counted as the plane counts them when they install; on the CPU each provider
    serves its eager kernel, read once for the run. The refused fp8 leaves of the sm86 plan serve
    bf16, and an 8-bit op library that failed for good is the reason int8 leaves serve torch's GEMM.
    A leaf's call records nothing: the forward is 0.26.4's."""
    torch = pytest.importorskip("torch")
    encoded = encode_int8_convrot256("w", weight(64, 512, 5))
    int8_roles = {
        "weight": torch.frombuffer(bytearray(encoded.payload), dtype=torch.int8).reshape(64, 512),
        "scale": torch.frombuffer(bytearray(encoded.companions[0].raw), dtype=torch.float32),
    }
    stand_in = probe._StandIn(512, 64)
    int8 = Int8ConvRotNativeLeaf(params=params_of(PARAMS["int8.convrot"]))
    leaves = {
        key: (int8.leaf(torch, int8_roles, stand_in, torch.bfloat16), None, ())
        for key in ("transformer.a.weight", "transformer.b.weight", "text_encoder.c.weight")
    }
    nvfp4 = Nvfp4NativeLeaf(
        params=params_of(
            {
                "block_size": 16,
                "global_scale": "multiply",
                "nibble_order": "low_first",
                "scale_layout": "row_major",
            }
        )
    ).leaf(
        torch, roles(torch, weight(256, 512, 6), "cpu"), probe._StandIn(512, 256), torch.bfloat16
    )
    leaves["transformer.d.weight"] = (nvfp4, None, ())
    before = set(vars(nvfp4))
    nvfp4(torch.randn(40, 512, dtype=torch.bfloat16))
    assert set(vars(nvfp4)) == before  # nothing written per call

    rows = [row(f"unet.{n}.weight", "fp8.rowwise", "bf16", (64, 256)) for n in "ab"]
    executor = Executor(None, tmp_path)  # type: ignore[arg-type]
    executor.backend = SimpleNamespace(gemm_sites=leaf_sites(leaves))
    executor.models = {"klein": object()}
    executor.refused_leaves = plan(rows, RTX_3090).refused_leaves()

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
        ("reason", f"{FP8_LEAF} needs cuda.sm89+ and this device is cuda.sm86"),
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


def test_a_windows_wheel_names_the_gemm_that_served_and_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Windows cozy-runtime wheel has no CUTLASS TMA GEMMs (cr-201): an RTX 50's fp8 leaves
    multiply in `_scaled_mm`, and the run's GEMM column says so and why. Ada's fp8 GEMM is
    `_scaled_mm` on every wheel, so it says nothing more. The wheel's one fact is stated:
    `_gemm.TMA` false."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(prebuilt, "ABSENT", "")
    monkeypatch.setattr(_kernels, "_gemm", SimpleNamespace(TMA=False), raising=False)
    reason = prebuilt.lost("fp8", 120)
    assert "cr-201" in reason
    assert prebuilt.lost("fp8", 89) == prebuilt.lost("int8", 86) == ""

    monkeypatch.setitem(int8_linear._LIBS, 0, prebuilt.Prebuilt(torch, 0, "mma.sync", False))
    monkeypatch.setitem(fp8_linear._OPS, 0, int8_linear._LIBS[0].fp8_linear)
    executor = Executor(None, tmp_path)  # type: ignore[arg-type]
    executor.backend = SimpleNamespace(gemm_sites={("transformer", ROWWISE_FP8): 3})
    rows = []
    for sm, name in ((120, "NVIDIA GeForce RTX 5080"), (89, "NVIDIA L4")):
        executor.device_facts = DeviceFacts("cuda", name, sm, "580.95.05", "", 0)
        record = Attempt("gemm", tmp_path, sm)
        Executor._emit_gemm(record, executor)
        rows.append([
            (row["name"], row["value"], row["fields"].get("reason", ""))  # type: ignore[union-attr]
            for row in record.ring.rows()
            if row["kind"] == "confession"
        ])  # fmt: skip
    selected = ("gemm.selected", fp8_linear.SCALED_MM, "")
    assert rows == [[selected, ("gemm.fallback", fp8_linear.SCALED_MM, reason)], [selected]]
