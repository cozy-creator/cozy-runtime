"""The shared operation validates policy against genuine native source artifacts."""

from __future__ import annotations

import math
import sys
import typing
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import msgspec
import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.author import (
    Invocation,
    ModelArtifact,
    UnsupportedInput,
    Weights,
    attempt,
    describe,
)
from cozy_runtime.author._errors import Outcome
from cozy_runtime.author._invoke import InvocationResult
from cozy_runtime.author._services import Attempt
from cozy_runtime.derive.operations import MAX_QUANTIZED_BYTES, QuantizationPlan, _selection, app
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import Discovered
from native_weights import NativeExecution
from test_quantize_artifact_native import _host, _run, _source


def _shared(
    root: Path,
    store: tensorfs.Store,
    source: ModelArtifact,
    plan: QuantizationPlan,
    encoding: str = "fp8-rowwise/1",
    *,
    request: str | None = None,
    epoch: int = 1,
    canceled: list[bool] | None = None,
    checkpoint: Callable[[Any], None] | None = None,
) -> tuple[InvocationResult | None, Outcome, Attempt]:
    with NativeExecution(
        store,
        Path(store.root).parent / "shared",
        request or root.name,
        {"source": source},
        {"model": MAX_QUANTIZED_BYTES},
        epoch=epoch,
        after_checkpoint=checkpoint,
    ) as host:
        describe(app)
        return attempt(
            app.get("quantize"),
            {
                "source": msgspec.to_builtins(source),
                "plan": msgspec.to_builtins(plan),
                "encoding": encoding,
            },
            Invocation(
                host.attempt.request_id,
                root,
                math.inf,  # no stopwatch: a test attempt ends on its own outcome
                sources={"source": Weights(source.manifest.digest)},
                tensorfs_source=host.client.source,
                tensorfs_output=host.client.open_output,
                tensorfs_adopt=host.client.adopt_model,
                cancel=lambda: bool(canceled and canceled[0]),
            ),
        )


def test_shared_operation_resumes_complete_native_groups(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    source = _source(store, "control")
    plan = QuantizationPlan(("control",))
    canceled = [False]
    checkpoints = []

    def stop(value: Any) -> None:
        checkpoints.append(value)
        canceled[0] = True

    result, outcome, _ = _shared(
        tmp_path / "first",
        store,
        source,
        plan,
        request="resume",
        canceled=canceled,
        checkpoint=stop,
    )
    assert result is None and outcome.terminal == "canceled" and len(checkpoints) == 1
    result, outcome, record = _shared(
        tmp_path / "resumed", store, source, plan, request="resume", epoch=2
    )
    assert outcome.terminal == "succeeded" and result is not None
    facts = next(
        row["fields"] for row in record.ring.rows() if row.get("name") == "quantization measurement"
    )
    assert isinstance(facts, dict) and facts["reused_keys"] == 1
    assert facts["source_bytes_read_this_run"] == 1024
    clean, outcome, _ = _shared(tmp_path / "clean", store, source, plan)
    assert outcome.terminal == "succeeded" and clean is not None
    assert clean.result.manifest == result.result.manifest


def test_builtin_interface_is_static_and_matches_its_actual_app() -> None:
    static = static_interface.build_builtin("operations")
    module = sys.modules[app.get("quantize").fn.__module__]
    actual = package_interface.build(
        Discovered(
            app,
            "cozy_runtime.derive.operations:app",
            Path(__file__).parent,
            module,
            describe(app),
            {},
        )
    )
    source = package_interface.source_facts(actual)
    assert package_interface.canonical_bytes(static) == package_interface.canonical_bytes(source)
    # Not memoized, but a writer: its identity keys TensorFS derivation replay.
    (installed,) = typing.cast(list[dict[str, Any]], actual["entrypoints"])
    assert installed["invocable"]["operation_identity"].startswith("sha256:")
    (row,) = typing.cast(list[dict[str, Any]], static["entrypoints"])
    assert row["name"] == "quantize" and row["accelerator"] is False
    assert row["models"][0]["weights"] is True
    assert row["weights_outputs"] == [
        {
            "output_id": "model",
            "mime_type": "application/vnd.cozy.model-manifest",
            "max_bytes": MAX_QUANTIZED_BYTES,
        }
    ]
    assert row["invocable"]["memoize"] is False


@pytest.mark.parametrize("component,dtype", [("unet", "f16"), ("transformer", "bf16")])
def test_shared_native_operation_preserves_existing_family_results(
    tmp_path: Path, component: str, dtype: str
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    source = _source(store, component, dtype)
    original, outcome, _ = _run(
        tmp_path / "legacy", _host(store, "legacy", source), source, component, "fp8-rowwise/1"
    )
    assert outcome.terminal == "succeeded" and original is not None
    result, outcome, _ = _shared(tmp_path / "shared", store, source, QuantizationPlan((component,)))
    assert outcome.terminal == "succeeded", outcome
    assert result is not None and result.result.manifest == original.manifest


def test_native_exact_selection_binds_geometry_and_construction_order(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    source = _source(store, "control", "f32")
    with (
        _host(store, "inspect", source) as owner,
        owner.client.source(source.manifest.digest) as capability,
    ):
        structure = capability.inspect()
    selected = structure.components["control"]["a.weight"]
    schema = canonical_json.digest(
        [
            [
                "control",
                "a.weight",
                selected.logical_dtype,
                list(selected.shape),
                [[name, p.dtype, list(p.shape)] for name, p in selected.parts.items()],
            ]
        ]
    )
    order = canonical_json.digest(
        [[component, key] for component, rows in structure.components.items() for key in rows]
    )
    plan = QuantizationPlan(
        ("control",),
        ("a.weight",),
        selected_schema_digest=schema,
        source_order_digests=(order,),
        output_precision="preserve",
    )
    result, outcome, _ = _shared(tmp_path / "preserve", store, source, plan)
    assert outcome.terminal == "succeeded", outcome
    assert result is not None
    source_header = store.manifest(source.manifest.digest)["header"]
    result_header = store.manifest(result.result.manifest.digest)["header"]
    assert source_header is not None and result_header is not None
    before = tensorfs.parse_header(source_header)
    after = tensorfs.parse_header(result_header)
    assert list(after["components"]["control"]) == list(structure.components["control"])
    for key in ("z.weight", "bias"):
        assert after["components"]["control"][key] == before["components"]["control"][key]
    assert after["components"]["control"]["a.weight"]["logical"]["logical_dtype"] == "f32"
    wrong = msgspec.structs.replace(
        plan,
        source_order_digests=(
            canonical_json.digest(
                [
                    [component, key]
                    for component, rows in reversed(tuple(structure.components.items()))
                    for key in reversed(tuple(rows))
                ]
            ),
        ),
    )
    result, outcome, _ = _shared(tmp_path / "wrong-order", store, source, wrong)
    assert (
        result is None and outcome.terminal == "refused" and "construction order" in outcome.message
    )
    for changed in [
        msgspec.structs.replace(plan, keys=("absent.weight",)),
        msgspec.structs.replace(plan, selected_schema_digest="sha256:" + "12" * 32),
    ]:
        with pytest.raises(UnsupportedInput):
            _selection(structure, changed)
    bad = replace(selected, parts={"value": replace(selected.parts["value"], dtype="bf16")})
    with pytest.raises(UnsupportedInput):
        _selection(
            replace(
                structure,
                components={"control": {**structure.components["control"], "a.weight": bad}},
            ),
            plan,
        )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"components":["control"],"family":"h3"}',
        b'{"components":["control"],"output_precision":"guess"}',
        b'{"components":true}',
    ],
)
def test_closed_plan_refuses_undeclared_fields_and_types(raw: bytes) -> None:
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(raw, type=QuantizationPlan)
