"""The shared operation validates policy against genuine native source artifacts."""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import msgspec
import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.author import Invocation, ModelArtifact, UnsupportedInput, attempt, describe
from cozy_runtime.author._errors import Outcome
from cozy_runtime.author._invoke import InvocationResult
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author._services import Attempt
from cozy_runtime.derive.operations import MAX_QUANTIZED_BYTES, QuantizationPlan, _selection, app
from cozy_runtime.derive.quantization import QuantizationSource
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import Discovered
from cozy_runtime.internal.weights_sink import WeightsTransactionHost
from test_quantize_artifact_native import _host, _run, _source


def _shared(
    root: Path,
    store: tensorfs.Store,
    source: ModelArtifact,
    plan: QuantizationPlan,
    encoding: str = "fp8-rowwise/1",
) -> tuple[InvocationResult | None, Outcome, Attempt]:
    host = WeightsTransactionHost(
        store=store,
        owner_scope="shared-quantize",
        request_id=root.name,
        invocation_spec_digest="sha256:" + "42" * 32,
        work_fingerprint="sha256:" + "43" * 32,
        writer_session_id=1,
        allowed_sources={source.manifest.digest: source.manifest.length},
        output_bounds={"model": MAX_QUANTIZED_BYTES},
    )
    describe(app)
    return attempt(
        app.get("quantize"),
        {
            "source": msgspec.to_builtins(source),
            "plan": msgspec.to_builtins(plan),
            "encoding": encoding,
        },
        Invocation(
            root.name,
            root,
            time.monotonic() + 30,
            models={"source": _derive_model(QuantizationSource, source.manifest.digest)},
            weights=host.open,
            weights_source_structure=host.structure,
        ),
    )


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
    assert package_interface.canonical_bytes(static) == package_interface.canonical_bytes(actual)
    jobs = static["jobs"]
    assert isinstance(jobs, list) and len(jobs) == 1
    row = jobs[0]
    assert isinstance(row, dict) and row["name"] == "quantize"
    assert row["weights_outputs"] == [
        {
            "output_id": "model",
            "mime_type": "application/vnd.cozy.model-manifest",
            "max_bytes": MAX_QUANTIZED_BYTES,
        }
    ]
    assert row["invocable"]["memoize"] is True


@pytest.mark.parametrize("component,dtype", [("unet", "f16"), ("transformer", "bf16")])
@pytest.mark.parametrize("encoding", ["fp8-rowwise/1", "mxfp8/1"])
def test_shared_native_operation_preserves_existing_family_results(
    tmp_path: Path, component: str, dtype: str, encoding: str
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    source = _source(store, component, dtype)
    original, outcome, _ = _run(
        tmp_path / "legacy", _host(store, "legacy", source), source, component, encoding
    )
    assert outcome.terminal == "succeeded" and original is not None
    result, outcome, _ = _shared(
        tmp_path / "shared", store, source, QuantizationPlan((component,)), encoding
    )
    assert outcome.terminal == "succeeded", outcome
    assert result is not None and result.result.manifest == original.manifest


def test_native_exact_selection_binds_geometry_and_construction_order(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    source = _source(store, "control", "f32")
    structure = _host(store, "inspect", source).structure(source.manifest.digest)
    selected = next(t for t in structure.tensors if t.key == "a.weight")
    schema = canonical_json.digest(
        [
            [
                selected.component,
                selected.key,
                selected.logical_dtype,
                list(selected.shape),
                [[p.name, p.dtype, list(p.shape)] for p in selected.parts],
            ]
        ]
    )
    order = canonical_json.digest([[t.component, t.key] for t in structure.tensors])
    plan = QuantizationPlan(("control",), ("a.weight",), schema, (order,), "preserve")
    result, outcome, _ = _shared(tmp_path / "preserve", store, source, plan)
    assert outcome.terminal == "succeeded", outcome
    assert result is not None
    source_header = store.manifest(source.manifest.digest)["header"]
    result_header = store.manifest(result.result.manifest.digest)["header"]
    assert source_header is not None and result_header is not None
    before = tensorfs.parse_header(source_header)
    after = tensorfs.parse_header(result_header)
    assert list(after["components"]["control"]) == [t.key for t in structure.tensors]
    for key in ("z.weight", "bias"):
        assert after["components"]["control"][key] == before["components"]["control"][key]
    assert after["components"]["control"]["a.weight"]["logical"]["logical_dtype"] == "f32"
    wrong = msgspec.structs.replace(
        plan,
        source_order_digests=(
            canonical_json.digest([[t.component, t.key] for t in reversed(structure.tensors)]),
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
    bad = replace(selected, parts=(replace(selected.parts[0], dtype="bf16"),))
    with pytest.raises(UnsupportedInput):
        _selection(
            replace(
                structure, tensors=tuple(bad if t.key == bad.key else t for t in structure.tensors)
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
