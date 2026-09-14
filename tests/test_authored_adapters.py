"""Authored calls preserve ordered adapters, exact payloads and real native custody."""

from __future__ import annotations

import copy
import sys
import types
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import msgspec
import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.author import ConformanceError, LoRA, ModelArtifact, ObjectRef, with_adapters
from cozy_runtime.author._calls import _Broker, _CallType
from cozy_runtime.author._model_bindings import read_binding
from cozy_runtime.author.fakes import fake_context
from cozy_runtime.internal import interface_wheel, package_interface, static_interface
from cozy_runtime.internal.lora import AdapterRefusal
from cozy_runtime.internal.worker import machine_models
from cozy_runtime.internal.worker.machine_child_target import Target
from cozy_runtime.internal.worker.machine_serving import (
    _call_arguments,
    _matches_bindings,
    _retain_models,
)
from cozy_runtime.internal.worker.package_prepare import (
    NativeAdapterSelection,
    PreparationRefusal,
    _slot_adapters,
)
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_serving_interfaces import project
from test_workspace_custody import produced
from test_workspace_memo import completed, release_original


class Arguments(msgspec.Struct, forbid_unknown_fields=True):
    first: ModelArtifact
    second: ModelArtifact
    payload: Any


class Result(msgspec.Struct):
    value: int


def artifact(value: str) -> ModelArtifact:
    return ModelArtifact(
        "producer", "model", ObjectRef("sha256:" + value * 64, 512), "sha256:" + "a" * 64
    )


def reserve(
    first: object, second: object, payload: Any = None, *, model_paths: Any = None
) -> bytes:
    key = ("sha256:" + "b" * 64, "fixture", "generate")
    binding = _CallType(
        *key, Arguments, Result, (("first",), ("second",)) if model_paths is None else model_paths
    )
    broker = _Broker("parent", {key: binding}, lambda *_: {})
    broker.bind(fake_context(request_id="parent"))
    return broker.reserve(*key, {"first": first, "second": second, "payload": payload}).payload


def test_empty_stack_preserves_plain_payload_and_artifact_identity() -> None:
    base = artifact("1")
    before = msgspec.json.encode(base)
    assert with_adapters(base, []) is base
    assert reserve(base, base, {"adapters": "ordinary user value"}) == reserve(
        with_adapters(base, []),
        base,
        {"adapters": "ordinary user value"},
        model_paths=(),
    )
    with_adapters(base, [LoRA(artifact("2"), "dit", 0)])
    assert msgspec.json.encode(base) == before


def test_order_zero_negative_and_independent_model_slots_join_payload_identity() -> None:
    base, a, b = artifact("1"), artifact("2"), artifact("3")
    stack = [LoRA(a, "dit", 0), LoRA(b, "dit", -0.25)]
    raw = reserve(with_adapters(base, stack), with_adapters(base, [LoRA(a, "text", 1)]))
    body = canonical_json.decode(raw)
    assert read_binding(body["first"])[0] == base
    assert [row.scale for row in read_binding(body["first"])[1]] == ["0", "-0.25"]
    assert read_binding(body["second"])[1][0].component == "text"
    variants = [
        stack[::-1],
        [LoRA(a, "dit", 1), stack[1]],
        [LoRA(a, "other", 0), stack[1]],
        [LoRA(b, "dit", 0), stack[1]],
    ]
    for rows in variants:
        assert raw != reserve(with_adapters(base, rows), with_adapters(base, [LoRA(a, "text", 1)]))


def test_adapter_wrapper_cannot_enter_a_job_or_arbitrary_payload_field() -> None:
    base = artifact("1")
    selected = with_adapters(base, [LoRA(artifact("2"), "dit")])
    with pytest.raises(ConformanceError, match="adapter_slot"):
        reserve(selected, base, model_paths=())
    with pytest.raises(ConformanceError, match="adapter_slot"):
        reserve(base, base, {"user": [selected]})


@pytest.mark.parametrize("strength", [float("inf"), float("-inf"), float("nan"), True, "0.5"])
def test_nonfinite_or_non_numeric_strength_refuses(strength: Any) -> None:
    with pytest.raises(ValueError):
        LoRA(artifact("2"), "dit", strength)


def test_untrusted_wire_cannot_hide_unknown_fields_or_noncanonical_scales() -> None:
    body = canonical_json.decode(
        reserve(with_adapters(artifact("1"), [LoRA(artifact("2"), "dit")]), artifact("1"))
    )["first"]
    for change in ({"scale": "1.0"}, {"scale": "NaN"}, {"unrecognized": "value"}):
        wrong = copy.deepcopy(body)
        wrong["adapters"][0].update(change)
        with pytest.raises((ValueError, msgspec.ValidationError)):
            read_binding(wrong)


def test_generated_serving_proxy_accepts_adapters_only_on_model_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration = static_interface.build(project(tmp_path))
    raw = package_interface.canonical_bytes(declaration)
    source = interface_wheel.generate(raw)["serving_fixture/__init__.py"]
    module = types.ModuleType("serving_fixture")
    monkeypatch.setitem(sys.modules, "serving_fixture", module)
    namespace = vars(module)
    exec(compile(source, "generated-caller.py", "exec"), namespace)
    binding = namespace["__cozy_bindings__"]["generate"]
    assert binding.model_paths == (("models", "model"),)
    broker = _Broker(
        "parent",
        {(binding.interface_digest, binding.module, binding.export): binding},
        lambda *_: {},
    )
    broker.bind(fake_context(request_id="parent"))
    call = broker.reserve(
        binding.interface_digest,
        binding.module,
        binding.export,
        {
            "payload": {"prompt": "same"},
            "models": {"model": with_adapters(artifact("1"), [LoRA(artifact("2"), "dit", 0.3)])},
        },
    )
    target = Target(
        revision="",
        interface_digest="",
        prepared_revision={},
        binding=None,
        digest="",
        entrypoint="generate",
        declaration=next(row for row in declaration["entrypoints"] if row["name"] == "generate"),
    )
    payload, models = _call_arguments(
        target, pb.ChildCallRequest(request_canonical_bytes=call.payload)
    )
    assert payload["prompt"] == "same" and "models" not in payload
    assert read_binding(models["model"])[1][0].scale == "0.3"
    assert (
        canonical_json.decode(call.payload)["models"]["model"]["format"] == "cozy.model-binding/1"
    )


def test_native_adapter_custody_outlives_prepare_and_original_result(tmp_path: Path) -> None:
    store, workspace, base = produced(tmp_path, request_id="base")
    _, _, adapter = produced(tmp_path, request_id="adapter")
    bound = canonical_json.decode(
        reserve(with_adapters(base.artifact, [LoRA(adapter.artifact, "dit", 0)]), base.artifact)
    )["first"]
    with ExitStack() as access:
        native, adapters = _retain_models(
            workspace,
            "owner",
            "child",
            "generate",
            [{"path": "generate.models.model"}],
            {"model": bound},
            access,
        )
    assert adapters["generate.models.model"][0].manifest == adapter.artifact.manifest.digest
    assert documents.body(native[0].manifest) == msgspec.to_builtins(base.artifact.manifest)
    release_original(workspace, completed(workspace, base), base)
    release_original(workspace, completed(workspace, adapter), adapter)
    tensorfs.gc(str(store.root))
    assert store.manifest(base.artifact.manifest.digest)
    assert store.manifest(adapter.artifact.manifest.digest)
    with workspace.locked() as db:
        rows = db.execute(
            "SELECT path FROM execution_model_holds "
            "WHERE owner='owner' AND recipient='child' ORDER BY path"
        ).fetchall()
        assert [row["path"] for row in rows] == ["input/model", "input/model/adapters/0"]
    with ExitStack() as access, pytest.raises(WorkspaceRefusal, match="provenance"):
        _retain_models(
            workspace,
            "foreign",
            "foreign-child",
            "generate",
            [{"path": "generate.models.model"}],
            {"model": bound},
            access,
        )
    machine_models.release(workspace, "owner", "child")


def test_native_preparation_checks_real_adapter_geometry_and_exact_length(tmp_path: Path) -> None:
    from test_adapter_integration import _fixture

    root, base, ref, _, _, _ = _fixture(tmp_path)
    store = tensorfs.Store.ensure(root)
    length = len(store.manifest(ref)["manifest"])
    row = NativeAdapterSelection("dit", ref, length, "adapter", "-0.25")
    models, slots = _slot_adapters({"manifest": base}, ["dit"], root, 1, native=[row])
    assert models[0].repo == "native/" + ref[7:]
    assert slots[0].scale == "-0.25" and models[0].manifest.length == length
    from dataclasses import replace

    with pytest.raises(PreparationRefusal, match="length"):
        _slot_adapters(
            {"manifest": base}, ["dit"], root, 1, native=[replace(row, manifest_length=length + 1)]
        )
    with pytest.raises(AdapterRefusal, match="adapter_component"):
        _slot_adapters(
            {"manifest": base}, ["dit"], root, 1, native=[replace(row, component="other")]
        )


def test_prepared_cache_requires_full_stack_and_rejects_zero_vs_omitted() -> None:
    base, adapter = artifact("1"), artifact("2")
    native = [
        pb.NativeModelBinding(
            slot="generate.models.model",
            manifest=pb.Ref(
                digest=documents.raw(base.manifest.digest), length=base.manifest.length
            ),
        )
    ]
    selected = NativeAdapterSelection(
        "dit", adapter.manifest.digest, adapter.manifest.length, "adapter", "0"
    )
    candidate = {
        "models": [
            {"id": "base", "manifest": msgspec.to_builtins(base.manifest)},
            {"id": "adapter", "manifest": msgspec.to_builtins(adapter.manifest)},
        ],
        "entrypoints": [
            {
                "name": "generate",
                "slots": [
                    {
                        "slot": "model",
                        "reference_model_id": "base",
                        "components": [{"component": "dit", "model_id": "base"}],
                        "adapters": [
                            {
                                "component": "dit",
                                "model_id": "adapter",
                                "source_component": "adapter",
                                "scale": "0",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    assert _matches_bindings(candidate, "generate", native, {native[0].slot: (selected,)})
    assert not _matches_bindings(candidate, "generate", native, {})
    from dataclasses import replace

    for changed in (
        replace(selected, scale="1"),
        replace(selected, component="text"),
        replace(selected, source_component="other"),
        replace(selected, manifest=base.manifest.digest),
    ):
        assert not _matches_bindings(candidate, "generate", native, {native[0].slot: (changed,)})


def test_canonical_strength_ignores_authored_decimal_context() -> None:
    from decimal import localcontext

    from cozy_runtime.author._model_bindings import scale_string

    with localcontext() as context:
        context.prec = 3
        assert scale_string(1.234567891234567) == "1.234567891234567"
        assert scale_string(1000000.0) == "1e+06"
        assert scale_string(-0.0) == "0"


def test_adapter_wrapper_inside_typed_user_payload_is_not_an_override() -> None:
    class UserData(msgspec.Struct):
        arbitrary: Any

    base = artifact("1")
    wrapped = with_adapters(base, [LoRA(artifact("2"), "dit", 0)])
    with pytest.raises(ConformanceError, match="adapter_slot"):
        reserve(base, base, UserData(wrapped))
