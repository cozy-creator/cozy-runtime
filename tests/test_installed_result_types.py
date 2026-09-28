"""Installed imports and generated managed callers agree on their result classes."""

from __future__ import annotations

import hashlib
import importlib
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import msgspec
import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author import (
    FileAsset,
    Invocation,
    ModelArtifact,
    attempt,
    script_app,
)
from cozy_runtime.author._call_results import decode
from cozy_runtime.author._calls import _Broker, _CallType
from cozy_runtime.internal import call_types, static_interface
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import CallInterface
from durable_seam import wire

SOURCE = """from __future__ import annotations
import msgspec
from cozy_runtime.author import App, Context, ModelArtifact, invocable

class Selection(msgspec.Struct, frozen=True):
    projection: ModelArtifact | None
    ready: bool

@invocable()
async def select(ctx: Context) -> Selection:
    raise AssertionError("the implementation executes in its child attempt")

app = App()
app.job(select)
"""


def installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ModuleType, tuple[CallInterface, ...], Executor]:
    module = "installed_selection"
    monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.delitem(sys.modules, module + "._cozy_runtime_callers", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "pyproject.toml").write_text('[project]\nname="selection"\nversion="1.0.0"\n')
    (tmp_path / "package.toml").write_text('[application]\nobject="installed_selection:app"\n')
    (tmp_path / f"{module}.py").write_text(SOURCE)
    importlib.invalidate_caches()
    document = static_interface.build(tmp_path)
    imported = importlib.import_module(module)
    rows = (CallInterface(module=module, export="select", interface_document=document),)
    engine = Executor(cast(Any, None), tmp_path)
    engine._install_call_proxies(rows)
    return imported, rows, engine


def test_plain_script_returns_installed_public_struct_after_managed_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported, rows, engine = installed(tmp_path, monkeypatch)
    binding = imported.__cozy_bindings__["select"]
    assert binding.result is not imported.Selection
    assert binding.python_result is imported.Selection
    # A second installation keeps the actual public type even though the export
    # is already a generated proxy with an Awaitable annotation.
    engine._install_call_proxies(rows)
    binding = imported.__cozy_bindings__["select"]
    assert binding.python_result is imported.Selection
    (tmp_path / "selection_client.py").write_text(
        "from installed_selection import Selection, select\n"
        "async def main() -> Selection:\n"
        "    return await select()\n"
    )
    monkeypatch.delitem(sys.modules, "selection_client", raising=False)
    artifact: dict[str, Any] = {
        "producer_request_id": "child",
        "output_slot": "model",
        "manifest": {"digest": "sha256:" + "1" * 64, "length": 163},
        "tensorfs_receipt_digest": "sha256:" + "2" * 64,
    }

    def exchange(kind: str, body: dict[str, Any]) -> dict[str, Any]:
        assert kind in {"child_call", "child_poll", "child_forget"}
        return {
            "ok": True,
            "state": "succeeded",
            "child_request_id": "child",
            "result": canonical_json.encode({"projection": artifact, "ready": False}).decode(),
        }

    broker = _Broker("parent", {(binding.module, binding.export): binding}, wire(exchange))
    app = script_app("selection_client")
    result, outcome, _ = attempt(
        app.get("main"),
        {},
        Invocation("parent", tmp_path / "result", time.monotonic() + 30, calls=broker),
    )
    assert outcome.terminal == "succeeded", outcome
    assert result is not None and type(result.result.value) is imported.Selection
    assert isinstance(result.result.value.projection, ModelArtifact)
    assert result.result.value.projection.manifest.digest == artifact["manifest"]["digest"]


@pytest.mark.parametrize(
    "value",
    [
        {"projection": None, "ready": "false"},
        {"projection": None},
    ],
)
def test_public_result_conversion_retains_generated_wire_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: dict[str, Any]
) -> None:
    imported, _, _ = installed(tmp_path, monkeypatch)
    binding = imported.__cozy_bindings__["select"]
    # Generated wire validation still rejects wrong types and missing fields.
    with pytest.raises(msgspec.ValidationError):
        decode(
            value,
            binding.result,
            [],
            request_id="parent",
            guard=lambda: None,
            observation=None,
            python_type=binding.python_result,
        )


def test_a_result_field_added_by_a_newer_child_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported, _, _ = installed(tmp_path, monkeypatch)
    binding = imported.__cozy_bindings__["select"]
    assert not binding.result.__struct_config__.forbid_unknown_fields
    value, _ = decode(
        {"projection": None, "ready": True, "added_by_newer_child": 1},
        binding.result,
        [],
        request_id="parent",
        guard=lambda: None,
        observation=None,
        python_type=binding.python_result,
    )
    assert type(value) is imported.Selection and value.ready is True


@pytest.mark.parametrize(
    "fields,options",
    [
        ([("projection", str | None), ("ready", bool)], {}),
        ([("projection", ModelArtifact | None), ("ready", bool), ("extra", int)], {}),
        ([("projection", ModelArtifact | None), ("ready", bool)], {"tag": "different"}),
        ([("projection", ModelArtifact | None), ("ready", bool)], {"array_like": True}),
    ],
)
def test_a_different_public_result_schema_keeps_the_generated_wire_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fields: list[Any], options: dict[str, Any]
) -> None:
    imported, _, _ = installed(tmp_path, monkeypatch)
    binding = imported.__cozy_bindings__["select"]
    cast(Any, imported).__cozy_bindings__ = {}
    cast(Any, imported).Selection = msgspec.defstruct("Selection", fields, **options)
    kept = call_types.public_result(imported, replace(binding, python_result=None))
    assert kept.python_result is None and kept.result is binding.result


def test_public_result_conversion_preserves_tag_validation_and_native_grants(
    tmp_path: Path,
) -> None:
    public = msgspec.defstruct("Report", [("file", FileAsset)], tag="report")
    wire = msgspec.defstruct(
        "Report", [("file", FileAsset)], tag="report", forbid_unknown_fields=True
    )
    module = ModuleType("native_report")
    cast(Any, module).Report = public
    binding = call_types.public_result(
        module, _CallType("", module.__name__, "produce", wire, wire)
    )
    data = b"retained native child bytes"
    path = tmp_path / "report.txt"
    path.write_bytes(data)
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    grant = {
        "output_id": "file",
        "kind": "file",
        "digest": digest,
        "length": len(data),
        "media_type": "text/plain",
        "local": str(path),
    }
    value = {
        "type": "report",
        "file": {
            "asset_ref": digest,
            "kind": "file",
            "digest": digest,
            "size_bytes": len(data),
            "media_type": "text/plain",
        },
    }
    token = object()
    decoded, _ = decode(
        value,
        wire,
        [grant],
        request_id="parent",
        guard=lambda: None,
        observation=None,
        grant_token=token,
        python_type=binding.python_result,
    )
    result = cast(Any, decoded)
    assert type(decoded) is public and result.file._grant_token is token
    assert result.file._local == path
    for wrong in ({**value, "type": "different"}, {**value, "unexpected": 1}):
        with pytest.raises(msgspec.ValidationError):
            decode(
                wrong,
                wire,
                [grant],
                request_id="parent",
                guard=lambda: None,
                observation=None,
                grant_token=token,
                python_type=binding.python_result,
            )
