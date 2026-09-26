"""Serving and job memo declarations bind only their named helpers/data/native metadata."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from cozy_runtime.author import Invocation, attempt, describe
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import Discovered
from cozy_runtime.internal.worker.machine_child_target import Target, computation


def load(name: str, path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def operation(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    serving: bool,
    delta: int = 1,
    caller: str = "caller",
) -> tuple[ModuleType, dict[str, str], bytes]:
    root.mkdir()
    source = """import msgspec
from cozy_runtime.author import App, Context, MemoDistribution, MemoResource, invocable
class Request(msgspec.Struct):
    value: int
class Result(msgspec.Struct):
    value: int
def helper(value):
    return value + DELTA
DEPENDENCIES = (
    helper, MemoResource("memo_data", "table.json"),
    MemoDistribution("memo-native-fixture"),
)
BODY
app = App()
REGISTRATION

def unrelated_caller():
    return CALLER
""".replace("DELTA", str(delta)).replace("CALLER", repr(caller))
    if serving:
        source = source.replace(
            "BODY",
            "def measure(ctx: Context, payload: Request) -> Result:\n"
            "    return Result(helper(payload.value))",
        )
        source = source.replace(
            "REGISTRATION", "app.entrypoint(measure, memoize=True, memo_dependencies=DEPENDENCIES)"
        )
    else:
        source = source.replace(
            "BODY",
            "@invocable(memoize=True, memo_dependencies=DEPENDENCIES)\n"
            "async def measure(ctx: Context, *, value: int) -> Result:\n"
            "    return Result(helper(value))",
        )
        source = source.replace("REGISTRATION", "app.job(measure)")
    path = root / "operation.py"
    path.write_text(source)
    module = load("memo_operation", path, monkeypatch)
    found = Discovered(module.app, "memo_operation:app", root, module, describe(module.app), {})
    document = package_interface.read_bytes(
        package_interface.canonical_bytes(package_interface.build(found))
    )
    declaration = document["entrypoints" if serving else "jobs"][0]
    metadata = declaration["invocable"]
    target = Target(
        root.name, "measure", declaration, {}, None, metadata.get("operation_identity", "")
    )
    return module, metadata, computation(target, {"value": 4}, b"n" * 32)


@pytest.mark.parametrize("serving", [False, True])
def test_helper_resource_and_native_build_changes_invalidate_without_caller_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    serving: bool,
) -> None:
    package = tmp_path / "memo_data"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "table.json").write_text('{"scale": 1}')
    monkeypatch.syspath_prepend(str(tmp_path))
    load("memo_data", package / "__init__.py", monkeypatch)
    dist = tmp_path / "memo_native_fixture-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: memo-native-fixture\nVersion: 1.0\n")
    (dist / "WHEEL").write_text("Wheel-Version: 1.0\nTag: cp312-abi3-linux_x86_64\n")

    def provenance(url: str, digest: str) -> None:
        (dist / "direct_url.json").write_text(
            json.dumps({"url": url, "archive_info": {"hashes": {"sha256": digest * 64}}})
        )

    provenance("file:///first/native.whl", "a")
    module, _, first = operation(tmp_path / "first", monkeypatch, serving=serving)
    provenance("file:///relocated/native.whl", "a")
    _, _, edited_caller = operation(
        tmp_path / "caller", monkeypatch, serving=serving, caller="edited caller"
    )
    _, _, edited_helper = operation(tmp_path / "helper", monkeypatch, serving=serving, delta=2)
    assert first and first == edited_caller and first != edited_helper
    (package / "table.json").write_text('{"scale": 2}')
    _, _, resource = operation(tmp_path / "resource", monkeypatch, serving=serving)
    assert resource and resource != first
    provenance("file:///relocated/native.whl", "b")
    _, _, build = operation(tmp_path / "native", monkeypatch, serving=serving)
    assert build and build != resource
    (dist / "direct_url.json").write_text(
        json.dumps({"url": "file:///editable", "dir_info": {"editable": True}})
    )
    editable, metadata, key = operation(tmp_path / "editable", monkeypatch, serving=serving)
    assert (
        not key
        and metadata["operation_identity_unavailable"] == "operation_native_identity_unavailable"
    )
    for index, selected in enumerate((module, editable)):
        result, outcome, _ = attempt(
            selected.app.get("measure"),
            {"value": 4},
            Invocation(
                "memo-" + str(index), tmp_path / ("output" + str(index)), time.monotonic() + 10
            ),
        )
        assert outcome.terminal == "succeeded", outcome
        assert result is not None and result.result.value == 5
