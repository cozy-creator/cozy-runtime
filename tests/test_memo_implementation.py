"""Real authored declarations bind callee logic without hashing their caller/install."""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.author import Invocation, attempt, describe
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import Discovered
from cozy_runtime.internal.worker.machine_child_target import Target, computation

SOURCE = """import msgspec
from cozy_runtime.author import App, Context, invocable
{imports}
class Result(msgspec.Struct):
    value: int

def helper(value: int) -> int:
    return value + {helper_delta}

@invocable(memoize=True{options})
async def calculate(ctx: Context, *, value: int) -> Result:
    ctx.raise_if_cancelled()
    return Result({expression})

app = App()
app.job(calculate)

def unrelated_caller() -> str:
    return {caller!r}
"""


def load_module(monkeypatch: pytest.MonkeyPatch, name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def authored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    *,
    options: str = "",
    expression: str = "value + 1",
    caller: str = "original caller",
    helper_delta: int = 1,
    helper_module: bool = False,
    opaque: bool = False,
) -> tuple[types.ModuleType, dict[str, Any], bytes]:
    root = tmp_path / label
    root.mkdir()
    imports = ""
    if helper_module:
        helper = root / "helper.py"
        helper.write_text(f"def adjust(value):\n    return value + {helper_delta}\n")
        load_module(monkeypatch, "memo_fixture_helper", helper)
        imports = "import memo_fixture_helper"
    source = SOURCE.format(
        imports=imports,
        options=options,
        expression=expression,
        caller=caller,
        helper_delta=helper_delta,
    )
    path = root / "operation.py"
    if opaque:
        module = types.ModuleType("memo_fixture_operation")
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(compile(source, "<opaque-operation>", "exec"), module.__dict__)
    else:
        path.write_text(source)
        module = load_module(monkeypatch, "memo_fixture_operation", path)
    found = Discovered(module.app, module.__name__ + ":app", root, module, describe(module.app), {})
    raw = package_interface.canonical_bytes(package_interface.build(found))
    declaration = package_interface.read_bytes(raw)["jobs"][0]
    identity = declaration["invocable"].get("operation_identity", "")
    target = Target("independent-install-" + label, "calculate", declaration, {}, None, identity)
    key = computation(target, {"value": 4}, b"n" * 32)
    return module, declaration["invocable"], key


def test_default_memo_identity_ignores_caller_but_tracks_callee(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, metadata, first = authored(tmp_path, monkeypatch, "original")
    _, _, edited_caller = authored(
        tmp_path, monkeypatch, "caller", caller="new logging/control flow"
    )
    changed, _, edited_callee = authored(tmp_path, monkeypatch, "callee", expression="value + 2")
    assert metadata["memoize"] and metadata.get("operation_identity") and first
    assert first == edited_caller
    assert first != edited_callee
    for index, (module, expected) in enumerate(((original, 5), (changed, 6))):
        result, outcome, _ = attempt(
            module.app.get("calculate"),
            {"value": 4},
            Invocation("memo-" + str(index), tmp_path / str(index), time.monotonic() + 10),
        )
        assert outcome.terminal == "succeeded", outcome
        assert result is not None and result.result.value == expected


@pytest.mark.parametrize("module_dependency", [False, True])
def test_declared_helper_changes_invalidate_without_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_dependency: bool,
) -> None:
    options = (
        ', memo_dependencies=("memo_fixture_helper",)'
        if module_dependency
        else ", memo_dependencies=(helper,)"
    )
    expression = "memo_fixture_helper.adjust(value)" if module_dependency else "helper(value)"
    _, _, first = authored(
        tmp_path,
        monkeypatch,
        "original",
        options=options,
        expression=expression,
        helper_module=module_dependency,
    )
    _, _, caller = authored(
        tmp_path,
        monkeypatch,
        "caller",
        options=options,
        expression=expression,
        helper_module=module_dependency,
        caller="edited",
    )
    _, _, helper = authored(
        tmp_path,
        monkeypatch,
        "helper",
        options=options,
        expression=expression,
        helper_module=module_dependency,
        helper_delta=2,
    )
    assert first and first == caller and first != helper


def test_explicit_version_remains_an_optional_invalidation_salt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, automatic = authored(tmp_path, monkeypatch, "automatic")
    _, _, empty = authored(tmp_path, monkeypatch, "empty", options=', memo_version=""')
    assert automatic == empty
    _, _, first = authored(
        tmp_path, monkeypatch, "first", options=', memo_version="native-format/1"'
    )
    _, _, second = authored(
        tmp_path, monkeypatch, "second", options=', memo_version="native-format/2"'
    )
    assert automatic and first and second and len({automatic, first, second}) == 3


def test_unavailable_source_disables_reuse_without_refusing_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, metadata, key = authored(tmp_path, monkeypatch, "opaque", opaque=True)
    assert metadata.get("operation_identity_unavailable") == "operation_source_unavailable"
    assert not key
    result, outcome, _ = attempt(
        module.app.get("calculate"),
        {"value": 4},
        Invocation("opaque", tmp_path / "output", time.monotonic() + 10),
    )
    assert outcome.terminal == "succeeded", outcome
    assert result is not None and result.result.value == 5
