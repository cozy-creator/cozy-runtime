from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import (
    interface_wheel,
    package_interface,
    static_interface,
)
from cozy_runtime.internal.discovery import discover

SOURCE = """from enum import Enum
from typing import Any
import msgspec
from cozy_runtime.author import App, Context, Model

class Mode(Enum):
    ONE = "one"
    TWO = "two"

class Request(msgspec.Struct, kw_only=True):
    prompt: str
    mode: Mode = Mode.ONE
    tags: dict[str, int] = msgspec.field(default_factory=dict)
    count: int = msgspec.field(name="wire_count", default=3)

class Result(msgspec.Struct):
    value: int

class FixtureModel(Model[object]):
    def load(self, loader: Any) -> None:
        raise RuntimeError("model construction belongs in the worker")

app = App()

@app.entrypoint
def generate(ctx: Context, payload: Request, model: FixtureModel) -> Result:
    raise RuntimeError("the serving implementation belongs in the worker")

@app.entrypoint
async def echo(payload: Request) -> Result:
    return Result(payload.count)
"""


def project(tmp_path: Path) -> Path:
    # The imported oracle uses one module cache; each fixture is a different source tree.
    sys.modules.pop("serving_fixture", None)
    root = tmp_path / "source"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="serving-fixture"\nversion="0.1.0"\n')
    (root / "package.toml").write_text('[application]\nobject="serving_fixture:app"\n')
    (root / "serving_fixture.py").write_text(SOURCE)
    return root


def test_ordinary_serving_request_model_name_collision_still_refuses(tmp_path: Path) -> None:
    root = project(tmp_path)
    (root / "serving_fixture.py").write_text(
        SOURCE.replace("    prompt: str", '    prompt: str\n    model: str = "user payload"')
    )
    raw = package_interface.canonical_bytes(static_interface.build(root))
    with pytest.raises(ConformanceError, match="collide"):
        interface_wheel.generate(raw)


def test_serving_metadata_keeps_original_handler_and_cli_schema(tmp_path: Path) -> None:
    root = project(tmp_path)
    source = static_interface.build(root)
    found = discover(root)
    assert source == package_interface.build(found)
    generate = found.app.get("generate")
    assert not generate.surface.is_async and not generate.invocable
    assert [p.name for p in generate.surface.params] == ["ctx", "payload", "model"]
    public = next(row for row in source["entrypoints"] if row["name"] == "generate")
    assert [row["name"] for row in public["request"]["fields"]] == [
        "prompt",
        "mode",
        "tags",
        "wire_count",
    ]
    assert public["invocable"]["memoize"] is False
    assert public["invocable"]["defaults"] == {
        "request/mode": "one",
        "request/tags": {},
        "request/wire_count": 3,
    }
    public["invocable"]["memoize"] = True
    public["invocable"]["capabilities"] = ["egress"]
    with pytest.raises(ConformanceError, match="external or secret"):
        package_interface.read_bytes(package_interface.canonical_bytes(source))


def test_serving_defaults_from_committed_choices_match_imported_interface(tmp_path: Path) -> None:
    root = project(tmp_path)
    source = SOURCE.replace("from typing import Any", "from typing import Any, Literal")
    source = source.replace(
        "from cozy_runtime.author import App, Context, Model",
        "from cozy_runtime.author import App, Context, Model, data_values\n"
        'STEPS = data_values(__file__, "steps.json", "plans", "steps")\n'
        "DEFAULT_STEPS = min(STEPS)",
    )
    source = source.replace(
        "    prompt: str",
        "    prompt: str\n"
        "    steps: Literal[STEPS] = DEFAULT_STEPS\n"
        "    highest: Literal[STEPS] = msgspec.field(default=max(STEPS))\n"
        "    values: list[int] = msgspec.field(default_factory=list)",
    )
    (root / "serving_fixture.py").write_text(source)
    (root / "steps.json").write_text('{"plans":[{"steps":30},{"steps":40},{"steps":50}]}')
    captured = static_interface.build(root)
    imported = package_interface.build(discover(root))
    assert package_interface.canonical_bytes(captured) == package_interface.canonical_bytes(
        imported
    )
    for entrypoint in captured["entrypoints"]:
        defaults = entrypoint["invocable"]["defaults"]
        assert defaults["request/steps"] == 30
        assert defaults["request/highest"] == 50
        assert defaults["request/values"] == []
        assert defaults["request/tags"] == {}


@pytest.mark.parametrize(
    "expression",
    [
        "custom_default()",
        "msgspec.field(default=custom_default())",
        "min((30, 40), key=custom_default)",
        "min(())",
        'min((30, "forty"))',
    ],
)
def test_unresolved_scalar_default_refuses_before_capture(tmp_path: Path, expression: str) -> None:
    root = project(tmp_path)
    source = SOURCE.replace(
        "class Request(msgspec.Struct, kw_only=True):",
        'raise AssertionError("static capture must not import this module")\n\n'
        "def custom_default():\n"
        '    raise AssertionError("static capture must not call user code")\n\n'
        "class Request(msgspec.Struct, kw_only=True):",
    ).replace('msgspec.field(name="wire_count", default=3)', expression)
    (root / "serving_fixture.py").write_text(source)
    with pytest.raises(static_interface.StaticRefusal) as caught:
        static_interface.build(root)
    assert caught.value.code == "static_computed"


def test_shadowed_min_is_not_executed_or_treated_as_a_builtin(tmp_path: Path) -> None:
    root = project(tmp_path)
    source = SOURCE.replace(
        "class Request(msgspec.Struct, kw_only=True):",
        "def min(values):\n"
        '    raise AssertionError("user function must never execute during capture")\n\n'
        "class Request(msgspec.Struct, kw_only=True):",
    ).replace('msgspec.field(name="wire_count", default=3)', "min((30, 40))")
    (root / "serving_fixture.py").write_text(source)
    with pytest.raises(static_interface.StaticRefusal) as caught:
        static_interface.build(root)
    assert caught.value.code == "static_computed"


def test_foldable_reducer_is_not_an_interface_type(tmp_path: Path) -> None:
    root = project(tmp_path)
    (root / "serving_fixture.py").write_text(SOURCE.replace("prompt: str", "prompt: min"))
    with pytest.raises(static_interface.StaticRefusal, match="not an interface type"):
        static_interface.build(root)


def test_deterministic_serving_measurement_explicitly_opts_into_memoization(tmp_path: Path) -> None:
    root = project(tmp_path)
    (root / "serving_fixture.py").write_text(
        SOURCE.replace(
            "@app.entrypoint\ndef generate", "@app.entrypoint(memoize=True)\ndef generate"
        )
    )
    doc = static_interface.build(root)
    imported = package_interface.build(discover(root))
    # Only imported code can name the memo implementation; it is not a source fence.
    generate = next(row for row in imported["entrypoints"] if row["name"] == "generate")
    identity = generate["invocable"].pop("operation_identity")
    assert identity.startswith("sha256:") and doc == imported
    rows = {row["name"]: row for row in doc["entrypoints"]}
    assert rows["generate"]["invocable"]["memoize"] is True
    assert rows["echo"]["invocable"]["memoize"] is False


def test_call_option_collision_does_not_change_cli_interface(tmp_path: Path) -> None:
    root = project(tmp_path)
    path = root / "serving_fixture.py"
    path.write_text(SOURCE.replace("    prompt: str", "    prompt: str\n    model: str"))
    raw = package_interface.canonical_bytes(static_interface.build(root))
    package_interface.read_bytes(raw)
    with pytest.raises(ConformanceError, match="collide"):
        interface_wheel.generate(raw)
