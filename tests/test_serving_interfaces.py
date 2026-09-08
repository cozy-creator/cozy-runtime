from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import interface_wheel, package_interface, static_interface
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


def test_deterministic_serving_measurement_explicitly_opts_into_memoization(tmp_path: Path) -> None:
    root = project(tmp_path)
    (root / "serving_fixture.py").write_text(
        SOURCE.replace(
            "@app.entrypoint\ndef generate", "@app.entrypoint(memoize=True)\ndef generate"
        )
    )
    doc = static_interface.build(root)
    assert doc == package_interface.build(discover(root))
    rows = {row["name"]: row for row in doc["entrypoints"]}
    assert rows["generate"]["invocable"]["memoize"] is True
    assert rows["echo"]["invocable"]["memoize"] is False


def test_generated_serving_caller_preserves_types_and_separates_call_options(
    tmp_path: Path,
) -> None:
    root = project(tmp_path)
    raw = package_interface.canonical_bytes(static_interface.build(root))
    output = tmp_path / "caller"
    for name, data in interface_wheel.generate(raw).items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    check = """import asyncio, inspect, json, sys
sys.path.insert(0, sys.argv[1])
import serving_fixture as client
from cozy_runtime.author import ActivationCapture, ModelArtifact, ObjectRef, PendingCall
from cozy_runtime.author._calls import _Broker, _current
from cozy_runtime.author.fakes import fake_context
assert "FixtureModel" not in vars(client)
assert list(inspect.signature(client.generate).parameters) == [
    "prompt", "mode", "tags", "wire_count", "model", "capture"]
binding = client.__cozy_bindings__["generate"]
sent = []
def exchange(kind, frame):
    index = frame["call_index"]
    if kind == "child_call":
        sent.append(frame)
    return {"ok": True, "state": "succeeded", "result": '{"value":7}',
            "child_request_id": "child-"+str(index)}
key = (binding.interface_digest, binding.module, binding.export)
broker = _Broker("parent", {key:binding}, exchange)
broker.bind(fake_context(request_id="parent"))
model = ModelArtifact("producer", "model", ObjectRef("sha256:"+"11"*32, 321), "sha256:"+"22"*32)
token = _current.set(broker)
async def run():
    capture = ActivationCapture(components=("unet",), steps=(0,))
    first = client.generate(prompt="same", model=model, capture=capture)
    assert isinstance(first, PendingCall)
    assert (await first).value == 7
    repeat = client.generate(prompt="same", model=model)
    assert (await repeat).value == 7
    assert first.request_id != repeat.request_id
    for _ in range(62):
        assert (await client.generate(prompt="same", model=model)).value == 7
        assert not broker.calls and not broker.activated
try:
    asyncio.run(run())
    broker.finish()
finally:
    _current.reset(token)
assert len(sent) == 64
assert [frame["call_index"] for frame in sent] == list(range(64))
body = json.loads(sent[0]["payload"])
assert set(body) == {"payload", "models"}
assert body["payload"] == {"prompt":"same", "mode":"one", "tags":{}, "wire_count":3}
assert body["models"]["model"]["producer_request_id"] == "producer"
assert "capture" not in body
assert sent[0]["capture"]["components"] == ["unet"]
assert "capture" not in sent[1]
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", check, str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    caller = output / "uses.py"
    caller.write_text(
        "from serving_fixture import generate, Result\n"
        "from cozy_runtime.author import ModelArtifact, PendingCall\n"
        "def run(model: ModelArtifact) -> PendingCall[Result]:\n"
        "    return generate(prompt='hello', model=model)\n"
    )
    typed = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", str(caller)],
        cwd=output,
        capture_output=True,
        text=True,
        check=False,
    )
    assert typed.returncode == 0, typed.stdout + typed.stderr


def test_call_option_collision_does_not_change_cli_interface(tmp_path: Path) -> None:
    root = project(tmp_path)
    path = root / "serving_fixture.py"
    path.write_text(SOURCE.replace("    prompt: str", "    prompt: str\n    model: str"))
    raw = package_interface.canonical_bytes(static_interface.build(root))
    package_interface.read_bytes(raw)
    with pytest.raises(ConformanceError, match="collide"):
        interface_wheel.generate(raw)
