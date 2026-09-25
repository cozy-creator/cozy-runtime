from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import interface_wheel, package_interface, static_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.worker.machine_child_target import Target
from cozy_runtime.internal.worker.machine_serving import arguments
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal

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


def test_modeled_invocable_serving_has_one_model_plane_for_self_and_generated_calls(
    tmp_path: Path,
) -> None:
    root = project(tmp_path)
    source = SOURCE.replace("App, Context, Model", "App, Context, Model, invocable")
    source = source.replace(
        "class Request(msgspec.Struct, kw_only=True):",
        "class Request(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):",
    )
    source = source.replace("    tags: dict[str, int] = msgspec.field(default_factory=dict)\n", "")
    source = source.replace(
        'count: int = msgspec.field(name="wire_count", default=3)', "count: int = 3"
    )
    source = source.replace(
        "@app.entrypoint\ndef generate(ctx: Context, payload: Request, model: FixtureModel)",
        '@invocable(memoize=False, defaults={"model":'
        '[{"gpu":"*","lane":"proof/adapter@1.0.0/base"}]})\n'
        "async def generate(ctx: Context, *, payload: Request, model: FixtureModel)",
    ).replace(
        "\n@app.entrypoint\nasync def echo",
        "\napp.entrypoint(generate)\n\n@app.entrypoint\nasync def echo",
    )
    (root / "serving_fixture.py").write_text(source)
    static = static_interface.build(root)
    imported = package_interface.build(discover(root))
    raw = package_interface.canonical_bytes(static)
    assert raw == package_interface.canonical_bytes(imported)
    entry = next(row for row in static["entrypoints"] if row["name"] == "generate")
    assert [field["name"] for field in entry["request"]["fields"]] == ["payload"]
    assert entry["invocable"]["parameters"] == ["payload"]
    output = tmp_path / "caller"
    for name, data in interface_wheel.generate(raw).items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    check = """import asyncio,inspect,json,sys
sys.path.insert(0,sys.argv[1])
import serving_fixture as client
from cozy_runtime.author import PendingCall,describe,ConformanceError
from cozy_runtime.author._calls import _Broker,_CallType,_current
from cozy_runtime.author.fakes import fake_context
if sys.argv[2]=="self":
    describe(client.app)
    surface=client.app.get("generate").surface
    binding=_CallType("sha256:"+"11"*32,"serving_fixture","generate",surface.payload_type,surface.result_type)
else:
    binding=client.__cozy_bindings__["generate"]
    assert list(inspect.signature(client.generate).parameters)==["payload","model","capture"]
sent=[]
def exchange(kind,frame):
    if kind=="child_call":sent.append(frame)
    return {"ok":True,"state":"succeeded","result":'{"value":7}',"child_request_id":"child"}
broker=_Broker("root",{(binding.interface_digest,binding.module,binding.export):binding,
                      ("",binding.module,binding.export):binding},exchange)
broker.bind(fake_context(request_id="root"))
token=_current.set(broker)
async def run():
    call=client.generate(payload=client.Request(prompt="typed"))
    assert isinstance(call,PendingCall) and (await call).value==7
    for invalid in ({"payload":{"prompt":5}},
                    {"payload":{"prompt":"typed","unknown":1}},
                    {"payload":client.Request(prompt="typed"),"model":"not an artifact"}):
        try:await client.generate(**invalid)
        except (TypeError,ValueError,ConformanceError):pass
        else:raise AssertionError("invalid request reached inference: "+repr(invalid))
try:
    asyncio.run(run());broker.finish()
finally:_current.reset(token)
assert len(sent)==1
print(sent[0]["payload"])
"""
    target = Target("revision", "generate", "interface", entry, {}, None, "digest")
    expected = None
    from dataclasses import replace

    for mode, module_path in (("self", root), ("imported", output)):
        result = subprocess.run(
            [sys.executable, "-I", "-c", check, str(module_path), mode],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        payload, models = arguments(
            replace(target, serving_envelope=mode == "imported"), result.stdout.strip().encode()
        )
        assert models.get("model") is None
        if expected is None:
            expected = payload
        assert payload == expected
        assert "model" not in canonical_json.decode(payload)


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
    assert (await client.generate(prompt="select default")).value == 7
    for _ in range(62):
        assert (await client.generate(prompt="same", model=model)).value == 7
        assert not broker.calls and not broker.activated
try:
    asyncio.run(run())
    broker.finish()
finally:
    _current.reset(token)
assert len(sent) == 65
assert [frame["call_index"] for frame in sent] == list(range(65))
body = json.loads(sent[0]["payload"])
assert set(body) == {"payload", "models"}
assert body["payload"] == {"prompt":"same", "mode":"one", "tags":{}, "wire_count":3}
assert body["models"]["model"]["producer_request_id"] == "producer"
assert "capture" not in body
assert sent[0]["capture"]["components"] == ["unet"]
assert "capture" not in sent[1]
assert json.loads(sent[2]["payload"])["models"] == {"model":None}
from pathlib import Path
Path(sys.argv[2]).write_text(sent[0]["payload"])
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", check, str(output), str(tmp_path / "serving-call.json")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    entry = next(
        row for row in static_interface.build(root)["entrypoints"] if row["name"] == "generate"
    )
    target = Target(
        "revision", "generate", "interface", entry, {}, None, "digest", serving_envelope=True
    )
    frozen = (tmp_path / "serving-call.json").read_bytes()
    payload, models = arguments(target, frozen)
    assert canonical_json.decode(payload) == {
        "prompt": "same",
        "mode": "one",
        "tags": {},
        "wire_count": 3,
    }
    assert models["model"]["producer_request_id"] == "producer"
    body = canonical_json.decode(frozen)
    for wrong in [
        body["payload"],
        {**body, "extra": 1},
        {**body, "models": {"foreign": None}},
        {**body, "payload": []},
    ]:
        with pytest.raises(WorkspaceRefusal, match="invalid payload or Model slots"):
            arguments(target, canonical_json.encode(wrong))
    for omitted in ({}, {"model": None}):
        assert arguments(target, canonical_json.encode({**body, "models": omitted}))[1] == {}
    # A self-call is the original invocable contract, even when its user fields
    # happen to be named payload and models. No heuristic peeks inside user data.
    from dataclasses import replace

    assert arguments(replace(target, serving_envelope=False), frozen) == (frozen, body)
    caller = output / "uses.py"
    caller.write_text(
        "from serving_fixture import generate, Result\n"
        "from cozy_runtime.author import ModelArtifact, PendingCall\n"
        "def run(model: ModelArtifact) -> PendingCall[Result]:\n"
        "    return generate(prompt='hello', model=model)\n"
        "def default() -> PendingCall[Result]:\n"
        "    return generate(prompt='hello')\n"
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


def test_caller_overlay_preserves_helpers_assets_and_dependencies(tmp_path: Path) -> None:
    root = project(tmp_path)
    raw = package_interface.canonical_bytes(static_interface.build(root))
    original = io.BytesIO()
    with zipfile.ZipFile(original, "w") as archive:
        archive.writestr("serving_fixture.py", SOURCE + "\ndef helper(): return 19\n")
        archive.writestr("serving_fixture_data/defaults.json", b'{"width":512}')
        archive.writestr("serving_fixture_helpers.py", "VALUE = 23\n")
        archive.writestr(
            "serving_fixture-0.1.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: serving-fixture\nVersion: 0.1.0\n"
            "Requires-Dist: msgspec>=0.19\nRequires-Dist: numpy>=1.26\n\nDescription.\n",
        )
        archive.writestr(
            "serving_fixture-0.1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
    filename, wheel = interface_wheel.build(
        raw,
        distribution="serving-fixture",
        version="0.1.0",
        implementation_digest="sha256:" + "11" * 32,
        implementation_wheel=original.getvalue(),
        implementation_filename="serving_fixture-0.1.0-py3-none-any.whl",
    )
    assert interface_wheel.build(
        raw,
        distribution="serving-fixture",
        version="0.1.0",
        implementation_digest="sha256:" + "11" * 32,
        implementation_wheel=original.getvalue(),
        implementation_filename="serving_fixture-0.1.0-py3-none-any.whl",
    ) == (filename, wheel)
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        archive.extractall(tmp_path / "overlay")
        metadata = archive.read("serving_fixture-0.1.0.dist-info/METADATA")
        assert b"Requires-Dist: numpy>=1.26\n" in metadata
        assert metadata.endswith(b"\n\nDescription.\n")
        assert archive.read("serving_fixture_data/defaults.json") == b'{"width":512}'
    from cozy_runtime.internal.installed_interfaces import read
    from cozy_runtime.internal.package_environment import InstalledEnvironment

    installed = InstalledEnvironment(
        tmp_path, tmp_path / "overlay", Path(sys.executable), b"", "", "sha256:" + "22" * 32, True
    )
    assert {row["export"] for row in read(installed)} == {"generate", "echo"}
    check = """import inspect, sys
sys.path.insert(0, sys.argv[1])
import serving_fixture as lib
from serving_fixture_helpers import VALUE
assert lib.helper() == 19 and VALUE == 23
assert lib.app.get("generate").fn is not lib.generate
assert "capture" in inspect.signature(lib.generate).parameters
assert "generate" in lib.__cozy_bindings__
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", check, str(tmp_path / "overlay")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    companion = next((tmp_path / "overlay").glob("_cozy_callers_*.py"))
    companion.write_bytes(companion.read_bytes() + b"# changed\n")
    with pytest.raises(ConformanceError, match="trusted generator"):
        read(installed)


def test_one_published_wheel_keeps_direct_serving_and_imported_callers(tmp_path: Path) -> None:
    """Publication seals this wheel once; both execution roles consume its same bytes."""
    source = """import msgspec
from cozy_runtime.author import App, Context
app = App()
FACTOR = 3
class Request(msgspec.Struct):
    value: int
class Result(msgspec.Struct):
    value: int
def helper(value):
    return value * FACTOR
@app.entrypoint
def generate(ctx: Context, payload: Request) -> Result:
    return Result(helper(payload.value))
@app.entrypoint(internal=True)
def secret(ctx: Context, payload: Request) -> Result:
    return Result(payload.value + 100)
"""
    root = tmp_path / "source"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="dual-fixture"\nversion="0.1.0"\n')
    (root / "package.toml").write_text('[application]\nobject="dual_fixture:app"\n')
    (root / "dual_fixture.py").write_text(source)
    raw = package_interface.canonical_bytes(static_interface.build(root))
    original = io.BytesIO()
    with zipfile.ZipFile(original, "w") as archive:
        archive.writestr("dual_fixture.py", source)
        archive.writestr(
            "dual_fixture-0.1.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: dual-fixture\nVersion: 0.1.0\n",
        )
        archive.writestr(
            "dual_fixture-0.1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
    import hashlib

    source_digest = "sha256:" + hashlib.sha256(original.getvalue()).hexdigest()
    _, wheel = interface_wheel.build(
        raw,
        distribution="dual-fixture",
        version="0.1.0",
        implementation_digest=source_digest,
        implementation_wheel=original.getvalue(),
        implementation_filename="dual_fixture-0.1.0-py3-none-any.whl",
    )
    _, repeated = interface_wheel.build(
        raw,
        distribution="dual-fixture",
        version="0.1.0",
        implementation_digest=source_digest,
        implementation_wheel=wheel,
        implementation_filename="dual_fixture-0.1.0-py3-none-any.whl",
    )
    assert repeated == wheel
    _, captured = interface_wheel.build(
        raw,
        distribution="dual-fixture",
        version="0.1.0",
        implementation_digest="sha256:" + "33" * 32,
        implementation_wheel=wheel,
        implementation_filename="dual_fixture-0.1.0-py3-none-any.whl",
    )
    with zipfile.ZipFile(io.BytesIO(captured)) as archive:
        assert archive.read("dual_fixture.py").count(b"from _cozy_callers_") == 1
    corrupted = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(wheel)) as original_wheel,
        zipfile.ZipFile(corrupted, "w") as altered,
    ):
        for name in original_wheel.namelist():
            body = original_wheel.read(name)
            if name.startswith("_cozy_callers_"):
                body += b"# tampered\n"
            altered.writestr(name, body)
    with pytest.raises(ConformanceError, match="generated source changed"):
        interface_wheel.build(
            raw,
            distribution="dual-fixture",
            version="0.1.0",
            implementation_digest=source_digest,
            implementation_wheel=corrupted.getvalue(),
            implementation_filename="dual_fixture-0.1.0-py3-none-any.whl",
        )
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        archive.extractall(tmp_path / "sealed")
    from cozy_runtime.internal.installed_interfaces import read
    from cozy_runtime.internal.package_environment import InstalledEnvironment

    installed = InstalledEnvironment(
        tmp_path, tmp_path / "sealed", Path(sys.executable), b"", "", "sha256:" + "22" * 32, True
    )
    inventory = read(installed)
    assert [(row["export"], row["implementation_digest"]) for row in inventory] == [
        ("generate", source_digest)
    ]
    from types import SimpleNamespace
    from typing import Any, cast

    from cozy_runtime.internal.worker.calls import Calls

    interface_path = tmp_path / "package-interface.json"
    interface_path.write_bytes(raw)
    parent = SimpleNamespace(
        job=SimpleNamespace(
            call_interfaces=tuple(inventory),
            python=sys.executable,
            package_interface=str(interface_path),
            build_id="sha256:" + "44" * 32,
        )
    )
    bindings = [row for row in Calls.bindings(cast(Any, parent)) if row["module"] == "dual_fixture"]
    assert [(row["export"], row.get("self", False)) for row in bindings] == [
        ("generate", False),
        ("secret", True),
    ]
    check = """import asyncio, json, sys
sys.path.insert(0,sys.argv[1])
import dual_fixture as lib
from cozy_runtime.author import PendingCall
from cozy_runtime.author._calls import _Broker, _current
from cozy_runtime.author.fakes import fake_context
ctx = fake_context()
assert lib.helper(4) == 12
raw = lib.app.get("generate").fn(ctx, lib.Request(7))
assert isinstance(raw, lib.Result) and raw.value == 21
assert lib.app.get("secret").fn(ctx, lib.Request(7)).value == 107
assert "secret" not in lib.__cozy_bindings__
binding=lib.__cozy_bindings__["generate"]
sent=[]
def exchange(kind,body):
    if kind=="child_call":
        sent.append(json.loads(body["payload"]))
        return {"ok":True,"child_request_id":"child"}
    return {"ok":True,"state":"succeeded","result":'{"value":21}'}
broker=_Broker(ctx.request_id,{(binding.interface_digest,binding.module,binding.export):binding},exchange)
broker.bind(ctx)
token=_current.set(broker)
async def run():
    pending=lib.generate(value=7)
    assert isinstance(pending,PendingCall)
    result=await pending
    assert result.value==21
try: asyncio.run(run()); broker.finish()
finally: _current.reset(token)
assert len(sent)==1 and sent[0]["payload"]=={"value":7}
# Calling through the registry still runs the body, not the proxy, after imported use.
assert lib.app.get("generate").fn(ctx, lib.Request(8)).value==24
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", check, str(tmp_path / "sealed")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("tag", "purelib"),
    [("cp312-abi3-manylinux_2_28_x86_64", "false"), ("py3-none-win_amd64", "true")],
)
def test_source_overlay_preserves_platform_build_metadata_and_native_bytes(
    tmp_path: Path, tag: str, purelib: str
) -> None:
    root = project(tmp_path)
    raw = package_interface.canonical_bytes(static_interface.build(root))
    filename = f"serving_fixture-0.1.0-2native-{tag}.whl"
    wheel_metadata = (
        "Wheel-Version: 1.0\r\nGenerator: original-native-builder\r\n"
        f"Root-Is-Purelib: {purelib}\r\nBuild: 2native\r\nTag: {tag}\r\n"
    ).encode()
    native = b"opaque native payload\x00\xff\x01"
    script = zipfile.ZipInfo("serving_fixture-0.1.0.data/scripts/native-helper")
    script.external_attr = 0o100755 << 16
    original = io.BytesIO()
    with zipfile.ZipFile(original, "w") as archive:
        archive.writestr("serving_fixture.py", SOURCE)
        archive.writestr("serving_fixture_native.abi3.so", native)
        archive.writestr(script, native)
        archive.writestr(
            "serving_fixture-0.1.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: serving-fixture\nVersion: 0.1.0\n",
        )
        archive.writestr("serving_fixture-0.1.0.dist-info/WHEEL", wheel_metadata)
    options = dict(
        distribution="serving-fixture",
        version="0.1.0",
        implementation_digest="sha256:" + "11" * 32,
        implementation_filename=filename,
    )
    actual_filename, overlay = interface_wheel.build(
        raw, implementation_wheel=original.getvalue(), **options
    )
    assert actual_filename == filename
    with zipfile.ZipFile(io.BytesIO(overlay)) as archive:
        assert archive.read("serving_fixture-0.1.0.dist-info/WHEEL") == wheel_metadata
        assert archive.read("serving_fixture_native.abi3.so") == native
        assert archive.read(script.filename) == native
        assert archive.getinfo(script.filename).external_attr == script.external_attr
    assert interface_wheel.build(raw, implementation_wheel=overlay, **options) == (
        filename,
        overlay,
    )
    with pytest.raises(ConformanceError, match="filename and WHEEL metadata disagree"):
        interface_wheel.build(
            raw,
            distribution="serving-fixture",
            version="0.1.0",
            implementation_digest="sha256:" + "11" * 32,
            implementation_filename="serving_fixture-0.1.0-py3-none-any.whl",
            implementation_wheel=original.getvalue(),
        )
