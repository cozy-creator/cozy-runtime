"""An explicitly selected installed wheel is described without importing its App."""

import base64
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

import cozy_runtime
from cozy_runtime.cli.io import CliError
from cozy_runtime.cli.main import dispatch, parse
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.config import Credentials, RuntimeConfig

SOURCE = """
from cozy_runtime.author import App, Context, invocable
import msgspec
raise RuntimeError("package imports must never run")
app = App()
class Result(msgspec.Struct, frozen=True):
    value: int
@invocable(memoize=True)
async def double(ctx: Context, *, value: int) -> Result:
    return Result(value * 2)
app.job(double)
"""


def installed(tmp_path: Path) -> tuple[Path, Path]:
    prefix = tmp_path / "venv"
    (prefix / "bin").mkdir(parents=True)
    python = prefix / "bin" / "python"
    python.touch()
    (prefix / "pyvenv.cfg").write_text("home = /unused\n")
    site = prefix / "lib" / "python3.12" / "site-packages"
    metadata = site / "example_app-1.0.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Name: example-app\nVersion: 1.0.0\n")
    (metadata / "entry_points.txt").write_text("[cozy.application]\ndefault = example_app:app\n")
    (metadata / "RECORD").write_text("example_app.py,,\n")
    (site / "example_app.py").write_text(SOURCE)
    return python, metadata


def test_describe_exact_installed_wheel_without_package_toml_or_imports(tmp_path: Path) -> None:
    python, _ = installed(tmp_path)
    body = static_interface.build_installed("Example.App", environment_python=python)
    assert body["application"] == "example_app:app"
    assert body["jobs"][0]["name"] == "double"
    assert body["jobs"][0]["invocable"]["memoize"] is True


def test_installed_description_reads_only_the_wheels_own_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dependency imported at module scope (Torch, Diffusers) is never scanned: describing
    an H3 environment's callees read 6,389 dependency files and took a minute per boot."""
    python, metadata = installed(tmp_path)
    site = metadata.parent
    (site / "heavy").mkdir()
    (site / "heavy" / "__init__.py").write_text("from heavy import deep\n")
    (site / "heavy" / "deep.py").write_text("VALUE = 1\n")
    (site / "example_app.py").write_text("import heavy\n" + SOURCE)
    loaded: list[str] = []
    load = static_interface.Reader._load

    def spy(self: static_interface.Reader, name: str, *args: Any) -> Any:
        loaded.append(name)
        return load(self, name, *args)

    monkeypatch.setattr(static_interface.Reader, "_load", spy)
    body = static_interface.build_installed("example-app", environment_python=python)
    assert body["jobs"][0]["name"] == "double"
    assert "example_app" in loaded
    assert not [name for name in loaded if name.startswith("heavy")]


@pytest.mark.parametrize(
    "mutation", ["missing", "ambiguous", "foreign", "dangling", "identity", "ambient"]
)
def test_refuses_invalid_selected_distribution(tmp_path: Path, mutation: str) -> None:
    python, metadata = installed(tmp_path)
    if mutation == "missing":
        (metadata / "entry_points.txt").unlink()
    elif mutation == "ambiguous":
        (metadata / "entry_points.txt").write_text(
            "[cozy.application]\none = example_app:app\ntwo = example_app:app\n"
        )
    elif mutation == "foreign":
        (metadata / "RECORD").write_text("another.py,,\n")
    elif mutation == "dangling":
        (metadata / "METADATA").unlink()
        (metadata / "METADATA").symlink_to(metadata / "absent")
    elif mutation == "identity":
        (metadata / "METADATA").write_text("Name: other\nVersion: 1.0.0\n")
    else:
        metadata.rename(metadata.with_name("another-1.0.0.dist-info"))
    with pytest.raises(static_interface.StaticRefusal):
        static_interface.build_installed("example-app", environment_python=python)


@pytest.mark.parametrize("flags", [[], ["--conformance"], ["--package-interface", "exact.json"]])
def test_installed_description_cannot_enter_importing_or_document_mode(flags: list[str]) -> None:
    args = ["describe", "--distribution", "example-app", *flags]
    if flags:
        args += ["--environment-python", "/unused/venv/bin/python"]
    verb, positional, opts, _ = parse(args)
    assert verb == "describe"
    with pytest.raises(CliError):
        dispatch(verb, positional, opts, RuntimeConfig(Path("/unused"), Credentials()))


def wheel(directory: Path) -> Path:
    info = "example_app-1.0.0.dist-info"
    members = {
        "example_app.py": SOURCE,
        f"{info}/METADATA": "Metadata-Version: 2.1\nName: example-app\nVersion: 1.0.0\n",
        f"{info}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{info}/entry_points.txt": "[cozy.application]\ndefault = example_app:app\n",
    }

    def row(name: str, text: str) -> str:
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=")
        return f"{name},sha256={digest.decode()},{len(text.encode())}\n"

    path = directory / "example_app-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)
        record = "".join(row(*member) for member in members.items()) + f"{info}/RECORD,,\n"
        archive.writestr(f"{info}/RECORD", record)
    return path


@pytest.mark.parametrize("mode", ["copy", "symlink"])
def test_an_app_installed_by_any_uv_link_mode_is_described(tmp_path: Path, mode: str) -> None:
    """A machine whose image seeds uv's cache installs with `--link-mode symlink`: every
    installed file links into that cache, and its callees are described all the same."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv installs the wheel")
    venv = tmp_path / "venv"
    subprocess.run([uv, "venv", "-q", "--python", sys.executable, str(venv)], check=True)
    python = venv / "bin" / "python"
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "-q",
            "--offline",
            "--no-deps",
            "--link-mode",
            mode,
            "--python",
            str(python),
            str(wheel(tmp_path)),
        ],
        check=True,
        env={"PATH": "/usr/bin:/bin", "UV_CACHE_DIR": str(tmp_path / "cache")},
    )
    site = next((venv / "lib").glob("python*/site-packages"))
    assert (site / "example_app.py").is_symlink() == (mode == "symlink")
    assert (site / "example_app-1.0.0.dist-info" / "METADATA").is_symlink() == (mode == "symlink")

    body = static_interface.build_installed("example-app", environment_python=python)
    assert body["application"] == "example_app:app"
    assert body["jobs"][0]["name"] == "double"

    # A caller described beside it reads the callee's types through the same links.
    caller = tmp_path / "caller"
    caller.mkdir()
    (caller / "pyproject.toml").write_text(
        '[project]\nname = "caller"\nversion = "0.1.0"\n'
        '[project.entry-points."cozy.application"]\ndefault = "caller:app"\n'
    )
    (caller / "package.toml").write_text('[application]\nobject = "caller:app"\n')
    (caller / "caller.py").write_text(
        "from cozy_runtime.author import App, Context\n"
        "from example_app import Result\n"
        "app = App()\n"
        "@app.entrypoint\n"
        "def relay(ctx: Context, payload: Result) -> Result:\n"
        "    return payload\n"
    )
    document = static_interface.build(caller, environment_python=python)
    assert document["entrypoints"][0]["request"]["fields"][0]["name"] == "value"


MEMO_SOURCE = """
import msgspec
from cozy_runtime.author import App, Context, MemoDistribution, invocable
import memo_helper
app = App()
class Result(msgspec.Struct, frozen=True):
    value: int
def scale(value: int) -> int:
    return value * 3
@invocable(
    memoize=True,
    memo_version="2",
    memo_dependencies=(scale, "memo_helper", MemoDistribution("msgspec")),
)
async def first(ctx: Context, *, value: int) -> Result:
    return Result(scale(value) + memo_helper.OFFSET)
@invocable(memoize=True)
async def second(ctx: Context, *, value: int) -> Result:
    return Result(value * 2)
@invocable(memoize=True, memo_dependencies=(MemoDistribution("msgspec" if True else "numpy"),))
async def computed(ctx: Context, *, value: int) -> Result:
    return Result(value)
app.job(first)
app.job(second)
app.job(computed)
"""


IMPORTED = """
import json, memo_app
from cozy_runtime.internal.memo_implementation import describe
# As the Runtime describes an App's callable: its implementation, which carries its declaration.
fns = {n: getattr(memo_app, n)._cozy_export.implementation for n in ("first", "second")}
print(json.dumps({n: describe(fn).get("operation_identity") for n, fn in fns.items()}))
"""


def memo_environment(tmp_path: Path, second_body: str = "value * 2") -> Path:
    """`memo_app` and its helper module, installed by uv from a real wheel."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv installs the wheel")
    project = tmp_path / "memo-project"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[project]\nname = "memo-app"\nversion = "1.0.0"\ndependencies = ["msgspec"]\n'
        '[project.entry-points."cozy.application"]\ndefault = "memo_app:app"\n'
        '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
        '[tool.hatch.build.targets.wheel]\nonly-include = ["memo_app.py", "memo_helper.py"]\n'
    )
    (project / "memo_app.py").write_text(MEMO_SOURCE.replace("value * 2", second_body))
    (project / "memo_helper.py").write_text("OFFSET = 7\n")
    venv = tmp_path / "memo-venv"
    subprocess.run([uv, "venv", "-q", "--python", sys.executable, str(venv)], check=True)
    python = venv / "bin" / "python"
    subprocess.run([uv, "pip", "install", "-q", "--python", str(python), str(project)], check=True)
    return python


def identities(document: dict[str, Any]) -> dict[str, object]:
    return {job["name"]: job["invocable"].get("operation_identity") for job in document["jobs"]}


def test_an_installed_memoized_callable_carries_the_identity_its_import_would(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The static identity is `describe`'s: one key whether a machine reads or imports it.
    A declaration the reader cannot follow carries none (its machine uses the package)."""
    python = memo_environment(tmp_path)

    def no_discovery_hooks(*args: object, **kwargs: object) -> None:
        raise AssertionError("static inspection must not execute installed distribution finders")

    monkeypatch.setattr(importlib.metadata, "distributions", no_discovery_hooks)
    found = identities(static_interface.build_installed("memo-app", environment_python=python))
    # The environment's own packages first, then this Runtime: describe() imports the App.
    site = next((python.parent.parent / "lib").glob("python*/site-packages"))
    runtime = Path(cozy_runtime.__file__).parents[1]
    imported = subprocess.run(
        [sys.executable, "-c", IMPORTED],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{site}:{runtime}"},
    )
    assert {k: found[k] for k in ("first", "second")} == json.loads(imported.stdout)
    assert all(str(found[k]).startswith("sha256:") for k in ("first", "second"))
    assert found["computed"] is None
    # A source reading carries no installed fact.
    assert all(
        "operation_identity" not in job["invocable"]
        for job in package_interface.source_facts(
            static_interface.build_installed("memo-app", environment_python=python)
        )["jobs"]
    )


def test_a_changed_function_changes_only_its_own_identity(tmp_path: Path) -> None:
    before = identities(
        static_interface.build_installed(
            "memo-app", environment_python=memo_environment(tmp_path / "a")
        )
    )
    after = identities(
        static_interface.build_installed(
            "memo-app", environment_python=memo_environment(tmp_path / "b", "value * 2 + 1")
        )
    )
    assert after["first"] == before["first"]
    assert after["second"] != before["second"]


def test_imported_memo_metadata_failure_keeps_typed_unavailability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cozy_runtime.author import MemoDistribution
    from cozy_runtime.internal.memo_implementation import describe

    class BrokenDistribution:
        version = "1.0"

        def read_text(self, name: str) -> str:
            raise OSError("metadata disappeared")

    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: BrokenDistribution())

    def work(value: int) -> int:
        return value * 2

    setattr(work, "_cozy_memo_declaration", ("", (MemoDistribution("native-helper"),)))
    assert describe(work) == {
        "operation_identity_unavailable": "operation_native_identity_unavailable"
    }
