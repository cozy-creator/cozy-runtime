"""An explicitly selected installed wheel is described without importing its App."""

import base64
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from cozy_runtime.cli.io import CliError
from cozy_runtime.cli.main import dispatch, parse
from cozy_runtime.internal import static_interface
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
