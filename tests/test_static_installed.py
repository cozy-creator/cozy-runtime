"""An explicitly selected installed wheel is described without importing its App."""

from pathlib import Path

import pytest

from cozy_runtime.cli.io import CliError
from cozy_runtime.cli.main import dispatch, parse
from cozy_runtime.internal import static_interface
from cozy_runtime.internal.config import Credentials, RuntimeConfig


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
    (site / "example_app.py").write_text("""
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
""")
    return python, metadata


def test_describe_exact_installed_wheel_without_package_toml_or_imports(tmp_path: Path) -> None:
    python, _ = installed(tmp_path)
    body = static_interface.build_installed("Example.App", environment_python=python)
    assert body["application"] == "example_app:app"
    assert body["jobs"][0]["name"] == "double"
    assert body["jobs"][0]["invocable"]["memoize"] is True


@pytest.mark.parametrize(
    "mutation", ["missing", "ambiguous", "foreign", "linked", "identity", "ambient"]
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
    elif mutation == "linked":
        original = metadata / "METADATA"
        retained = metadata / "original"
        original.rename(retained)
        original.symlink_to(retained)
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
