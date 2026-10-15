"""Internal visibility preserves same-package execution without opening a root API."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import interface_wheel, package_interface, static_interface
from cozy_runtime.internal.discovery import discover

SOURCE = """import msgspec
from cozy_runtime.author import App, Context, invocable
app = App()
class Request(msgspec.Struct):
    n: int
class Result(msgspec.Struct):
    value: int
@app.entrypoint
def public(payload: Request) -> Result:
    return Result(payload.n)
@app.entrypoint(internal=True)
def segment(payload: Request) -> Result:
    return Result(payload.n)
@invocable
async def child(ctx: Context, *, n: int) -> Result:
    return Result(n)
app.entrypoint(child, internal=True, accelerator=False)
@app.entrypoint(hidden=True)
def unfinished(payload: Request) -> Result:
    return Result(payload.n)
"""


def project(tmp_path: Path, source: str = SOURCE) -> Path:
    sys.modules.pop("internal_fixture", None)
    root = tmp_path / "source"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="internal-fixture"\nversion="0.1.0"\n')
    (root / "package.toml").write_text('[application]\nobject="internal_fixture:app"\n')
    (root / "internal_fixture.py").write_text(source)
    return root


def test_internal_interface_is_identical_for_static_and_imported_discovery(tmp_path: Path) -> None:
    root = project(tmp_path)
    static = static_interface.build(root)
    imported = package_interface.build(discover(root))
    raw = package_interface.canonical_bytes(static)
    assert raw == package_interface.canonical_bytes(imported)
    document = package_interface.read_bytes(raw)
    entries = {entry["name"]: entry for entry in document["entrypoints"]}
    assert list(entries) == ["child", "public", "segment"]
    assert entries["segment"]["internal"] is True and entries["child"]["internal"] is True
    assert "internal" not in entries["public"]
    generated = interface_wheel.generate(raw)["internal_fixture/__init__.py"].decode()
    assert "def public(" in generated
    assert "def segment(" not in generated and "def child(" not in generated


@pytest.mark.parametrize("value", ['"yes"', "1"])
def test_internal_requires_a_boolean_in_both_readers(tmp_path: Path, value: str) -> None:
    root = project(tmp_path, SOURCE.replace("internal=True", "internal=" + value))
    with pytest.raises(ConformanceError, match="internal must be a boolean"):
        static_interface.build(root)
    with pytest.raises(ConformanceError, match="internal must be a boolean"):
        discover(root)


def test_hidden_and_internal_are_distinct_and_mutually_exclusive(tmp_path: Path) -> None:
    root = project(tmp_path, SOURCE.replace("hidden=True", "internal=True, hidden=True"))
    for reader in (static_interface.build, discover):
        with pytest.raises(ConformanceError, match="cannot also be hidden"):
            reader(root)


@pytest.mark.parametrize("value", ["true", 1, None])
def test_interface_reader_rejects_nonboolean_internal(tmp_path: Path, value: object) -> None:
    document = static_interface.build(project(tmp_path))
    document["entrypoints"][1]["internal"] = value
    with pytest.raises(ConformanceError, match="internal"):
        package_interface.read_bytes(canonical_json.encode(document))
