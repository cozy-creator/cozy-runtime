"""A job's own execution device reaches the package interface from either reader."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import discover

SOURCE = """import msgspec
from cozy_runtime.author import App, Context, invocable
app = App()
class Request(msgspec.Struct):
    n: int
class Result(msgspec.Struct):
    value: int
@app.job(accelerator=False)
def convert(payload: Request) -> Result:
    return Result(payload.n)
@invocable
async def tables(ctx: Context, *, n: int) -> Result:
    return Result(n)
app.job(tables, accelerator=True)
@app.job
def undeclared(payload: Request) -> Result:
    return Result(payload.n)
"""


def project(tmp_path: Path, source: str = SOURCE) -> Path:
    sys.modules.pop("accelerator_fixture", None)
    root = tmp_path / "source"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="accelerator-fixture"\nversion="0.1.0"\n')
    (root / "package.toml").write_text('[application]\nobject="accelerator_fixture:app"\n')
    (root / "accelerator_fixture.py").write_text(source)
    return root


def test_job_accelerator_is_identical_for_static_and_imported_discovery(tmp_path: Path) -> None:
    root = project(tmp_path)
    static = static_interface.build(root)
    imported = package_interface.build(discover(root))
    raw = package_interface.canonical_bytes(static)
    assert raw == package_interface.canonical_bytes(imported)
    jobs = {job["name"]: job for job in package_interface.read_bytes(raw)["jobs"]}
    assert jobs["convert"]["accelerator"] is False
    assert jobs["tables"]["accelerator"] is True
    assert "accelerator" not in jobs["undeclared"]


@pytest.mark.parametrize("value", ['"cpu"', "0"])
def test_job_accelerator_requires_a_boolean_in_both_readers(tmp_path: Path, value: str) -> None:
    root = project(tmp_path, SOURCE.replace("accelerator=False", "accelerator=" + value))
    with pytest.raises(ConformanceError, match="accelerator must be a boolean"):
        static_interface.build(root)
    with pytest.raises(ConformanceError, match="accelerator must be a boolean"):
        discover(root)


@pytest.mark.parametrize("value", ["false", 0, None])
def test_interface_reader_rejects_nonboolean_accelerator(tmp_path: Path, value: object) -> None:
    document = static_interface.build(project(tmp_path))
    document["jobs"][0]["accelerator"] = value
    with pytest.raises(ConformanceError, match="accelerator"):
        package_interface.read_bytes(canonical_json.encode(document))
