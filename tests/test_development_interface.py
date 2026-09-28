"""A local run derives its development PackageInterface with the worker's one builder.

The CLI once built the document by importing the app while the worker re-derived it from
source; for a package with a memoized job only the import added `operation_identity`, so
every local run of it was refused `development_source_changed`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import discover

PACKAGE = """\
import msgspec
from cozy_runtime.author import App, Context, invocable

app = App()


class Request(msgspec.Struct):
    value: int


class Result(msgspec.Struct):
    value: int


@app.entrypoint
def serve(payload: Request) -> Result:
    return Result(payload.value + 1)


@invocable(memoize=True)
async def twice(ctx: Context, *, value: int) -> Result:
    return Result(value * 2)


app.job(twice)
"""


def test_conformance_compares_what_a_source_reading_can_know(tmp_path: Path) -> None:
    """The import states a memoized job's installed operation identity; source never can."""
    (tmp_path / "package.toml").write_text('[application]\nobject = "memo_conformance:app"\n')
    (tmp_path / "memo_conformance.py").write_text(PACKAGE)
    static = static_interface.build(tmp_path)
    imported = package_interface.build(discover(tmp_path))
    job = imported["jobs"][0]["invocable"]
    assert job["memoize"] and job["operation_identity"].startswith("sha256:")
    with pytest.raises(package_interface.StalePackageInterface, match="operation_identity"):
        package_interface.compare(static, imported, ("source", "import"))
    package_interface.compare(
        static, package_interface.source_facts(imported), ("source", "import")
    )
