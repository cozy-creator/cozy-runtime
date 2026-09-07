"""An ordinary job-only package must install without a fake serving entrypoint."""

from __future__ import annotations

from pathlib import Path

import pytest

from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.local import development_placement_documents
from cozy_runtime.protocol import documents


def test_job_only_project_produces_a_development_placement(tmp_path: Path) -> None:
    (tmp_path / "package.toml").write_text('[application]\nobject = "only_job:app"\n')
    (tmp_path / "only_job.py").write_text(
        "import msgspec\n"
        "from cozy_runtime.author import App\n"
        "app = App()\n"
        "class Request(msgspec.Struct):\n    value: int\n"
        "class Result(msgspec.Struct):\n    value: int\n"
        "@app.job\n"
        "def run(payload: Request) -> Result:\n    return Result(payload.value + 1)\n"
    )
    found = discover(tmp_path)
    interface = package_interface.build(found)
    assert not interface["entrypoints"]
    assert len(interface["jobs"]) == 1
    result = development_placement_documents(
        interface,
        package="local/only-job",
        release="0.0.0",
        source_digest="sha256:" + "a" * 64,
        found=found,
    )
    assert not result.placement.entrypoints
    assert not result.placement.models
    assert result.placement.development.package == "local/only-job"
    assert result.package_interface == package_interface.canonical_bytes(interface)
    assert documents.body(result.placement)["entrypoints"] == []


def test_empty_interface_still_refuses() -> None:
    with pytest.raises(RuntimeError, match="at least one visible"):
        development_placement_documents(
            {"entrypoints": [], "jobs": []},
            package="local/empty",
            release="0.0.0",
            source_digest="sha256:" + "a" * 64,
        )
