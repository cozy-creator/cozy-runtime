"""An edited implementation cannot replace an older request's executable plan."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import job_plan, package_environment, package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.local import development_source_digest, stage_job_plans
from cozy_runtime.internal.worker.package_prepare import _stage_job_plans
from cozy_runtime.internal.worker.plan import JobBinding


def test_two_implementations_with_one_interface_keep_exact_resumable_plans(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "package.toml").write_text('[application]\nobject="revision_job:app"\n')
    program = """import msgspec
from cozy_runtime.author import App
app=App()
class Input(msgspec.Struct):
    value:int
class Output(msgspec.Struct):
    value:int
@app.job
def compute(payload:Input)->Output:
    return Output(payload.value + INCREMENT)
"""
    source = project / "revision_job.py"
    source.write_text(program.replace("INCREMENT", "1"))
    interface = package_interface.build(discover(project))
    first = development_source_digest(project)
    source.write_text(program.replace("INCREMENT", "2"))
    second = development_source_digest(project)
    assert first != second
    exact = package_interface.canonical_bytes(interface)
    assert package_interface.canonical_bytes(package_interface.build(discover(project))) == exact
    interface_path = tmp_path / "interface.json"
    interface_path.write_bytes(exact)
    descriptor = package_interface.job_descriptor_id(interface, "compute")

    def installed(build: str) -> package_environment.InstalledEnvironment:
        generation = tmp_path / build.removeprefix("sha256:")
        generation.mkdir(exist_ok=True)
        return package_environment.InstalledEnvironment(
            generation, generation, Path(sys.executable), b"", "", build, True
        )

    root = tmp_path / "remote-plans"
    for build in (first, second):
        _stage_job_plans(
            interface,
            root=root,
            build_id=build,
            interface_path=interface_path,
            installed=installed(build),
        )
    old = job_plan.path(root, first, descriptor)
    new = job_plan.path(root, second, descriptor)
    old_bytes, new_bytes = old.read_bytes(), new.read_bytes()
    assert old != new
    assert json.loads(old_bytes)["build_id"] == first
    assert json.loads(new_bytes)["build_id"] == second
    # A Host replay points at its immutable prepared B1 file even though B2 was
    # prepared later with the identical callable schema. No descriptor-only
    # fallback can select the other implementation.
    old_record = json.loads(old_bytes)
    new_record = json.loads(new_bytes)
    # No package interface dependencies in this ordinary fixture.
    old_record["python"] = new_record["python"] = "/nonexistent/not-a-venv/python"
    assert JobBinding.read(old_record).build_id == first
    assert JobBinding.read(new_record).build_id == second
    local = tmp_path / "local"
    stage_job_plans(local, [old_record])
    stage_job_plans(local, [new_record])
    assert job_plan.path(local / "job-plans", first, descriptor).exists()
    assert job_plan.path(local / "job-plans", second, descriptor).exists()
    _stage_job_plans(
        interface,
        root=root,
        build_id=first,
        interface_path=interface_path,
        installed=installed(first),
    )
    assert old.read_bytes() == old_bytes and new.read_bytes() == new_bytes
    assert not (root / (descriptor.removeprefix("sha256:") + ".json")).exists()
    original = json.loads(old_bytes)
    inode = old.stat().st_ino
    job_plan.write(root, original)
    assert old.stat().st_ino == inode
    original["python"] = "/relocated/interpreter"
    job_plan.write(root, original)
    assert json.loads(old.read_bytes())["python"] == "/relocated/interpreter"
    original["job"] = "another"
    with pytest.raises(ValueError, match="another callable"):
        job_plan.write(root, original)
