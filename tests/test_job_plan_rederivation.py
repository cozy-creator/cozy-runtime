"""A job plan is derived from its installation and descriptor: newer derivations replace it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cozy_runtime.internal import job_plan
from cozy_runtime.internal.worker.plan import JobBinding

DESCRIPTOR = "sha256:" + "a" * 64


def _record(**changes: object) -> dict[str, object]:
    return {
        "job_descriptor_id": DESCRIPTOR,
        "installation_id": "install-1",
        "application": "package:app",
        "package_interface": "/plans/interface.json",
        "python": "/venv/bin/python",
        "job": "compute",
        "publishes": False,
        "emits_media": False,
        "gpu_rate_micro_usd_per_hour": 0,
        "cap_micro_usd": 0,
        **changes,
    }


def test_a_rederived_plan_replaces_the_old_bytes_and_still_reads(tmp_path: Path) -> None:
    target = job_plan.write(tmp_path, _record())
    inode = target.stat().st_ino
    assert job_plan.write(tmp_path, _record()) == target and target.stat().st_ino == inode
    job_plan.write(tmp_path, _record(python="/relocated/bin/python", added_by_newer=True))
    held = json.loads(target.read_bytes())
    assert held["python"] == "/relocated/bin/python" and target.stat().st_mode & 0o222 == 0
    assert JobBinding.read(held).job == "compute"


def test_another_callable_under_the_same_key_refuses(tmp_path: Path) -> None:
    target = job_plan.write(tmp_path, _record())
    with pytest.raises(ValueError, match="another callable"):
        job_plan.write(tmp_path, _record(job="other"))
    assert json.loads(target.read_bytes())["job"] == "compute"
