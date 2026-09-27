"""A package environment's executor is admitted by protocol, not Runtime package metadata."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

import cozy_runtime
from cozy_runtime.internal.worker import lanes
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from test_device_lanes import _config, _workspace
from test_end_to_end import NO_EXECUTOR


def _shadowing_python(root: Path) -> Path:
    """An interpreter that imports this checkout's Runtime but whose metadata says 0.0.1:
    the shape of an environment whose lock carried another Runtime."""
    venv = root / "shadow"
    uv = shutil.which("uv") or "uv"
    subprocess.run([uv, "venv", "-q", "--python", sys.executable, str(venv)], check=True)
    site = next(venv.glob("lib/python*/site-packages"))
    info = site / "cozy_runtime-0.0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.3\nName: cozy-runtime\nVersion: 0.0.1\n")
    source = Path(cozy_runtime.__file__).parents[1]
    (site / "zz_worker_checkout.pth").write_text(f"{source}\n{sysconfig.get_path('purelib')}\n")
    return venv / "bin" / "python"


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_executor_handshake_accepts_compatible_protocol_with_other_package_metadata() -> None:
    with _workspace() as root:
        worker = Worker(
            _config(root / "home"),
            WorkerOptions(root=root / "worker", devices="0"),
            InMemoryControlHost(),
        )
        try:
            lane = lanes.DeviceLane("lane-0", (0,), "0", worker_pid=os.getpid())
            same = worker.supervision.spawn(imposed=worker.imposed(lane))
            assert same.hello["runtime_version"] == cozy_runtime.__version__
            worker.supervision.retire_current(same, "handshake proof")
            worker.supervision.python = str(_shadowing_python(root))
            compatible = worker.supervision.spawn(imposed=worker.imposed(lane))
            assert compatible.hello["runtime_version"] == "0.0.1"
            assert compatible.hello["executor_protocol_revision"] == 1
            worker.supervision.retire_current(compatible, "compatible protocol proof")
        finally:
            worker.supervision.close()
