"""A real old Worker excludes its successor until its authority is drained."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

WORKER = """
import importlib.metadata,json,sys
from pathlib import Path
from cozy_runtime.internal.config import RuntimeConfig,Credentials
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker,WorkerOptions
root=Path(sys.argv[1])
worker=Worker(RuntimeConfig(cozy_home=root/'home',credentials=Credentials()),
    WorkerOptions(root=root/'worker',tensorfs_root=root/'store',no_warm=True,
                  worker_id='upgrade-proof',accelerator_backend='none'),InMemoryControlHost())
try:
    print('READY '+json.dumps({'runtime':importlib.metadata.version('cozy-runtime')}),flush=True)
    if sys.argv[2]=='hold':sys.stdin.readline()
finally:
    worker.shutdown()
"""


def test_legacy_runtime_lease_blocks_workspace_before_migration(tmp_path: Path) -> None:
    legacy_python = os.environ.get("COZY_LEGACY_RUNTIME_PYTHON")
    if not legacy_python:
        pytest.skip("set COZY_LEGACY_RUNTIME_PYTHON to the installed public Runtime0.3.0")
    old = subprocess.Popen(
        [legacy_python, "-c", WORKER, str(tmp_path), "hold"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert old.stdout is not None
        for line in old.stdout:
            if line.startswith("READY "):
                assert json.loads(line[6:])["runtime"] == "0.3.0"
                break
        else:
            assert old.stderr is not None
            raise AssertionError(old.stderr.read())
        refused = subprocess.run(
            [sys.executable, "-c", WORKER, str(tmp_path), "once"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert refused.returncode != 0
        assert "already supervised by another live worker" in refused.stderr
        assert "READY " not in refused.stdout
        # The successor cannot expose WorkspaceRPC or start staging the handoff
        # before acquiring the existing fixed-root Runtime lifetime lease.
        assert not (tmp_path / "store/.cozy-workspace").exists()
        assert old.poll() is None
        assert old.stdin is not None
        old.stdin.close()
        assert old.wait(timeout=30) == 0
        successor = subprocess.run(
            [sys.executable, "-c", WORKER, str(tmp_path), "once"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert successor.returncode == 0, successor.stdout + successor.stderr
        assert '"runtime": "0.4.0"' in successor.stdout
        assert (tmp_path / "store/.cozy-workspace").is_dir()
    finally:
        if old.poll() is None:
            old.kill()
            old.wait(timeout=10)
        for pipe in (old.stdin, old.stdout, old.stderr):
            if pipe is not None:
                pipe.close()
