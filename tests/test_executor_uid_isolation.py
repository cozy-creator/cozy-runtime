"""The pod's real root-to-executor UID boundary, including actual rank transport."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from cozy_runtime.internal.worker.child import ExecutorSupervision


@pytest.mark.skipif(os.geteuid() != 0, reason="requires root to launch the pod executor UID")
@pytest.mark.parametrize("degree", [2, 4])
def test_two_requests_under_pod_executor_uid(degree: int) -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory(prefix="cozy-uid-") as directory:
        root = Path(directory)
        root.chmod(0o755)
        worker = root / "worker"
        requests = root / "requests"
        requests.mkdir(mode=0o700)
        os.chown(requests, 65533, 65533)
        supervision = ExecutorSupervision(
            root=worker,
            python=sys.executable,
            base_env=tuple(os.environ.items()),
            cozy_home=root,
            executor_uid=65533,
            executor_gid=65533,
        )
        (worker / "private").write_text("worker-owned record")
        (worker / "private").chmod(0o600)
        try:
            assert worker.stat().st_uid == 0 and worker.stat().st_mode & 0o777 == 0o710
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / "testdata" / "uid_isolation.py"),
                    "--root",
                    str(worker),
                    "--requests",
                    str(requests),
                    "--world",
                    str(degree),
                ],
                user=65533,
                group=65533,
                extra_groups=[],
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            assert result.returncode == 0, result.stdout
            assert '"requests": 2' in result.stdout
            # Provisioned for an older executor's `Model.warm` spool: still the executor's.
            warm = worker / "warm"
            assert warm.stat().st_uid == 65533 and warm.stat().st_mode & 0o777 == 0o700
            assert worker.stat().st_mode & 0o777 == 0o710
        finally:
            supervision.close()
