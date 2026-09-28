"""A new worker waits for the previous generation to release its root (run 1486)."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest

from cozy_runtime.internal.worker.child import ExecutorGone, ExecutorSupervision

HOLD = """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
print("held", flush=True)
sys.stdin.read()
"""


def supervision(tmp_path: Path) -> ExecutorSupervision:
    return ExecutorSupervision(
        root=tmp_path / "worker",
        python=sys.executable,
        base_env=(),
        cozy_home=tmp_path / "home",
        executor_uid=-1,
        executor_gid=-1,
    )


def test_a_new_worker_waits_for_the_previous_generation_to_release_its_root(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "worker").mkdir()
    previous = subprocess.Popen(
        [sys.executable, "-c", HOLD, str(tmp_path / "worker" / "worker.lock")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert previous.stdout is not None and previous.stdin is not None
    assert previous.stdout.readline().strip() == "held"
    started: list[ExecutorSupervision] = []
    booting = threading.Thread(target=lambda: started.append(supervision(tmp_path)))
    booting.start()
    booting.join(0.5)
    assert booting.is_alive() and not started  # waiting, not refused
    previous.stdin.close()  # the previous generation exits
    previous.wait()
    booting.join()
    assert started, "the new worker did not take its root once the previous one released it"
    assert f"waiting for the previous worker (pid {previous.pid})" in capfd.readouterr().err
    # In one process a root is still supervised once: the second is refused, never waited on.
    with pytest.raises(ExecutorGone, match="already supervised"):
        supervision(tmp_path)
    started[0].close()
