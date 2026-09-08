"""Canceled real gRPC watches release RPC capacity without a new Claim."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cozy_runtime.internal.worker.control import WatchFanout


def test_canceled_watchers_leave_runtime_unary_services_available() -> None:
    child = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "testdata/progress_cancellation.py")],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "Exception in thread control-stream" not in child.stderr
    rows = [json.loads(line) for line in child.stdout.splitlines() if line.startswith("{")]
    assert rows[0]["phase"] == "fixed" and rows[0]["remaining"] == 0
    assert rows[1]["phase"] == "recovered" and not rows[1]["worker_stopped"]


def test_already_ended_watch_leaves_no_fanout_registration() -> None:
    fanout = WatchFanout(2)
    watch = fanout.open("request", 1)
    watch.end()
    assert list(watch.frames(fanout)) == []
    assert not fanout.watchers
