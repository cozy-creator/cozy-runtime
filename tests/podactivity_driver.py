"""The WORKER half of `test_pod_activity_file`, run as its own process.

The pod supervisor reads the activity file from outside the worker's process tree — it is the
worker's PARENT — and that placement is not incidental. `activity.meter` folds in every
thread of the worker and every descendant of it, and the worker arms itself as a child
subreaper, so anything an in-process arm spawns to watch the file lands back inside the tree
and its own reads show up as the worker's work. A wedge observed from in there never looks
still. So the arm puts the worker where the pod puts it: below the observer.

argv: <root> <activity-path> <mode>, where mode is `plain` or `wedge`. One JSON line goes to
stdout at the end — the run's terminal, and for `wedge` the publisher's fresh-sample
boundary after confirmed SIGSTOP and its tick at SIGCONT. A reading already in progress
when the executor stops is outside the frozen window.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from cozy_runtime.internal import child_env, package_interface
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.local import (
    LocalRecordOwner,
    LocalRequest,
    run_slice,
)
from cozy_runtime.internal.worker import session
from cozy_runtime.internal.worker.session import Worker, WorkerOptions

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "examples" / "marco-polo"
#: The reporter tick this arm runs at, so a short run publishes many samples. It is
#: RESOLUTION, not contract — the rule under test is the same at any cadence.
TICK = 0.05
#: How long the executor is held stopped, in ticks.
FROZEN_TICKS = 40


def main(argv: list[str]) -> int:
    root, path, mode = Path(argv[0]), Path(argv[1]), argv[2]
    session.REPORT_SECONDS = TICK
    home = root / "home"
    config = RuntimeConfig(
        cozy_home=home,
        credentials=Credentials(),
        child_base_env=tuple(
            sorted(
                (k, v)
                for k, v in os.environ.items()
                if not child_env.erased(k) and k != "COZY_BOOTSTRAP_CREDENTIAL"
            )
        ),
    )
    found = discover(PACKAGE)
    body = package_interface.build(found)
    workspace = root / ".run"
    workspace.mkdir(parents=True, exist_ok=True)
    installation_id = "local-" + uuid.uuid4().hex
    request = LocalRequest(
        entrypoint="marco",
        payload={"message": "marco"},
        outputs=(),
        request_id="pod-activity",
        kind="entrypoint",
    )
    window: list[int] = []
    hook = _freeze_on_entry(window) if mode == "wedge" else None
    with contextlib.chdir(PACKAGE), contextlib.redirect_stdout(sys.stderr):
        outcome = run_slice(
            config,
            request,
            workspace=workspace,
            package_installation_id=installation_id,
            package_interface_document=body,
            project=PACKAGE,
            development_tensorfs_root=home / "tensorfs",
            options=WorkerOptions(
                root=workspace / "worker",
                release_id=installation_id,
                activity_path=path,
            ),
            record_owner_hook=hook,
        )
    print(
        json.dumps(
            {
                "ended": outcome.ended,
                "error": outcome.error,
                "result": outcome.result,
                "window": window,
            }
        )
    )
    return 0


def _freeze_on_entry(window: list[int]) -> Any:
    """SIGSTOP the executor the instant the attempt ENTERS the device: the attempt is
    accepted, the worker is fine, and nothing is moving — a process alive, owning its scope,
    burning nothing. SIGCONT afterwards, so the run still reaches its terminal and the arm
    proves the stopped worker was RESUMABLE and not merely broken."""
    executors: list[int] = []

    def hook(owner: LocalRecordOwner, worker: Worker) -> None:
        spawn = worker.supervision.spawn

        def remember(**kwargs: Any) -> Any:
            executor = spawn(**kwargs)
            executors.append(executor.pid)
            return executor

        worker.supervision.spawn = remember  # type: ignore[method-assign]
        enter = worker.enter

        def resume_after(pid: int) -> None:
            # This fixture's clock-only helper is not the subject's work, just like
            # the production reporter/watchdog threads excluded from the meter.
            tid = threading.get_native_id()
            worker.pollers.add(tid)
            try:
                time.sleep(FROZEN_TICKS * TICK)
                assert worker.activity_file is not None
                window.append(worker.activity_file.sequence)
                os.kill(pid, signal.SIGCONT)
            finally:
                worker.pollers.discard(tid)

        def stop_after_entry(attempt: Any) -> bool:
            accepted = enter(attempt)
            if accepted and executors and not window:
                os.kill(executors[0], signal.SIGSTOP)
                stopped = os.waitid(os.P_PID, executors[0], os.WSTOPPED | os.WEXITED | os.WNOWAIT)
                assert stopped is not None and stopped.si_code == os.CLD_STOPPED
                assert worker.activity_file is not None
                # meter() runs before publish() increments sequence. The next
                # published tick may therefore carry a pre-stop reading.
                window.append(worker.activity_file.sequence + 1)
                threading.Thread(
                    target=resume_after, args=(executors[0],), daemon=True, name="resume"
                ).start()
            return accepted

        worker.enter = stop_after_entry  # type: ignore[method-assign]

    return hook


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
