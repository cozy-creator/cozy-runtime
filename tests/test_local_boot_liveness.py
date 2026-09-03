"""A local run ends on OBSERVED stillness, never on a readiness clock (xs-007 row 7).

`cozy run` used to carry a 1800 s readiness deadline: a worker still downloading or loading
a large model at minute thirty was killed and the person in front of it read a hang report
for a boot that was working. These arms run the REAL `local.run_slice` — the same in-process
RecordOwner, the same worker, a real spawned executor over `examples/marco-polo` — and
assert the two outcomes that replace that clock: a boot that keeps working completes however
long it takes, and a boot that has provably stopped is ended, promptly, with the measurement
that ended it.

The injection is the harness's, not the runtime's (decision #249): `record_owner_hook` hands
the live worker over before it starts, and the arms wrap `supervision.spawn` — the one call
the convergence lane makes on its way to readiness. The slow arm does real work on that
lane before spawning; the wedged arm stops the real executor with SIGSTOP the moment it has
dialled back, which is what a process stuck in the kernel looks like from outside: alive,
owning its device scope, burning nothing.

The sampling constants are turned down because they are RESOLUTION, not contract. The rule
under test is "still for longer than this boot's own longest pause, and never for less than
the floor derived from the pull ledger's", and it is the same rule at any cadence.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import signal
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import child_env, liveness, package_interface, pull
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.local import (
    LocalRecordOwner,
    LocalRequest,
    SliceOutcome,
    development_source_digest,
    run_slice,
)
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from test_end_to_end import NO_EXECUTOR

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "examples" / "marco-polo"

needs_executor = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")

#: The resolution the arms run at. The pull ledger's floor becomes 0.5 s and the readiness
#: watch's derived floor `STILL_FACTOR` times that.
SAMPLE = 0.1
STALL_SAMPLES = 5


@pytest.fixture(autouse=True)
def fine_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(liveness, "SAMPLE_SECONDS", SAMPLE)
    monkeypatch.setattr(pull, "STALL_SAMPLE_SECONDS", SAMPLE)
    monkeypatch.setattr(pull, "STALL_SAMPLES", STALL_SAMPLES)


def pull_floor() -> float:
    return pull.STALL_SAMPLES * pull.STALL_SAMPLE_SECONDS


def boot_floor() -> float:
    return liveness.STILL_FACTOR * pull_floor()


@contextlib.contextmanager
def _short_tmp() -> Iterator[Path]:
    """A short root: the executor's control socket lives under it and `sun_path` is 108 B."""
    import shutil
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="cz.", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _run(root: Path, hook: Any) -> tuple[SliceOutcome, float]:
    """One `marco` request through the whole real path, exactly as `cozy run` drives it."""
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
    workspace.mkdir(parents=True)
    digest = development_source_digest(PACKAGE)
    request = LocalRequest(
        entrypoint="marco",
        payload={"message": "marco"},
        outputs=(),
        request_id="cli-marco",
        kind="entrypoint",
    )
    started = time.monotonic()
    with contextlib.chdir(PACKAGE), contextlib.redirect_stdout(sys.stderr):
        outcome = run_slice(
            config,
            request,
            workspace=workspace,
            package_code_digest=digest,
            package_interface_document=body,
            found=found,
            development_tensorfs_root=home / "tensorfs",
            options=WorkerOptions(root=workspace / "worker", release_id=digest, no_warm=True),
            record_owner_hook=hook,
        )
    return outcome, time.monotonic() - started


@needs_executor
def test_a_boot_that_keeps_working_is_never_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    """The slow arm: a boot far longer than the old cap's equivalent completes.

    The old 1800 s cap was 60 times the pull ledger's 30 s floor. At this resolution that is
    60 times 0.5 s, so the convergence lane is made to work — hash, pause, hash — for that
    long before it spawns the executor: a boot that is moving the whole time and never, in
    any one pause, still for longer than the derived floor. The run must reach its terminal.
    """
    long_boot = 60 * pull_floor()

    def hook(owner: LocalRecordOwner, worker: Worker) -> None:
        spawn = worker.supervision.spawn

        def slow_spawn(**kwargs: Any) -> Any:
            digest = b"cold weights"
            until = time.monotonic() + long_boot
            while time.monotonic() < until:
                burst = time.monotonic() + 0.05
                while time.monotonic() < burst:
                    digest = hashlib.sha256(digest).digest()
                time.sleep(0.2)
            return spawn(**kwargs)

        monkeypatch.setattr(worker.supervision, "spawn", slow_spawn)

    with _short_tmp() as root:
        outcome, elapsed = _run(root, hook)

    assert outcome.ended == "terminal", (outcome.ended, outcome.error)
    assert outcome.result == {"message": "polo"}
    assert elapsed >= long_boot, f"the boot took {elapsed:.1f}s: the slow arm did not run"


@needs_executor
def test_a_boot_that_stops_working_is_ended_on_the_measurement() -> None:
    """The wedged arm: an executor that stops dead is observed, and the run ends on that.

    SIGSTOP the moment the executor has dialled back: the worker's `prepare` then blocks on
    the seam with nothing burning anywhere — no cursor moving, no thread working, no child
    working. The run must end `worker_wedged`, naming the open boot step and the measured
    stillness, after the derived floor and NOT on any larger clock; and the workspace's
    executor must be reclaimed on the way out, which is `shutdown` doing its real job over a
    stopped process.
    """
    stopped: list[int] = []

    def hook(owner: LocalRecordOwner, worker: Worker) -> None:
        spawn = worker.supervision.spawn

        def stop_after_spawn(**kwargs: Any) -> Any:
            executor = spawn(**kwargs)
            os.kill(executor.pid, signal.SIGSTOP)
            stopped.append(executor.pid)
            return executor

        worker.supervision.spawn = stop_after_spawn  # type: ignore[method-assign]

    with _short_tmp() as root:
        outcome, elapsed = _run(root, hook)

    assert stopped, "the executor was never spawned, so nothing was wedged"
    assert outcome.ended == "worker_wedged", (outcome.ended, outcome.error)
    assert "no measurable progress for" in outcome.error
    assert "load:prepare:marco" in outcome.error, outcome.error
    assert elapsed > boot_floor(), f"{elapsed:.1f}s: ended before the derived floor"
    # Promptly: within a few floors, which the rule explains. The old clock was 60 floors.
    assert elapsed < 4 * boot_floor(), f"{elapsed:.1f}s: the run waited on a clock"
    with pytest.raises(ProcessLookupError):
        os.kill(stopped[0], 0)
