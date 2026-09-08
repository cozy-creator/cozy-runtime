"""The pod supervisor's window into this worker, proven on the real worker (th-142).

A pod's supervisor is the only process that can end the rental — the pod cannot delete
itself — and after th-142 it is no longer in the middle of the control stream, so what it
releases the pod on is the file this worker publishes. These arms run the REAL `run_slice`
path (the same in-process RecordOwner, the same worker, a real spawned executor over
`examples/marco-polo`) with the pod's `activity_path` set, in its OWN PROCESS, and read the
file the reporter actually wrote from the process above it — which is where the supervisor
reads it from, and the only place an observer does not measure itself as the worker's work.

* the facts track the run: an unclaimed worker holds nothing, an owner is attached while it
  runs, and the meter CHANGES sample to sample — which is what a slow attempt looks like,
  and why one is never mistaken for a wedge;
* SIGSTOP the executor the instant the attempt is admitted and the file keeps claiming an
  attempt in flight while the meter goes STILL — the exact signature tensorhub's
  `activityWatch` calls wedged, and the case today's `OwnerAttached() && InFlightAttempts()`
  signal reports as a pod in use, forever.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import podactivity_driver
from cozy_runtime.internal import child_env, liveness
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker import activity
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from test_end_to_end import NO_EXECUTOR

DRIVER = Path(podactivity_driver.__file__).resolve()
#: The document the pod supervisor parses, byte for byte, in BOTH repositories. tensorhub
#: keeps the same bytes at `pod-supervisor/testdata/activity.json` and parses them with the
#: production type; changing the shape on one side and not the other is what these two
#: copies exist to catch.
FIXTURE = Path(__file__).resolve().parent / "testdata" / "pod-activity.json"

needs_executor = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")


@contextlib.contextmanager
def _short_tmp() -> Iterator[Path]:
    """A short root: the executor's control socket lives under it and `sun_path` is 108 B."""
    import shutil

    root = Path(tempfile.mkdtemp(prefix="cz-act.", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _config(home: Path) -> RuntimeConfig:
    return RuntimeConfig(
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


def _drive(root: Path, mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the worker below this process and poll its file the way the supervisor does."""
    path = root / "activity"
    every = podactivity_driver.TICK / 2
    child = subprocess.Popen(
        [sys.executable, str(DRIVER), str(root), str(path), mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONPATH": str(DRIVER.parent)},
    )
    samples: list[dict[str, Any]] = []
    try:
        while child.poll() is None:
            with contextlib.suppress(OSError, ValueError):
                document = json.loads(path.read_bytes())
                if not samples or document != samples[-1]:
                    samples.append(document)
            time.sleep(every)
        stdout, _ = child.communicate(timeout=60)
    finally:
        if child.poll() is None:  # pragma: no cover - only on a harness failure
            child.kill()
            child.wait()
    summary = json.loads(stdout.decode().strip().splitlines()[-1])
    return summary, samples


# --------------------------------------------------------------- the file, on its own


def test_the_published_document_is_the_shape_the_supervisor_parses() -> None:
    """A real `Worker`'s own publish carries exactly the fixture's keys and types, and the
    patience it declares is this runtime's own derivation — not a number chosen in Go."""
    with _short_tmp() as root:
        worker = Worker(
            _config(root / "home"),
            WorkerOptions(root=root / "worker", activity_path=root / "activity"),
            InMemoryControlHost(),
        )
        worker.publish_activity()
        published = json.loads((root / "activity").read_bytes())
    fixture = json.loads(FIXTURE.read_bytes())
    assert sorted(published) == sorted(fixture), (sorted(published), sorted(fixture))
    for key, sample in fixture.items():
        assert isinstance(published[key], type(sample)), (key, published[key], sample)
    assert published["version"] == activity.VERSION == fixture["version"]
    assert published["still_factor"] == liveness.STILL_FACTOR == fixture["still_factor"]
    assert (
        published["still_floor_seconds"]
        == liveness.STILL_FACTOR * liveness.noise_floor()
        == fixture["still_floor_seconds"]
    )
    # An unclaimed worker with nothing running is a pod nobody is holding.
    assert published["owner_attached"] is False
    assert published["attempts_in_flight"] == 0
    assert published["sequence"] == 1


def test_each_publish_replaces_the_file_atomically_and_advances_the_tick() -> None:
    """The supervisor polls four times a second against a writer it never coordinates with:
    every read must be a whole document, and the tick must say the writer RAN."""
    with _short_tmp() as root:
        published = activity.ActivityFile(root / "activity")
        for expected in (1, 2, 3):
            published.publish(owner_attached=False, attempts_in_flight=0, meter="0:0")
            document = json.loads((root / "activity").read_bytes())
            assert document["sequence"] == expected
        assert [p.name for p in root.iterdir() if p.name.startswith(".activity")] == []


def test_an_unreadable_meter_is_absent_and_never_a_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """`progress_burn`'s rule carried across the file: a reading nobody could take is null,
    which the supervisor treats as deciding nothing — never a stand-in number, which would
    read as a subject that has stopped and release a pod that was working.

    The refusal is the one a `/proc` mounted `hidepid=2` gives a reader that does not own the
    process, raised at the site that mount refuses. `liveness.burn` beside it shows the same
    rule one layer down, unpatched: a pid that does not exist reads as None, never 0.
    """
    assert liveness.burn(2**22 - 1) is None
    with _short_tmp() as root:
        worker = Worker(
            _config(root / "home"),
            WorkerOptions(root=root / "worker"),
            InMemoryControlHost(),
        )
        assert activity.meter(worker) is not None

        def refuse(_path: str) -> list[str]:
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(os, "listdir", refuse)
        assert activity.meter(worker) is None


# ------------------------------------------------------- the file, over a real run


@needs_executor
def test_a_real_run_publishes_an_attached_owner_and_a_moving_meter() -> None:
    """The producing half of the pod's release decision, over the real path.

    While the run is under way the file says an owner is attached and its meter CHANGES from
    sample to sample — which is what the supervisor's rule reads as work, and why an attempt
    that is merely slow is never mistaken for one that has stopped.
    """
    with _short_tmp() as root:
        summary, samples = _drive(root, "plain")

    assert summary["ended"] == "terminal", (summary["ended"], summary["error"])
    assert summary["result"] == {"message": "polo"}
    assert samples, "the reporter published nothing over a whole real run"

    attached = [s for s in samples if s["owner_attached"]]
    assert attached, "no sample saw the local RecordOwner attached"

    # The meter moved: work, observed, not assumed.
    meters = [s["meter"] for s in attached]
    assert len(set(meters)) > 1, f"the meter never changed across {len(attached)} owned samples"
    assert all(m is not None for m in meters)

    # The tick is monotone over the whole run: the writer says it RAN, every time.
    ticks = [s["sequence"] for s in samples]
    assert ticks == sorted(set(ticks)), ticks


@needs_executor
def test_a_stopped_executor_leaves_an_attempt_in_flight_with_a_meter_that_has_stopped() -> None:
    """THE WEDGE, as the supervisor sees it.

    Over the window the executor is stopped, the file keeps saying `attempts_in_flight >= 1`
    while its meter stays IDENTICAL sample after sample. That is what tensorhub's
    `activityWatch` calls wedged, and what today's signal reports as a pod in use forever.
    The tick keeps advancing through it, so the arm also proves the two are distinguishable:
    this is a wedged ATTEMPT, not a wedged process.
    """
    with _short_tmp() as root:
        summary, samples = _drive(root, "wedge")

    window = summary["window"]
    assert len(window) == 2, "the attempt was never admitted, so nothing was stopped"
    assert summary["ended"] == "terminal", (summary["ended"], summary["error"])
    # The frozen window is the ticks published while the executor was stopped, from the
    # first that says an attempt is in flight — the next tick after acceptance.
    frozen = [s for s in samples if window[0] < s["sequence"] <= window[1]]
    held = list(itertools.dropwhile(lambda s: s["attempts_in_flight"] == 0, frozen))
    assert len(held) > 4, f"only {len(held)} of {len(frozen)} samples saw the stopped attempt"
    # The attempt is still claimed — the pod looks BUSY on the two facts alone.
    assert all(s["attempts_in_flight"] > 0 for s in held), [s["attempts_in_flight"] for s in held]
    # And the meter has stopped, which is the only thing that tells the two apart.
    assert len({s["meter"] for s in held}) == 1, sorted({s["meter"] for s in held})
    # The writer kept running.
    assert len({s["sequence"] for s in held}) > 1, "the reporter itself stopped ticking"
