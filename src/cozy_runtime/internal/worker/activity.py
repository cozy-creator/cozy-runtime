"""Runtime lifecycle facts for the fixed Host that supervises this worker.

The reporter separates active work from observer attachment and result retention.
Only work in progress protects a rental from idle release. Retained results,
paused work and connected clients do not renew its idle deadline.

The versioned file is atomically replaced in the Host's bootstrap directory. Its
sequence identifies a fresh reporter sample. The opaque meter and timing fields
remain available to compatible observers; they are not execution or custody state.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cozy_runtime.internal import liveness, proctree
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cozy_runtime.internal.worker.session import Worker

#: The document's shape. The pod supervisor parses these exact keys
#: (tensorhub `pod-supervisor/activity.go`); a change here is a change there, and the
#: fixture in each repo is what catches a drift in one of them.
VERSION = 2
MAX_HOLDING, MAX_REASON = 12, 96


class ActivityFile:
    """The publisher. One atomic replace per tick, no fsync — `/run` is tmpfs and the
    reader is a live process on the same machine, so durability buys nothing and a
    per-tick fsync is a syscall for no one.

    `owner` hands each published file to the supervisor's uid, the same handoff
    `readiness.publish` makes for the receipt: this worker is root and the supervisor has
    already dropped to the request-plane uid by the time it reads.
    """

    def __init__(self, path: Path, *, owner: tuple[int, int] | None = None) -> None:
        self.path = path
        self.owner = owner
        self.sequence = 0
        # The reporter and a submission's strict publish share one staged name; a second
        # writer reopening the first's 0400 file is EACCES, which refused the submission.
        self._lock = threading.Lock()

    def publish(
        self,
        *,
        owner_attached: bool,
        attempts_in_flight: int,
        meter: str | None,
        active_work: bool,
        execution_retention_required: bool = False,
        holding: list[str] | None = None,
    ) -> None:
        with self._lock:
            self._publish(
                owner_attached,
                attempts_in_flight,
                meter,
                active_work,
                execution_retention_required,
                holding or [],
            )

    def _publish(
        self,
        owner_attached: bool,
        attempts_in_flight: int,
        meter: str | None,
        active_work: bool,
        execution_retention_required: bool,
        holding: list[str],
    ) -> None:
        self.sequence += 1
        document: dict[str, Any] = {
            "version": VERSION,
            # The tick's own count. It says the reporter RAN, which is the one thing the
            # facts below cannot say about themselves: a worker wedged hard enough to stop
            # reporting keeps claiming an attached owner forever, and that claim is exactly
            # what used to hold a dead pod open.
            "sequence": self.sequence,
            "owner_attached": owner_attached,
            "attempts_in_flight": attempts_in_flight,
            "active_work": active_work,
            # What keeps the machine from idling, for whoever asks why it has not: bounded so
            # the document stays inside the supervisor's 4 KiB read.
            "holding": [reason[:MAX_REASON] for reason in holding[:MAX_HOLDING]],
            "execution_retention_required": execution_retention_required,
            "meter": meter,
            "still_factor": liveness.STILL_FACTOR,
            "still_floor_seconds": liveness.STILL_FACTOR * liveness.noise_floor(),
        }
        payload = (json.dumps(document, sort_keys=True) + "\n").encode()
        staged = self.path.parent / f".activity-{os.getpid()}"
        with contextlib.suppress(FileNotFoundError):
            os.unlink(staged)  # a crashed predecessor's 0400 file under a reused pod pid
        descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o400)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
            if self.owner is not None:
                os.chown(staged, self.owner[0], self.owner[1])
            os.replace(staged, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(staged)
            raise


def holding(worker: Worker) -> list[str]:
    """What this machine is doing that keeps it from idling: a supervised unit, a model
    preparation under way, the preparation lock, an activating placement. Every open
    execution has a live unit unless the worker FAILED, and a FAILED worker failed them all,
    so it holds nothing. A queued preparation holds nothing until it starts.

    Do not take the control lock: model preparation can hold it for minutes while
    the reporter must keep telling the supervisor that preparation is active.
    """
    if worker.phase == pb.WorkerPhase.WORKER_PHASE_FAILED:
        return []
    held = [f"unit:{key}" for key in worker.supervisor.keys("")]
    calls = worker.machine_calls
    if calls is not None:
        held += [f"preparation:{key}" for key in calls.serving.preparations.started()]
    if worker.preparation_lock.locked():
        held.append("preparation_lock")
    # A reclaimed replica stays ACTIVATING until its next grant: demand eligibility, not
    # work. Its actual revival is held by `_activating` and its supervised unit.
    placements = [worker.placement, *(h.placement for h in tuple(worker.hosted.values()))]
    held += [
        f"activating:{p.placement_id}"
        for p in placements
        if p
        and p.serving == pb.SERVING_STATE_ACTIVATING
        and (p.placement_id not in worker._dead_replicas or p.placement_id in worker._activating)
    ]
    return held


def active_work(worker: Worker) -> bool:
    return bool(holding(worker))


def meter(worker: Worker) -> str | None:
    """One reading of everything this worker is doing, or None when it cannot be read.

    A boot's readiness composite, widened to the
    whole worker: every thread's `progress_burn` except the lanes that wake on a CLOCK
    (`Worker.pollers` — the reporter that writes this file among them, so the writer's own
    cost can never look like work), PLUS every descendant process, PLUS the observation
    cursors' positions and ends, PLUS each held attempt's state. A thread or child ending
    lowers the sum and an ending is progress, which is why the rule reading this compares
    for CHANGE and not for growth.

    An unreadable reading is None, and an unreadable meter decides nothing — `progress_burn`'s
    own rule, restated across the file.
    """
    me = os.getpid()
    quiet = worker.pollers | {threading.get_native_id()}
    try:
        threads = [int(name) for name in os.listdir(f"/proc/{me}/task")]
        working = sum(liveness.burn(me, tid) or 0 for tid in threads if tid not in quiet)
        below = sum(liveness.burn(pid) or 0 for pid in proctree.descendants(me))
        moving = [
            [cursor.subject, cursor.position, cursor.ended]
            for cursor in sorted(worker.monitor.cursors.values(), key=lambda c: c.subject)
        ]
        held = sorted(
            [attempt.request_id, attempt.attempt, attempt.state]
            for attempt in list(worker.engine.history.values())
        )
    except (OSError, proctree.ProcessTreeUnsupported, RuntimeError, ValueError):
        return None
    fingerprint = hashlib.sha256(json.dumps([moving, held]).encode()).hexdigest()[:16]
    return f"{working + below}:{fingerprint}"
