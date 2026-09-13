"""Runtime lifecycle facts for the fixed Host that supervises this worker.

The reporter publishes observer attachment, active attempts and required execution
retention independently. Accepted work and uncollected results remain owned when
no client is connected; a quiet progress meter does not relinquish that retention.
The Host still enforces its external rental allowance and shutdown authority.

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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cozy_runtime.internal.worker.session import Worker

#: The document's shape. The pod supervisor parses these exact keys
#: (tensorhub `pod-supervisor/activity.go`); a change here is a change there, and the
#: fixture in each repo is what catches a drift in one of them.
VERSION = 1


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

    def publish(
        self,
        *,
        owner_attached: bool,
        attempts_in_flight: int,
        meter: str | None,
        execution_retention_required: bool = False,
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
            "execution_retention_required": execution_retention_required,
            "meter": meter,
            "still_factor": liveness.STILL_FACTOR,
            "still_floor_seconds": liveness.STILL_FACTOR * liveness.noise_floor(),
        }
        payload = (json.dumps(document, sort_keys=True) + "\n").encode()
        staged = self.path.parent / f".activity-{os.getpid()}"
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


def meter(worker: Worker) -> str | None:
    """One reading of everything this worker is doing, or None when it cannot be read.

    The same composite `local.run_slice`'s readiness watch folds for a boot, widened to the
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
