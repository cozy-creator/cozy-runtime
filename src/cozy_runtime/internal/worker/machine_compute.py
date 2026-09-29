"""Owned-machine idle compute policy, enforced beside Runtime's durable admission.

The existing machine idle grace is fifteen minutes. It releases idle executor scopes,
never the resident coordinator or durable work. Rentals may separately be released by
their machine agent. Observer attachment and retained products are not runnable demand.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from cozy_runtime.protocol import worker_pb2 as pb

from . import activity

if TYPE_CHECKING:
    from .session import Worker

IDLE_SECONDS = 900


def park_if_idle(worker: Worker) -> bool:
    """Retire only idle compute after grace, serialized with root intake and preparation."""
    if not worker.options.sole_supervisor or worker.stop.is_set():
        return False
    # Package/model preparation takes this lock before the control lock too. Holding
    # both prevents an idle verdict from racing an accepted request or fresh preparation.
    if not worker.preparation_lock.acquire(blocking=False):
        worker.compute_idle_since = time.monotonic()
        return False
    try:
        with worker.control_lock:
            now = time.monotonic()
            active = (
                # Our own guard holds preparation_lock; every other hold is real demand.
                any(item != "preparation_lock" for item in activity.holding(worker))
                or any(
                    attempt.state in (*worker.engine.PRE_RELEASE, "released")
                    for attempt in worker.engine.history.values()
                )
                or (
                    worker.executions is not None
                    and worker.executions.active(worker.fence.record_owner_id)
                )
            )
            if active:
                worker.compute_idle_since = now
                return False
            if now - worker.compute_idle_since < IDLE_SECONDS:
                return False
            worker.prespawns.close()
            worker._apply_desired_state(
                pb.DesiredWorkerState(
                    revision=worker.accepted.accepted_desired_state_revision + 1,
                    wire_minor=worker.fence.wire_minor,
                )
            )
            for slot in worker.warm_cpu.values():
                slot.supervision.close()
            worker.warm_cpu.clear()
            for name, slot in list(worker.job_slots.items()):
                if name.startswith("cpu-"):
                    slot.supervision.close()
                    worker.job_slots.pop(name, None)
                    worker.lanes.composition.pop(name, None)
                    worker.lanes.by_id.pop(name, None)
            if worker.supervision.current is not None or any(
                hosted.supervision.current is not None for hosted in worker.hosted.values()
            ):
                worker.note("compute_park_refused", "executor reclamation is not yet proven")
                return False
            worker._dead_replicas.clear()
            worker.compute_idle_since = now
            worker.note(
                "compute_parked", "idle executor scopes released; coordinator remains ready"
            )
            return True
    finally:
        worker.preparation_lock.release()
