"""Bounded CPU composition uses the same supervised job slots and lane consumers."""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

from cozy_runtime.author._calls import MAX_ACTIVE_CALLS
from cozy_runtime.internal.worker.child import ExecutorSupervision
from cozy_runtime.internal.worker.job_slots import JobSlot
from cozy_runtime.internal.worker.lanes import DeviceLane
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .session import Worker


def key(request: str) -> str:
    return "cpu-" + hashlib.sha256(request.encode()).hexdigest()[:16]


def ensure(worker: Worker, request: str, directive: pb.JobDirective, binding: JobBinding) -> bool:
    """Return false for ordinary model/weight/GPU jobs; CPU tasks get a bounded own lane."""
    declaration = worker.engine._job_declaration(binding, binding.job_descriptor_id)
    if (
        directive.device_count
        or directive.resource_caps.device_required
        or directive.resource_caps.max_device_memory_bytes
        or declaration.get("models")
        or declaration.get("weights_outputs")
    ):
        return False
    slot_key = key(request)
    old = worker.job_slots.get(slot_key)
    if old is not None:
        if old.binding != binding or old.directive != directive:
            raise WorkspaceRefusal("CPU invocation changed its admitted slot")
        if old.ready():
            return True
        release(worker, request)
    if len(worker.lanes.composition) >= MAX_ACTIVE_CALLS:
        raise WorkspaceRefusal("CPU composition process capacity is exhausted")
    # Host RAM is never an admission gate: a CPU task that outgrows memory pages to disk
    # like any other process, and its declared caps are sizing facts, not refusals.
    lane = DeviceLane(slot_key, (), "", worker_pid=os.getpid())
    supervision = ExecutorSupervision(
        root=worker.root / "placements" / slot_key,
        python=binding.python,
        base_env=worker.config.child_base_env,
        cozy_home=worker.config.cozy_home,
        executor_uid=worker._slot_executor_uid(slot_key),
        executor_gid=worker.options.executor_gid,
        jit_pod_scope=f"{worker.fence.worker_boot_id}-{slot_key}",
        kernel_cache=worker.config.kernel_cache,
    )
    supervision.use_environment(binding.python, binding.installation_id)
    supervision.on_change = lambda executor: lane.row("").ledger.begin_generation(
        executor.pid if executor else 0
    )
    supervision.on_lane_failure = worker.fail_machine
    copied = pb.JobDirective()
    copied.CopyFrom(directive)
    worker.job_slots[slot_key] = JobSlot(copied, binding, supervision, lane)
    worker.lanes.composition[slot_key] = lane
    worker.lanes.by_id[slot_key] = lane
    try:
        supervision.sweep_orphans()
        executor = supervision.spawn(imposed=worker.imposed(lane))
        executor.reserved_for_job = True
        worker.jobs[binding.job_descriptor_id] = binding
        worker.engine.jobs = worker.jobs
        if worker.accepted.mode != "serving":
            worker.accepted.mode = "job"
        worker.set_job_ready(True)
        lane.settled.set()
        worker._sync_lane_consumers()
        return True
    except BaseException:
        release(worker, request)
        raise


def release(worker: Worker, request: str, *, ordinal: int | None = None) -> None:
    with worker.control_lock:
        if ordinal is not None and any(
            attempt.request_id == request
            and attempt.attempt > ordinal
            and attempt.state != "closed"
            for attempt in worker.engine.history.values()
        ):
            # Native finalization of an older attempt may finish after its retry was admitted.
            return
        slot_key = key(request)
        slot = worker.job_slots.get(slot_key)
        if slot is None:
            return
        slot.supervision.close()
        worker.job_slots.pop(slot_key, None)
        worker.lanes.composition.pop(slot_key, None)
        worker.lanes.by_id.pop(slot_key, None)
        worker._sync_lane_consumers()
