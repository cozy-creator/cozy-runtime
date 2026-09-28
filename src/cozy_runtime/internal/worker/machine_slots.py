"""Bounded CPU composition uses the same supervised job slots and lane consumers."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cozy_runtime.author._calls import MAX_ACTIVE_CALLS
from cozy_runtime.internal.worker.child import Executor, ExecutorSupervision
from cozy_runtime.internal.worker.job_slots import JobSlot
from cozy_runtime.internal.worker.lanes import DeviceLane
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .session import Worker


def key(request: str) -> str:
    return "cpu-" + hashlib.sha256(request.encode()).hexdigest()[:16]


def admits(directive: pb.JobDirective, declaration: Mapping[str, Any]) -> bool:
    """Whether a job takes the CPU slot. The device grant alone decides: the declaration's
    Models are derive-only Manifest views and its weights outputs are TensorFS writes."""
    del declaration  # no rule reads it
    return not (
        directive.device_count
        or directive.resource_caps.device_required
        or directive.resource_caps.max_device_memory_bytes
    )


def _warm_key(binding: JobBinding, directive: pb.JobDirective) -> tuple[str, str, bytes]:
    return (
        binding.installation_id,
        binding.job_descriptor_id,
        directive.SerializeToString(deterministic=True),
    )


def ensure(worker: Worker, request: str, directive: pb.JobDirective, binding: JobBinding) -> bool:
    """Return false for jobs granted a device; CPU tasks get a bounded own lane.

    The lane is the request's; its executor is the previous request's of the same job when
    one is idle, so a composer's imports are paid once per worker, not once per request.
    """
    declaration = worker.engine._job_declaration(binding, binding.job_descriptor_id)
    if not admits(directive, declaration):
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
    warm = worker.warm_cpu.pop(_warm_key(binding, directive), None)
    if warm is not None and not (warm.binding == binding and warm.ready()):
        warm.supervision.close()
        warm = None
    if warm is None and _memory_stalled():
        # A new process needs memory the idle ones are holding.
        for idle in list(worker.warm_cpu):
            worker.warm_cpu.pop(idle).supervision.close()
    if warm is not None:
        supervision = warm.supervision
    else:
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

    def begin(executor: Executor | None) -> None:
        lane.row("").ledger.begin_generation(executor.pid if executor else 0)

    supervision.on_change = begin
    supervision.on_lane_failure = worker.fail_machine
    copied = pb.JobDirective()
    copied.CopyFrom(directive)
    worker.job_slots[slot_key] = JobSlot(copied, binding, supervision, lane)
    worker.lanes.composition[slot_key] = lane
    worker.lanes.by_id[slot_key] = lane
    try:
        if warm is None:
            supervision.sweep_orphans()
            executor = supervision.spawn(imposed=worker.imposed(lane))
            executor.reserved_for_job = True
        else:
            begin(supervision.current)
        worker.jobs[binding.job_descriptor_id] = binding
        worker.engine.jobs = worker.jobs
        if worker.accepted.mode != "serving":
            worker.accepted.mode = "job"
        worker.set_job_ready(True)
        lane.settled.set()
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
        warm = _warm_key(slot.binding, slot.directive)
        if slot.ready() and warm not in worker.warm_cpu:
            # Idle, not closed: the next request of this job adopts its process.
            worker.warm_cpu[warm] = slot
        else:
            slot.supervision.close()
        worker.job_slots.pop(slot_key, None)
        worker.lanes.composition.pop(slot_key, None)
        worker.lanes.by_id.pop(slot_key, None)


def _memory_stalled() -> bool:
    """Whether the kernel measured every runnable task stalled on memory in its last 10 s."""
    try:
        full = Path("/proc/pressure/memory").read_text().splitlines()[1].split()
        return float(full[1].removeprefix("avg10=")) > 0
    except (OSError, IndexError, ValueError):
        return False  # No pressure measurement on this host: nothing says to vacate.


def forget(worker: Worker, installation_id: str) -> None:
    """Close the idle executors of an installation that is going away."""
    for warm in [key for key in worker.warm_cpu if key[0] == installation_id]:
        worker.warm_cpu.pop(warm).supervision.close()
