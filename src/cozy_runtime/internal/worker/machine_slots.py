"""Bounded CPU composition uses the same supervised job slots and lane consumers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from cozy_runtime.author._calls import MAX_ACTIVE_CALLS
from cozy_runtime.internal import proctree
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


def free_memory() -> int:
    """Current host/cgroup availability already charges suspended parent processes."""
    rows = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(rows["MemAvailable"].split()[0]) * 1024
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if not line.startswith("0::"):
            continue
        relative = line.partition("0::")[2].lstrip("/")
        group = Path("/sys/fs/cgroup") / relative
        for directory in (group, *group.parents):
            if not directory.is_relative_to("/sys/fs/cgroup"):
                break
            maximum, current = directory / "memory.max", directory / "memory.current"
            if maximum.is_file() and current.is_file():
                cap = maximum.read_text().strip()
                if cap != "max":
                    available = min(available, int(cap) - int(current.read_text()))
    return max(0, available)


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
    remaining = 0
    for name, slot in worker.job_slots.items():
        if not name.startswith("cpu-"):
            continue
        executor = slot.supervision.current
        resident = proctree.rss_bytes(executor.pid) if executor is not None else 0
        if resident < 0:
            raise WorkspaceRefusal("CPU composition memory is unreadable")
        remaining += max(0, slot.directive.resource_caps.max_rss_bytes - resident)
    needed = directive.resource_caps.max_rss_bytes or proctree.rss_bytes(os.getpid())
    if needed <= 0 or needed + remaining > free_memory():
        raise WorkspaceRefusal("CPU composition has insufficient host memory")
    lane = DeviceLane(slot_key, (), "", worker_pid=os.getpid())
    supervision = ExecutorSupervision(
        root=worker.root / "placements" / slot_key,
        python=binding.python,
        base_env=worker.config.child_base_env,
        cozy_home=worker.config.cozy_home,
        executor_uid=worker._slot_executor_uid(slot_key),
        executor_gid=worker.options.executor_gid,
        jit_pod_scope=f"{worker.fence.worker_boot_id}-{slot_key}",
    )
    supervision.use_environment(binding.python, binding.environment_content_digest)
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
            # A delayed old acknowledgement cannot retire the new retry's slot.
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
