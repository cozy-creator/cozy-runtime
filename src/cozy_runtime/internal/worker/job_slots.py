"""The two explicitly admitted private job roles; no independent scheduler."""

from __future__ import annotations

from dataclasses import dataclass

from cozy_runtime.internal.worker.child import ExecutorSupervision
from cozy_runtime.internal.worker.lanes import DeviceLane
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.protocol import worker_pb2 as pb


@dataclass(slots=True)
class JobSlot:
    directive: pb.JobDirective
    binding: JobBinding
    supervision: ExecutorSupervision
    lane: DeviceLane

    def ready(self) -> bool:
        executor = self.supervision.current
        return bool(executor is not None and not executor.poisoned and executor.alive())
