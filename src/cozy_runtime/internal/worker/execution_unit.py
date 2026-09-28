"""One machine execution, root or child, as its own supervised unit.

The unit runs its execution start to finish on one thread: a child's call phase, the queued
wait for its GPU grant, its attempt INLINE (staged beside the device, on the device, its post
phase) and its settlement into its parent. It blocks only in `wait_for`, woken by what it
waits on (its grant, a control, a preparation, its own deadline), and re-reads its row on
every wake. Its exception fails this execution and nothing else (`crashed`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from cozy_runtime.internal.worker import machine_lanes
from cozy_runtime.internal.worker.attempts import safe
from cozy_runtime.internal.worker.gpu_scheduler import Want, width_for
from cozy_runtime.internal.worker.supervisor import Unit, fault_text
from cozy_runtime.internal.worker.workspace_calls import CallFenced
from cozy_runtime.internal.worker.workspace_executions import (
    ExecutionChanged,
    ExecutionRow,
    StaleExecutionGeneration,
)
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from cozy_runtime.internal.worker.machine_calls import Calls
    from cozy_runtime.internal.worker.session import Worker

EXECUTION = "execution:"
#: A durable fence moved under this unit: it re-reads its row and acts again.
FENCES = (ExecutionChanged, StaleExecutionGeneration, CallFenced)


def key(request: str) -> str:
    return EXECUTION + request


class Execution:
    def __init__(self, worker: Worker, calls: Calls, owner: str, request: str) -> None:
        assert worker.executions is not None
        self.worker, self.calls, self.owner, self.request = worker, calls, owner, request
        self.executions = worker.executions
        #: set by a crash: the unit only ends the execution FAILED with this, then settles
        self.failure = ""
        self.crashes = 0
        self.root_open = False
        self.is_root: bool | None = None
        self.shape: machine_lanes.Shape | None = None
        self.shaped = 0

    # ------------------------------------------------------------------ supervision

    def run(self, unit: Unit) -> None:
        try:
            while not self.worker.stop.is_set():
                try:
                    if not self._step(unit):
                        return
                except FENCES as exc:
                    self.worker.note("execution", f"{self.request} re-reads: {exc}")
        finally:
            self._close_root()

    def crashed(self, exc: BaseException) -> bool:
        """Record this execution's own failure; run once more to settle it. A second crash
        while it is still open leaves it with no owner, which the supervisor escalates."""
        self.crashes += 1
        row = self.executions.row(self.owner, self.request)
        if row is None:
            self.calls.fail_call(self.owner, self.request, exc)
            return False
        if row.terminal:
            return self.crashes == 1
        if self.crashes > 1:
            raise RuntimeError(f"{self.request} could not be ended: {fault_text(exc)}") from exc
        self.failure = safe(fault_text(exc), 4096)
        return True

    # ------------------------------------------------------------------ the phases

    def _step(self, unit: Unit) -> bool:
        """One pass over this execution's durable state. False when the unit is done."""
        row = self.executions.row(self.owner, self.request)
        if row is None:  # a child call before its execution exists
            return self.calls.call(unit, self.owner, self.request)
        self._track_root(row)
        if row.terminal:
            self._settle(unit, row)
            return False
        if not self.executions.assigned_here(
            self.owner, self.request, row.ordinal, self.worker.fence.worker_boot_id
        ):
            return False  # another live Runtime process holds this attempt
        held = self.worker.engine.history.get((self.request, row.ordinal))
        if held is not None and held.state not in ("outcome", "closed"):
            unit.wait()  # its attempt ends through the thread that holds it
            return True
        if row.dispatched:
            # Dispatched, and no thread here holds it: its outcome is recorded and only
            # unprojected, or its attempt was lost (it failed), never run again.
            self.executions.reconcile(self.owner, self.request)
            now = self.executions.row(self.owner, self.request)
            if now is not None and not now.terminal:
                self.worker.fail_execution(
                    self.owner, self.request, "the dispatched attempt is held by no one"
                )
            return True
        self.failure = self.failure or self.worker.failure_reason
        if self.failure:
            self.worker.fail_execution(self.owner, self.request, self.failure)
            return True
        if row.desired != "run":
            self.executions.stop_queued(self.owner, self.request)
            return True
        if action := self.calls.ancestor_stop(self.owner, self.request):
            # Its parent stopped (a control, a failure, a restart): this child follows it.
            self.executions.control(
                self.owner,
                self.request,
                f"ancestor-{action}.{row.generation}",
                row.generation,
                action,
            )
            return True
        self._queued(unit, row)
        return True

    def _queued(self, unit: Unit, row: ExecutionRow) -> None:
        offered = row.attempt_offer()
        spec = documents.parse(offered.invocation_spec_canonical_bytes, pb.InvocationSpec)
        due = _due(spec.deadline_unix_ms)
        if due is not None and due <= time.monotonic():
            self.executions.stop_queued(self.owner, self.request, deadline=True)
            return
        shape = self._shaped(row)
        if shape.refusal:
            self._refuse(offered, shape.refusal.partition(":")[0], shape.refusal)
            return
        grant_key = f"{self.request}#{row.ordinal}"
        ordinals: tuple[int, ...] = ()
        attempt = None
        try:
            if shape.kind != "cpu":
                want = self._want(shape)
                if isinstance(want, str):
                    self._refuse(offered, want.partition(":")[0], want)
                    return
                broker = self.worker.gpu
                broker.want(grant_key, want)
                unit.wait_for(
                    lambda: broker.granted(grant_key) is not None or self._moved(row), due
                )
                granted = broker.granted(grant_key)
                if granted is None:
                    return  # moved or past its deadline: the next pass decides
                ordinals = granted
            attempt = self.worker.dispatch_machine(
                unit, offered, ordinals, lambda: self._moved(row)
            )
            if attempt is not None:
                self.worker.run_attempt(attempt)
        finally:
            if attempt is None and shape.kind != "cpu":
                self.worker.gpu.release(grant_key)

    def _settle(self, unit: Unit, row: ExecutionRow) -> None:
        """A terminal execution delivers to its parent, stops its children, and releases
        what it retains. The row is already terminal: a failure here is noted, never the
        worker's."""
        worker, owner, request = self.worker, self.owner, self.request
        steps: tuple[tuple[str, Callable[[], None]], ...] = (
            ("custody", lambda: self.calls.deliver(unit, owner, request, row)),
            ("children", lambda: worker.stop_children(owner, request, row)),
            ("retention", lambda: worker.release_retention(owner, request, row)),
        )
        for name, step in steps:
            try:
                step()
            except FENCES:
                raise
            except Exception as exc:
                worker.settle_fault(owner, request, name, exc)

    # ------------------------------------------------------------------ helpers

    def _track_root(self, row: ExecutionRow) -> None:
        """A root's GPU lease exists exactly while its row is open and wants to run."""
        open_now = row.desired == "run" and row.state in ("queued", "running")
        if open_now and not self.root_open and self._is_root():
            self.worker.gpu.open_root(self.request, row.priority)
            self.root_open = True
        elif not open_now:
            self._close_root()

    def _close_root(self) -> None:
        if self.root_open:
            self.root_open = False
            self.worker.gpu.close_root(self.request)

    def _is_root(self) -> bool:
        if self.is_root is None:
            self.is_root = self.executions.scheduling_root(self.owner, self.request)[0] == (
                self.request
            )
        return self.is_root

    def _shaped(self, row: ExecutionRow) -> machine_lanes.Shape:
        if self.shape is None or self.shaped != row.ordinal:
            try:
                self.shape = machine_lanes.shape(self.worker, self.owner, self.request)
            except Exception as exc:  # only this execution is refused
                self.shape = machine_lanes.Shape(
                    self.request, "cpu", refusal=f"gpu_shape_unreadable: {fault_text(exc)}"
                )
            self.shaped = row.ordinal
        return self.shape

    def _want(self, shape: machine_lanes.Shape) -> Want | str:
        devices = len(self.worker.lanes.entries)
        if shape.kind != "serving":  # a device job is worker-wide
            return Want(shape.root, devices)
        readable = self.worker.readable_gpus()
        if refusal := machine_lanes.unrunnable(shape, readable):
            return refusal
        return Want(
            shape.root,
            width_for(shape.degrees, len(readable), shape.gpus),
            shape.template,
            tuple(sorted(set(range(devices)) - set(readable))),
        )

    def _moved(self, row: ExecutionRow) -> bool:
        """Whether what this queued wait rests on changed: a control, or its parent."""
        now = self.executions.row(self.owner, self.request)
        return (
            now is None
            or (now.state, now.desired, now.generation) != (row.state, row.desired, row.generation)
            or bool(self.calls.ancestor_stop(self.owner, self.request))
        )

    def _refuse(self, offered: pb.AttemptOffer, code: str, detail: str) -> None:
        self.worker.refuse(offered, code, detail, pb.CAUSE_CODE_CONSTRAINT_INFEASIBLE)


def _due(deadline_unix_ms: int) -> float | None:
    """The caller's absolute deadline as a monotonic instant, or None for no deadline."""
    if not deadline_unix_ms:
        return None
    return time.monotonic() + deadline_unix_ms / 1000 - time.time()
