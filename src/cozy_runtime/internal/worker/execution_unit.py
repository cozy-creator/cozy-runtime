"""One machine execution, root or child, as its own supervised unit.

The unit runs its execution start to finish on one thread: a child's call phase, the wait
until the stage scheduler lets its GPU call go (placed and next on its GPUs, or holding its
whole turn), its attempt inline, and its settlement into its parent. It re-reads its row on
every wake. Its exception fails this execution and nothing else (`crashed`).
"""

from __future__ import annotations

import contextlib
import functools
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from cozy_runtime.internal.worker import machine_lanes
from cozy_runtime.internal.worker.attempts import safe
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
        #: the attempt whose call record this unit journaled on its root
        self.recorded = 0

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
            # No thread here holds it: its outcome is only unprojected, or its attempt was
            # lost and it fails. It never runs again.
            self.executions.reconcile(self.owner, self.request)
            now = self.executions.row(self.owner, self.request)
            if now is not None and not now.terminal:
                self._fail("the dispatched attempt is held by no one")
            return True
        self.failure = self.failure or self.worker.failure_reason
        if self.failure:
            self._fail(self.failure)
            return True
        if row.desired != "run":
            self.executions.stop_queued(self.owner, self.request)
            return True
        if action := self.calls.journal.ancestor_stop(self.owner, self.request):
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
        self.calls.timing.phase(self.owner, self.request, row.ordinal, "queued")
        offered = row.attempt_offer()
        deadline = documents.parse(
            offered.invocation_spec_canonical_bytes, pb.InvocationSpec
        ).deadline_unix_ms
        due = time.monotonic() + deadline / 1000 - time.time() if deadline else None
        if due is not None and due <= time.monotonic():
            self.executions.stop_queued(self.owner, self.request, deadline=True)
            return
        shape = self._shaped(row)
        if shape.refusal:
            self._refuse(offered, shape.refusal)
            return
        grant_key = f"{self.request}#{row.ordinal}"
        ordinals: tuple[int, ...] = ()
        attempt = None
        stages = self.worker.stages
        try:
            if shape.kind != "cpu":
                want = machine_lanes.want(self.worker, shape)
                if isinstance(want, str):
                    self._refuse(offered, want)
                    return
                stages.want(grant_key, want)
                if shape.kind == "serving":
                    journal = functools.partial(self.worker._journal_row, self.request, row.ordinal)
                    self.worker.prespawns.waiting(grant_key, want, shape.interface, journal)
                unit.wait_for(
                    lambda: (
                        stages.ready(grant_key)
                        or stages.refused(grant_key) is not None
                        or self._moved(row)
                    ),
                    due,
                )
                if (refused := stages.refused(grant_key)) is not None:
                    self.worker.refuse(
                        offered, "no_capacity", refused.detail, pb.CAUSE_CODE_NO_CAPACITY
                    )
                    return
                placed = stages.placed(grant_key)
                if placed is None:
                    return  # moved or past its deadline: the next pass decides
                ordinals = placed
            self.calls.timing.phase(self.owner, self.request, row.ordinal, "preparing")
            attempt = self.worker.dispatch_machine(
                unit, offered, ordinals, lambda: self._moved(row)
            )
            if attempt is not None:
                self.worker.run_attempt(attempt)
        finally:
            if attempt is None and shape.kind != "cpu":
                stages.release(grant_key)

    def _fail(self, why: str) -> None:
        """End this execution FAILED: an undispatched attempt gets the terminal; a recorded
        one is projected."""
        self.worker.note("execution", f"{self.request} failed: {why[:400]}")
        try:
            self.executions.fail(self.owner, self.request, why)
        except ExecutionChanged:
            self.executions.reconcile(self.owner, self.request)

    def _settle(self, unit: Unit, row: ExecutionRow) -> None:
        """A terminal execution delivers to its parent, stops its children and releases what
        it retains. The row is already terminal: a failed step is noted, never the worker's."""
        steps: tuple[tuple[str, Callable[[], None]], ...] = (
            ("record", lambda: self._record(row)),
            ("custody", lambda: self.calls.deliver(unit, self.owner, self.request, row)),
            ("children", lambda: self._stop_children(row)),
            ("retention", lambda: self._release_retention(row)),
        )
        for name, step in steps:
            try:
                step()
            except FENCES:
                raise
            except Exception as exc:
                why = fault_text(exc)
                self.worker.note("execution", f"{self.request} settle {name} faulted: {why}"[:400])
                fault = {"step": name, "fault": why[:1024]}
                with contextlib.suppress(Exception):
                    self.executions.record(self.owner, self.request, "settle_fault", fault)

    def _record(self, row: ExecutionRow) -> None:
        """Once per attempt, before custody wakes its parent (the call's label is the
        parent's until then)."""
        if self.recorded != row.ordinal:
            self.recorded = row.ordinal
            self.calls.record(self.owner, self.request, row)

    def _stop_children(self, row: ExecutionRow) -> None:
        """Open children follow their terminal parent: cancelled with it, else paused. A
        pending call's unit learns that no parent awaits it."""
        action: Literal["pause", "cancel"] = "cancel" if row.state == "canceled" else "pause"
        for call in self.calls.journal.children(self.owner, self.request):
            if not self.executions.owns(self.owner, call.child_request):
                self.worker.supervisor.poke(key(call.child_request))
            elif row.state != "succeeded":
                self.worker.control_tree(self.owner, call.child_request, action, row.generation)
        self.calls.serving.forget(self.request, row.ordinal)

    def _release_retention(self, row: ExecutionRow) -> None:
        """A canceled execution, or one whose hold this boot releases, relinquishes what it
        retains once no effect is in doubt."""
        boot = self.request in self.worker.releasing
        if (
            (row.desired == "cancel" or boot)
            and not row.retention_waived
            and self.executions.release(self.owner, self.request, boot=boot)
        ):
            self.worker.releasing.discard(self.request)

    def _track_root(self, row: ExecutionRow) -> None:
        """A root's GPU lease exists exactly while its row is open and wants to run."""
        if row.desired != "run" or row.state not in ("queued", "running"):
            self._close_root()
            return
        if self.is_root is None:
            root, _ = self.executions.scheduling_root(self.owner, self.request)
            self.is_root = root == self.request
        if self.is_root and not self.root_open:
            self.worker.stages.open_root(self.request, row.priority)
            self.root_open = True

    def _close_root(self) -> None:
        if self.root_open:
            self.root_open = False
            self.worker.stages.close_root(self.request)

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

    def _moved(self, row: ExecutionRow) -> bool:
        """Whether what this queued wait rests on changed: a control, or its parent."""
        now = self.executions.row(self.owner, self.request)
        return (
            now is None
            or (now.state, now.desired, now.generation) != (row.state, row.desired, row.generation)
            or bool(self.calls.journal.ancestor_stop(self.owner, self.request))
        )

    def _refuse(self, offered: pb.AttemptOffer, detail: str) -> None:
        code = detail.partition(":")[0]
        self.worker.refuse(offered, code, detail, pb.CAUSE_CODE_CONSTRAINT_INFEASIBLE)
