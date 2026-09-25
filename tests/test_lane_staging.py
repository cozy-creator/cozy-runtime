"""Queue depth 2 on a device lane (proto-061 G): one attempt running, one staged.

The REAL worker and a REAL executor process, over the real Claim/Snapshot/DesiredState/Offer
conversation, serving `corpus/queue` — a weightless entrypoint that holds the device for as
long as its payload asks. The owner offers the way a hub does: against the lane's reported
`available_attempt_slots`, one offer in flight at a time. The arms prove that the second
attempt is ACCEPTED (hydrated, journaled, visible as QUEUED) while the first is still on the
device, that it enters the device right after the first's release, and that a staged attempt
cancels without ever reaching the device.
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from conftest import sample_python
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.local import (
    LOCAL_PLACEMENT,
    LocalRecordOwner,
    LocalRequest,
    development_source_digest,
    stage_development_placement,
)
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import _config
from test_end_to_end import NO_EXECUTOR

needs_executor = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")

PACKAGE = Path(__file__).resolve().parent.parent / "corpus" / "queue"
#: release -> next device entry, measured in the worker. The staged attempt's fetch, hydrate
#: and journal are already done; what remains is request resolution in the executor.
GAP_BOUND_MS = 150.0


class QueueOwner(LocalRecordOwner):
    """`LocalRecordOwner`'s conversation over a QUEUE of requests: an offer goes out whenever
    the placement's lane reports a free seat and no offer of this owner is unanswered."""

    def __init__(self, requests: list[LocalRequest], *args: Any, **kwargs: Any) -> None:
        super().__init__(requests[0], *args, **kwargs)
        self.pending = list(requests)
        self.in_flight = False
        self.lock = threading.Lock()
        self.accepted_at: dict[str, float] = {}
        self.outcomes: dict[str, dict[str, Any]] = {}
        self.outcome_at: dict[str, float] = {}
        self.queued_seen: list[tuple[str, str]] = []
        self.expected = len(requests)
        self.on_accept: Callable[[str], None] | None = None

    def _on_observed_state(self, report: pb.ObservedWorkerState) -> None:
        self.reports.append(report)
        self.admission_epoch = report.admission_epoch
        if report.worker_phase == pb.WorkerPhase.WORKER_PHASE_FAILED:
            self._end("worker_error", "the worker failed")
            return
        states = {
            held.request_id: pb.AttemptState.Name(held.state) for held in report.held_attempts
        }
        for request_id, state in states.items():
            if state == "ATTEMPT_STATE_QUEUED":
                running = [r for r, s in states.items() if s == "ATTEMPT_STATE_RUNNING"]
                self.queued_seen.extend((request_id, other) for other in running)
        status = next((p for p in report.placements if p.placement_id == LOCAL_PLACEMENT), None)
        lane = next(
            (
                lane
                for lane in report.lanes
                if status is not None and lane.lane_id == status.device_lane_id
            ),
            None,
        )
        ready = (
            status is not None
            and report.admission_state == pb.AdmissionState.ADMISSION_STATE_OPEN
            and status.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
            and lane is not None
            and lane.available_attempt_slots > 0
        )
        with self.lock:
            if not ready or self.in_flight or not self.pending:
                return
            self.request = self.pending.pop(0)
            self.in_flight = True
            self.dispatched = True
        self._dispatch()

    def _on_accepted(self, accepted: pb.AttemptAccepted) -> None:
        self.accepted_at[accepted.request_id] = time.perf_counter()
        with self.lock:
            self.in_flight = False
        if self.on_accept is not None:
            self.on_accept(accepted.request_id)

    def _on_outcome(self, outcome: pb.AttemptOutcome) -> None:
        self._send(
            pb.RecordOwnerFrame(
                outcome_ack=self._env(
                    pb.AttemptOutcomeAck(
                        request_id=outcome.request_id,
                        attempt_ordinal=outcome.attempt_ordinal,
                        invocation_spec_digest=outcome.invocation_spec_digest,
                        outcome_id=outcome.outcome_id,
                        outcome_digest=outcome.outcome_digest,
                    )
                )
            )
        )
        body = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
        self.outcome_at[outcome.request_id] = time.perf_counter()
        with self.lock:
            if outcome.request_id not in self.accepted_at:
                self.in_flight = False  # refused before acceptance: the seat is free again
            self.outcomes[outcome.request_id] = body
            finished = len(self.outcomes) == self.expected
        if finished:
            self._end("terminal", "")

    def _end(self, ended: str, error: str) -> None:
        self.ended, self.error = ended, error
        self.done.set()
        self.host.close()


@contextlib.contextmanager
def _workspace() -> Iterator[Path]:
    # Short: the executor's control socket lives under it and `sun_path` holds 108 bytes.
    with tempfile.TemporaryDirectory(prefix="cz-stage.", dir="/tmp") as raw:
        yield Path(raw)


def _request(tag: str, seconds: float) -> LocalRequest:
    return LocalRequest(
        entrypoint="hold",
        payload={"tag": tag, "seconds": seconds},
        outputs=(),
        request_id=f"stage-{tag}",
    )


def _serve(
    root: Path,
    requests: list[LocalRequest],
    hook: Callable[[QueueOwner, Worker], None] | None = None,
) -> tuple[QueueOwner, Worker]:
    config = _config(root / "home")
    body = package_interface.build(discover(PACKAGE))
    workspace = root / "run"
    workspace.mkdir(parents=True, exist_ok=True)
    staged = stage_development_placement(
        config.cozy_home,
        body,
        package="local/development-package",
        release="0.0.0",
        source_digest=development_source_digest(PACKAGE),
        artifact_cache=workspace / "artifact-cache",
        tensorfs_root=config.cozy_home / "tensorfs",
        project=PACKAGE,
    )
    owner = QueueOwner(
        requests,
        dict(staged.bindings),
        workspace,
        package_code_digest=documents.spell(staged.placement.development.source_digest),
        placement=staged.placement,
    )
    options = WorkerOptions(
        root=workspace / "worker",
        devices="0",
        python=str(sample_python()),
        instance_id="ins-lane-staging",
        tensorfs_root=staged.tensorfs_root,
        artifact_cache=staged.artifact_cache,
        development_project=PACKAGE,
        development_package=staged.placement.development.package,
        development_release=staged.placement.development.release,
        development_source_digest=documents.spell(staged.placement.development.source_digest),
    )
    worker = Worker(config, options, owner.host)
    owner.watch(worker)
    if hook is not None:
        hook(owner, worker)
    with contextlib.chdir(PACKAGE), contextlib.redirect_stdout(sys.stderr):
        worker.run()
    return owner, worker


@needs_executor
def test_second_attempt_is_staged_while_the_first_runs_and_enters_at_release() -> None:
    first, second = _request("first", 1.5), _request("second", 0.0)
    with _workspace() as root:
        owner, worker = _serve(root, [first, second])
    assert owner.ended == "terminal", (owner.ended, owner.error)
    for request in (first, second):
        body = owner.outcomes[request.request_id]
        assert int(body["status"]) == pb.OutcomeStatus.OUTCOME_STATUS_SUCCEEDED, body
    ran = worker.engine.history[(first.request_id, 1)]
    staged = worker.engine.history[(second.request_id, 1)]
    # ACCEPTED BEFORE THE FIRST'S DEVICE RELEASE, and seen by the owner as QUEUED behind it
    assert owner.accepted_at[second.request_id] < ran.resulted_at
    assert (second.request_id, first.request_id) in owner.queued_seen, owner.queued_seen
    # ONE DEVICE: the second entered only after the first released ...
    assert staged.dispatched_at >= ran.resulted_at
    # ... and right after it
    gap_ms = (staged.dispatched_at - ran.resulted_at) * 1000
    print(f"release -> next device entry: {gap_ms:.1f} ms", file=sys.stderr)
    assert gap_ms < GAP_BOUND_MS, gap_ms
    # the same executor process ran both
    assert ran.executor_pid == staged.executor_pid != 0


@needs_executor
def test_a_staged_attempt_cancels_without_reaching_the_device() -> None:
    first, second = _request("long", 1.5), _request("cancelled", 0.0)

    def hook(owner: QueueOwner, worker: Worker) -> None:
        def on_accept(request_id: str) -> None:
            if request_id != second.request_id:
                return
            spec = owner.spec_digest
            owner._send(
                pb.RecordOwnerFrame(
                    cancel_attempt=owner._env(
                        pb.CancelAttempt(
                            request_id=second.request_id,
                            attempt_ordinal=1,
                            invocation_spec_digest=spec,
                            reason=pb.CancelReason.CANCEL_REASON_CLIENT,
                        )
                    )
                )
            )

        owner.on_accept = on_accept

    with _workspace() as root:
        owner, worker = _serve(root, [first, second], hook)
    assert owner.ended == "terminal", (owner.ended, owner.error)
    ran = worker.engine.history[(first.request_id, 1)]
    cancelled = worker.engine.history[(second.request_id, 1)]
    body = owner.outcomes[second.request_id]
    assert int(body["status"]) == pb.OutcomeStatus.OUTCOME_STATUS_CANCELED, body
    assert cancelled.executions == 0 and cancelled.dispatched_at == 0.0
    # the cancel was answered while the first still held the device
    assert owner.outcome_at[second.request_id] < ran.resulted_at
    assert int(owner.outcomes[first.request_id]["status"]) == (
        pb.OutcomeStatus.OUTCOME_STATUS_SUCCEEDED
    )
    assert ran.executions == 1
