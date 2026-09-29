"""One Runtime request history survives client loss, process death and native release."""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import grpc
import pytest
import tensorfs

import signed_claims
from cozy_runtime import canonical_json
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR


def offer(request: str = "root", *, job: bool = True) -> pb.AttemptOffer:
    invocation, digest = documents.identity(
        pb.InvocationSpec(
            job=pb.JobInvocationSpec(
                installation_id="local-" + "11" * 16, job_descriptor_id="sha256:" + "12" * 32
            )
        )
        if job
        else pb.InvocationSpec()
    )
    return pb.AttemptOffer(
        request_id=request,
        attempt_ordinal=1,
        invocation_spec_digest=digest,
        invocation_spec_canonical_bytes=invocation,
    )


def complete(
    executions: Executions,
    request: str,
    value: Any = None,
    status: pb.OutcomeStatus = pb.OUTCOME_STATUS_SUCCEEDED,
    cause: pb.CauseCode = pb.CAUSE_CODE_UNSPECIFIED,
    origin: pb.CauseOrigin = pb.CAUSE_ORIGIN_WORKER,
) -> pb.AttemptOutcome:
    selected = executions.offer("owner", request)
    body, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id=request,
            attempt_ordinal=selected.attempt_ordinal,
            invocation_spec_digest=documents.spell(selected.invocation_spec_digest),
            status=status,
            cause=pb.OutcomeCause(code=cause, origin=origin),
            safe_message="finished",
            result=pb.ResultEnvelope(inline_result=canonical_json.encode(value)),
        )
    )
    terminal = pb.AttemptOutcome(
        request_id=request,
        attempt_ordinal=selected.attempt_ordinal,
        invocation_spec_digest=selected.invocation_spec_digest,
        outcome_id="out-" + str(selected.attempt_ordinal),
        outcome_digest=digest,
        outcome_canonical_bytes=body,
    )
    executions.workspace.outcome("owner", terminal)
    executions.reconcile("owner", request)
    return terminal


def settle(ready: Callable[[], bool]) -> None:
    """What an execution's own unit does next, as soon as it has done it."""
    bound = time.monotonic() + 60  # a hang bound on a loaded box, not a budget
    while not ready():
        assert time.monotonic() < bound
        time.sleep(0.01)


def ack(outcome: pb.AttemptOutcome, *, retain: bool = False) -> pb.AttemptOutcomeAck:
    return pb.AttemptOutcomeAck(
        request_id=outcome.request_id,
        attempt_ordinal=outcome.attempt_ordinal,
        invocation_spec_digest=outcome.invocation_spec_digest,
        outcome_id=outcome.outcome_id,
        outcome_digest=outcome.outcome_digest,
        retain_work=retain,
    )


@pytest.mark.parametrize("value", [None, 0, False, "", [], {}, 3.25, {"result": [1, "two"]}])
def test_exact_results_survive_lost_replies_collection_and_native_release(
    tmp_path: Path, value: Any
) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    submitted = offer()
    receipt = executions.submit(
        "owner",
        "submission",
        b"c" * 32,
        submitted,
        expected_execution_workspace_id=executions.workspace_id,
    )
    # A reconnect's stamps are not another semantic request.
    submitted.record_owner_epoch = 7
    submitted.control_stream_epoch = 9
    submitted.worker_boot_id = "later-stream"
    submitted.admission_epoch = 11
    assert (
        executions.submit(
            "owner",
            "submission",
            b"c" * 32,
            submitted,
            expected_execution_workspace_id=executions.workspace_id,
        )
        == receipt
    )
    terminal = complete(executions, "root", value)
    executions.progress("owner", "root", 1, {"type": "progress", "payload": {"position": 100}})
    reopened = Executions(Workspace(tmp_path / "store"))
    assert reopened.collect("owner", "root") == terminal
    assert reopened.retention_required("owner")
    with pytest.raises(WorkspaceRefusal, match="uncollected"):
        workspace.acknowledge("owner", ack(terminal))
    assert reopened.acknowledge_collection("owner", ack(terminal)).collected
    assert reopened.acknowledge_collection("owner", ack(terminal)).collected
    workspace.acknowledge("owner", ack(terminal))
    assert reopened.collect("owner", "root") == terminal
    assert not reopened.retention_required("owner")
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM executions").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM operation_cache").fetchone()[0] == 0


def test_submission_conflicts_do_not_rebind_root_or_an_existing_attempt(tmp_path: Path) -> None:
    e = Executions(Workspace(tmp_path / "store"))
    original = offer()
    e.submit(
        "owner", "submission", b"c" * 32, original, expected_execution_workspace_id=e.workspace_id
    )
    for submission, capture, changed in [
        ("submission", b"d" * 32, original),
        ("submission", b"c" * 32, offer("other")),
        ("different", b"c" * 32, original),
    ]:
        with pytest.raises(WorkspaceRefusal, match="identity changed"):
            e.submit(
                "owner",
                submission,
                capture,
                changed,
                expected_execution_workspace_id=e.workspace_id,
            )
    original.placement_id = "another-semantic-target"
    with pytest.raises(WorkspaceRefusal, match="identity changed"):
        e.submit(
            "owner",
            "submission",
            b"c" * 32,
            original,
            expected_execution_workspace_id=e.workspace_id,
        )
    with pytest.raises(WorkspaceRefusal, match="this owner"):
        e.status("foreign", "root")
    e.workspace.accept("owner", offer("legacy"))
    with pytest.raises(WorkspaceRefusal, match="another execution path"):
        e.submit(
            "owner",
            "legacy",
            b"c" * 32,
            offer("legacy"),
            expected_execution_workspace_id=e.workspace_id,
        )


def test_controls_fence_generation_and_replay_commands_without_another_attempt(
    tmp_path: Path,
) -> None:
    e = Executions(Workspace(tmp_path / "store"))
    e.submit("owner", "s", b"c" * 32, offer(), expected_execution_workspace_id=e.workspace_id)
    terminal = complete(e, "root", status=pb.OUTCOME_STATUS_FAILED)
    assert e.acknowledge_collection("owner", ack(terminal)).collected
    assert e.retention_required("owner")  # reading an error does not abandon interrupted work
    second = e.control("owner", "root", "resume-1", 1, "resume")
    assert second.attempt_ordinal == 2 and second.generation == 2
    assert e.control("owner", "root", "resume-1", 1, "resume") == second
    with pytest.raises(WorkspaceRefusal, match="identity changed"):
        e.control("owner", "root", "resume-1", 2, "cancel")
    with pytest.raises(WorkspaceRefusal, match="stale"):
        e.control("owner", "root", "old-cancel", 1, "cancel")
    canceled = e.control("owner", "root", "cancel", 2, "cancel")
    assert canceled.state == "canceling" and canceled.generation == 3
    # Finishing the already-authorized attempt is allowed after the cancellation fence.
    current = offer()
    current.attempt_ordinal = 2
    body, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id="root",
            attempt_ordinal=2,
            invocation_spec_digest=documents.spell(current.invocation_spec_digest),
            status=pb.OUTCOME_STATUS_CANCELED,
        )
    )
    e.workspace.outcome(
        "owner",
        pb.AttemptOutcome(
            request_id="root",
            attempt_ordinal=2,
            invocation_spec_digest=current.invocation_spec_digest,
            outcome_id="canceled",
            outcome_digest=digest,
            outcome_canonical_bytes=body,
        ),
    )
    e.reconcile("owner", "root")
    assert e.status("owner", "root").state == "canceled"
    with pytest.raises(WorkspaceRefusal, match="only paused or failed"):
        e.control("owner", "root", "resume-canceled", 3, "resume")
    assert e.collect("owner", "root", 1) == terminal
    assert not e.acknowledge_collection("owner", ack(terminal)).collected


def test_progress_compaction_reports_gaps_but_preserves_logs_and_days_of_retention(
    tmp_path: Path,
) -> None:
    now = [1_000]
    e = Executions(Workspace(tmp_path / "store"), clock_ms=lambda: now[0])
    e.submit("owner", "s", b"c" * 32, offer(), expected_execution_workspace_id=e.workspace_id)
    e.progress("owner", "root", 1, {"type": "log", "payload": "durable author log"})
    for index in range(300):
        e.progress("owner", "root", 1, {"type": "progress", "payload": {"position": index}})
    now[0] += 7 * 24 * 60 * 60 * 1000  # scoped service clock, never the host clock
    first = e.events("owner", "root")
    assert first.compacted_through > 0 and first.events[1].kind == "log"
    assert e.retention_required("owner")
    last = first.events[-1].sequence
    assert e.events("owner", "root", last).events[0].sequence > last
    with pytest.raises(WorkspaceRefusal, match="ahead"):
        e.events("owner", "root", 10_000)
    terminal = complete(e, "root")
    assert e.collect("owner", "root") == terminal
    wrong = ack(terminal)
    wrong.outcome_digest = b"x" * 32
    with pytest.raises(WorkspaceRefusal, match="changed"):
        e.acknowledge_collection("owner", wrong)
    assert e.retention_required("owner")


def test_deferred_telemetry_precedes_the_transition_that_follows_it(tmp_path: Path) -> None:
    e = Executions(Workspace(tmp_path / "store"))
    e.submit("owner", "s", b"c" * 32, offer(), expected_execution_workspace_id=e.workspace_id)
    for index in range(50):
        e.progress("owner", "root", 1, {"type": "log", "payload": {"line": index}})
    e.progress("owner", "absent", 1, {"type": "log", "payload": "not held here"})
    complete(e, "root")
    events = e.events("owner", "root").events
    assert [event.kind for event in events[-51:]] == ["log"] * 50 + ["outcome"]
    assert [canonical_json.decode(event.body)["payload"]["line"] for event in events[-51:-1]] == [
        *range(50)
    ]


def test_pending_telemetry_never_holds_process_exit(tmp_path: Path) -> None:
    # A stream abandoned mid-read keeps the journal held until the process ends.
    script = (
        "import sys\nfrom pathlib import Path\n"
        "from cozy_runtime.internal.worker.workspace import Workspace\n"
        "w = Workspace(Path(sys.argv[1]))\n"
        "def held():\n    with w.locked():\n        yield\n"
        "stream = held(); next(stream)\n"
        "w.defer(lambda db: None)\n"
    )
    subprocess.run([sys.executable, "-c", script, str(tmp_path / "store")], check=True, timeout=60)


CHILD = """
import base64,sys
from pathlib import Path
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import worker_pb2 as pb
e=Executions(Workspace(Path(sys.argv[1])))
e.submit('owner','submission',b'c'*32,pb.AttemptOffer.FromString(base64.b64decode(sys.argv[2])),expected_execution_workspace_id=e.workspace_id)
if sys.argv[3] == 'dispatched':
    e.dispatched('owner','root',1)
print('accepted',flush=True)
sys.stdin.readline()
"""


@pytest.mark.parametrize("dispatched", [False, True])
def test_a_dead_process_requeues_only_work_it_never_dispatched(
    tmp_path: Path, dispatched: bool
) -> None:
    """The one automatic requeue: an attempt its Runtime process never handed to an
    executor runs once more, at the next ordinal, once. Dispatched work is never rerun."""
    root = tmp_path / "store"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            CHILD,
            str(root),
            base64.b64encode(offer().SerializeToString()).decode(),
            "dispatched" if dispatched else "queued",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "accepted"
        e = Executions(Workspace(root))
        e.workspace.recover("owner")
        e.reconcile("owner", "root", requeue="boot-next")
        assert e.status("owner", "root").attempt_ordinal == 1  # a live origin is not reclaimed
        assert not e.assigned_here("owner", "root", 1, "")
        child.kill()
        child.communicate(timeout=10)
        e.workspace.recover("owner")  # boot reconcile, after the process is gone
        e.reconcile("owner", "root", requeue="boot-next")
        lost = documents.parse(
            e.collect("owner", "root", 1).outcome_canonical_bytes, pb.AttemptOutcomeBody
        )
        assert (lost.status, lost.cause.origin) == (
            pb.OUTCOME_STATUS_ABANDONED,
            pb.CAUSE_ORIGIN_INFRA,
        )
        assert lost.execution_started == dispatched
        state = e.status("owner", "root")
        if dispatched:
            assert (state.state, state.attempt_ordinal) == ("failed", 1)
            return
        assert (state.state, state.attempt_ordinal) == ("queued", 2)
        assert e.assigned_here("owner", "root", 2, "boot-next")
        # Lost before dispatch a second time: failed, never a third attempt.
        with e.workspace.locked() as db:
            db.execute(
                "UPDATE attempts SET process_owner=? WHERE request='root' AND ordinal=2",
                (child_identity(child.pid),),
            )
        e.workspace.recover("owner")
        e.reconcile("owner", "root", requeue="boot-next")
        assert (e.status("owner", "root").state, e.status("owner", "root").attempt_ordinal) == (
            "failed",
            2,
        )
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


def child_identity(pid: int) -> bytes:
    """A Runtime process birth that has ended (the killed child's pid, a later tick)."""
    from cozy_runtime.internal.worker import workspace_recovery

    identity = canonical_json.decode(workspace_recovery.process_identity())
    return canonical_json.encode({**identity, "pid": pid})


@pytest.mark.parametrize("abandon", [False, True])
def test_native_result_has_separate_collection_and_consumer_custody(
    tmp_path: Path, abandon: bool
) -> None:
    from cozy_runtime.internal import canonical
    from cozy_runtime.internal.weights_sink import protocol_receipt
    from test_workspace_custody import produced, retention

    value = json.loads(
        (
            Path(__file__).parent / "testdata/worker-protocol/canonical/invocation_spec_job.json"
        ).read_text()
    )
    for outputs in (value["outputs"], value["job"]["publication_contract"]["outputs"]):
        for output in outputs:
            output["mime_type"] = "application/vnd.cozy.model-manifest"
    invocation = canonical.write(value)
    offered = pb.AttemptOffer(
        request_id="producer",
        attempt_ordinal=1,
        invocation_spec_digest=hashlib.sha256(invocation).digest(),
        invocation_spec_canonical_bytes=invocation,
    )
    e = Executions(Workspace(tmp_path / "store"))
    e.submit("owner", "native", b"c" * 32, offered, expected_execution_workspace_id=e.workspace_id)
    store, workspace, receipt = produced(tmp_path)
    independent = workspace.retain("owner", retention(receipt, 42))
    reference, _, _ = protocol_receipt(
        receipt,
        owner_scope="owner",
        request_id="producer",
        invocation_spec_digest=documents.spell(offered.invocation_spec_digest),
    )
    body, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=documents.spell(offered.invocation_spec_digest),
            status=pb.OUTCOME_STATUS_SUCCEEDED,
            weights_receipts=[reference],
            result=pb.ResultEnvelope(inline_result=b"null"),
        )
    )
    terminal = pb.AttemptOutcome(
        request_id="producer",
        attempt_ordinal=1,
        invocation_spec_digest=offered.invocation_spec_digest,
        outcome_id="native",
        outcome_digest=digest,
        outcome_canonical_bytes=body,
    )
    workspace.outcome("owner", terminal)
    e.reconcile("owner", "producer")
    assert e.collect("owner", "producer") == terminal
    replacement = Workspace(tmp_path / "replacement-store")
    with workspace.locked() as db:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shutil.copyfile(
        workspace.directory / "journal.sqlite3", replacement.directory / "journal.sqlite3"
    )
    copied = Executions(replacement)
    assert copied.workspace_id == e.workspace_id
    with pytest.raises(Exception) as lost:
        copied.collect("owner", "producer")
    assert getattr(lost.value, "code", None) in {"ROOT_ABSENT", "OBJECT_ABSENT"}
    tensorfs.gc(str(store.root))
    assert store.manifest(documents.spell(independent.manifest.digest))["manifest"]
    if abandon:
        e.control("owner", "producer", "abandon", 1, "cancel")
        assert e.release("owner", "producer")
        assert not e.status("owner", "producer").collected
        assert not e.retention_required("owner")
    else:
        e.acknowledge_collection("owner", ack(terminal))
        workspace.release_result(
            "owner",
            pb.DerivedResultReleaseRequest(
                weights_transaction_id=receipt.weights_transaction_id,
                tensorfs_receipt_digest=documents.raw(receipt.tensorfs_receipt_digest),
            ),
        )
        workspace.acknowledge("owner", ack(terminal))
    tensorfs.gc(str(store.root))
    assert store.manifest(documents.spell(independent.manifest.digest))["manifest"]
    assert Executions(Workspace(tmp_path / "store")).collect("owner", "producer") == terminal


def test_worker_intake_and_terminal_are_durable_without_an_observer(tmp_path: Path) -> None:
    from cozy_runtime.internal.config import Credentials, RuntimeConfig
    from cozy_runtime.internal.worker.attempts import InlineResult
    from cozy_runtime.internal.worker.control import InMemoryControlHost
    from cozy_runtime.internal.worker.session import Worker, WorkerOptions

    worker = Worker(
        RuntimeConfig(
            cozy_home=tmp_path / "home",
            credentials=Credentials(),
            record_owner_public_key=signed_claims.PUBLIC_KEY,
        ),
        WorkerOptions(
            **signed_claims.IDENTITY,
            root=tmp_path / "worker",
            tensorfs_root=tmp_path / "store",
            activity_path=tmp_path / "activity",
        ),
        InMemoryControlHost(),
    )
    frames: list[pb.WorkerFrame] = []
    claim = signed_claims.claim()
    try:
        assert worker.accept_claim(claim, frames.append)[0] == 1
        # No Control stream or watcher is running. Real admission produces a typed terminal.
        submitted = offer("refused", job=False)
        receipt = worker.submit_execution(
            claim,
            "refused",
            b"c" * 32,
            submitted,
            expected_execution_workspace_id=worker._execution_service(claim)[0].workspace_id,
        )
        settle(lambda: worker.execution_status(claim, "refused").state == "failed")
        assert receipt.request_id == "refused"
        assert json.loads((tmp_path / "activity").read_bytes())["execution_retention_required"]
        assert worker.execution_status(claim, "refused").state == "failed"
        refused = worker.collect_execution(claim, "refused")
        assert documents.read(refused.outcome_canonical_bytes, pb.AttemptOutcomeBody)[
            "safe_message"
        ]
        # Durable terminal publication precedes the engine's lane/custody close.
        # No observer or external acknowledgement is needed for that eventual close.
        settle(lambda: worker.engine.history[("refused", 1)].state == "closed")
        assert worker.engine.history[("refused", 1)].state == "closed"
        assert (
            worker.submit_execution(
                claim,
                "refused",
                b"c" * 32,
                submitted,
                expected_execution_workspace_id=worker._execution_service(claim)[0].workspace_id,
            )
            == receipt
        )
        # Exercise the real engine's successful terminal path independently of its compute phase.
        assert worker.executions is not None
        submitted = offer("scalar", job=False)
        submitted.grant.invocation_spec_digest = submitted.invocation_spec_digest
        worker.executions.submit(
            "owner",
            "scalar",
            b"d" * 32,
            submitted,
            expected_execution_workspace_id=worker.executions.workspace_id,
        )
        attempt = worker.engine.offer(submitted)
        worker.engine.hold(attempt)
        terminal = worker.engine.outcome(
            attempt,
            pb.OUTCOME_STATUS_SUCCEEDED,
            pb.CAUSE_CODE_UNSPECIFIED,
            pb.CAUSE_ORIGIN_RUNTIME,
            "completed None",
            result=InlineResult(
                canonical=b"null", schema_digest="sha256:" + "34" * 32, adjustments=()
            ),
        )
        worker.settle(terminal)
        assert (
            attempt.state == "closed"
            and worker.execution_status(claim, "scalar").state == "succeeded"
        )
        assert worker.collect_execution(claim, "scalar") == terminal
        assert worker.acknowledge_execution_collection(claim, ack(terminal)).collected
        worker.executions.progress(
            "owner", "refused", 1, {"type": "log", "payload": "ignored after terminal"}
        )
        assert worker.execution_events(claim, "refused").head_sequence > 0
        foreign = signed_claims.claim("foreign", epoch=2)
        assert worker.accept_claim(foreign, frames.append)[0] == 0
        assert worker.fence.record_owner_id == "owner"
        with pytest.raises(WorkspaceRefusal, match="authenticated owner"):
            worker.execution_status(foreign, "scalar")
        worker.control_execution(claim, "refused", "abandon-refused", 1, "cancel")
        executions = worker.executions
        settle(lambda: not executions.retention_required("owner"))
        assert not worker.execution_status(claim, "refused").collected
        assert not worker.executions.retention_required("owner")
        worker.publish_activity()
        assert not json.loads((tmp_path / "activity").read_bytes())["execution_retention_required"]
    finally:
        worker.shutdown()


def test_every_recorded_outcome_is_final_and_only_a_command_runs_again(tmp_path: Path) -> None:
    """No automatic retry: an executor fault, an author error, an executor lost after dispatch
    or a refusal ends the execution with its own reason. Only its owner's explicit resume
    mints another attempt."""
    e = Executions(Workspace(tmp_path / "store"))
    for request, status, cause, state in [
        ("transient", pb.OUTCOME_STATUS_FAILED, pb.CAUSE_CODE_EXECUTOR_FAULT, "failed"),
        ("author", pb.OUTCOME_STATUS_FAILED, pb.CAUSE_CODE_AUTHOR_EXCEPTION, "failed"),
        ("abandoned", pb.OUTCOME_STATUS_ABANDONED, pb.CAUSE_CODE_EXECUTOR_INVALIDATED, "failed"),
        ("canceled", pb.OUTCOME_STATUS_CANCELED, pb.CAUSE_CODE_CLIENT_CANCEL, "canceled"),
        ("structural", pb.OUTCOME_STATUS_REFUSED, pb.CAUSE_CODE_LOCAL_SAFETY, "failed"),
    ]:
        e.submit(
            "owner",
            request,
            b"c" * 32,
            offer(request),
            expected_execution_workspace_id=e.workspace_id,
        )
        complete(e, request, status=status, cause=cause)
        e.reconcile("owner", request, requeue="boot")  # a boot pass changes nothing either
        assert (e.status("owner", request).state, e.status("owner", request).attempt_ordinal) == (
            state,
            1,
        )
    assert e.control("owner", "transient", "user-resume", 1, "resume").attempt_ordinal == 2


def test_a_failed_attempt_stays_failed_when_a_pause_reaches_it_first(tmp_path: Path) -> None:
    """A worker integrity failure fails every held attempt while their parents' stop pauses
    the children: whichever lands first, a failed outcome ends FAILED, and only the stop a
    pause itself caused (a canceled outcome) ends paused."""
    e = Executions(Workspace(tmp_path / "store"))
    for request, status, state in [
        ("failed", pb.OUTCOME_STATUS_FAILED, "failed"),
        ("refused", pb.OUTCOME_STATUS_REFUSED, "failed"),
        ("stopped", pb.OUTCOME_STATUS_CANCELED, "paused"),
    ]:
        e.submit(
            "owner",
            request,
            b"c" * 32,
            offer(request),
            expected_execution_workspace_id=e.workspace_id,
        )
        e.dispatched("owner", request, 1)
        # The pause lands first, while the attempt still runs ...
        assert e.control("owner", request, "pause", 1, "pause").state == "pausing"
        # ... and the attempt's own outcome arrives after it.
        pinned = offer(request)
        body, digest = documents.identity(
            pb.AttemptOutcomeBody(
                request_id=request,
                attempt_ordinal=1,
                invocation_spec_digest=documents.spell(pinned.invocation_spec_digest),
                status=status,
                cause=pb.OutcomeCause(
                    code=pb.CAUSE_CODE_LOCAL_SAFETY, origin=pb.CAUSE_ORIGIN_WORKER
                ),
                safe_message="the worker failed",
            )
        )
        e.workspace.outcome(
            "owner",
            pb.AttemptOutcome(
                request_id=request,
                attempt_ordinal=1,
                invocation_spec_digest=pinned.invocation_spec_digest,
                outcome_id="out-1",
                outcome_digest=digest,
                outcome_canonical_bytes=body,
            ),
        )
        e.reconcile("owner", request)
        assert e.status("owner", request).state == state, request


def test_queued_cancellation_and_absolute_deadline_have_exact_terminals(tmp_path: Path) -> None:
    e = Executions(Workspace(tmp_path / "store"), clock_ms=lambda: 2000)
    e.submit(
        "owner",
        "cancel",
        b"c" * 32,
        offer("cancel"),
        expected_execution_workspace_id=e.workspace_id,
    )
    e.control("owner", "cancel", "cancel", 1, "cancel")
    e.stop_queued("owner", "cancel")
    assert e.status("owner", "cancel").state == "canceled"
    assert not documents.parse(
        e.collect("owner", "cancel").outcome_canonical_bytes, pb.AttemptOutcomeBody
    ).execution_started
    submitted = offer("expired")
    spec = pb.InvocationSpec(
        job=pb.JobInvocationSpec(
            installation_id="local-" + "11" * 16, job_descriptor_id="sha256:" + "12" * 32
        ),
        deadline_unix_ms=1000,
    )
    submitted.invocation_spec_canonical_bytes, submitted.invocation_spec_digest = (
        documents.identity(spec)
    )
    e.submit(
        "owner", "expired", b"c" * 32, submitted, expected_execution_workspace_id=e.workspace_id
    )
    complete(e, "expired", status=pb.OUTCOME_STATUS_FAILED)
    with pytest.raises(WorkspaceRefusal, match="deadline"):
        e.control("owner", "expired", "resume", 1, "resume")


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
@pytest.mark.parametrize("origin", ["private", "published", "published-sdk"])
def test_real_executor_and_next_queued_root_finish_without_control_stream(
    origin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conftest import image_python
    from cozy_runtime.internal import (
        child_env,
        package_installation,
        package_interface,
        static_interface,
    )
    from cozy_runtime.internal.config import Credentials, RuntimeConfig
    from cozy_runtime.internal.worker.control import GrpcControlHost
    from cozy_runtime.internal.worker.plan import JobBinding
    from cozy_runtime.internal.worker.session import Worker, WorkerOptions
    from cozy_runtime.protocol import WIRE_MINOR
    from cozy_runtime.protocol import worker_pb2_grpc as rpc
    from local_owner import LocalRecordOwner, LocalRequest
    from test_end_to_end import _run, _wheel_rows
    from test_job_preparation_isolation import package
    from test_publish_derivation import _download_set, _SimpleIndex

    # Published fixtures use the same SDK-wheel directory a real installed machine has.
    sdk_wheels = image_python().parent.parent.parent / "wheels"
    monkeypatch.setattr(package_installation, "machine_wheels", lambda _: sdk_wheels)
    with tempfile.TemporaryDirectory(prefix="cz-machine.") as raw:
        root = Path(raw)
        environment, preparation = package(root, "offline")
        _, second_preparation = package(root, "different")
        project = root / "project_offline"
        source = project / "prepare_offline.py"
        source.write_text(
            source.read_text()
            + """
from cozy_runtime.author import invocable
@invocable(memoize=True)
async def helper(ctx: Context, *, n: int) -> Result:
    return Result(n)
app.job(helper, internal=True)
@app.entrypoint
def serve(payload: Request) -> Result:
    path = Path(payload.entered)
    count = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(count))
    return Result(count)
"""
        )
        source.write_text(
            source.read_text()
            .replace("def offline(", "async def offline(")
            .replace("return Result(7)", "return await helper(n=7)")
        )
        uv = shutil.which("uv")
        assert uv is not None
        wheels = environment / ".stage/offline/wheels"
        _run(uv, "build", "--wheel", "--out-dir", str(wheels), str(project))
        preparation.ClearField("files")
        preparation.files.extend(_wheel_rows(list(wheels.glob("*.whl"))))
        sdk_python = None
        if origin == "published-sdk":
            sdk = root / "sdk"
            _run(uv, "venv", "--python", str(image_python()), str(sdk))
            sdk_python = sdk / "bin/python"
            runtime_wheel = next(
                image_python().parent.parent.parent.glob("wheels/cozy_runtime-*.whl")
            )
            _run(uv, "pip", "install", "--python", str(sdk_python), str(runtime_wheel))
        indexes = []
        (root / "artifacts").mkdir()
        worker = Worker(
            RuntimeConfig(
                cozy_home=root / "home",
                credentials=Credentials(),
                record_owner_public_key=signed_claims.PUBLIC_KEY,
                child_base_env=tuple(
                    sorted(
                        (k, v)
                        for k, v in os.environ.items()
                        if not child_env.erased(k) and k != "PYTHONPATH"
                    )
                ),
            ),
            WorkerOptions(
                **signed_claims.IDENTITY,
                root=root / "worker",
                devices="",
                accelerator_backend="none",
                python=str(image_python()),
                environment_python=sdk_python,
                install_root=environment,
                artifact_cache=root / "artifacts",
                tensorfs_root=root / "store",
                grant_roots=(str(root),),
            ),
            GrpcControlHost("127.0.0.1:0", root / "control.addr"),
        )
        thread = threading.Thread(target=worker.run, daemon=True)
        try:
            captures = {}
            placements = {}
            for current_preparation in (preparation, second_preparation):
                if origin == "private":
                    prepared = worker.prepare_local_package(current_preparation)
                else:
                    wheel, runtime = (
                        next(
                            Path(row.path)
                            for row in current_preparation.files
                            if row.filename.startswith(prefix)
                        )
                        for prefix in ("prepare_", "cozy_runtime-")
                    )
                    package_name = "published/" + wheel.name.split("-")[0].replace("_", "-")
                    index = _SimpleIndex(
                        root / ("index-" + current_preparation.operation_id), [wheel, runtime]
                    )
                    indexes.append(index)
                    # The complete locked closure: this release, its Runtime, and their
                    # registry dependencies. The loopback index has priority over PyPI.
                    locked = (
                        "--index-url https://pypi.org/simple\n"
                        f"--extra-index-url {index.url}\n"
                        f"{package_name.split('/')[1]}==1.0.0 "
                        f"--hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}\n"
                        f"cozy-runtime=={runtime.name.split('-')[1]} "
                        f"--hash=sha256:{hashlib.sha256(runtime.read_bytes()).hexdigest()}\n"
                    ).encode() + bytes(current_preparation.dependency_requirements)
                    project_root = root / ("project_" + current_preparation.operation_id)
                    (project_root / "package.toml").write_text(
                        '[application]\nobject="prepare_'
                        + current_preparation.operation_id
                        + ':app"\n'
                    )
                    prepared = worker.prepare_package_set(
                        pb.PreparePackageSetRequest(
                            download_delegation=_download_set(package_name, "1.0.0"),
                            install_root=str(environment),
                            application="prepare_" + current_preparation.operation_id + ":app",
                            image_inventory=None
                            if sdk_python
                            else pb.ImageInventory(
                                profile="test",
                                python="3.12",
                                distributions=[
                                    pb.ImageDistribution(
                                        distribution="cozy-runtime",
                                        version=worker.base.distribution_versions["cozy-runtime"],
                                    )
                                ],
                            ),
                            locked_requirements=locked,
                            package_interface=package_interface.canonical_bytes(
                                static_interface.build(project_root)
                            ),
                        )
                    )
                placement_body = documents.read(
                    prepared.placement_set.placement_set_canonical_bytes, pb.PlacementSet
                )["placements"][0]
                installation_id = placement_body["installation_id"]
                assert prepared.installed_package.installation_id == installation_id
                capture, capture_digest = documents.identity(
                    pb.MachineExecutionCapture(
                        root_installation_id=installation_id,
                        installed_packages=[prepared.installed_package],
                        bindings=[
                            pb.MachineCallableBinding(
                                caller_installation_id=installation_id,
                                module="prepare_offline",
                                export="helper",
                                callee_installation_id=installation_id,
                                entrypoint="helper",
                            )
                        ]
                        if current_preparation.operation_id == "offline"
                        else [],
                    )
                )
                captures[current_preparation.operation_id] = (capture, capture_digest)
                placements[current_preparation.operation_id] = prepared.placement_set
            if worker.supervision.current is not None:
                worker.supervision.retire_current(
                    worker.supervision.current, "begin worker lifecycle"
                )
            plans = [
                json.loads(path.read_bytes()) for path in (root / "home/job-plans").glob("*/*.json")
            ]
            binding = JobBinding.read(next(row for row in plans if row.get("job") == "offline"))
            second_binding = JobBinding.read(
                next(row for row in plans if row.get("job") == "different")
            )
            thread.start()
            until = time.monotonic() + 90
            while not (root / "control.addr").exists():
                assert thread.is_alive() and time.monotonic() < until
                time.sleep(0.02)
            claim = signed_claims.claim()
            # End a real stream immediately. Neither a SnapshotAck nor a sender survives it.
            worker.serve_stream(iter([pb.RecordOwnerFrame(claim=claim)]), lambda frame: None)
            channel = grpc.insecure_channel((root / "control.addr").read_text().strip())
            client = rpc.WorkerControlStub(channel)
            request = LocalRequest(
                entrypoint="offline",
                payload={"entered": str(root / "entered"), "gate": str(root / "gate")},
                outputs=(),
                kind="job",
                request_id="first",
                job_descriptor_id=binding.job_descriptor_id,
                timeout_ms=60_000,
            )
            generator = LocalRecordOwner(
                request, {}, root / "grants-first", package_installation_id=binding.installation_id
            )
            worker._apply_desired_state(generator.desired_state())
            while not worker.job_ready:
                assert time.monotonic() < until, worker.latched
                time.sleep(0.02)
            incoming: queue.Queue[pb.RecordOwnerFrame | None] = queue.Queue()
            observed: list[pb.WorkerFrame] = []

            def frames() -> Iterator[pb.RecordOwnerFrame]:
                yield pb.RecordOwnerFrame(claim=claim)
                while (frame := incoming.get()) is not None:
                    yield frame

            control = client.Control(frames(), timeout=30)
            admitted = next(control).claim_ack
            assert admitted.accepted
            next(control)  # the snapshot

            def watch_control() -> None:
                with contextlib.suppress(grpc.RpcError):
                    observed.extend(control)

            observer = threading.Thread(target=watch_control, daemon=True)
            observer.start()
            for identifier in ("first", "second", "canceled", "paused"):
                selected = second_binding if identifier == "second" else binding
                capture, capture_digest = captures[
                    "different" if identifier == "second" else "offline"
                ]
                generator = LocalRecordOwner(
                    dataclasses.replace(
                        request,
                        request_id=identifier,
                        entrypoint=selected.job,
                        job_descriptor_id=selected.job_descriptor_id,
                        payload=request.payload if identifier == "first" else {},
                    ),
                    {},
                    root / ("grants-" + identifier),
                    package_installation_id=selected.installation_id,
                )
                message = generator.offer()
                submission = pb.MachineExecutionSubmit(
                    claim=claim,
                    submission_id=identifier,
                    capture_digest=capture_digest,
                    capture_canonical_bytes=capture,
                    offer=message,
                    prepared_state=generator.desired_state(),
                    payload_canonical_bytes=canonical_json.encode(generator.request.payload),
                    expected_execution_workspace_id=client.GetMachineExecutionWorkspace(
                        pb.MachineExecutionWorkspaceQuery(claim=claim)
                    ).execution_workspace_id,
                )
                # Remove the old transport payload before intake: only the inline captured
                # arguments are available when the queued executor eventually starts.
                for path in generator.grant_dir.glob("*.json"):
                    path.unlink()
                if identifier == "first":
                    helper = JobBinding.read(
                        next(row for row in plans if row.get("job") == "helper")
                    )
                    internal_generator = LocalRecordOwner(
                        dataclasses.replace(
                            request,
                            request_id="internal-root",
                            entrypoint=helper.job,
                            job_descriptor_id=helper.job_descriptor_id,
                            payload={"n": 7},
                        ),
                        {},
                        root / "internal-grants",
                        package_installation_id=helper.installation_id,
                    )
                    internal_offer = internal_generator.offer()
                    with pytest.raises(grpc.RpcError, match="internal_callable") as private_root:
                        client.SubmitMachineExecution(
                            pb.MachineExecutionSubmit(
                                claim=claim,
                                submission_id="internal-root",
                                capture_digest=capture_digest,
                                capture_canonical_bytes=capture,
                                offer=internal_offer,
                                prepared_state=internal_generator.desired_state(),
                                payload_canonical_bytes=canonical_json.encode({"n": 7}),
                                expected_execution_workspace_id=client.GetMachineExecutionWorkspace(
                                    pb.MachineExecutionWorkspaceQuery(claim=claim)
                                ).execution_workspace_id,
                            ),
                            timeout=30,
                        )
                    assert private_root.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                    assert worker.executions is not None
                    assert not worker.executions.owns("owner", "internal-root")
                    invalid = pb.MachineExecutionSubmit()
                    invalid.CopyFrom(submission)
                    invalid.offer.request_id = "rejected-input"
                    invalid.submission_id = "rejected-input"
                    invalid.payload_canonical_bytes = b"null"
                    with pytest.raises(grpc.RpcError) as denied:
                        client.SubmitMachineExecution(invalid, timeout=5)
                    assert (
                        dict(denied.value.trailing_metadata())["cozy-error-code"]
                        == "execution_submission_refused"
                    )
                    invalid.CopyFrom(submission)
                    invalid.offer.request_id = "rejected-authority"
                    invalid.submission_id = invalid.offer.request_id
                    invalid.publication_authorization_id = "019aaaab-0000-7000-8000-000000000003"
                    with pytest.raises(grpc.RpcError) as ungranted:
                        client.SubmitMachineExecution(invalid, timeout=30)
                    assert ungranted.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                    assert "no independent publication" in ungranted.value.details()
                    assert not worker.executions.owns("owner", invalid.offer.request_id)
                if identifier == "first":
                    for field in ("uninstalled", "duplicate"):
                        invalid.CopyFrom(submission)
                        invalid.offer.request_id = "rejected-capture-" + field
                        invalid.submission_id = invalid.offer.request_id
                        damaged = documents.read(capture, pb.MachineExecutionCapture)
                        if field == "uninstalled":
                            damaged["installed_packages"][0]["installation_id"] = "local-absent"
                        else:
                            damaged["installed_packages"] *= 2
                        invalid.capture_canonical_bytes = canonical_json.encode(damaged)
                        invalid.capture_digest = documents.digest_of(
                            invalid.capture_canonical_bytes
                        )
                        with pytest.raises(grpc.RpcError) as wrong_capture:
                            client.SubmitMachineExecution(invalid, timeout=5)
                        assert wrong_capture.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                if identifier == "first":
                    from cozy_runtime.internal.worker.machine_publication import (
                        PublicationAuthority,
                    )

                    worker.options = dataclasses.replace(
                        worker.options,
                        hubs=(
                            PublicationAuthority(
                                "https://selected.example",
                                worker_id="fixture",
                                worker_token="fixture",
                            ),
                        ),
                    )
                    submission.hub = "https://selected.example"
                    submission.source_credentials.append(
                        pb.SourceCredential(
                            provider=pb.NATIVE_SOURCE_OPERATION_CIVITAI,
                            credential="initial-fixture-credential",
                        )
                    )
                    original_execution = worker.execution

                    def observe_admission(
                        owner: str,
                        request_id: str,
                        execute: Callable[[str, str], Any] = original_execution,
                    ) -> Any:
                        if request_id == "first":
                            assert worker.executions is not None and worker.executions.owns(
                                owner, request_id
                            )
                            assert (
                                worker.machine_sources.credentials[(owner, request_id)][
                                    pb.NATIVE_SOURCE_OPERATION_CIVITAI
                                ]
                                == "initial-fixture-credential"
                            )
                        return execute(owner, request_id)

                    monkeypatch.setattr(worker, "execution", observe_admission)
                receipt = client.SubmitMachineExecution(submission, timeout=30)
                assert client.SubmitMachineExecution(submission, timeout=30) == receipt
                if identifier == "first":
                    assert worker.executions is not None
                    retained = worker.executions.preparation("owner", identifier)
                    assert retained["hub"] == "https://selected.example"
                    monkeypatch.setattr(worker, "execution", original_execution)

                    class WaitingSource:
                        pokes = 0

                        def poke(self) -> None:
                            self.pokes += 1

                    waiting = WaitingSource()
                    with worker.machine_sources.lock:
                        worker.machine_sources.waiting["scope-replay-proof"] = cast(Any, waiting)
                    # Receipt reconciliation needs its durable scope, not a fresh
                    # content bearer, and accepts harmless origin normalization.
                    held_options = worker.options
                    worker.options = dataclasses.replace(worker.options, hubs=())
                    try:
                        alias = pb.MachineExecutionSubmit()
                        alias.CopyFrom(submission)
                        alias.hub = "https://SELECTED.example:443"
                        assert client.SubmitMachineExecution(alias, timeout=5) == receipt
                        alias.hub = "https://another.example"
                        alias.source_credentials[0].credential = "rejected-fixture-credential"
                        before_pokes = waiting.pokes
                        with pytest.raises(
                            grpc.RpcError, match="changed its Hub scope"
                        ) as changed_hub:
                            client.SubmitMachineExecution(alias, timeout=5)
                        assert changed_hub.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                        assert "cozy-error-code" not in dict(changed_hub.value.trailing_metadata())
                        assert worker.executions.preparation("owner", identifier) == retained
                        assert waiting.pokes == before_pokes
                        assert (
                            worker.machine_sources.credentials[("owner", identifier)][
                                pb.NATIVE_SOURCE_OPERATION_CIVITAI
                            ]
                            == "initial-fixture-credential"
                        )
                        alias.hub = submission.hub
                        alias.source_credentials[0].credential = "renewed-fixture-credential"
                        assert client.SubmitMachineExecution(alias, timeout=5) == receipt
                        assert waiting.pokes > before_pokes
                        assert (
                            worker.machine_sources.credentials[("owner", identifier)][
                                pb.NATIVE_SOURCE_OPERATION_CIVITAI
                            ]
                            == "renewed-fixture-credential"
                        )
                    finally:
                        worker.options = held_options
                        with worker.machine_sources.lock:
                            del worker.machine_sources.waiting["scope-replay-proof"]
                if identifier == "first":
                    invalid.CopyFrom(submission)
                    invalid.payload_canonical_bytes = b"null"
                    with pytest.raises(grpc.RpcError) as ambiguous:
                        client.SubmitMachineExecution(invalid, timeout=5)
                    assert "cozy-error-code" not in dict(ambiguous.value.trailing_metadata())
            query = pb.MachineExecutionQuery(
                claim=claim,
                request_id="second",
                expected_execution_workspace_id=receipt.execution_workspace_id,
            )
            assert client.GetMachineExecution(query, timeout=5).state == "queued"
            while not (root / "entered").exists():
                assert time.monotonic() < until
                time.sleep(0.02)
            assert worker.execution_status(claim, "second").state == "queued"
            assert all(
                frame.WhichOneof("msg") in {"claim_ack", "snapshot", "observed_state"}
                for frame in observed
            )
            incoming.put(None)
            control.cancel()
            observer.join(5)
            worker.control_execution(claim, "canceled", "cancel", 1, "cancel")
            while worker.execution_status(claim, "canceled").state != "canceled":
                assert time.monotonic() < until
                time.sleep(0.02)
            assert worker.execution_status(claim, "canceled").state == "canceled"
            worker.control_execution(claim, "paused", "pause", 1, "pause")
            while worker.execution_status(claim, "paused").state != "paused":
                assert time.monotonic() < until
                time.sleep(0.02)
            assert worker.execution_status(claim, "paused").state == "paused"
            (root / "gate").touch()
            for identifier in ("first", "second"):
                while worker.execution_status(claim, identifier).state not in (
                    "succeeded",
                    "failed",
                ):
                    assert time.monotonic() < until
                    time.sleep(0.02)
                outcome = worker.collect_execution(claim, identifier)
                body = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
                assert body["status"] == pb.OUTCOME_STATUS_SUCCEEDED, body
                while worker.engine.history[(identifier, 1)].state != "closed":
                    assert time.monotonic() < until
                    time.sleep(0.02)
                assert worker.execution_status(claim, identifier).state == "succeeded"
                query.request_id = identifier
                assert (
                    client.CollectMachineExecution(
                        pb.MachineExecutionCollect(execution=query), timeout=5
                    )
                    == outcome
                )
                assert (
                    client.ListMachineExecutionEvents(
                        pb.MachineExecutionEventsQuery(execution=query), timeout=5
                    ).head_sequence
                    > 0
                )
                assert client.AcknowledgeMachineExecutionCollection(
                    pb.MachineExecutionCollectionAck(execution=query, outcome=ack(outcome)),
                    timeout=5,
                ).collected
            assert thread.is_alive()  # completed results still await collection
            query.expected_execution_workspace_id = "replaced-workspace"
            with pytest.raises(grpc.RpcError) as mismatched:
                client.GetMachineExecution(query, timeout=5)
            assert mismatched.value.code() == grpc.StatusCode.FAILED_PRECONDITION
            if origin != "private":
                # Published inference uses the exact prepared placement and remains fresh.
                serving_placement = documents.read(
                    placements["offline"].placement_set_canonical_bytes, pb.PlacementSet
                )["placements"][0]
                entrypoint = next(
                    row for row in serving_placement["entrypoints"] if row["name"] == "serve"
                )
                payload = canonical_json.encode({"entered": str(root / "inference-count")})
                payload_digest = documents.spell(documents.digest_of(payload))
                serving_spec, serving_digest = documents.identity(
                    pb.InvocationSpec(
                        installation_id=serving_placement["installation_id"],
                        payload_digest=payload_digest,
                        inputs=[
                            pb.InputBinding(
                                input_id="payload",
                                digest=payload_digest,
                                length=len(payload),
                                kind_mime="application/json",
                            )
                        ],
                        serving=pb.ServingInvocationSpec(
                            entrypoint_binding_digest=entrypoint["entrypoint_binding_digest"],
                            attempt_binding_id=entrypoint["entrypoint_binding_digest"],
                            bindings_digest=serving_placement["bindings_digest"],
                        ),
                    )
                )
                serving_capture, serving_capture_digest = captures["offline"]
                for ordinal in (1, 2):
                    serving_offer = pb.AttemptOffer(
                        request_id=f"inference-{ordinal}",
                        attempt_ordinal=1,
                        placement_id=serving_placement["placement_id"],
                        invocation_spec_canonical_bytes=serving_spec,
                        invocation_spec_digest=serving_digest,
                        grant=pb.DeliveryGrant(
                            invocation_spec_digest=serving_digest,
                            inputs=[
                                pb.InputAccess(
                                    input_id="payload",
                                    url="data:application/json;base64,"
                                    + base64.b64encode(payload).decode(),
                                )
                            ],
                        ),
                    )
                    client.SubmitMachineExecution(
                        pb.MachineExecutionSubmit(
                            claim=claim,
                            submission_id=serving_offer.request_id,
                            capture_digest=serving_capture_digest,
                            capture_canonical_bytes=serving_capture,
                            offer=serving_offer,
                            payload_canonical_bytes=payload,
                            prepared_state=pb.DesiredWorkerState(
                                revision=1,
                                wire_minor=WIRE_MINOR,
                                posture=pb.POSTURE_ACCEPTING,
                                placement_set=placements["offline"],
                            ),
                            expected_execution_workspace_id=client.GetMachineExecutionWorkspace(
                                pb.MachineExecutionWorkspaceQuery(claim=claim)
                            ).execution_workspace_id,
                        ),
                        timeout=30,
                    )
                    while worker.execution_status(claim, serving_offer.request_id).state not in (
                        "succeeded",
                        "failed",
                    ):
                        assert time.monotonic() < until, worker.latched
                        time.sleep(0.02)
                    inference = worker.collect_execution(claim, serving_offer.request_id)
                    result = documents.read(
                        inference.outcome_canonical_bytes, pb.AttemptOutcomeBody
                    )
                    assert result["status"] == pb.OUTCOME_STATUS_SUCCEEDED, result
                    assert json.loads(base64.b64decode(result["result"]["inline_result"])) == {
                        "value": ordinal
                    }
            channel.close()
            # Replace the entire Worker and gRPC server. No client preparation is
            # replayed, and the old wheel upload is no longer available.
            config, options = worker.config, worker.options
            worker.request_stop()
            worker.host.stop()
            thread.join(30)
            assert not thread.is_alive()
            for path in (environment / ".stage").rglob("*.whl"):
                path.unlink()
            for index in indexes:
                index.close()
            indexes.clear()
            (root / "control.addr").unlink()
            worker = Worker(config, options, GrpcControlHost("127.0.0.1:0", root / "control.addr"))
            assert not worker.prepared_installations and not worker.engine.jobs
            thread = threading.Thread(target=worker.run, daemon=True)
            thread.start()
            while not (root / "control.addr").exists():
                assert thread.is_alive() and time.monotonic() < until
                time.sleep(0.02)
            worker.serve_stream(iter([pb.RecordOwnerFrame(claim=claim)]), lambda frame: None)
            channel = grpc.insecure_channel((root / "control.addr").read_text().strip())
            client = rpc.WorkerControlStub(channel)
            query.expected_execution_workspace_id = receipt.execution_workspace_id
            assert client.SubmitMachineExecution(submission, timeout=5) == receipt
            assert not worker.prepared_installations and not worker.engine.jobs
            query.request_id = "second"
            assert (
                client.CollectMachineExecution(
                    pb.MachineExecutionCollect(execution=query), timeout=5
                )
                == outcome
            )
            query.request_id = "paused"
            resume = pb.MachineExecutionControl(
                execution=query,
                command_id="resume-after-worker-replacement",
                expected_generation=2,
                action=pb.MACHINE_EXECUTION_ACTION_RESUME,
            )
            stale = pb.MachineExecutionControl()
            stale.CopyFrom(resume)
            stale.command_id = "stale-resume"
            stale.expected_generation = 1
            with pytest.raises(grpc.RpcError) as conflict:
                client.ControlMachineExecution(stale, timeout=5)
            assert conflict.value.code() == grpc.StatusCode.ABORTED
            assert (
                dict(conflict.value.trailing_metadata())["cozy-error-code"]
                == "execution_generation_stale"
            )
            resumed = client.ControlMachineExecution(resume, timeout=30)
            assert resumed.attempt_ordinal == 2
            while client.GetMachineExecution(query, timeout=5).state not in ("succeeded", "failed"):
                assert time.monotonic() < until
                time.sleep(0.02)
            assert worker.workspace is not None
            with worker.workspace.locked() as db:
                child_failures = [
                    dict(row)
                    for row in db.execute(
                        "SELECT safe_code,safe_detail FROM execution_calls "
                        "WHERE parent_request='paused'"
                    )
                ]
            assert client.GetMachineExecution(query, timeout=5).state == "succeeded", child_failures
            assert client.ControlMachineExecution(resume, timeout=5) == resumed
            assert worker.prepared_installations and worker.engine.jobs
            channel.close()
        finally:
            for index in indexes:
                index.close()
            worker.request_stop()
            worker.host.stop()
            if thread.ident is not None:
                thread.join(30)
            else:
                worker.shutdown()
            assert not thread.is_alive()
