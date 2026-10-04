"""One Runtime request history survives client loss, process death and native release."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


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
