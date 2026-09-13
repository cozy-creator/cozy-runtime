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
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import grpc
import pytest
import tensorfs

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
                build_id="sha256:" + "11" * 32, job_descriptor_id="sha256:" + "12" * 32
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
) -> pb.AttemptOutcome:
    selected = executions.offer("owner", request)
    body, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id=request,
            attempt_ordinal=selected.attempt_ordinal,
            invocation_spec_digest=documents.spell(selected.invocation_spec_digest),
            status=status,
            cause=pb.OutcomeCause(code=cause, origin=pb.CAUSE_ORIGIN_WORKER),
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
    executions.observe("owner", pb.WorkerFrame(attempt_outcome=terminal))
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
    receipt = executions.submit("owner", "submission", b"c" * 32, submitted)
    # A reconnect's stamps are not another semantic request.
    submitted.record_owner_epoch = 7
    submitted.control_stream_epoch = 9
    submitted.worker_boot_id = "later-stream"
    submitted.admission_epoch = 11
    assert executions.submit("owner", "submission", b"c" * 32, submitted) == receipt
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
    e.submit("owner", "submission", b"c" * 32, original)
    for submission, capture, changed in [
        ("submission", b"d" * 32, original),
        ("submission", b"c" * 32, offer("other")),
        ("different", b"c" * 32, original),
    ]:
        with pytest.raises(WorkspaceRefusal, match="identity changed"):
            e.submit("owner", submission, capture, changed)
    original.placement_id = "another-semantic-target"
    with pytest.raises(WorkspaceRefusal, match="identity changed"):
        e.submit("owner", "submission", b"c" * 32, original)
    with pytest.raises(WorkspaceRefusal, match="this owner"):
        e.status("foreign", "root")
    e.workspace.accept("owner", offer("legacy"))
    with pytest.raises(WorkspaceRefusal, match="another execution path"):
        e.submit("owner", "legacy", b"c" * 32, offer("legacy"))


def test_controls_fence_generation_and_replay_commands_without_another_attempt(
    tmp_path: Path,
) -> None:
    e = Executions(Workspace(tmp_path / "store"))
    e.submit("owner", "s", b"c" * 32, offer())
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
    e.submit("owner", "s", b"c" * 32, offer())
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


CHILD = """
import base64,sys
from pathlib import Path
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import worker_pb2 as pb
e=Executions(Workspace(Path(sys.argv[1])))
e.submit('owner','submission',b'c'*32,pb.AttemptOffer.FromString(base64.b64decode(sys.argv[2])),max_attempts=2)
print('accepted',flush=True)
sys.stdin.readline()
"""


def test_actual_process_death_recovers_without_duplicate_execution(tmp_path: Path) -> None:
    root = tmp_path / "store"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            CHILD,
            str(root),
            base64.b64encode(offer().SerializeToString()).decode(),
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
        assert e.status("owner", "root").state == "queued"  # live origin cannot be reclaimed
        assert not e.assigned_here("owner", "root", 1, "")
        child.kill()
        child.communicate(timeout=10)
        assert e.status("owner", "root").state == "retrying"
        with pytest.raises(WorkspaceRefusal, match="not terminal"):
            e.collect("owner", "root")
        original = e.collect("owner", "root", 1)
        assert (
            documents.read(original.outcome_canonical_bytes, pb.AttemptOutcomeBody)["status"]
            == pb.OUTCOME_STATUS_ABANDONED
        )
        assert (
            e.submit("owner", "submission", b"c" * 32, offer(), max_attempts=2).request_id == "root"
        )
        assert e.status("owner", "root").attempt_ordinal == 1
        e.retry_due("owner")
        assert e.status("owner", "root").attempt_ordinal == 2
        assert e.assigned_here("owner", "root", 2, "")
        assert not e.assigned_here("owner", "root", 2, "foreign-boot")
        assert e.collect("owner", "root", 1) == original
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


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
    e.submit("owner", "native", b"c" * 32, offered)
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
    assert e.collect("owner", "producer") == terminal
    replacement = Workspace(tmp_path / "replacement-store")
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
        assert e.abandon("owner", "producer")
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
    from cozy_runtime.protocol import WIRE_MINOR

    worker = Worker(
        RuntimeConfig(
            cozy_home=tmp_path / "home",
            credentials=Credentials(),
            bootstrap_credential="test-bootstrap",
        ),
        WorkerOptions(
            root=tmp_path / "worker",
            tensorfs_root=tmp_path / "store",
            activity_path=tmp_path / "activity",
        ),
        InMemoryControlHost(),
    )
    frames: list[pb.WorkerFrame] = []
    claim = pb.Claim(
        record_owner_id="owner",
        record_owner_epoch=1,
        wire_minor=WIRE_MINOR,
        proof=b"test-bootstrap",
    )
    try:
        assert worker.accept_claim(claim, frames.append)[0] == 1
        # No Control stream or watcher is running. Real admission produces a typed terminal.
        submitted = offer("refused", job=False)
        receipt = worker.submit_execution(claim, "refused", b"c" * 32, submitted)
        worker.drive_executions()
        assert receipt.request_id == "refused"
        assert json.loads((tmp_path / "activity").read_bytes())["execution_retention_required"]
        assert worker.execution_status(claim, "refused").state == "failed"
        refused = worker.collect_execution(claim, "refused")
        assert documents.read(refused.outcome_canonical_bytes, pb.AttemptOutcomeBody)[
            "safe_message"
        ]
        assert worker.engine.history[("refused", 1)].state == "closed"
        assert worker.submit_execution(claim, "refused", b"c" * 32, submitted) == receipt
        # Exercise the real engine's successful terminal path independently of its compute phase.
        assert worker.executions is not None
        submitted = offer("scalar", job=False)
        submitted.grant.invocation_spec_digest = submitted.invocation_spec_digest
        worker.executions.submit("owner", "scalar", b"d" * 32, submitted)
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
        worker.send(pb.WorkerFrame(attempt_outcome=terminal))
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
        foreign = pb.Claim(
            record_owner_id="foreign",
            record_owner_epoch=2,
            wire_minor=WIRE_MINOR,
            proof=b"test-bootstrap",
        )
        assert worker.accept_claim(foreign, frames.append)[0] == 0
        assert worker.fence.record_owner_id == "owner"
        with pytest.raises(WorkspaceRefusal, match="authenticated owner"):
            worker.execution_status(foreign, "scalar")
        worker.control_execution(claim, "refused", "abandon-refused", 1, "cancel")
        worker.drive_executions()
        assert not worker.execution_status(claim, "refused").collected
        assert not worker.executions.retention_required("owner")
        worker.publish_activity()
        assert not json.loads((tmp_path / "activity").read_bytes())["execution_retention_required"]
    finally:
        worker.shutdown()


def test_schema_upgrade_preserves_existing_custody_rows(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    workspace.accept("owner", offer("legacy"))
    with workspace.locked() as db:
        for table in ("execution_calls", "execution_commands", "execution_events", "executions"):
            db.execute("DROP TABLE " + table)
        db.execute("ALTER TABLE workspace_state DROP COLUMN execution_workspace_id")
        db.execute("PRAGMA user_version=6")
    workspace = Workspace(tmp_path / "store")
    assert workspace.contains("owner", "legacy", 1)
    e = Executions(workspace)
    assert len(e.workspace_id) == 32
    assert Executions(Workspace(tmp_path / "store")).workspace_id == e.workspace_id
    assert Executions(Workspace(tmp_path / "other-store")).workspace_id != e.workspace_id
    e.submit("owner", "new", b"c" * 32, offer("new"))
    with workspace.locked() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 8
        assert db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 2


def test_retry_policy_never_retries_authored_errors_or_user_cancellation(tmp_path: Path) -> None:
    now = [1000]
    e = Executions(Workspace(tmp_path / "store"), clock_ms=lambda: now[0])
    for request, status, cause in [
        ("transient", pb.OUTCOME_STATUS_FAILED, pb.CAUSE_CODE_EXECUTOR_FAULT),
        ("author", pb.OUTCOME_STATUS_FAILED, pb.CAUSE_CODE_AUTHOR_EXCEPTION),
        ("abandoned", pb.OUTCOME_STATUS_ABANDONED, pb.CAUSE_CODE_EXECUTOR_INVALIDATED),
        ("canceled", pb.OUTCOME_STATUS_CANCELED, pb.CAUSE_CODE_CLIENT_CANCEL),
        ("structural", pb.OUTCOME_STATUS_REFUSED, pb.CAUSE_CODE_LOCAL_SAFETY),
    ]:
        e.submit("owner", request, b"c" * 32, offer(request), max_attempts=2, retry_delay_ms=500)
        complete(e, request, status=status, cause=cause)
    e.retry_due("owner")
    assert e.status("owner", "transient").attempt_ordinal == 1
    assert e.status("owner", "transient").state == "retrying"
    with pytest.raises(WorkspaceRefusal, match="not terminal"):
        e.collect("owner", "transient")
    now[0] += 500
    e.retry_due("owner")
    assert e.status("owner", "transient").attempt_ordinal == 2
    e.retry_due("owner")
    assert all(
        e.status("owner", request).attempt_ordinal == 1
        for request in ("author", "canceled", "structural")
    )
    assert e.status("owner", "abandoned").attempt_ordinal == 2
    complete(e, "transient", status=pb.OUTCOME_STATUS_FAILED, cause=pb.CAUSE_CODE_EXECUTOR_FAULT)
    now[0] += 500
    e.retry_due("owner")
    assert e.status("owner", "transient").attempt_ordinal == 2
    # An explicit command authorizes another attempt without quietly expanding auto retries.
    assert e.control("owner", "transient", "user-resume", 2, "resume").attempt_ordinal == 3


def test_queued_cancellation_and_absolute_deadline_have_exact_terminals(tmp_path: Path) -> None:
    e = Executions(Workspace(tmp_path / "store"), clock_ms=lambda: 2000)
    e.submit("owner", "cancel", b"c" * 32, offer("cancel"))
    e.control("owner", "cancel", "cancel", 1, "cancel")
    e.stop_queued("owner", "cancel")
    assert e.status("owner", "cancel").state == "canceled"
    assert not documents.read(
        e.collect("owner", "cancel").outcome_canonical_bytes, pb.AttemptOutcomeBody
    ).get("execution_started", False)
    submitted = offer("expired")
    spec = pb.InvocationSpec(
        job=pb.JobInvocationSpec(
            build_id="sha256:" + "11" * 32, job_descriptor_id="sha256:" + "12" * 32
        ),
        deadline_unix_ms=1000,
    )
    submitted.invocation_spec_canonical_bytes, submitted.invocation_spec_digest = (
        documents.identity(spec)
    )
    e.submit("owner", "expired", b"c" * 32, submitted)
    complete(e, "expired", status=pb.OUTCOME_STATUS_FAILED)
    with pytest.raises(WorkspaceRefusal, match="deadline"):
        e.control("owner", "expired", "resume", 1, "resume")


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_real_executor_and_next_queued_root_finish_without_control_stream() -> None:
    from conftest import image_python
    from cozy_runtime.internal import child_env
    from cozy_runtime.internal.config import Credentials, RuntimeConfig
    from cozy_runtime.internal.local import LocalRecordOwner, LocalRequest
    from cozy_runtime.internal.worker.control import GrpcControlHost
    from cozy_runtime.internal.worker.plan import JobBinding
    from cozy_runtime.internal.worker.session import Worker, WorkerOptions
    from cozy_runtime.protocol import WIRE_MINOR
    from cozy_runtime.protocol import worker_pb2_grpc as rpc
    from test_end_to_end import _run, _wheel_rows
    from test_job_preparation_isolation import measure, package

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
app.job(helper)
"""
        )
        uv = shutil.which("uv")
        assert uv is not None
        wheels = environment / ".stage/offline/wheels"
        _run(uv, "build", "--wheel", "--out-dir", str(wheels), str(project))
        preparation.ClearField("wheels")
        preparation.wheels.extend(_wheel_rows(list(wheels.glob("*.whl"))))
        (root / "artifacts").mkdir()
        worker = Worker(
            RuntimeConfig(
                cozy_home=root / "home",
                credentials=Credentials(),
                bootstrap_credential="offline",
                child_base_env=tuple(
                    sorted(
                        (k, v)
                        for k, v in os.environ.items()
                        if not child_env.erased(k)
                        and k not in ("PYTHONPATH", "COZY_BOOTSTRAP_CREDENTIAL")
                    )
                ),
            ),
            WorkerOptions(
                root=root / "worker",
                devices="",
                accelerator_backend="none",
                python=str(image_python()),
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
            for current_preparation in (preparation, second_preparation):
                measure(worker, current_preparation)
                prepared = worker.prepare_local_package(current_preparation)
                placement_body = documents.read(
                    prepared.placement_set.placement_set_canonical_bytes, pb.PlacementSet
                )["placements"][0]
                revision = pb.LocalPackageRevision(
                    package=current_preparation.package.package,
                    release=current_preparation.package.release,
                    source_digest=current_preparation.package.source_digest,
                    package_interface=pb.Ref(
                        digest=documents.raw(placement_body["package_interface"]["digest"]),
                        length=placement_body["package_interface"]["length"],
                    ),
                    files=[
                        pb.LocalPackageFileRef(
                            filename=w.filename, digest=w.digest, length=w.length
                        )
                        for w in current_preparation.wheels
                    ],
                )
                capture, capture_digest = documents.identity(
                    pb.MachineExecutionCapture(
                        root_revision_digest=current_preparation.package.local_revision_digest,
                        revisions=[revision],
                        bindings=[
                            pb.MachineCallableBinding(
                                caller_revision_digest=current_preparation.package.local_revision_digest,
                                interface_digest=revision.package_interface.digest,
                                module="prepare_offline",
                                export="helper",
                                callee_revision_digest=current_preparation.package.local_revision_digest,
                                entrypoint="helper",
                            )
                        ]
                        if current_preparation.operation_id == "offline"
                        else [],
                    )
                )
                captures[current_preparation.operation_id] = (capture, capture_digest)
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
            claim = pb.Claim(
                record_owner_id="owner",
                record_owner_epoch=1,
                wire_minor=WIRE_MINOR,
                proof=b"offline",
            )
            # End a real stream immediately. Neither a SnapshotAck nor a sender survives it.
            worker.serve_stream(iter([pb.RecordOwnerFrame(claim=claim)]), lambda frame: None)
            channel = grpc.insecure_channel((root / "control.addr").read_text().strip())
            client = rpc.WorkerControlStub(channel)
            assert not worker.connected.is_set() and not worker.fence.dispatch_open
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
                request, {}, root / "grants-first", package_code_digest=binding.build_id
            )
            worker._apply_desired_state(generator._desired_state())
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
            snapshot = next(control).snapshot
            incoming.put(
                pb.RecordOwnerFrame(
                    snapshot_ack=pb.SnapshotAck(
                        record_owner_epoch=admitted.record_owner_epoch,
                        control_stream_epoch=admitted.control_stream_epoch,
                        worker_boot_id=admitted.worker_boot_id,
                        snapshot_id=snapshot.snapshot_id,
                        snapshot_digest=snapshot.snapshot_digest,
                    )
                )
            )

            def watch_control() -> None:
                with contextlib.suppress(grpc.RpcError):
                    observed.extend(control)

            observer = threading.Thread(target=watch_control, daemon=True)
            observer.start()
            while not worker.connected.is_set():
                assert time.monotonic() < until
                time.sleep(0.02)
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
                    package_code_digest=selected.build_id,
                )
                generator.host.inbound.get_nowait()  # helper's Claim is not transmitted
                generator._dispatch()
                message = generator.host.inbound.get_nowait()
                assert message is not None
                submission = pb.MachineExecutionSubmit(
                    claim=claim,
                    submission_id=identifier,
                    capture_digest=capture_digest,
                    capture_canonical_bytes=capture,
                    offer=message.attempt_offer,
                    prepared_state=generator._desired_state(),
                    payload_canonical_bytes=canonical_json.encode(generator.request.payload),
                )
                # Remove the old transport payload before intake: only the inline captured
                # arguments are available when the queued executor eventually starts.
                for path in generator.grant_dir.glob("*.json"):
                    path.unlink()
                if identifier == "first":
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
                    for minor, detail in ((51, "wire52"), (52, "no independent publication")):
                        assert worker.executions is not None
                        invalid.CopyFrom(submission)
                        invalid.offer.request_id = f"rejected-authority-{minor}"
                        invalid.submission_id = invalid.offer.request_id
                        invalid.claim.wire_minor = minor
                        invalid.publication_authorization_id = (
                            "019aaaab-0000-7000-8000-000000000003"
                        )
                        with pytest.raises(grpc.RpcError) as ungranted:
                            client.SubmitMachineExecution(invalid, timeout=30)
                        assert ungranted.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                        assert detail in ungranted.value.details()
                        assert not worker.executions.owns("owner", invalid.offer.request_id)
                receipt = client.SubmitMachineExecution(submission, timeout=30)
                assert client.SubmitMachineExecution(submission, timeout=30) == receipt
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
            incoming.put(
                pb.RecordOwnerFrame(
                    desired_state=pb.DesiredWorkerState(
                        record_owner_epoch=admitted.record_owner_epoch,
                        control_stream_epoch=admitted.control_stream_epoch,
                        worker_boot_id=admitted.worker_boot_id,
                        revision=999,
                        wire_minor=WIRE_MINOR,
                        posture=pb.POSTURE_ACCEPTING,
                    )
                )
            )
            while not any(
                fault.reason == "machine_execution_owns_worker" for fault in worker.engine.faults
            ):
                assert time.monotonic() < until
                time.sleep(0.02)
            assert worker.accepted.accepted_desired_state_revision == 1
            assert not any(
                frame.HasField("attempt_accepted") or frame.HasField("attempt_outcome")
                for frame in observed
            )
            incoming.put(None)
            control.cancel()
            observer.join(5)
            while worker.connected.is_set():
                assert time.monotonic() < until
                time.sleep(0.02)
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
            assert not worker.connected.is_set() and not worker.fence.dispatch_open
            assert thread.is_alive()  # completed results still await collection
            query.expected_execution_workspace_id = "replaced-workspace"
            with pytest.raises(grpc.RpcError) as mismatched:
                client.GetMachineExecution(query, timeout=5)
            assert mismatched.value.code() == grpc.StatusCode.FAILED_PRECONDITION
            channel.close()
            # Replace the entire Worker and gRPC server. No client preparation is
            # replayed, and the old wheel upload is no longer available.
            config, options = worker.config, dataclasses.replace(worker.options, python="")
            old_boot = worker.fence.worker_boot_id
            worker.request_stop()
            worker.host.stop()
            thread.join(30)
            assert not thread.is_alive()
            for path in (environment / ".stage").rglob("*.whl"):
                path.unlink()
            (root / "control.addr").unlink()
            worker = Worker(config, options, GrpcControlHost("127.0.0.1:0", root / "control.addr"))
            assert not worker.local_revisions and not worker.engine.jobs
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
            assert not worker.local_revisions and not worker.engine.jobs
            query.request_id = "second"
            assert (
                client.CollectMachineExecution(
                    pb.MachineExecutionCollect(execution=query), timeout=5
                )
                == outcome
            )
            assert client.GetMachineExecution(query, timeout=5).worker_boot_id != old_boot
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
            assert client.GetMachineExecution(query, timeout=5).state == "succeeded"
            assert client.ControlMachineExecution(resume, timeout=5) == resumed
            assert worker.local_revisions and worker.engine.jobs
            assert not worker.connected.is_set() and not worker.fence.dispatch_open
            channel.close()
        finally:
            worker.request_stop()
            worker.host.stop()
            if thread.ident is not None:
                thread.join(30)
            else:
                worker.shutdown()
            assert not thread.is_alive()


def test_queued_offer_prevents_preparation_switch_before_lane_accepts() -> None:
    from types import SimpleNamespace

    from cozy_runtime.internal.worker.attempts import AttemptEngine
    from cozy_runtime.internal.worker.session import Worker

    # No lane consumer has accepted this offer yet. A second captured package must
    # leave the first package's preparation and admission epoch intact during that gap.
    worker = SimpleNamespace(
        accepted=SimpleNamespace(accepted_desired_state_revision=1, desired_state_body=b"first"),
        engine=SimpleNamespace(
            history={("first", 1): SimpleNamespace(state="queued")},
            ON_DEVICE=AttemptEngine.ON_DEVICE,
            HELD=AttemptEngine.HELD,
        ),
    )
    state = pb.DesiredWorkerState(job=pb.JobDirective(build_id="different"))
    preparation = {"state": base64.b64encode(state.SerializeToString()).decode()}
    assert Worker._prepare_execution(worker, preparation) is False  # type: ignore[arg-type]
