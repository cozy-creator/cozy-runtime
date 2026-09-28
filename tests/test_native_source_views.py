"""Supplemental native custody checks; the client CLI owns function qualification."""

from __future__ import annotations

import hashlib
import queue
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal import source_interfaces
from cozy_runtime.internal.worker import source_views, workspace_sources
from cozy_runtime.internal.worker import workspace_byte_outputs as byte_outputs
from cozy_runtime.internal.worker.source_calls import SourceCalls
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_workspace_native_memo import accepted, completed


def run_view(workspace: Workspace, command: pb.NativeSourceCommand) -> pb.NativeSourceStatus:
    frames: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
    calls = SourceCalls(workspace, lambda: "owner", frames.put, lambda _: True)
    calls.handle(command)
    with calls.lock:
        done = calls.tasks.get(command.service_id)
    result = frames.get(timeout=30)
    if done is not None:
        assert done.wait(30)
    return result


def finish_parent(workspace: Workspace, command: pb.NativeSourceCommand, *, retain: bool) -> None:
    call = command.parent_call
    body, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id=call.parent_request_id,
            attempt_ordinal=call.parent_attempt_ordinal,
            invocation_spec_digest=documents.spell(call.parent_invocation_spec_digest),
            status=pb.OUTCOME_STATUS_FAILED if retain else pb.OUTCOME_STATUS_SUCCEEDED,
            execution_started=True,
        )
    )
    workspace.outcome(
        "owner",
        pb.AttemptOutcome(
            request_id=call.parent_request_id,
            attempt_ordinal=call.parent_attempt_ordinal,
            invocation_spec_digest=call.parent_invocation_spec_digest,
            outcome_id=f"done-{call.parent_attempt_ordinal}",
            outcome_digest=digest,
            outcome_canonical_bytes=body,
        ),
    )
    workspace.acknowledge(
        "owner",
        pb.AttemptOutcomeAck(
            request_id=call.parent_request_id,
            attempt_ordinal=call.parent_attempt_ordinal,
            invocation_spec_digest=call.parent_invocation_spec_digest,
            outcome_id=f"done-{call.parent_attempt_ordinal}",
            outcome_digest=digest,
            retain_work=retain,
        ),
    )


def reattach(workspace: Workspace, command: pb.NativeSourceCommand) -> pb.NativeSourceCommand:
    call = command.parent_call
    with workspace.locked() as db:
        raw = db.execute(
            "SELECT invocation FROM attempts WHERE request=? AND ordinal=?",
            (call.parent_request_id, call.parent_attempt_ordinal),
        ).fetchone()[0]
    finish_parent(workspace, command, retain=True)
    copied = pb.NativeSourceCommand()
    copied.CopyFrom(command)
    copied.parent_call.parent_attempt_ordinal += 1
    ordinal = copied.parent_call.parent_attempt_ordinal
    workspace.accept(
        "owner",
        pb.AttemptOffer(
            request_id=call.parent_request_id,
            attempt_ordinal=ordinal,
            invocation_spec_digest=call.parent_invocation_spec_digest,
            invocation_spec_canonical_bytes=raw,
        ),
    )
    workspace.mark_running(
        "owner",
        pb.AttemptAccepted(
            request_id=call.parent_request_id,
            attempt_ordinal=ordinal,
            invocation_spec_digest=call.parent_invocation_spec_digest,
        ),
    )
    return copied


def view_call(
    source: pb.NativeSourceCommand, value: bytes, *, index: int = 1
) -> pb.NativeSourceCommand:
    parent = source.parent_call
    arguments = {"source": canonical_json.decode(value)}
    intent = canonical_json.encode(
        {
            "module": source_interfaces.MODULE,
            "export": "source_files",
            "request": arguments,
        }
    )
    return pb.NativeSourceCommand(
        service_id=workspace_sources.identity("owner", parent.parent_request_id, index),
        operation=pb.NATIVE_SOURCE_OPERATION_SOURCE_FILES,
        phase=pb.NATIVE_SOURCE_PHASE_EXECUTE,
        parent_call=pb.ChildCallRequest(
            parent_request_id=parent.parent_request_id,
            parent_attempt_ordinal=parent.parent_attempt_ordinal,
            parent_invocation_spec_digest=parent.parent_invocation_spec_digest,
            call_index=index,
            module=source_interfaces.MODULE,
            export="source_files",
            request_canonical_bytes=canonical_json.encode(arguments),
            intent_digest=hashlib.sha256(intent).digest(),
        ),
    )


def test_source_view_replays_one_producer_and_retained_child_survives_parent_ack(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "store")
    source = accepted(workspace, "script")
    value, _ = completed(workspace, source, b"metadata bytes")
    command = view_call(source, value)
    frames: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
    detached_before_reply: list[bool] = []

    def receive(status: pb.NativeSourceStatus) -> None:
        with calls.lock:
            detached_before_reply.append(command.service_id not in calls.tasks)
        frames.put(status)

    calls = SourceCalls(workspace, lambda: "owner", receive, lambda _: True)
    calls.handle(command)
    first = frames.get(timeout=30)
    assert detached_before_reply == [True]
    assert first.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, first.safe_code
    assert first.HasField("byte_output")
    assert first.byte_output_attempt_ordinal == 1
    assert (
        first.byte_output_invocation_spec_digest == source.parent_call.parent_invocation_spec_digest
    )
    assert first.byte_output.content_bytes == len(b"metadata bytes")
    calls.handle(command)
    repeated = frames.get(timeout=30)
    assert repeated.byte_output == first.byte_output
    assert repeated.native_receipt_canonical_bytes == first.native_receipt_canonical_bytes
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM byte_outputs").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM operation_cache").fetchone()[0] == 0
    recipient = pb.NativeByteRetentionRequest(
        source=first.byte_output, retention_id="sha256:" + "ab" * 32
    )
    byte_outputs.change_hold(workspace, "owner", recipient, release=False)
    spec = source.parent_call.parent_invocation_spec_digest
    body, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id="script",
            attempt_ordinal=1,
            invocation_spec_digest=documents.spell(spec),
            status=pb.OUTCOME_STATUS_SUCCEEDED,
            execution_started=True,
        )
    )
    workspace.outcome(
        "owner",
        pb.AttemptOutcome(
            request_id="script",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            outcome_id="done",
            outcome_digest=digest,
            outcome_canonical_bytes=body,
        ),
    )
    workspace.acknowledge(
        "owner",
        pb.AttemptOutcomeAck(
            request_id="script",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            outcome_id="done",
            outcome_digest=digest,
        ),
    )
    tensorfs.gc(str(workspace.store_root))
    store = tensorfs.Store.open(str(workspace.store_root))
    observed = store.tree_root(first.byte_output.producer_root_id)
    assert observed is not None and observed["released"]
    with byte_outputs.leased(workspace, "owner", recipient) as (_, lease, members):
        data = bytearray(len(b"metadata bytes"))
        lease.read_into("sha256:" + members[0].blob.sha256, len(data), 0, len(data), data)
        assert data == b"metadata bytes"
    byte_outputs.change_hold(workspace, "owner", recipient, release=True)
    byte_outputs.change_hold(workspace, "owner", recipient, release=True)


def test_source_view_rejects_forged_producer_without_reserving_bytes(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    source = accepted(workspace, "script")
    value, _ = completed(workspace, source)
    forged = canonical_json.decode(value)
    forged["producer_request_id"] = "another-source"
    command = view_call(source, canonical_json.encode(forged))
    frames: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
    calls = SourceCalls(workspace, lambda: "owner", frames.put, lambda _: True)
    calls.handle(command)
    assert frames.get(timeout=30).state == pb.NATIVE_SOURCE_STATE_FAILED
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM byte_outputs").fetchone()[0] == 0


def test_author_cannot_forge_source_view_namespace(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    source = accepted(workspace, "script")
    with pytest.raises(WorkspaceRefusal, match="namespace"):
        byte_outputs.commit(
            workspace,
            "owner",
            "script",
            1,
            source.parent_call.parent_invocation_spec_digest,
            byte_outputs.native_slot("source_files", 1),
            b"{}",
            [],
            64 << 20,
        )


@pytest.mark.parametrize("index", [1, 80])
def test_source_view_lost_reply_reattaches_without_rebinding_original_producer(
    tmp_path: Path,
    index: int,
) -> None:
    workspace = Workspace(tmp_path / "store")
    source = accepted(workspace, "script")
    value, _ = completed(workspace, source, b"original metadata")
    command = view_call(source, value, index=index)
    first = run_view(workspace, command)
    assert first.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, first.safe_code
    retried = reattach(workspace, command)
    restarted = Workspace(workspace.store_root)
    second = run_view(restarted, retried)
    assert second.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, second.safe_code
    assert second.parent_attempt_ordinal == 2
    assert second.byte_output_attempt_ordinal == 1
    assert second.byte_output == first.byte_output
    assert second.byte_output_invocation_spec_digest == first.byte_output_invocation_spec_digest
    recipient = pb.NativeByteRetentionRequest(
        source=second.byte_output, retention_id="sha256:" + "cd" * 32
    )
    byte_outputs.change_hold(restarted, "owner", recipient, release=False)
    finish_parent(restarted, retried, retain=False)
    tensorfs.gc(str(workspace.store_root))
    store = tensorfs.Store.open(str(workspace.store_root))
    observed = store.tree_root(first.byte_output.producer_root_id)
    assert observed is not None and observed["released"]
    with byte_outputs.leased(restarted, "owner", recipient) as (_, lease, members):
        data = bytearray(len(b"original metadata"))
        lease.read_into("sha256:" + members[0].blob.sha256, len(data), 0, len(data), data)
        assert data == b"original metadata"
    byte_outputs.change_hold(restarted, "owner", recipient, release=True)


@pytest.mark.parametrize("field,value", [("ordinal", 8), ("spec", b"wrong" * 7)])
def test_source_view_rejects_forged_recorded_producer(
    tmp_path: Path, field: str, value: int | bytes
) -> None:
    workspace = Workspace(tmp_path / "store")
    source = accepted(workspace, "script")
    data, _ = completed(workspace, source)
    command = view_call(source, data)
    first = run_view(workspace, command)
    assert first.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, first.safe_code
    with workspace.locked() as db:
        db.execute(f"UPDATE byte_outputs SET {field}=?", (value,))
    with pytest.raises(WorkspaceRefusal, match="producer"):
        source_views.replay(workspace, "owner", command)


def test_source_view_stop_before_journal_completion_releases_native_root(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    source = accepted(workspace, "script")
    data, _ = completed(workspace, source)
    command = view_call(source, data)
    workspace_sources.accepted(workspace, "owner", command)
    envelope = source_views.prepare(workspace, "owner", command, b"k" * 32)
    store = tensorfs.Store.open(str(workspace.store_root))
    manifest = envelope.manifest
    result = store.create_tree_root(envelope.view_owner, manifest.digest, manifest.length)
    workspace_sources.stopped(workspace, "owner", command)
    with pytest.raises(WorkspaceRefusal, match="current native call"):
        source_views.complete(
            workspace,
            "owner",
            command,
            b"{}",
            b"k" * 32,
            result["receipt"],
            envelope.view_owner,
        )
    source_views.abort_unfinished(workspace, "owner", command)
    tensorfs.gc(str(workspace.store_root))
    observed = store.tree_root(envelope.view_owner)
    assert observed is not None and observed["released"]


def test_source_view_failed_completion_retries_original_intent_and_native_root(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "store")
    source = accepted(workspace, "script")
    data, _ = completed(workspace, source)
    command = view_call(source, data)
    workspace_sources.accepted(workspace, "owner", command)
    envelope = source_views.prepare(workspace, "owner", command, b"k" * 32)
    store = tensorfs.Store.open(str(workspace.store_root))
    manifest = envelope.manifest
    native = store.create_tree_root(envelope.view_owner, manifest.digest, manifest.length)
    with workspace.locked() as db:
        db.execute(
            "UPDATE native_calls SET state='failed' WHERE service_id=?", (command.service_id,)
        )
    source_views.abort_unfinished(workspace, "owner", command)
    observed = store.tree_root(envelope.view_owner)
    assert observed is not None and not observed["released"]
    retried = reattach(workspace, command)
    result = run_view(Workspace(workspace.store_root), retried)
    assert result.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, result.safe_code
    assert result.parent_attempt_ordinal == 2
    assert result.byte_output_attempt_ordinal == 1
    assert result.byte_output.producer_root_id == envelope.view_owner
    assert result.native_receipt_canonical_bytes == native["receipt"]
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM byte_outputs").fetchone()[0] == 1


def test_source_view_explicit_cancel_cannot_restart_same_logical_call(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    source = accepted(workspace, "script")
    data, _ = completed(workspace, source)
    command = view_call(source, data)
    workspace_sources.accepted(workspace, "owner", command)
    envelope = source_views.prepare(workspace, "owner", command, b"k" * 32)
    workspace_sources.stopped(workspace, "owner", command)
    source_views.abort_unfinished(workspace, "owner", command)
    retried = reattach(workspace, command)
    result = run_view(workspace, retried)
    assert result.state == pb.NATIVE_SOURCE_STATE_CANCELED
    with workspace.locked() as db:
        assert (
            db.execute(
                "SELECT state FROM byte_outputs WHERE id=?", (envelope.view_owner,)
            ).fetchone()[0]
            == "released"
        )
