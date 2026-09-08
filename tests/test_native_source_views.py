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


def view_call(source: pb.NativeSourceCommand, value: bytes) -> pb.NativeSourceCommand:
    parent = source.parent_call
    arguments = {"source": canonical_json.decode(value)}
    intent = canonical_json.encode(
        {
            "interface_digest": source_interfaces.DIGEST,
            "module": source_interfaces.MODULE,
            "export": "source_files",
            "request": arguments,
        }
    )
    return pb.NativeSourceCommand(
        service_id=workspace_sources.identity("owner", parent.parent_request_id, 1),
        operation=pb.NATIVE_SOURCE_OPERATION_SOURCE_FILES,
        phase=pb.NATIVE_SOURCE_PHASE_EXECUTE,
        parent_call=pb.ChildCallRequest(
            parent_request_id=parent.parent_request_id,
            parent_attempt_ordinal=parent.parent_attempt_ordinal,
            parent_invocation_spec_digest=parent.parent_invocation_spec_digest,
            call_index=1,
            interface_digest=documents.raw(source_interfaces.DIGEST),
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
    frames: queue.Queue[pb.WorkerFrame] = queue.Queue()
    calls = SourceCalls(workspace, lambda: "owner", frames.put, lambda _: True)
    calls.handle(command)
    first = frames.get(timeout=30).native_source_status
    assert first.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, first.safe_code
    assert first.HasField("byte_output")
    assert first.byte_output_attempt_ordinal == 1
    assert (
        first.byte_output_invocation_spec_digest == source.parent_call.parent_invocation_spec_digest
    )
    assert first.byte_output.content_bytes == len(b"metadata bytes")
    calls.handle(command)
    repeated = frames.get(timeout=30).native_source_status
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
    assert store.tree_root(first.byte_output.producer_root_id)["released"]
    with byte_outputs.leased(workspace, "owner", recipient) as (_, lease, members):
        data = bytearray(len(b"metadata bytes"))
        lease.read_into("sha256:" + members[0]["blob"]["sha256"], len(data), 0, len(data), data)
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
    frames: queue.Queue[pb.WorkerFrame] = queue.Queue()
    calls = SourceCalls(workspace, lambda: "owner", frames.put, lambda _: True)
    calls.handle(command)
    assert frames.get(timeout=30).native_source_status.state == pb.NATIVE_SOURCE_STATE_FAILED
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
            source_views.slot(1),
            b"{}",
            [],
            64 << 20,
        )
