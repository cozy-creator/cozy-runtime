"""Supplemental custody/refusal checks; actual function proof uses Creator's CLI."""

from __future__ import annotations

import base64
import hashlib
import queue
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal import source_interfaces
from cozy_runtime.internal.pathkey import opaque_key
from cozy_runtime.internal.worker import commit_files, workspace_sources
from cozy_runtime.internal.worker import workspace_byte_outputs as outputs
from cozy_runtime.internal.worker.source_calls import SourceCalls
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_workspace_native_memo import accepted


def pending(tmp_path: Path) -> tuple[Workspace, pb.NativeSourceCommand, Path, bytes]:
    workspace = Workspace(tmp_path / "store")
    source = accepted(workspace, "script")
    data = canonical_json.encode({"report": "observed" * 12000})
    spool = tmp_path / "spool" / opaque_key("attempt-input-spool", "script", 1)
    spool.mkdir(parents=True)
    path = spool / "file-0001"
    path.write_bytes(data)
    arguments = {
        "handle": "attempt:script#1/file/0001",
        "digest": documents.spell(hashlib.sha256(data).digest()),
        "size_bytes": len(data),
        "media_type": "application/json",
    }
    intent = canonical_json.encode(
        {
            "interface_digest": source_interfaces.DIGEST,
            "module": source_interfaces.MODULE,
            "export": "commit_file",
            "request": arguments,
        }
    )
    command = pb.NativeSourceCommand(
        service_id=workspace_sources.identity("owner", "script", 1),
        operation=pb.NATIVE_SOURCE_OPERATION_COMMIT_FILE,
        phase=pb.NATIVE_SOURCE_PHASE_EXECUTE,
        parent_call=pb.ChildCallRequest(
            parent_request_id="script",
            parent_attempt_ordinal=1,
            parent_invocation_spec_digest=source.parent_call.parent_invocation_spec_digest,
            call_index=1,
            interface_digest=documents.raw(source_interfaces.DIGEST),
            module=source_interfaces.MODULE,
            export="commit_file",
            request_canonical_bytes=canonical_json.encode(arguments),
            intent_digest=hashlib.sha256(intent).digest(),
        ),
    )
    return workspace, command, path, data


def test_commit_snapshots_large_file_replays_without_spool_and_ignores_parent_result(
    tmp_path: Path,
) -> None:
    workspace, command, path, data = pending(tmp_path)
    frames: queue.Queue[pb.WorkerFrame] = queue.Queue()
    calls = SourceCalls(
        workspace, lambda: "owner", frames.put, lambda _: True, spool_root=tmp_path / "spool"
    )
    calls.handle(command)
    first = frames.get(timeout=30).native_source_status
    assert first.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, first.safe_code
    assert first.byte_output.content_bytes == len(data) > 48 << 10
    assert first.byte_output_attempt_ordinal == 1
    assert (
        first.byte_output_invocation_spec_digest
        == command.parent_call.parent_invocation_spec_digest
    )
    assert (
        canonical_json.decode(first.result_canonical_bytes)["file"]["media_type"]
        == "application/json"
    )
    path.unlink()
    calls.handle(command)
    replay = frames.get(timeout=30).native_source_status
    assert replay.byte_output == first.byte_output
    recipient = pb.NativeByteRetentionRequest(
        source=first.byte_output, retention_id="sha256:" + "ab" * 32
    )
    outputs.change_hold(workspace, "owner", recipient, release=False)
    spec = command.parent_call.parent_invocation_spec_digest
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
    with outputs.leased(workspace, "owner", recipient) as (_, lease, members):
        actual = bytearray(len(data))
        lease.read_into("sha256:" + members[0]["blob"]["sha256"], len(data), 0, len(data), actual)
        assert actual == data
    outputs.change_hold(workspace, "owner", recipient, release=True)


@pytest.mark.parametrize("fault", ["symlink", "directory", "changed", "oversize"])
def test_mutated_or_nonregular_pending_file_cannot_become_retained(
    tmp_path: Path,
    fault: str,
) -> None:
    workspace, command, path, data = pending(tmp_path)
    workspace_sources.accepted(workspace, "owner", command)
    plan = commit_files.prepare(workspace, "owner", command, b"k" * 32, tmp_path / "spool")
    if fault == "symlink":
        target = tmp_path / "foreign"
        target.write_bytes(data)
        path.unlink()
        path.symlink_to(target)
    elif fault == "directory":
        path.unlink()
        path.mkdir()
    else:
        path.write_bytes(b"x" * (len(data) + (fault == "oversize")))
    store = tensorfs.Store.open(str(workspace.store_root))
    with pytest.raises((OSError, WorkspaceRefusal)):
        commit_files.execute(
            store,
            {
                **plan,
                "arguments": canonical_json.decode(command.parent_call.request_canonical_bytes),
            },
        )
    assert store.tree_root(plan["view_owner"]) is None


def test_cancel_before_import_releases_only_after_permanent_stop(tmp_path: Path) -> None:
    workspace, command, _, _ = pending(tmp_path)
    workspace_sources.accepted(workspace, "owner", command)
    plan = commit_files.prepare(workspace, "owner", command, b"k" * 32, tmp_path / "spool")
    with workspace.locked() as db:
        db.execute(
            "UPDATE native_calls SET state='failed' WHERE service_id=?", (command.service_id,)
        )
    with pytest.raises(WorkspaceRefusal, match="permanent stop"):
        outputs.release_native(workspace, "owner", command.service_id)
    with workspace.locked() as db:
        db.execute(
            "UPDATE native_calls SET state='stopped' WHERE service_id=?", (command.service_id,)
        )
    outputs.release_native(workspace, "owner", command.service_id)
    outputs.release_native(workspace, "owner", command.service_id)
    with workspace.locked() as db:
        row = db.execute(
            "SELECT state FROM byte_outputs WHERE id=?", (plan["view_owner"],)
        ).fetchone()
        assert row["state"] == "released"


def test_file_completion_fenced_after_native_import(tmp_path: Path) -> None:
    workspace, command, _, _ = pending(tmp_path)
    workspace_sources.accepted(workspace, "owner", command)
    plan = commit_files.prepare(workspace, "owner", command, b"k" * 32, tmp_path / "spool")
    result = commit_files.execute(
        tensorfs.Store.open(str(workspace.store_root)),
        {**plan, "arguments": canonical_json.decode(command.parent_call.request_canonical_bytes)},
    )
    with workspace.locked() as db:
        db.execute(
            "UPDATE native_calls SET state='stopped' WHERE service_id=?", (command.service_id,)
        )
    with pytest.raises(WorkspaceRefusal, match="exact current"):
        outputs.complete_native(
            workspace,
            "owner",
            command,
            canonical_json.encode(result["result"]),
            b"k" * 32,
            base64.b64decode(result["native_receipt"]),
            plan["view_owner"],
        )
    outputs.release_native(workspace, "owner", command.service_id)
