"""Supplemental custody/refusal checks; actual function proof uses Creator's CLI."""

from __future__ import annotations

import hashlib
import queue
import time
from pathlib import Path
from typing import Any

import msgspec
import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.author import App, FileAsset, Invocation, OutputError, Outputs, attempt
from cozy_runtime.author._calls import _Broker
from cozy_runtime.internal import source_interfaces
from cozy_runtime.internal.pathkey import opaque_key
from cozy_runtime.internal.worker import byte_outputs as final_outputs
from cozy_runtime.internal.worker import commit_files, workspace_sources
from cozy_runtime.internal.worker import workspace_byte_outputs as outputs
from cozy_runtime.internal.worker.source_calls import SourceCalls
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from durable_seam import wire
from test_native_source_views import finish_parent, reattach
from test_workspace_native_memo import accepted


class Empty(msgspec.Struct):
    pass


@pytest.mark.parametrize(
    "fault", ["", "foreign", "wrong_attempt", "unreturned", "return_committed"]
)
def test_owned_commit_facade_keeps_budget_and_explicit_result_boundary(
    tmp_path: Path, fault: str
) -> None:
    app = App()
    data = b"parent generated file" * 4000
    ready_path = tmp_path / "received"
    ready_path.write_bytes(data)
    digest = documents.spell(hashlib.sha256(data).digest())
    submitted = 0

    @app.job
    async def write(payload: Empty, out: Outputs) -> Empty:
        saved = out.save_bytes(data, media_type="application/json")
        if fault == "foreign":
            await out.commit(FileAsset(saved.ref))
        if fault == "wrong_attempt":
            saved._attempt = "another#1"
        ready = await out.commit(saved)
        assert ready.read_bytes() == data
        assert await out.commit(saved) is ready
        assert await out.commit(ready) is ready
        with pytest.raises(OutputError, match="aggregate limit"):
            out.save_bytes(b"x", media_type="application/json")
        if fault == "unreturned":
            out.save_bytes(b"", media_type="application/json")
        if fault == "return_committed":
            return ready  # type: ignore[return-value] # intentionally wrong R
        return Empty()

    def exchange(kind: str, body: dict[str, Any]) -> dict[str, Any]:
        nonlocal submitted
        if kind == "child_call":
            submitted += 1
            assert body["export"] == "commit_file"
            args = canonical_json.decode(body["payload"].encode())
            assert args["digest"] == digest and args["size_bytes"] == len(data)
            assert args["slot"] == "file/0001"
            return {"ok": True}
        if kind == "child_forget":
            return {"ok": True}
        assert kind == "child_poll"
        value = {
            "file": {
                "asset_ref": digest,
                "kind": "file",
                "digest": digest,
                "size_bytes": len(data),
                "media_type": "application/json",
            }
        }
        return {
            "ok": True,
            "state": "succeeded",
            "result": canonical_json.encode(value).decode(),
            "byte_grants": [
                {
                    "output_id": "file",
                    "kind": "file",
                    "digest": digest,
                    "length": len(data),
                    "local": str(ready_path),
                    "media_type": "application/json",
                }
            ],
        }

    broker = _Broker(
        "script#1",
        {(b.module, b.export): b for b in source_interfaces.bindings().values()},
        wire(exchange),
    )
    _, outcome, _ = attempt(
        app.get("write"),
        {},
        Invocation(
            "script#1", tmp_path, time.monotonic() + 30, calls=broker, max_output_bytes=len(data)
        ),
    )
    if fault:
        assert (
            outcome.code
            == {
                "foreign": "output_owner",
                "wrong_attempt": "output_owner",
                "unreturned": "unreturned_handle",
                "return_committed": "result_schema",
            }[fault]
        ), outcome
    else:
        assert outcome.terminal == "succeeded", outcome
    assert submitted == (0 if fault in ("foreign", "wrong_attempt") else 1)


def pending(
    tmp_path: Path, data: bytes | None = None
) -> tuple[Workspace, pb.NativeSourceCommand, Path, bytes]:
    workspace = Workspace(tmp_path / "store")
    source = accepted(workspace, "script")
    if data is None:
        data = canonical_json.encode({"report": "observed" * 12000})
    spool = tmp_path / "spool" / opaque_key("attempt-input-spool", "script", 1)
    spool.mkdir(parents=True)
    path = spool / "file-0001"
    path.write_bytes(data)
    arguments = {
        "slot": "file/0001",
        "digest": documents.spell(hashlib.sha256(data).digest()),
        "size_bytes": len(data),
        "media_type": "application/json",
    }
    intent = canonical_json.encode(
        {
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
    frames: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
    calls = SourceCalls(
        workspace, lambda: "owner", frames.put, lambda _: True, spool_root=tmp_path / "spool"
    )
    calls.handle(command)
    first = frames.get(timeout=30)
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
    replay = frames.get(timeout=30)
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
        lease.read_into("sha256:" + members[0].blob.sha256, len(data), 0, len(data), actual)
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
        commit_files.execute(store, plan)
    assert store.tree_root(plan.view_owner) is None


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
        row = db.execute("SELECT state FROM byte_outputs WHERE id=?", (plan.view_owner,)).fetchone()
        assert row["state"] == "released"


def replace_arguments(command: pb.NativeSourceCommand, arguments: dict[str, Any]) -> None:
    call = command.parent_call
    call.request_canonical_bytes = canonical_json.encode(arguments)
    call.intent_digest = hashlib.sha256(
        canonical_json.encode(
            {
                "module": source_interfaces.MODULE,
                "export": "commit_file",
                "request": arguments,
            }
        )
    ).digest()


@pytest.mark.parametrize("slot", ["../file-0001", "attempt:other#1/file/0001", "file/00001"])
def test_native_file_slot_cannot_override_current_parent_scope(tmp_path: Path, slot: str) -> None:
    workspace, command, _, _ = pending(tmp_path)
    args = canonical_json.decode(command.parent_call.request_canonical_bytes)
    args["slot"] = slot
    replace_arguments(command, args)
    workspace_sources.accepted(workspace, "owner", command)
    with pytest.raises(WorkspaceRefusal, match="own pending file"):
        commit_files.prepare(workspace, "owner", command, b"k" * 32, tmp_path / "spool")
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM byte_outputs").fetchone()[0] == 0


def test_accepted_slot_cannot_reuse_a_result_for_changed_bytes(tmp_path: Path) -> None:
    workspace, command, _, _ = pending(tmp_path)
    workspace_sources.accepted(workspace, "owner", command)
    plan = commit_files.prepare(workspace, "owner", command, b"k" * 32, tmp_path / "spool")
    args = canonical_json.decode(command.parent_call.request_canonical_bytes)
    args["digest"] = "sha256:" + "00" * 32
    replace_arguments(command, args)
    with pytest.raises(WorkspaceRefusal, match="another native source intent"):
        workspace_sources.accepted(workspace, "owner", command)
    with workspace.locked() as db:
        rows = db.execute("SELECT id,state FROM byte_outputs").fetchall()
        assert len(rows) == 1 and rows[0]["id"] == plan.view_owner


def test_file_completion_fenced_after_native_import(tmp_path: Path) -> None:
    workspace, command, _, _ = pending(tmp_path)
    workspace_sources.accepted(workspace, "owner", command)
    plan = commit_files.prepare(workspace, "owner", command, b"k" * 32, tmp_path / "spool")
    result = commit_files.execute(tensorfs.Store.open(str(workspace.store_root)), plan)
    with workspace.locked() as db:
        db.execute(
            "UPDATE native_calls SET state='stopped' WHERE service_id=?", (command.service_id,)
        )
    with pytest.raises(WorkspaceRefusal, match="exact current"):
        outputs.complete_native(
            workspace,
            "owner",
            command,
            canonical_json.encode(result.result),
            b"k" * 32,
            result.native_receipt,
            plan.view_owner,
        )
    outputs.release_native(workspace, "owner", command.service_id)


@pytest.mark.parametrize("imported", [False, True])
def test_interrupted_commit_reattaches_original_file_and_producer(
    tmp_path: Path, imported: bool
) -> None:
    workspace, command, path, data = pending(tmp_path)
    workspace_sources.accepted(workspace, "owner", command)
    plan = commit_files.prepare(workspace, "owner", command, b"k" * 32, tmp_path / "spool")
    if imported:
        commit_files.execute(tensorfs.Store.open(str(workspace.store_root)), plan)
        path.unlink()
    with workspace.locked() as db:
        db.execute(
            "UPDATE native_calls SET state='failed' WHERE service_id=?", (command.service_id,)
        )
    retry = reattach(workspace, command)
    spool = tmp_path / "spool" / opaque_key("attempt-input-spool", "script", 2)
    spool.mkdir(parents=True)
    (spool / "file-0001").write_bytes(data)
    frames: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
    restarted = Workspace(workspace.store_root)
    calls = SourceCalls(
        restarted, lambda: "owner", frames.put, lambda _: True, spool_root=tmp_path / "spool"
    )
    calls.handle(retry)
    result = frames.get(timeout=30)
    assert result.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, result.safe_code
    assert result.parent_attempt_ordinal == 2 and result.byte_output_attempt_ordinal == 1
    assert result.byte_output.producer_root_id == plan.view_owner
    assert result.byte_output.content_bytes == len(data)
    with restarted.locked() as db:
        assert db.execute("SELECT count(*) FROM byte_outputs").fetchone()[0] == 1
    finish_parent(restarted, retry, retain=False)
    store = tensorfs.Store.open(str(workspace.store_root))
    root = store.tree_root(plan.view_owner)
    assert root is not None and root["released"]


@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_explicit_reexport_retains_final_bytes_before_parent_service_release(
    tmp_path: Path, tree: bool, empty: bool
) -> None:
    workspace, command, path, data = pending(tmp_path, b"" if empty else None)
    frames: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
    calls = SourceCalls(
        workspace, lambda: "owner", frames.put, lambda _: True, spool_root=tmp_path / "spool"
    )
    calls.handle(command)
    produced = frames.get(timeout=30)
    assert produced.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, produced.safe_code
    received = pb.NativeByteRetentionRequest(
        source=produced.byte_output, retention_id="sha256:" + "a1" * 32
    )
    outputs.change_hold(workspace, "owner", received, release=False)
    path.unlink()
    spec = command.parent_call.parent_invocation_spec_digest
    returned = final_outputs.reexport(
        workspace,
        "owner",
        "script",
        1,
        spec,
        "returned",
        received,
        tree=tree,
        max_bytes=200000,
        media_type="application/json",
    )
    assert returned.native_tree.producer_root_id != produced.byte_output.producer_root_id
    assert returned.native_tree.manifest == produced.byte_output.manifest
    final_hold = pb.NativeByteRetentionRequest(
        source=returned.native_tree, retention_id="sha256:" + "a2" * 32
    )
    outputs.change_hold(workspace, "owner", final_hold, release=False)
    body, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id="script",
            attempt_ordinal=1,
            invocation_spec_digest=documents.spell(spec),
            status=pb.OUTCOME_STATUS_SUCCEEDED,
            execution_started=True,
            output_manifest=pb.OutputManifest(outputs=[returned]),
        )
    )
    workspace.outcome(
        "owner",
        pb.AttemptOutcome(
            request_id="script",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            outcome_id="returned",
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
            outcome_id="returned",
            outcome_digest=digest,
        ),
    )
    outputs.change_hold(workspace, "owner", received, release=True)
    tensorfs.gc(str(workspace.store_root))
    with outputs.leased(workspace, "owner", final_hold) as (_, lease, members):
        actual = bytearray(len(data))
        lease.read_into("sha256:" + members[0].blob.sha256, len(data), 0, len(data), actual)
        assert actual == data
    with pytest.raises(WorkspaceRefusal, match="retention"):
        final_outputs.reexport(
            workspace,
            "owner",
            "script",
            1,
            spec,
            "late",
            received,
            tree=tree,
            max_bytes=200000,
            media_type="application/json",
        )
    outputs.change_hold(workspace, "owner", final_hold, release=True)
