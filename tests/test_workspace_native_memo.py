"""Source service memoization uses the existing index and actual native tree custody."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal import source_interfaces
from cozy_runtime.internal.worker import workspace_memo, workspace_sources
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def accepted(workspace: Workspace, parent: str) -> pb.NativeSourceCommand:
    raw = documents.canonical_bytes(
        pb.InvocationSpec(
            job=pb.JobInvocationSpec(
                build_id="sha256:" + "22" * 32,
                job_descriptor_id="sha256:" + "33" * 32,
            )
        )
    )
    spec = hashlib.sha256(raw).digest()
    workspace.accept(
        "owner",
        pb.AttemptOffer(
            request_id=parent,
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            invocation_spec_canonical_bytes=raw,
        ),
    )
    workspace.mark_running(
        "owner",
        pb.AttemptAccepted(
            request_id=parent,
            attempt_ordinal=1,
            invocation_spec_digest=spec,
        ),
    )
    args = {"repository": "example/model", "revision": "a" * 40}
    intent = canonical_json.encode(
        {
            "interface_digest": source_interfaces.DIGEST,
            "module": source_interfaces.MODULE,
            "export": "download_huggingface",
            "request": args,
        }
    )
    command = pb.NativeSourceCommand(
        service_id=workspace_sources.identity("owner", parent, 0),
        operation=pb.NATIVE_SOURCE_OPERATION_HUGGINGFACE,
        parent_call=pb.ChildCallRequest(
            parent_request_id=parent,
            parent_attempt_ordinal=1,
            parent_invocation_spec_digest=spec,
            call_index=0,
            interface_digest=documents.raw(source_interfaces.DIGEST),
            module=source_interfaces.MODULE,
            export="download_huggingface",
            intent_digest=hashlib.sha256(intent).digest(),
            request_canonical_bytes=canonical_json.encode(args),
        ),
    )
    workspace_sources.accepted(workspace, "owner", command)
    return command


def completed(
    workspace: Workspace, command: pb.NativeSourceCommand, data: bytes = b"weights"
) -> tuple[bytes, dict[str, Any]]:
    store = tensorfs.Store.open(str(workspace.store_root))
    digest = tensorfs.object_id(data)
    store.put_reader(io.BytesIO(data), digest, len(data))
    manifest = canonical_json.encode(
        {
            "entries": [
                {
                    "kind": "file",
                    "path": "weights.safetensors",
                    "blob": {"sha256": digest.removeprefix("sha256:"), "length": len(data)},
                }
            ]
        }
    )
    manifest_id = tensorfs.object_id(manifest)
    store.put_manifest(manifest, manifest_id, len(manifest))
    root = store.create_tree_root(
        workspace_sources.native_owner("owner", command.service_id), manifest_id, len(manifest)
    )
    value = canonical_json.encode(
        {
            "producer_request_id": command.service_id,
            "output_slot": "source",
            "manifest": {"digest": manifest_id, "length": len(manifest)},
            "tensorfs_receipt_digest": root["receipt_digest"],
        }
    )
    workspace_sources.record_complete(
        workspace, "owner", command, value, b"k" * 32, root["receipt"]
    )
    assert workspace_memo.retain_native_result(workspace, "owner", command.service_id) == value
    return value, dict(root)


def test_native_tree_hit_survives_producer_release_and_preserves_original_result(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "store")
    command = accepted(workspace, "first")
    value, root = completed(workspace, command)
    assert workspace_memo.record_native(workspace, "owner", command.service_id)
    store = tensorfs.Store.open(str(workspace.store_root))
    store.release_tree_root(root["owner"])
    tensorfs.gc(str(store.root))
    restarted = Workspace(Path(store.root))
    second = accepted(restarted, "second")
    hit = workspace_memo.lookup_native(restarted, "owner", second.service_id, b"k" * 32)
    assert hit is not None and hit["result"] == value and hit["native_receipt"] == root["receipt"]
    assert workspace_memo.record_reuse(restarted, "owner", second.service_id) == value
    with restarted.locked() as db:
        access = db.execute("SELECT * FROM source_access WHERE request='second'").fetchone()
        assert access["state"] == "held" and access["producer"] == root["owner"]
        assert db.execute("SELECT count(*) FROM operation_cache").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM weights").fetchone()[0] == 0
    with pytest.raises(WorkspaceRefusal, match="terminal kind"):
        workspace_memo.lookup(
            restarted,
            "owner",
            pb.LookupOperationCall(
                computation_digest=b"k" * 32,
                consumer_request_id="wrong-terminal",
            ),
        )
    workspace_memo.release_native_lookup(restarted, "owner", second.service_id)
    released = store.tree_root(access["retention_id"])
    assert released is not None and released["released"]
    with pytest.raises(WorkspaceRefusal, match="current accepted call"):
        workspace_memo.lookup_native(restarted, "owner", second.service_id, b"k" * 32)


def test_native_mapping_conflict_disables_reuse_without_rewriting_prior_results(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "store")
    first = accepted(workspace, "first")
    value, _ = completed(workspace, first)
    assert workspace_memo.record_native(workspace, "owner", first.service_id)
    second = accepted(workspace, "second")
    other, _ = completed(workspace, second, b"different weights")
    assert other != value
    assert not workspace_memo.record_native(workspace, "owner", second.service_id)
    third = accepted(workspace, "third")
    assert workspace_memo.lookup_native(workspace, "owner", third.service_id, b"k" * 32) is None
    with workspace.locked() as db:
        assert db.execute("SELECT state FROM operation_cache").fetchone()[0] == "disabled"
        results = {
            row[0] for row in db.execute("SELECT result FROM native_calls WHERE state='complete'")
        }
        assert results == {value, other}


def test_cancel_before_native_recipient_creation_prevents_late_tree_retain(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    first = accepted(workspace, "first")
    _, root = completed(workspace, first)
    assert workspace_memo.record_native(workspace, "owner", first.service_id)
    second = accepted(workspace, "second")
    with workspace.locked() as db:
        cached = db.execute("SELECT * FROM operation_cache").fetchone()
        payload = canonical_json.decode(cached["body"])
        (hold,) = payload["holds"]
        hold["retention_id"] = workspace_memo._hold_id(
            "consumer", "owner", second.service_id, b"k" * 32, hold["transaction_id"]
        )
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT INTO operation_lookups(owner,consumer,key,cache_id,body,state) "
            "VALUES(?,?,?,?,?,'retaining')",
            ("owner", second.service_id, b"k" * 32, cached["id"], canonical_json.encode(payload)),
        )
        workspace_memo._reserve_holds(workspace, db, "owner", payload["holds"])
        db.commit()
    workspace_memo.release_native_lookup(workspace, "owner", second.service_id)
    store = tensorfs.Store.open(str(workspace.store_root))
    released = store.tree_root(hold["retention_id"])
    assert released is not None and released["released"]
    with pytest.raises(tensorfs.errors.Refusal):
        store.retain_tree_root(root["owner"], hold["retention_id"])
