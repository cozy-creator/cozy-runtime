"""Retained native results read across Runtime/TensorFS versions by the identity they use."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import msgspec
import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author._services import MAX_OUTPUT_BYTES
from cozy_runtime.internal.worker import workspace_byte_outputs as byte_outputs
from cozy_runtime.internal.worker import workspace_memo, workspace_sources
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from test_owned_file_commit import pending
from test_workspace_custody import produced
from test_workspace_memo import completed as outcome
from test_workspace_native_memo import accepted, completed


def _additive(value: dict[str, Any]) -> dict[str, Any]:
    """The same artifact as a version with more members writes it."""
    return {
        **value,
        "retention_hint": "warm",
        "manifest": {**value["manifest"], "encoding": "tree/2"},
    }


def test_native_source_result_with_additive_members_completes_and_memoizes(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "store")
    first = accepted(workspace, "first")
    value, root = completed(workspace, first, shape=_additive)
    assert workspace_memo.record_native(workspace, "owner", first.service_id)

    # A replayed completion from a version without those members is the same result.
    current = {
        key: val for key, val in canonical_json.decode(value).items() if key != "retention_hint"
    }
    current["manifest"] = {
        "digest": current["manifest"]["digest"],
        "length": current["manifest"]["length"],
    }
    workspace_sources.record_complete(
        workspace, "owner", first, canonical_json.encode(current), b"k" * 32, root["receipt"]
    )

    # The same bytes computed by the current version agree with the retained memo entry.
    second = accepted(workspace, "second")
    completed(workspace, second)
    assert workspace_memo.record_native(workspace, "owner", second.service_id)
    third = accepted(workspace, "third")
    hit = workspace_memo.lookup_native(workspace, "owner", third.service_id, b"k" * 32)
    assert hit is not None
    with workspace.locked() as db:
        assert db.execute("SELECT state FROM operation_cache").fetchone()[0] != "disabled"

    for tampered in (
        {
            **current,
            "manifest": {**current["manifest"], "length": current["manifest"]["length"] + 1},
        },
        {**current, "producer_request_id": "another-producer"},
        {**current, "tensorfs_receipt_digest": "sha256:" + "ab" * 32},
    ):
        with pytest.raises(WorkspaceRefusal, match="native receipt"):
            workspace_sources.record_complete(
                workspace,
                "owner",
                first,
                canonical_json.encode(tampered),
                b"k" * 32,
                root["receipt"],
            )


def test_file_commit_manifest_with_additive_members_reserves_its_accepted_blob(
    tmp_path: Path,
) -> None:
    workspace, command, _, data = pending(tmp_path)
    workspace_sources.accepted(workspace, "owner", command)
    entry: dict[str, Any] = {
        "path": "payload",
        "kind": "file",
        "mode": "0644",
        "blob": {"sha256": hashlib.sha256(data).hexdigest(), "length": len(data), "chunks": 1},
    }
    tampered = {**entry, "blob": {**entry["blob"], "length": len(data) + 1}}
    with pytest.raises(WorkspaceRefusal, match="accepted call"):
        byte_outputs.reserve_native(
            workspace,
            "owner",
            command,
            canonical_json.encode({"entries": [tampered]}),
            MAX_OUTPUT_BYTES,
        )
    row = byte_outputs.reserve_native(
        workspace,
        "owner",
        command,
        canonical_json.encode({"entries": [entry]}),
        MAX_OUTPUT_BYTES,
    )
    assert row.state == "intent" and row.content_bytes == len(data)


def test_typed_model_result_with_additive_members_records_and_agrees(tmp_path: Path) -> None:
    schema = b'{"input":"model"}'
    _, workspace, first = produced(tmp_path, result_schema=schema)
    value = _additive(msgspec.to_builtins(first.artifact))
    original = outcome(workspace, first, result=canonical_json.encode(value))
    assert workspace_memo.record(workspace, "owner", original).recorded
    _, _, second = produced(tmp_path, request_id="another-producer", result_schema=schema)
    equivalent = outcome(workspace, second, result=msgspec.json.encode(second.artifact))
    assert workspace_memo.record(workspace, "owner", equivalent).recorded
    with workspace.locked() as db:
        assert db.execute("SELECT state FROM operation_cache").fetchone()[0] != "disabled"

    _, _, third = produced(tmp_path, request_id="tampered-producer", result_schema=schema)
    value = msgspec.to_builtins(third.artifact)
    value["manifest"]["length"] += 1
    with pytest.raises(WorkspaceRefusal, match="differs from its native receipt"):
        workspace_memo.record(
            workspace, "owner", outcome(workspace, third, result=canonical_json.encode(value))
        )
