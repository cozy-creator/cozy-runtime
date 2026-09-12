"""Pressure cleanup uses genuine native roots, custody and collection counts."""

from __future__ import annotations

import io
from pathlib import Path

import tensorfs

from cozy_runtime.internal.worker import workspace_memo as memo
from cozy_runtime.protocol import worker_pb2 as pb
from test_workspace_custody import produced
from test_workspace_memo import completed, release_original


def test_pressure_keeps_active_producer_then_drops_only_unused_cache(tmp_path: Path) -> None:
    store, workspace, receipt = produced(tmp_path)
    call = completed(workspace, receipt)
    assert memo.record(workspace, "owner", call).recorded
    protected = memo.reclaim_unused(workspace, target_bytes=1)
    assert protected.removed_entries == protected.reclaimed_bytes == 0
    assert store.contains(tensorfs.object_id(b"\x31" * 2048))
    release_original(workspace, call, receipt)
    pruned = memo.reclaim_unused(workspace, target_bytes=1)
    assert pruned.removed_entries == 1 and pruned.reclaimed_bytes > 0
    assert not pruned.store_busy
    assert not memo.lookup(
        workspace,
        "owner",
        pb.LookupOperationCall(
            computation_digest=call.computation_digest, consumer_request_id="after-pressure"
        ),
    ).found


def test_pressure_defers_to_native_readers_without_dropping_mappings(tmp_path: Path) -> None:
    store, workspace, receipt = produced(tmp_path)
    call = completed(workspace, receipt)
    assert memo.record(workspace, "owner", call).recorded
    lease = store.acquire_manifest(receipt.artifact.manifest.digest)
    try:
        release_original(workspace, call, receipt)
        result = memo.reclaim_unused(workspace, target_bytes=1)
        assert result.store_busy and result.reclaimed_bytes == result.removed_entries == 0
        with workspace.locked() as db:
            assert db.execute("SELECT state FROM operation_cache").fetchone()[0] == "ready"
    finally:
        lease.release()
    assert memo.reclaim_unused(workspace, target_bytes=1).removed_entries == 1


def test_pressure_sweeps_unreferenced_bytes_without_eviction_candidates(tmp_path: Path) -> None:
    store, workspace, receipt = produced(tmp_path)
    call = completed(workspace, receipt)
    release_original(workspace, call, receipt)
    orphan = b"orphan" * 1024
    identity = tensorfs.object_id(orphan)
    store.put_reader(io.BytesIO(orphan), identity, len(orphan))
    result = memo.reclaim_unused(workspace, target_bytes=1)
    assert result.removed_entries == 0 and result.reclaimed_bytes >= len(orphan)
    assert not store.contains(identity)


def test_pressure_does_not_wait_behind_foreground_workspace_transition(tmp_path: Path) -> None:
    _, workspace, _ = produced(tmp_path)
    with workspace.locked():
        result = memo.reclaim_unused(workspace, target_bytes=1)
        assert result.store_busy and result.removed_entries == result.reclaimed_bytes == 0
