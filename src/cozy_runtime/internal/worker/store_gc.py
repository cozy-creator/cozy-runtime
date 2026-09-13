"""The fixed worker store's native collector; roots and live readers decide safety."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import Worker

from cozy_runtime.internal import fill
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def collect(
    tensorfs_root: Path, *, keep_manifests: list[str] | None = None
) -> pb.CollectStoreGarbageResult:
    if not tensorfs_root.is_absolute():
        raise ValueError("store collection requires the configured absolute store root")
    tensorfs = fill.tensorfs_module()
    try:
        result = (
            tensorfs.gc(str(tensorfs_root))
            if keep_manifests is None
            else tensorfs.gc(str(tensorfs_root), evict_cached=True, keep_manifests=keep_manifests)
        )
    except Exception as exc:
        if getattr(exc, "code", None) == "STORE_BUSY":
            return pb.CollectStoreGarbageResult(store_busy=True)
        raise
    reclaimed = result["reclaimed_bytes"]
    if type(reclaimed) is not int or reclaimed < 0:
        raise ValueError("native collection returned invalid accounting")
    return pb.CollectStoreGarbageResult(reclaimed_bytes=reclaimed)


def collect_worker(worker: Worker) -> pb.CollectStoreGarbageResult:
    """Hold the existing placement/preparation generation through native reclamation.

    The supervisor separately fences its download-to-activation handoff. Shared local
    stores have no exclusive worker owner and remain ordinary unreferenced-object GC.
    A busy foreground lane wins immediately, without a waiter blocking control progress.
    """
    root = Path(worker.options.tensorfs_root or "")
    workspace = getattr(worker, "workspace", None)
    if workspace is not None:
        from cozy_runtime.internal import local_storage_admission

        from . import workspace_memo

        target = local_storage_admission.pressure_target(root)
        if target:
            pruned = workspace_memo.reclaim_unused(workspace, target_bytes=target)
            if pruned.store_busy or pruned.removed_entries:
                return pb.CollectStoreGarbageResult(
                    reclaimed_bytes=pruned.reclaimed_bytes, store_busy=pruned.store_busy
                )
    if not worker.options.owns_tensorfs_store:
        return collect(root)
    if not worker.preparation_lock.acquire(blocking=False):
        return pb.CollectStoreGarbageResult(store_busy=True)
    try:
        if not worker.control_lock.acquire(blocking=False):
            return pb.CollectStoreGarbageResult(store_busy=True)
        try:
            if worker.accepted.accepted_desired_state_revision == 0:
                # Host preparation/activation facts outlive this Runtime incarnation.
                # Until the owner replays desired state, those selected models are unknown
                # here and may not be made disposable from an empty in-memory census.
                return collect(root)
            keep: set[str] = set()
            placements = [worker.placement, worker.pending]
            placements.extend(hosted.placement for hosted in worker.hosted.values())
            placements.extend(worker.local_revisions.values())
            placements.extend(worker.local_operations.values())
            docs = [placement.document for placement in placements if placement is not None]
            docs.extend(worker.pending_hosted)
            if worker.accepted.desired_state_body:
                desired = pb.DesiredWorkerState.FromString(worker.accepted.desired_state_body)
                if desired.HasField("placement_set"):
                    body = documents.read(
                        desired.placement_set.placement_set_canonical_bytes, pb.PlacementSet
                    )
                    docs.extend(body.get("placements", []))
            for document in docs:
                for model in document.get("models", []):
                    keep.add(model["manifest"]["digest"])
            result = collect(root, keep_manifests=sorted(keep))
            if not result.store_busy:
                # Routing must stop advertising bytes a successful pressure pass removed.
                worker.held_manifests.rebuild(fill.open_store(fill.tensorfs_module(), root))
            return result
        finally:
            worker.control_lock.release()
    finally:
        worker.preparation_lock.release()
