"""The fixed worker store's native collector; roots and live readers decide safety."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import Worker

from cozy_runtime.internal import fill
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

#: Collections run by this process: bytes verified at one generation are still here while
#: it holds. Another process's eviction still meets admission's native checks.
generation = 0


def _collected() -> None:
    global generation
    generation += 1


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
        _collected()  # a failed pass may still have removed objects
        raise
    _collected()
    # Collection already ran; unreadable accounting reports nothing reclaimed.
    reclaimed = result.get("reclaimed_bytes") if isinstance(result, dict) else None
    return pb.CollectStoreGarbageResult(
        reclaimed_bytes=reclaimed if type(reclaimed) is int and reclaimed >= 0 else 0
    )


def collect_worker(worker: Worker) -> pb.CollectStoreGarbageResult:
    """Hold the existing placement/preparation generation through native reclamation.

    Prepared metadata is reusable, but does not own model bytes. Accepted desired
    state protects acquisition; a cache miss before acceptance asks the owner to
    ensure the exact models again. Shared local
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
                # Until desired-state replay establishes this incarnation's input owners,
                # an empty in-memory census cannot authorize cache eviction.
                return collect(root)
            keep: set[str] = set()
            placements = [worker.placement, worker.pending]
            placements.extend(hosted.placement for hosted in worker.hosted.values())
            docs = [placement.document for placement in placements if placement is not None]
            docs.extend(worker.pending_hosted)
            if worker.accepted.desired_state_body:
                desired = pb.DesiredWorkerState.FromString(worker.accepted.desired_state_body)
                if desired.HasField("placement_set"):
                    docs.extend(
                        documents.parse(
                            desired.placement_set.placement_set_canonical_bytes, pb.PlacementSet
                        ).placements
                    )
            for document in docs:
                for model in document.models:
                    keep.add(documents.spell(model.manifest.digest))
            result = collect(root, keep_manifests=sorted(keep))
            if not result.store_busy:
                # Routing must stop advertising bytes a successful pressure pass removed.
                worker.held_manifests.rebuild(fill.open_store(root))
            return result
        finally:
            worker.control_lock.release()
    finally:
        worker.preparation_lock.release()
