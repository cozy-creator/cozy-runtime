"""The fixed worker store's native collector; roots and live readers decide safety."""

from __future__ import annotations

from pathlib import Path

from cozy_runtime.internal import fill
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
