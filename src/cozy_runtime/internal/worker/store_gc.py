"""The fixed worker store's native collector; roots and live readers decide safety."""

from pathlib import Path

from cozy_runtime.internal import fill
from cozy_runtime.protocol import worker_pb2 as pb


def collect(tensorfs_root: Path) -> pb.CollectStoreGarbageResult:
    if not tensorfs_root.is_absolute():
        raise ValueError("store collection requires the configured absolute store root")
    tensorfs = fill.tensorfs_module()
    try:
        result = tensorfs.gc(str(tensorfs_root))
    except Exception as exc:
        if getattr(exc, "code", None) == "STORE_BUSY":
            return pb.CollectStoreGarbageResult(store_busy=True)
        raise
    reclaimed = result["reclaimed_bytes"]
    if type(reclaimed) is not int or reclaimed < 0:
        raise ValueError("native collection returned invalid accounting")
    return pb.CollectStoreGarbageResult(reclaimed_bytes=reclaimed)
