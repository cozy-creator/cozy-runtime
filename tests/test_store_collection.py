"""Explicit collection observes native roots and live read leases on one fixed store."""

from __future__ import annotations

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import tensorfs

from cozy_runtime.internal.worker import store_gc
from cozy_runtime.internal.worker.control import _PreparationServicer
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from test_weights_resume_native import _committed


def test_loopback_collection_is_fixed_store_and_defers_while_readers_are_live(
    tmp_path: Path,
) -> None:
    store, _, _, receipt = _committed(tmp_path)
    orphan = b"orphan" * 1024
    digest = "sha256:" + hashlib.sha256(orphan).hexdigest()
    store.put_reader(io.BytesIO(orphan), digest, len(orphan))
    other = tensorfs.Store.init(tmp_path / "other")
    other.put_reader(io.BytesIO(orphan), digest, len(orphan))
    lease = store.acquire_manifest(receipt.artifact.manifest.digest)
    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    rpc.add_RuntimePreparationServicer_to_server(
        _PreparationServicer(
            None,
            None,
            None,
            None,
            store_collector=lambda: store_gc.collect(Path(store.root)),
        ),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            client = rpc.RuntimePreparationStub(channel)
            busy = client.CollectStoreGarbage(pb.CollectStoreGarbageRequest(), timeout=5)
            assert busy.store_busy and busy.reclaimed_bytes == 0
            assert store.contains(digest)
            lease.release()
            collected = client.CollectStoreGarbage(pb.CollectStoreGarbageRequest(), timeout=5)
            assert not collected.store_busy and collected.reclaimed_bytes >= len(orphan)
            assert not store.contains(digest)
            assert store.manifest(receipt.artifact.manifest.digest)
            assert other.contains(digest)
    finally:
        if lease.live:
            lease.release()
        server.stop(0).wait()
