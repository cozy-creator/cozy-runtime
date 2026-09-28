"""Four actual gRPC uploads share one native lease; no synthetic held rows."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tempfile
import threading
from concurrent import futures
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import grpc
import tensorfs
from tensorfs.derived import Config as NativeConfig
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.internal.weights_sink import (
    protocol_receipt,
    receipt_from_native,
    weights_transaction_id,
)
from cozy_runtime.internal.worker.control import _RuntimeWeightsServicer
from cozy_runtime.internal.worker.weights import WeightsExchange
from cozy_runtime.protocol import worker_pb2 as p
from cozy_runtime.protocol import worker_pb2_grpc as g

with tempfile.TemporaryDirectory(prefix="runtime-native-uploads-") as directory:
    root = Path(directory)
    store = tensorfs.Store.init(root / "store")
    plain = next(digest for name, digest in tensorfs.seed_digests() if name == "plain/1")
    length = 8 << 20
    bodies = {f"weight_{i}": bytes([i + 1]) * length for i in range(4)}
    output_bound = 4 * length + 4096
    transaction = store.begin_derived(
        weights_transaction_id("upload-owner", "upload-request", "sha256:" + "a" * 64, "model"),
        1,
        *Derivation(
            sources={},
            targets={
                "model": Target(
                    add={
                        name: Tensor(
                            logical_dtype="f32",
                            shape=(length // 4,),
                            encoding=plain,
                            parts={"value": Part(dtype="f32", shape=(length // 4,))},
                        )
                        for name in bodies
                    }
                )
            },
            configs={"model": NativeConfig("add")},
            order=tuple(("model", name) for name in bodies),
        ).native_arguments(output_bound),
        work_fingerprint="sha256:" + "b" * 64,
    )
    for name, body in bodies.items():
        transaction.add_part("model", name, "value", io.BytesIO(body))
    transaction.add_config("model", io.BytesIO(b"{}"))
    facts = transaction.commit()
    receipt = receipt_from_native(
        "model", facts["transaction_id"], facts, replayed=False, request_id="upload-request"
    )
    reference, _, _ = protocol_receipt(
        receipt,
        owner_scope="upload-owner",
        request_id="upload-request",
        invocation_spec_digest="sha256:" + "a" * 64,
    )
    exchange = WeightsExchange(
        store_root=root / "store",
        stop=threading.Event(),
        owner_scope=lambda: "upload-owner",
        allow_private_egress=True,
    )
    held = exchange._hold(receipt.weights_transaction_id, 1, reference)
    expected = {"sha256:" + hashlib.sha256(body).hexdigest(): body for body in bodies.values()}
    received: dict[str, bytes] = {}
    received_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_PUT(self) -> None:
            object_id = self.path.removeprefix("/")
            body = self.rfile.read(int(self.headers["Content-Length"]))
            assert body == expected[object_id]
            assert self.headers["If-None-Match"] == "*"
            assert (
                self.headers["X-Amz-Checksum-Sha256"]
                == base64.b64encode(hashlib.sha256(body).digest()).decode()
            )
            with received_lock:
                received[object_id] = body
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=origin.serve_forever, daemon=True).start()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    g.add_RuntimeWeightsServicer_to_server(_RuntimeWeightsServicer(exchange), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    client = g.RuntimeWeightsStub(channel)
    barrier = threading.Barrier(4)

    def upload(object_id: str) -> Any:
        source = held.objects[object_id]
        request = p.WeightsUploadRequest(
            weights_transaction_id=receipt.weights_transaction_id,
            writer_epoch=1,
            object_id=object_id,
            source_ref=source.source_ref,
            length=source.length,
            operation_id="upload-" + object_id[7:],
            grant_revision=1,
            grant=p.WeightsUploadGrant(
                object_id=object_id,
                length=source.length,
                url=f"http://127.0.0.1:{origin.server_port}/{object_id}",
                required_headers=[
                    p.WeightsUploadHeader(name="if-none-match", value="*"),
                    p.WeightsUploadHeader(
                        name="x-amz-checksum-sha256",
                        value=base64.b64encode(bytes.fromhex(object_id[7:])).decode(),
                    ),
                ],
            ),
        )
        barrier.wait()
        result = client.Upload(request, timeout=10)
        assert result.outcome == p.WEIGHTS_UPLOAD_OUTCOME_UPLOADED, result
        assert result.transferred_bytes == length and result.checksum_sha256 == object_id
        return object_id

    print(json.dumps({"phase": "held", "objects": len(expected), "bytes": 4 * length}), flush=True)
    try:
        with futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(upload, expected))
        assert received == expected
        print(
            json.dumps({"phase": "complete", "uploads": len(results), "bytes": 4 * length}),
            flush=True,
        )
    finally:
        channel.close()
        server.stop(0)
        origin.shutdown()
        origin.server_close()
        exchange.close()
