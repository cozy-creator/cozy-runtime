"""Actual public Worker/gRPC cancellation starvation and same-process Claim recovery."""

import json
import queue
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import grpc
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.control import GrpcControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as p
from cozy_runtime.protocol import worker_pb2_grpc as g


def await_for(predicate: Callable[[], bool], why: str) -> None:
    until = time.monotonic() + 5
    while not predicate():
        if time.monotonic() > until:
            raise AssertionError(why)
        threading.Event().wait(0.01)


def frames(q: queue.Queue[Any]) -> Iterator[Any]:
    while True:
        frame = q.get()
        if frame is None:
            return
        yield frame


with tempfile.TemporaryDirectory(prefix="runtime-watch-starvation-") as directory:
    root = Path(directory)
    key = Ed25519PrivateKey.generate()
    bound = threading.Event()
    host = GrpcControlHost("127.0.0.1:0", root / "control.addr", on_bound=lambda _: bound.set())
    worker = Worker(
        RuntimeConfig(
            cozy_home=root / "home",
            credentials=Credentials(),
            record_owner_public_key=key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw),
        ),
        WorkerOptions(
            root=root / "worker",
            tensorfs_root=root / "store",
            worker_id="test-worker",
            worker_boot_id="same-boot",
            worker_tls_certificate_digest="sha256:" + "a" * 64,
        ),
        host,
    )
    host.weights = worker.weights
    host.checkpoint_pager = worker.checkpoint_page
    server = threading.Thread(target=host.serve, args=(worker,), daemon=True)
    server.start()
    assert bound.wait(5)
    channel = grpc.insecure_channel((root / "control.addr").read_text().strip())
    control = g.WorkerControlStub(channel)
    preparation = g.RuntimePreparationStub(channel)
    claim = p.Claim(
        record_owner_epoch=1,
        record_owner_id="owner",
        worker_id="test-worker",
        worker_boot_id="same-boot",
        wire_minor=WIRE_MINOR,
    )
    claim.proof = key.sign(
        documents.canonical_bytes(
            p.ClaimProof(
                record_owner_epoch=1,
                worker_boot_id="same-boot",
                worker_id="test-worker",
                worker_tls_certificate_digest=b"\xaa" * 32,
            )
        )
    )
    queues: list[queue.Queue[Any]] = []
    calls: list[Any] = []

    def connect() -> tuple[p.ClaimAck, Any]:
        q: queue.Queue[Any] = queue.Queue()
        queues.append(q)
        q.put(p.RecordOwnerFrame(claim=claim))
        call = control.Control(frames(q), timeout=15)
        calls.append(call)
        ack = next(call).claim_ack
        assert ack.accepted
        return ack, call

    try:
        ack, old_control = connect()
        watchers = []
        for _index in range(7):
            watch = control.WatchProgress(
                p.ProgressOpen(
                    record_owner_epoch=1,
                    control_stream_epoch=ack.control_stream_epoch,
                    worker_boot_id="same-boot",
                ),
                timeout=15,
            )
            watchers.append(watch)
            calls.append(watch)
            await_for(lambda: len(worker.watches.watchers) == len(watchers), "watch never opened")
        for watch in watchers:
            watch.cancel()
        result = preparation.CheckpointPage(p.CheckpointPageRequest(), timeout=3)
        assert result.safe_code
        await_for(lambda: not worker.watches.watchers, "canceled watchers remain registered")
        assert (
            list(
                worker.watch(
                    p.ProgressOpen(
                        record_owner_epoch=1,
                        control_stream_epoch=ack.control_stream_epoch,
                        worker_boot_id="same-boot",
                    ),
                    on_cancel=lambda _: False,
                )
            )
            == []
        )
        assert not worker.watches.watchers
        # Both deadline expiration and closing the client's connection take the same
        # cancellation path, without another progress frame or a new control Claim.
        opened = p.ProgressOpen(
            record_owner_epoch=1,
            control_stream_epoch=ack.control_stream_epoch,
            worker_boot_id="same-boot",
        )
        expired = control.WatchProgress(opened, timeout=2)
        calls.append(expired)
        await_for(lambda: len(worker.watches.watchers) == 1, "deadline watch never opened")
        assert expired.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        await_for(lambda: not worker.watches.watchers, "expired watch remains registered")
        with grpc.insecure_channel((root / "control.addr").read_text().strip()) as detached:
            disconnected = g.WorkerControlStub(detached).WatchProgress(opened, timeout=15)
            calls.append(disconnected)
            await_for(lambda: len(worker.watches.watchers) == 1, "client watch never opened")
        await_for(lambda: not worker.watches.watchers, "disconnected watch remains registered")
        assert preparation.CheckpointPage(p.CheckpointPageRequest(), timeout=3).safe_code
        print(
            json.dumps(
                {
                    "phase": "fixed",
                    "canceled_watchers": 7,
                    "deadline_and_disconnect": True,
                    "remaining": len(worker.watches.watchers),
                    "unary_safe_code": result.safe_code,
                }
            ),
            flush=True,
        )
        # Exactly what Host.endSession does before opening the new control session.
        old_control.cancel()
        fresh, new_control = connect()
        assert fresh.control_stream_epoch == ack.control_stream_epoch + 1
        result = preparation.CheckpointPage(p.CheckpointPageRequest(), timeout=3)
        assert result.safe_code
        assert all(w.ended.is_set() for w in worker.watches.watchers)
        print(
            json.dumps(
                {
                    "phase": "recovered",
                    "stream": fresh.control_stream_epoch,
                    "same_worker": True,
                    "same_boot": True,
                    "unary_safe_code": result.safe_code,
                    "worker_stopped": worker.stop.is_set(),
                }
            ),
            flush=True,
        )
        assert not worker.stop.is_set()
    finally:
        for call in calls:
            call.cancel()
        for q in queues:
            q.put(None)
        worker.watches.end_all()
        worker.shutdown()
        channel.close()
        server.join(5)
        assert not server.is_alive(), "control host did not finish transport teardown"
        assert not any(
            thread.name == "control-stream" and thread.is_alive()
            for thread in threading.enumerate()
        ), "control reader survived its host"
