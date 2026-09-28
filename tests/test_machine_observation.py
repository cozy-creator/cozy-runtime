"""Authenticated observation does not require opening the dispatch barrier."""

from __future__ import annotations

import queue
import tempfile
import threading
import time
from pathlib import Path

import grpc

import signed_claims
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.control import GrpcControlHost
from cozy_runtime.internal.worker.session import REPORT_SECONDS, Worker, WorkerOptions
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc


def test_claim_only_stream_reports_without_acknowledging_dispatch() -> None:
    with tempfile.TemporaryDirectory(prefix="cz-observe.") as raw:
        root = Path(raw)
        worker = Worker(
            RuntimeConfig(
                cozy_home=root / "home",
                credentials=Credentials(),
                record_owner_public_key=signed_claims.PUBLIC_KEY,
            ),
            WorkerOptions(
                **signed_claims.IDENTITY,
                root=root / "worker",
                devices="",
                accelerator_backend="none",
            ),
            GrpcControlHost("127.0.0.1:0", root / "address"),
        )
        thread = threading.Thread(target=worker.run, daemon=True)
        inbound: queue.Queue[pb.RecordOwnerFrame | None] = queue.Queue()
        thread.start()
        channel = None
        stream = None
        try:
            deadline = time.monotonic() + 15
            while not (root / "address").exists():
                assert thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.02)
            channel = grpc.insecure_channel((root / "address").read_text().strip())
            inbound.put(pb.RecordOwnerFrame(claim=signed_claims.claim()))
            stream = rpc.WorkerControlStub(channel).Control(
                iter(inbound.get, None), timeout=3 * REPORT_SECONDS + 5
            )
            ack = next(stream).claim_ack
            assert ack.accepted and ack.control_stream_epoch > 0
            assert next(stream).HasField("snapshot")
            report = next(stream)
            assert report.HasField("observed_state")
            observed = report.observed_state
            assert observed.control_stream_epoch == ack.control_stream_epoch
            assert observed.worker_boot_id == ack.worker_boot_id
            assert observed.worker_phase == pb.WORKER_PHASE_ONLINE
            assert observed.admission_state == pb.ADMISSION_STATE_CLOSED
            assert not worker.engine.history
        finally:
            inbound.put(None)
            if stream is not None:
                stream.cancel()
            if channel is not None:
                channel.close()
            worker.request_stop()
            worker.host.stop()
            thread.join(15)
            assert not thread.is_alive()
