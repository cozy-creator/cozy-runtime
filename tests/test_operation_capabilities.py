"""An owner below the package floor keeps its claim; only the operations it cannot use fail."""

from __future__ import annotations

import queue
from pathlib import Path

from cozy_runtime.internal import canonical
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import CAPABILITY_UNAVAILABLE_CODE, MIN_COMPATIBLE_WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb


def test_old_minor_offer_is_refused_as_capability_unavailable(tmp_path: Path) -> None:
    worker = Worker(
        RuntimeConfig(cozy_home=tmp_path / "home", credentials=Credentials()),
        WorkerOptions(root=tmp_path / "worker", tensorfs_root=tmp_path / "store", worker_id="w"),
        InMemoryControlHost(),
    )
    try:
        worker.fence.wire_minor = MIN_COMPATIBLE_WIRE_MINOR - 1
        spec = canonical.write({"format": "cozy.worker.v1.InvocationSpec/1"})
        worker.on_offer(
            pb.AttemptOffer(
                request_id="old-owner",
                attempt_ordinal=1,
                admission_epoch=worker.admission_epoch,
                invocation_spec_canonical_bytes=spec,
                invocation_spec_digest=documents.digest_of(spec),
            )
        )
        frames: list[pb.WorkerFrame] = []
        while True:
            try:
                frame = worker.outbound.get_nowait()
            except queue.Empty:
                break
            if frame is not None:
                frames.append(frame)
        (outcome,) = [
            frame.attempt_outcome for frame in frames if frame.HasField("attempt_outcome")
        ]
        body = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
        assert body["cause"]["code"] == pb.CAUSE_CODE_CAPABILITY_UNAVAILABLE
        assert body["safe_message"].startswith(CAPABILITY_UNAVAILABLE_CODE + ": execution needs")
        assert "update Creator" in body["safe_message"]
        assert not worker.latched
    finally:
        worker.shutdown()
