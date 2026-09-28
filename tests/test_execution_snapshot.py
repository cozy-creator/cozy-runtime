"""Reclaiming a control stream cannot turn Runtime executions into owner assignments."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from pathlib import Path

import signed_claims
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_execution import ack, complete, offer


def test_reconnect_preserves_machine_collection_and_legacy_recovery(tmp_path: Path) -> None:
    worker = Worker(
        RuntimeConfig(
            cozy_home=tmp_path / "home",
            credentials=Credentials(),
            record_owner_public_key=signed_claims.PUBLIC_KEY,
        ),
        WorkerOptions(
            **signed_claims.IDENTITY, root=tmp_path / "worker", tensorfs_root=tmp_path / "store"
        ),
        InMemoryControlHost(),
    )
    claim = signed_claims.claim()
    observed: queue.Queue[pb.WorkerFrame] = queue.Queue()
    try:
        assert worker.accept_claim(claim, observed.put)[0] == 1
        assert worker.executions is not None and worker.workspace is not None
        executions, workspace = worker.executions, worker.workspace
        for request in ("machine-active", "machine-retained"):
            executions.submit(
                "owner",
                request,
                b"c" * 32,
                offer(request, job=False),
                expected_execution_workspace_id=executions.workspace_id,
                worker_boot=worker.fence.worker_boot_id,
            )
        retained = complete(executions, "machine-retained", {"value": 7})
        # Real engine admissions exercise the live projection independently of retained rows.
        for request in ("machine-active", "legacy-active"):
            submitted = offer(request, job=False)
            submitted.grant.invocation_spec_digest = submitted.invocation_spec_digest
            if request == "legacy-active":
                workspace.accept("owner", submitted)
            attempt = worker.engine.offer(submitted)
            worker.send(pb.WorkerFrame(attempt_accepted=worker.engine._accepted(attempt)))

        submitted = offer("legacy-retained", job=False)
        workspace.accept("owner", submitted)
        body, digest = documents.identity(
            pb.AttemptOutcomeBody(
                request_id=submitted.request_id,
                attempt_ordinal=1,
                invocation_spec_digest=documents.spell(submitted.invocation_spec_digest),
                status=pb.OUTCOME_STATUS_SUCCEEDED,
            )
        )
        legacy_terminal = pb.AttemptOutcome(
            request_id=submitted.request_id,
            attempt_ordinal=1,
            invocation_spec_digest=submitted.invocation_spec_digest,
            outcome_id="legacy-terminal",
            outcome_digest=digest,
            outcome_canonical_bytes=body,
        )
        workspace.outcome("owner", legacy_terminal)

        def receive(
            incoming: queue.Queue[pb.RecordOwnerFrame | None],
        ) -> Iterator[pb.RecordOwnerFrame]:
            while (frame := incoming.get()) is not None:
                yield frame

        for epoch in (2, 3):
            incoming: queue.Queue[pb.RecordOwnerFrame | None] = queue.Queue()
            incoming.put(pb.RecordOwnerFrame(claim=claim))
            stream = threading.Thread(
                target=worker.serve_stream,
                args=(receive(incoming), observed.put),
                daemon=True,
            )
            stream.start()
            try:
                accepted = observed.get(timeout=5).claim_ack
                assert accepted.accepted and accepted.control_stream_epoch == epoch
                snapshot = observed.get(timeout=5).snapshot
                assert (
                    documents.digest_of(snapshot.snapshot_canonical_bytes)
                    == snapshot.snapshot_digest
                )
                document = documents.read(snapshot.snapshot_canonical_bytes, pb.WorkerSnapshotBody)
                held = {row["request_id"]: row for row in document["held_attempts"]}
                assert set(held) == {"legacy-active", "legacy-retained"}
                assert held["legacy-active"]["state"] == pb.ATTEMPT_STATE_RUNNING
                assert held["legacy-retained"]["state"] == pb.ATTEMPT_STATE_OUTCOME_PENDING_ACK
                incoming.put(
                    pb.RecordOwnerFrame(
                        snapshot_ack=pb.SnapshotAck(
                            record_owner_epoch=accepted.record_owner_epoch,
                            control_stream_epoch=accepted.control_stream_epoch,
                            worker_boot_id=accepted.worker_boot_id,
                            snapshot_id=snapshot.snapshot_id,
                            snapshot_digest=snapshot.snapshot_digest,
                        )
                    )
                )
                replayed = []
                while True:
                    frame = observed.get(timeout=5)
                    if frame.HasField("observed_state"):
                        break
                    if frame.HasField("attempt_outcome"):
                        replayed.append(frame.attempt_outcome.request_id)
                assert replayed == ["legacy-retained"]
                assert worker.connected.is_set() and worker.fence.dispatch_open

            finally:
                incoming.put(None)
                stream.join(5)
                assert not stream.is_alive()
            assert not worker.connected.is_set()
            assert worker.collect_execution(claim, "machine-retained") == retained
            assert worker.execution_status(claim, "machine-active").state == "running"
            assert not worker.execution_status(claim, "machine-retained").collected
        assert worker.acknowledge_execution_collection(claim, ack(retained)).collected
        assert worker.collect_execution(claim, "machine-retained") == retained
    finally:
        worker.shutdown()
