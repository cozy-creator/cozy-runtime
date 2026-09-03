"""The worker owns the durable attempt record (th-142 child 2), proved by killing one.

The pod's supervisor used to keep this record — a SQLite ledger of every attempt it forwarded
— and rode the survivors beside the worker's snapshot as a second document. th-142 measured
that document and found nothing in it the worker could not observe, so the record moved here.
A record only exists if it survives the process, and the only honest proof of that is to end
the process while it holds one.

So the first leg SIGKILLs a REAL worker that is REALLY holding a REAL marco-polo attempt: a
child process runs the whole `run_slice` path, gets an executor, accepts the attempt (which is
what writes the durable row) and then holds it; the parent kills -9 from outside. A fresh
worker then boots on the same record root and runs the REAL stream conversation — Claim,
ClaimAck, WorkerSnapshot, SnapshotAck, replay, ack — over the real in-process control host.
What it must show:

* the snapshot names the killed attempt, OUTCOME_PENDING_ACK, with a terminal the dead
  process never authored;
* that terminal replays as an `AttemptOutcome` after the barrier and not before it;
* the acknowledgement closes it, and a third boot on the same root owes nothing;
* and a boot BETWEEN those two replays the SAME outcome id and the SAME bytes — an ordinal
  is not re-terminated once per restart.

Nothing here is mocked. The one thing the harness does is hold the attempt open, which is what
a worker that is about to be killed looks like from the inside.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

from cozy_runtime.internal.worker import durable
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.records import WorkerRecords
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "examples" / "marco-polo"

#: A failsafe only. Every assertion below is on an observed event — a file that appears, a
#: frame that arrives, a process that is gone — never on elapsed time.
WAIT = 600.0

needs_executor = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")


def _until(condition: Any, why: str) -> None:
    deadline = time.monotonic() + WAIT
    while not condition():
        assert time.monotonic() < deadline, why
        time.sleep(0.05)


# --------------------------------------------------------------------------- the fresh boot


class Reclaim:
    """A RecordOwner that knows nothing but the protocol: it claims, reconciles what the
    worker reports, acknowledges the exact snapshot, and takes the replay."""

    def __init__(self, worker: Worker, host: InMemoryControlHost) -> None:
        self.worker = worker
        self.host = host
        self.frames: list[pb.WorkerFrame] = []
        self.arrived = threading.Event()
        host.on_frame = self._observe
        self.thread = threading.Thread(target=host.serve, args=(worker,), daemon=True)

    def _observe(self, frame: pb.WorkerFrame) -> None:
        self.frames.append(frame)
        self.arrived.set()

    def of_kind(self, kind: str) -> list[Any]:
        return [getattr(f, kind) for f in list(self.frames) if f.WhichOneof("msg") == kind]

    def start(self) -> None:
        self.host.deliver(
            pb.RecordOwnerFrame(
                claim=pb.Claim(
                    record_owner_epoch=1, record_owner_id="reclaim", wire_minor=WIRE_MINOR
                )
            )
        )
        self.thread.start()

    def snapshot(self) -> pb.WorkerSnapshot:
        _until(lambda: self.of_kind("snapshot"), "the worker never sent a WorkerSnapshot")
        return cast(pb.WorkerSnapshot, self.of_kind("snapshot")[0])

    def acknowledge(self, snapshot: pb.WorkerSnapshot) -> None:
        self.host.deliver(
            pb.RecordOwnerFrame(
                snapshot_ack=pb.SnapshotAck(
                    record_owner_epoch=snapshot.record_owner_epoch,
                    control_stream_epoch=snapshot.control_stream_epoch,
                    worker_boot_id=snapshot.worker_boot_id,
                    snapshot_id=snapshot.snapshot_id,
                    snapshot_digest=snapshot.snapshot_digest,
                )
            )
        )

    def outcome(self) -> pb.AttemptOutcome:
        _until(
            lambda: self.of_kind("attempt_outcome"),
            "the recovered terminal never replayed after the barrier",
        )
        return cast(pb.AttemptOutcome, self.of_kind("attempt_outcome")[0])

    def close(self, outcome: pb.AttemptOutcome, snapshot: pb.WorkerSnapshot) -> None:
        self.host.deliver(
            pb.RecordOwnerFrame(
                outcome_ack=pb.AttemptOutcomeAck(
                    record_owner_epoch=snapshot.record_owner_epoch,
                    control_stream_epoch=snapshot.control_stream_epoch,
                    worker_boot_id=snapshot.worker_boot_id,
                    request_id=outcome.request_id,
                    attempt_ordinal=outcome.attempt_ordinal,
                    invocation_spec_digest=outcome.invocation_spec_digest,
                    outcome_id=outcome.outcome_id,
                    outcome_digest=outcome.outcome_digest,
                )
            )
        )

    def stop(self) -> None:
        self.host.close()
        self.thread.join(timeout=WAIT)
        self.worker.stop.set()
        with contextlib.suppress(Exception):
            self.worker.weights.close()


def _boot(root: Path, record_root: Path, name: str) -> Reclaim:
    """A worker with nothing but its record root: no placement, no executor, no history in
    memory. Everything it reports about the killed attempt came off the disk."""
    from test_device_lanes import _config

    options = WorkerOptions(
        root=root / name,
        worker_id="wrk-recovery",
        instance_id=name,
        devices="0",
        no_warm=True,
        record_root=record_root,
    )
    host = InMemoryControlHost()
    worker = Worker(_config(root / "home"), options, host)
    return Reclaim(worker, host)


# --------------------------------------------------------------------------- the killed leg


@needs_executor
def test_a_killed_worker_still_owes_the_attempt_it_was_holding() -> None:
    with tempfile.TemporaryDirectory(prefix="cz-record.", dir="/tmp") as raw:
        root = Path(raw)
        record_root = root / "record"
        holding = root / "holding"

        child = subprocess.Popen(
            [sys.executable, str(Path(__file__)), str(root)],
            cwd=str(PACKAGE),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent)},
        )
        try:
            _until(
                lambda: holding.is_file() or child.poll() is not None,
                "the worker never reached an accepted attempt",
            )
            assert child.poll() is None, f"the worker exited early ({child.returncode})"
            killed = json.loads(holding.read_text())
            # THE INTERRUPTION. Not a shutdown, not a cancel: the process holding the
            # attempt stops existing, exactly as a pod's worker does when it is reclaimed.
            child.send_signal(signal.SIGKILL)
        finally:
            child.wait(timeout=WAIT)
        assert child.returncode == -signal.SIGKILL, child.returncode

        # The record is on the disk and nothing else is: the process that wrote it is gone.
        journal = record_root / durable.FILENAME
        assert journal.is_file(), "the killed worker left no durable record"
        kinds = [json.loads(line)["kind"] for line in journal.read_text().splitlines() if line]
        assert "accepted" in kinds and "outcome" not in kinds, kinds

        # ---- the fresh boot names it, replays it, and closes it -----------------------
        owner = _boot(root, record_root, "second")
        try:
            owner.start()
            snapshot = owner.snapshot()
            body = documents.read(snapshot.snapshot_canonical_bytes, pb.WorkerSnapshotBody)
            held = [
                row
                for row in body["held_attempts"]
                if row["request_id"] == killed["request_id"]
                and row["attempt_ordinal"] == killed["attempt_ordinal"]
            ]
            assert len(held) == 1, body["held_attempts"]
            assert held[0]["state"] == pb.ATTEMPT_STATE_OUTCOME_PENDING_ACK, held[0]
            assert held[0]["outcome_id"], "a recovered attempt with no terminal owes nothing"

            # The replay is AFTER the barrier and not before it.
            assert not owner.of_kind("attempt_outcome"), "a terminal replayed before the ack"
            owner.acknowledge(snapshot)
            outcome = owner.outcome()
            assert outcome.request_id == killed["request_id"]
            assert outcome.attempt_ordinal == killed["attempt_ordinal"]
            assert documents.spell(outcome.outcome_digest) == held[0]["outcome_digest"]
            terminal = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
            assert terminal["status"] == pb.OUTCOME_STATUS_ABANDONED, terminal
            assert terminal["cause"]["code"] == pb.CAUSE_CODE_EXECUTOR_INVALIDATED, terminal
            assert terminal["execution_started"] is True, terminal
        finally:
            owner.stop()

        # ---- a boot in between replays the SAME terminal, byte for byte ---------------
        again = _boot(root, record_root, "third")
        try:
            again.start()
            snapshot_again = again.snapshot()
            again.acknowledge(snapshot_again)
            replay = again.outcome()
            assert replay.outcome_id == outcome.outcome_id, "the ordinal was re-terminated"
            assert replay.outcome_canonical_bytes == outcome.outcome_canonical_bytes
            # ...and THIS is the boot that acknowledges it.
            again.close(replay, snapshot_again)
            _until(
                lambda: (
                    "closed"
                    in [
                        json.loads(line)["kind"]
                        for line in journal.read_text().splitlines()
                        if line
                    ]
                    or not journal.read_text().strip()
                ),
                "the acknowledgement never reached the durable record",
            )
        finally:
            again.stop()

        # ---- and after the ack, nothing is owed --------------------------------------
        after = _boot(root, record_root, "fourth")
        try:
            after.start()
            final = documents.read(after.snapshot().snapshot_canonical_bytes, pb.WorkerSnapshotBody)
            # An empty repeated field is ABSENT from a canonical document, not an empty list.
            assert "held_attempts" not in final, final["held_attempts"]
        finally:
            after.stop()


def test_the_snapshot_carries_the_worker_s_own_rows_and_no_second_document() -> None:
    """`HostSnapshotBody/1` is deleted at wire 33: there is no field left to carry it.

    This is the schema half of the same fact, and it needs no worker: a second document beside
    the worker's cannot be sent because `WorkerSnapshot` has no slot for one, and `SnapshotAck`
    has no slot to echo its digest.
    """
    assert WIRE_MINOR >= 33, WIRE_MINOR
    assert not hasattr(pb, "HostSnapshotBody")
    snapshot = {field.name for field in pb.WorkerSnapshot.DESCRIPTOR.fields}
    assert "host_snapshot_digest" not in snapshot
    assert "host_snapshot_canonical_bytes" not in snapshot
    assert "host_snapshot_digest" not in {f.name for f in pb.SnapshotAck.DESCRIPTOR.fields}
    body = {field.name for field in pb.WorkerSnapshotBody.DESCRIPTOR.fields}
    assert {"held_attempts", "weights_transactions"} <= body


def test_a_worker_with_no_record_root_recovers_nothing(tmp_path: Path) -> None:
    """The local door keeps no record, and that is not a degraded mode: a run whose
    RecordOwner is in this process has no restart to survive."""
    records = WorkerRecords(journal=durable.Journal.under(None))
    records.append(
        "accepted",
        {
            "request_id": "req-local",
            "attempt": 1,
            "invocation_spec_digest": documents.spell(bytes(32)),
        },
    )
    assert list(tmp_path.iterdir()) == []
    recovery = durable.recover(durable.Journal.under(None), boot_id="b", instance_id="i")
    assert recovery.attempts == [] and recovery.rows == []


def test_the_file_holds_the_work_in_flight_and_not_the_work_ever_done(tmp_path: Path) -> None:
    """A long-lived worker serves for weeks. The record must be bounded by what it OWES, not by
    what it has done, or the boundedness the wire gets from a digest is undone by the disk."""
    records = WorkerRecords(journal=durable.Journal.under(tmp_path))
    spec = documents.spell(bytes(range(32)))
    for ordinal in range(1, durable.COMPACT_AFTER + 2):
        identity = {"request_id": f"req-{ordinal}", "attempt": 1}
        records.append("accepted", {**identity, "invocation_spec_digest": spec})
        records.append(
            "outcome",
            {
                **identity,
                "invocation_spec_digest": spec,
                "outcome_id": f"out-{ordinal}",
                "outcome_digest": spec,
                "outcome_canonical_b64": base64.b64encode(b"{}").decode(),
            },
        )
        records.append("closed", {**identity, "outcome_id": f"out-{ordinal}"})
    path = tmp_path / durable.FILENAME
    lines = [line for line in path.read_text().splitlines() if line]
    assert len(lines) < durable.COMPACT_AFTER, f"the file kept {len(lines)} dead rows"
    recovery = durable.recover(durable.Journal.under(tmp_path), boot_id="b", instance_id="i")
    assert recovery.attempts == [], recovery.attempts


def test_a_torn_tail_is_not_a_record(tmp_path: Path) -> None:
    """The crash this file exists for happens mid-write. Bytes that did not all land never
    became a fact, and recovery reads the fold, not the file."""
    journal = durable.Journal.under(tmp_path)
    records = WorkerRecords(journal=journal)
    records.append(
        "accepted",
        {
            "request_id": "req-torn",
            "attempt": 1,
            "invocation_spec_digest": documents.spell(bytes(range(32))),
            "placement_id": "plc-1",
        },
    )
    path = tmp_path / durable.FILENAME
    with path.open("ab") as handle:  # a line whose bytes stopped arriving
        handle.write(b'{"seq":2,"kind":"outcome","at_unix_ms":1,"bo')
    recovery = durable.recover(durable.Journal.under(tmp_path), boot_id="b2", instance_id="i2")
    assert len(recovery.attempts) == 1, recovery.attempts
    assert recovery.interrupted == [("req-torn", 1)], recovery.interrupted
    terminal = documents.read(
        base64.b64decode(recovery.attempts[0]["outcome_canonical_b64"]), pb.AttemptOutcomeBody
    )
    assert terminal["status"] == pb.OUTCOME_STATUS_ABANDONED


# --------------------------------------------------------------------------- the driver


def drive(root: Path) -> None:
    """The child process: a REAL worker, a REAL executor, a REAL accepted attempt, held open.

    The ordinary `run_slice` path with one thing added by the harness and nothing changed
    inside the runtime: once the attempt is ACCEPTED — the moment the durable row is written
    and `AttemptAccepted` leaves — the lane thread stops instead of running, so the process is
    killable while it holds an obligation. Fault injection belongs to the harness (#249).
    """
    from cozy_runtime.internal import package_interface
    from cozy_runtime.internal.discovery import discover
    from cozy_runtime.internal.local import LocalRequest, development_source_digest, run_slice
    from test_device_lanes import _config

    holding = root / "holding"
    found = discover(PACKAGE)
    body = package_interface.build(found)
    workspace = root / "run"
    workspace.mkdir(parents=True, exist_ok=True)

    def capture(_owner: Any, worker: Worker) -> None:
        def held(attempt: Any) -> Any:
            holding.write_text(
                json.dumps({"request_id": attempt.request_id, "attempt_ordinal": attempt.attempt})
            )
            threading.Event().wait()  # until the parent kills this process

        # Fault injection belongs to the harness (#249): the runtime has no plant switch.
        worker.engine.execute = held  # type: ignore[method-assign]

    request = LocalRequest(
        entrypoint="marco",
        payload={"message": "marco"},
        outputs=(),
        request_id="record-recovery",
        kind="entrypoint",
    )
    with contextlib.chdir(PACKAGE), contextlib.redirect_stdout(sys.stderr):
        run_slice(
            _config(root / "home"),
            request,
            workspace=workspace,
            package_code_digest=development_source_digest(PACKAGE),
            package_interface_document=body,
            found=found,
            development_tensorfs_root=root / "home" / "tensorfs",
            options=WorkerOptions(
                root=root / "worker",
                devices="0",
                no_warm=True,
                record_root=root / "record",
            ),
            record_owner_hook=capture,
        )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    drive(Path(sys.argv[1]))
