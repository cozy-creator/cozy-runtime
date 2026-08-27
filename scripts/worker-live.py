#!/usr/bin/env python
"""cr-007's live verification: a REAL two-process worker, driven over REAL protocol bytes.

Not a test suite (#160). Every section below starts the actual system — a RecordOwner
process (this script), a no-CUDA worker process, and a disposable CUDA executor process
— and prints what it observed. `cl-001` does not exist yet, so the RecordOwner side of
th-024 is a scripted gRPC SERVER here: it speaks the committed `cozy.worker.v1` service
over a unix socket with the copied bindings, and its canonical documents are checked against
worker-protocol's own frozen fixtures (`fixtures` section) so "real bytes" is a fact.

    nice -n 19 corpus/.venv/bin/python scripts/worker-live.py fixtures
    nice -n 19 corpus/.venv/bin/python scripts/worker-live.py attempt
    nice -n 19 corpus/.venv/bin/python scripts/worker-live.py kill
    nice -n 19 corpus/.venv/bin/python scripts/worker-live.py rollback
    nice -n 19 corpus/.venv/bin/python scripts/worker-live.py nocuda
    nice -n 19 corpus/.venv/bin/python scripts/worker-live.py arms
    nice -n 19 corpus/.venv/bin/python scripts/worker-live.py bench

The bytes come from `~/cozy_v2/tensorfs-bench`, cr-005's real SDXL CAS, READ-ONLY.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import queue
import re
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import grpc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cozy_runtime.internal.readiness import WORKER_TLS_SERVER_NAME  # noqa: E402
from cozy_runtime.protocol import (  # noqa: E402
    SCHEMA_DIGEST,
    SCHEMA_DIGEST_RAW,
    WIRE_MINOR,
    WIRE_SCHEMA_REV,
    documents,
)
from cozy_runtime.protocol import worker_pb2 as pb  # noqa: E402
from cozy_runtime.protocol import worker_pb2_grpc as pb_grpc  # noqa: E402

BENCH = Path("/home/fidika/cozy_v2/tensorfs-bench")
STORE = BENCH / "store"
CONFIG = BENCH / "hf" / "sdxl" / "unet" / "config.json"
CHECKPOINT = BENCH / "results" / "checkpoint.txt"
WORK = Path("/tmp/cozy-slice-store")
RELEASE = "cozy/sdxl-unet@cr-005"
COMPONENT = "unet"
FIXTURES = Path("/home/fidika/cozy_v2/worker-protocol/fixtures")

PASS = 0
FAIL = 0
MIB = 1 << 20
GIB = 1 << 30


def head(title: str) -> None:
    print(f"\n=== {title}", flush=True)


def check(label: str, ok: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {label}" + (f" — {detail}" if detail else ""), flush=True)
    else:
        FAIL += 1
        print(f"  FAIL {label}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def _number(text: str, pattern: str, default: int = 0) -> int:
    found = re.search(pattern, text)
    return int(found.group(1)) if found else default


#: A quiet card, in MiB. The real UNet is 4.78 GiB on an 8 GiB device, so a section only
#: means anything when nothing else is resident: this box is shared, and a neighbour holding
#: 5 GiB turns a correct run into an OOM that says nothing about this code.
QUIET_MIB = 400


def require_quiet_card(seconds: float = 3600.0) -> int:
    """Block until the card is effectively idle, saying so while it waits."""
    started = time.monotonic()
    announced = False
    while time.monotonic() - started < seconds:
        used = gpu_used_mib()
        if used < 0 or used <= QUIET_MIB:
            if announced:
                print(f"  the card is quiet again after {time.monotonic() - started:.0f} s")
            return used
        if not announced:
            announced = True
            print(
                f"  waiting for a quiet card: {used} MiB in use by another process on this "
                f"shared box (need <= {QUIET_MIB} MiB)",
                flush=True,
            )
        time.sleep(5.0)
    return gpu_used_mib()


def gpu_used_mib() -> int:
    out = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()[0]
    )
    return int(out)


# --------------------------------------------------------------------------- the owner

#: The scripted RecordOwner's one PLACEMENT routing key. Owner-minted; never identity.
PLACEMENT = "plc-live"


@dataclass
class Hub:
    """The scripted OWNER of th-024 — it DIALS the worker's hosted `WorkerControl` (#436;
    the worker-dials arrangement this class used to serve is retired). cl-001's real local
    client speaks the same conversation: Claim -> ClaimAck -> snapshot -> SnapshotAck ->
    directive -> dispatch, under the three-field envelope."""

    seen: list[pb.WorkerFrame] = field(default_factory=list)
    lock: threading.Condition = field(default_factory=threading.Condition)
    progress: list[pb.AttemptProgress] = field(default_factory=list)
    boots: list[str] = field(default_factory=list)
    #: The snapshot document's `held_attempts` rows, as the JSON values they are.
    #: NOT re-parsed into messages: the document is the subject of the digest, and a
    #: second representation beside it is free to disagree with the fenced bytes.
    snapshot: list[dict[str, Any]] = field(default_factory=list)
    snapshot_document: dict[str, Any] = field(default_factory=dict)
    #: the admission generation this owner last OBSERVED; every offer echoes it (§6)
    admission_generation: int = 0
    directive: pb.DesiredWorkerState | None = None
    withhold_acks: set[str] = field(default_factory=set)
    wrong_ack: set[str] = field(default_factory=set)
    auto_ack: bool = True
    block_progress: threading.Event = field(default_factory=threading.Event)
    streams: int = 0
    revision: int = 0
    process: Any = None
    log: Any = None
    #: the owner's accepted envelope, learned from ClaimAck
    epoch: int = 1
    generation: int = 0
    boot_id: str = ""
    record_owner_id: str = "hub-live"
    outbound: queue.Queue[pb.RecordOwnerFrame | None] = field(default_factory=queue.Queue)
    channel: Any = None
    watch_channel: Any = None
    call: Any = None
    root_cert: bytes | None = None
    token: str = ""
    connected: threading.Event = field(default_factory=threading.Event)

    # ------------------------------------------------------------ dialing

    def connect(
        self,
        addr: str,
        *,
        claim_only: bool = False,
        root_cert: bytes | None = None,
        token: str = "",
    ) -> None:
        """Dial the worker and CLAIM it. Reconnecting (a new worker on the same address, or
        the same worker after a stream drop) is the same call again: `seen` accumulates, so
        `since` cursors stay valid across a restart.

        THE REMOTE LEG (cl-015/#445): `root_cert` pins the worker's own (self-signed)
        certificate — trusting exactly that PEM IS the fingerprint pin — and `token` is the
        bearer owner token, presented as `Claim.proof` and checked worker-side against the
        provisioned credential, constant-time. Never mTLS.
        """
        self.streams += 1
        self.connected.clear()
        # A FRESH queue per stream, and the request iterator closes over ITS OWN. Reading
        # `self.outbound` from inside the iterator made every previous stream's iterator a
        # consumer of the CURRENT stream's queue: a superseded iterator still parked in
        # `get()` would take the new stream's Claim and write it to a channel that was
        # closing, and the reconnect then waited forever for an ack to a claim the worker
        # never received. Same defect as the worker's own outbound lane, same shape of fix.
        outbound: queue.Queue[pb.RecordOwnerFrame | None] = queue.Queue()
        self.outbound = outbound
        if self.channel is not None:
            self.channel.close()
        if self.watch_channel is not None:
            self.watch_channel.close()
        self.root_cert, self.token = root_cert, token
        self.channel = self._dial(addr)
        stub = pb_grpc.WorkerControlStub(self.channel)
        self.call = stub.Control(self._outbound_iter(outbound))
        threading.Thread(target=self._read, args=(self.call,), daemon=True).start()
        claim = pb.Claim(
            record_owner_epoch=self.epoch,
            record_owner_id=self.record_owner_id,
            wire_minor=WIRE_MINOR,
            proof=token.encode(),
            # THE SCHEMA FENCE, sent by BOTH ends and checked before any other body field
            # (#536b). A harness that omitted it would be indistinguishable from a stale
            # pre-rev-2 RecordOwner, and the worker would refuse it — which is the arm.
            wire_schema_digest=SCHEMA_DIGEST_RAW,
        )
        outbound.put(pb.RecordOwnerFrame(claim=claim))
        if claim_only:
            return
        # The claim handshake completes on the reader thread (ClaimAck -> snapshot ->
        # SnapshotAck -> directive); dispatch is open once `connected` sets. Observed, not
        # timed: the worker process dying is the only failing shape, and wait() names it.
        # A BootFailure INSTEAD of ClaimAck (01) ends the dial: the verdict is delivered
        # and the worker's exit code is the section's to observe.
        try:
            outcome = self.wait(
                lambda m: m if m.WhichOneof("msg") in ("claim_ack", "boot_failure") else None,
                120,
                "ClaimAck",
                len(self.seen),
            )
        except RuntimeError:
            return  # the worker died before claiming — its exit IS the observation
        if outcome.WhichOneof("msg") == "boot_failure":
            return
        # WatchProgress rides a PHYSICALLY separate connection (01): a second channel.
        self.watch_channel = self._dial(addr)
        watch_stub = pb_grpc.WorkerControlStub(self.watch_channel)
        opened = self._env(pb.ProgressOpen())
        threading.Thread(
            target=self._watch, args=(watch_stub.WatchProgress(opened),), daemon=True
        ).start()

    def _dial(self, addr: str) -> Any:
        if self.root_cert is not None:
            return grpc.secure_channel(
                addr,
                grpc.ssl_channel_credentials(self.root_cert),
                options=(
                    ("grpc.ssl_target_name_override", WORKER_TLS_SERVER_NAME),
                    ("grpc.default_authority", WORKER_TLS_SERVER_NAME),
                ),
            )
        return grpc.insecure_channel(addr)

    def _outbound_iter(self, outbound: queue.Queue[pb.RecordOwnerFrame | None]) -> Any:
        while True:
            frame = outbound.get()
            if frame is None:
                return
            yield frame

    def _env(self, message: Any) -> Any:
        message.record_owner_epoch = self.epoch
        message.control_stream_generation = self.generation
        message.worker_boot_id = self.boot_id
        return message

    def _watch(self, frames: Any) -> None:
        try:
            for item in frames:
                # The saturation arm holds this lane still: the client stops READING, gRPC
                # flow control backs the stream up, and the worker's watch ring sheds oldest
                # frames — Control must not notice.
                while self.block_progress.is_set():
                    time.sleep(0.05)
                with self.lock:
                    self.progress.append(item)
                    self.lock.notify_all()
        except grpc.RpcError:
            pass

    def _read(self, call: Any) -> None:
        try:
            for frame in call:
                kind = frame.WhichOneof("msg")
                with self.lock:
                    self.seen.append(frame)
                    self.lock.notify_all()
                if kind == "observed_state":
                    # THE RECORDOWNER DISPATCHES AGAINST THE GENERATION IT LAST SAW (§6).
                    # Tracked on EVERY observed state, not only on the ones an arm waits
                    # for: a fence the owner only reads when it is looking is not a fence
                    # the owner is actually holding.
                    self.admission_generation = frame.observed_state.admission_generation
                if kind == "claim_ack":
                    self._on_claim_ack(frame.claim_ack)
                elif kind == "snapshot":
                    self._on_snapshot(frame.snapshot)
                elif kind == "attempt_outcome":
                    self._on_outcome(frame.attempt_outcome)
        except grpc.RpcError:
            pass

    def _on_claim_ack(self, ack: pb.ClaimAck) -> None:
        if not ack.accepted:
            print(
                f"[hub] Claim REFUSED: {pb.ClaimRejection.Name(ack.rejection)}",
                flush=True,
            )
            return
        self.generation = ack.control_stream_generation
        self.boot_id = ack.worker_boot_id
        self.boots.append(ack.worker_boot_id)
        # THE OTHER DIRECTION OF THE FENCE (#536b): the worker answers with its own digest,
        # and a RecordOwner that does not check it has only fenced one side.
        if ack.wire_schema_digest != SCHEMA_DIGEST_RAW:
            raise RuntimeError(
                f"the worker speaks schema sha256:{ack.wire_schema_digest.hex()} and this "
                f"harness speaks {SCHEMA_DIGEST}"
            )
        print(
            f"[hub] ClaimAck boot={ack.worker_boot_id} "
            f"generation={ack.control_stream_generation} minor={ack.wire_minor} "
            f"schema_rev={WIRE_SCHEMA_REV} device={ack.resources.device_name!r} "
            f"backend={ack.resources.backend!r} unreadable={list(ack.resources.unreadable)}",
            flush=True,
        )

    def _on_snapshot(self, snapshot: pb.WorkerSnapshot) -> None:
        """RECOMPUTE before parsing, ack by id AND digest, then send the desired state.

        There is no entry count to reconcile any more: the snapshot is ONE canonical
        document bounded by its own digest, so a truncated or re-ordered one cannot match.
        """
        recomputed = documents.digest_of(snapshot.snapshot_canonical_bytes)
        if recomputed != snapshot.snapshot_digest:
            print(
                f"[hub] snapshot {snapshot.snapshot_id} DIGEST MISMATCH: carried "
                f"{snapshot.snapshot_digest.hex()[:16]}, bytes hash to "
                f"{recomputed.hex()[:16]} — NOT acknowledged, the barrier stays closed",
                flush=True,
            )
            return
        document = documents.read(snapshot.snapshot_canonical_bytes, pb.WorkerSnapshotBody)
        self.snapshot_document = document
        self.snapshot = list(document.get("held_attempts", []))
        self.admission_generation = int(document.get("admission_generation", 0))
        print(
            f"[hub] snapshot {snapshot.snapshot_id}: {len(self.snapshot)} held attempt(s), "
            f"phase={document.get('worker_phase', 0)} "
            f"admission={document.get('admission_state', 0)}/"
            f"{document.get('available_attempt_slots', 0)} slots at generation "
            f"{self.admission_generation}, {len(snapshot.snapshot_canonical_bytes)} B",
            flush=True,
        )
        self.send(
            pb.RecordOwnerFrame(
                snapshot_ack=self._env(
                    pb.SnapshotAck(
                        snapshot_id=snapshot.snapshot_id,
                        snapshot_digest=snapshot.snapshot_digest,
                    )
                )
            )
        )
        if self.directive is not None:
            directive = pb.DesiredWorkerState()
            directive.CopyFrom(self.directive)
            self._env(directive)
            self.send(pb.RecordOwnerFrame(desired_state=directive))
        self.connected.set()

    def _on_outcome(self, outcome: pb.AttemptOutcome) -> None:
        body = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
        print(
            f"[hub] AttemptOutcome {outcome.request_id}#{outcome.attempt_ordinal} "
            f"status={body['status']} cause={body.get('cause', {}).get('code')} "
            f"execution_started={body.get('execution_started', False)} "
            f"id={outcome.outcome_id} digest={outcome.outcome_digest.hex()[:12]}",
            flush=True,
        )
        key = f"{outcome.request_id}#{outcome.attempt_ordinal}"
        if key in self.withhold_acks or not self.auto_ack:
            return
        digest = outcome.outcome_digest
        if key in self.wrong_ack:
            digest = bytes(32)
        self.send(
            pb.RecordOwnerFrame(
                outcome_ack=self._env(
                    pb.AttemptOutcomeAck(
                        request_id=outcome.request_id,
                        attempt_ordinal=outcome.attempt_ordinal,
                        invocation_spec_digest=outcome.invocation_spec_digest,
                        outcome_id=outcome.outcome_id,
                        outcome_digest=digest,
                    )
                )
            )
        )

    # ------------------------------------------------------------ driving

    def send(self, frame: pb.RecordOwnerFrame | None) -> None:
        self.outbound.put(frame)

    def session(self) -> str:
        """The claimed worker's BOOT identity (the old session_id's successor, #437)."""
        return self.boot_id

    def incarnation(self) -> int:
        """The placement's executor generation the worker last REPORTED (#446/#454)."""
        for report in reversed(self.reports()):
            for status in report.placements:
                if status.placement_id == PLACEMENT:
                    return int(status.executor_generation)
        return 1

    def mark(self) -> int:
        """A cursor into the observed stream, so a wait can require a FRESH frame."""
        return len(self.seen)

    def wait(self, predicate: Any, seconds: float = 120.0, what: str = "", since: int = 0) -> Any:
        deadline = time.monotonic() + seconds
        with self.lock:
            while True:
                for message in list(self.seen)[since:]:
                    found = predicate(message)
                    if found:
                        return found
                if self.process is not None and self.process.poll() is not None:
                    tail = ""
                    if self.log is not None:
                        tail = "\n".join(self.log.read_text(errors="replace").splitlines()[-12:])
                    raise RuntimeError(
                        f"the worker exited {self.process.returncode} while waiting for "
                        f"{what}\n{tail}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"waiting for {what or predicate}")
                self.lock.wait(min(remaining, 0.25))

    def terminals(self, request_id: str = "", attempt: int = 0) -> list[pb.AttemptOutcome]:
        out = []
        for message in list(self.seen):
            if message.WhichOneof("msg") != "attempt_outcome":
                continue
            outcome = message.attempt_outcome
            if request_id and outcome.request_id != request_id:
                continue
            if attempt and outcome.attempt_ordinal != attempt:
                continue
            out.append(outcome)
        return out

    def accepted(self, request_id: str = "") -> list[pb.AttemptAccepted]:
        return [
            m.attempt_accepted
            for m in list(self.seen)
            if m.WhichOneof("msg") == "attempt_accepted"
            and (not request_id or m.attempt_accepted.request_id == request_id)
        ]

    def reports(self) -> list[pb.ObservedWorkerState]:
        return [
            m.observed_state for m in list(self.seen) if m.WhichOneof("msg") == "observed_state"
        ]

    def deployment(self, report: pb.ObservedWorkerState) -> pb.PlacementStatus:
        for status in report.placements:
            if status.placement_id == PLACEMENT:
                return status
        return pb.PlacementStatus()

    def ready(self, plan_id: str, seconds: float = 300.0, since: int = 0) -> pb.ObservedWorkerState:
        """DISPATCHABLE, on the two axes plus the one admission fence — not "READY".

        Three facts, checked together because each says something the others do not: the
        placement's SERVING axis, the plan's presence in `dispatchable_plan_ids`, and an
        OPEN admission window with a free seat.
        """

        def dispatchable(m: Any) -> Any:
            if m.WhichOneof("msg") != "observed_state":
                return None
            report = m.observed_state
            status = self.deployment(report)
            if (
                status.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
                and plan_id in status.dispatchable_plan_ids
                and report.admission_state == pb.AdmissionState.ADMISSION_STATE_OPEN
                and report.available_attempt_slots > 0
            ):
                self.admission_generation = report.admission_generation
                return report
            return None

        report: pb.ObservedWorkerState = self.wait(dispatchable, seconds, "DISPATCHABLE", since)
        return report

    def close(self) -> None:
        self.send(None)
        if self.channel is not None:
            self.channel.close()
        if self.watch_channel is not None:
            self.watch_channel.close()


def read_addr(root: Path, process: Any, budget: float = 60.0) -> str:
    """The worker publishes its bound address to `control.addr` (file handoff); the owner
    reads exactly that. Observed, never guessed: the wait ends on the file appearing or the
    process dying, and `budget` bounds a FAILING run only."""
    addr_file = root / "control.addr"
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        if addr_file.is_file():
            text = addr_file.read_text().strip()
            if text:
                return text
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"the worker exited {process.returncode} before binding")
        time.sleep(0.02)
    raise TimeoutError(f"no control.addr under {root}")


def serving_directive(
    release: str, plan_ids: list[str], posture: Any, plan_root: Path, revision: int = 1
) -> pb.DesiredWorkerState:
    """The full-replace DesiredWorkerState carrying a CONTENT-ADDRESSED DesiredPlacementSet.

    The set travels as its exact canonical `PlacementSet/1` bytes plus their digest; the
    worker RECOMPUTES before it parses a field (§4). The duplicate structured form the
    frozen wire carried is deleted, so there is no second place for the set to disagree
    with itself.
    """
    subjects = []
    for plan_id in sorted(plan_ids):
        source = plan_root / f"{plan_id.split(':')[-1]}.json"
        raw = source.read_bytes()
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        (plan_root / f"{digest.removeprefix('sha256:')}.json").write_bytes(raw)
        subjects.append(
            pb.ArtifactSubject(
                subject_id=plan_id,
                kind="plan",
                digest=documents.raw(digest),
                length=len(raw),
            )
        )
    placement_set = pb.PlacementSet(
        placements=[
            pb.Placement(
                placement_id=PLACEMENT,
                spec=pb.PlacementSpec(
                    endpoint_release_id=release,
                    binding_plans=subjects,
                ),
            )
        ]
    )
    canonical_bytes, set_digest = documents.identity(placement_set)
    return pb.DesiredWorkerState(
        revision=revision,
        posture=posture,
        wire_minor=WIRE_MINOR,
        placement_set=pb.DesiredPlacementSet(
            placement_set_digest=set_digest,
            placement_set_canonical_bytes=canonical_bytes,
        ),
    )


# --------------------------------------------------------------------------- fixtures


def snapshot() -> str:
    for line in CHECKPOINT.read_text().strip().split("\n"):
        name, _, value = line.partition("\t")
        if name == "checkpoint":
            return value
    raise SystemExit("no checkpoint digest in results/checkpoint.txt")


def binding_record(plan_id: str, entrypoint: str) -> dict[str, Any]:
    return {
        "entrypoint_binding_plan_id": plan_id,
        "project": str(ROOT / "corpus" / "endpoint"),
        "model_class": "SdxlUnetModel",
        "model_binding_path": f"{entrypoint}.models.model",
        "model_parameter_name": "model",
        "artifact_ref": RELEASE,
        "components": [COMPONENT],
        "release": RELEASE,
        "host_bytes": 2 * GIB,
        "pinned_bytes": 256 * MIB,  # tfs-007 measured shape: 16 slots x 16 MiB
        "entrypoint": entrypoint,
        "model_construction_digest": "",
    }


def install_row(home: Path, facts: dict[str, Any]) -> None:
    """This harness's artifact index (#567e): the ref the record names, resolved HERE.

    A binding record carries no store root any more, because a store root is a fact about
    the machine that serves and the record's writer may not be it. These arms therefore
    install the row the way `pull` installs it in production, and the worker looks the ref
    up in the index of the COZY_HOME it was launched under.
    """
    from cozy_runtime.internal import artifacts

    artifacts.install(
        home,
        artifacts.Artifact(
            ref=str(facts.get("ref", RELEASE)),
            store=str(facts.get("store") or working_store(WORK / "store")),
            config=str(facts.get("config", CONFIG)),
            snapshots=dict(facts.get("snapshots") or {COMPONENT: snapshot()}),
        ),
    )


def prefill_journal(root: Path, target_bytes: int) -> int:
    """Grow a REAL journal to just under the planted file-size limit, using the real writer.

    The ENOSPC plant used to be `RLIMIT_FSIZE=440` against an empty journal, on the reasoning
    that the journal was the one regular file the worker wrote. Since tfs-007 that is no
    longer true: registering a GC-safe read hold updates the STORE's own state file (502 B),
    so a 440 B limit now kills PREPARATION and the arm never reaches its subject. Growing the
    journal first keeps the limit comfortably above everything else the worker writes while
    still making the ACCEPTANCE append the write that fails — which is the arm.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from cozy_runtime.internal.journal import Journal

    path = root / "journal.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    journal = Journal(path)
    while path.stat().st_size < target_bytes:
        journal.append("session", {"session_id": "pre-fill", "instance_id": "pre-fill"})
    journal.close()
    return path.stat().st_size


def working_store(at: Path) -> str:
    """A HARDLINKED working copy of the read-only bench CAS: a TensorFS lease registers a
    GC-safe hold, which is a write, and the bench store is read-only by standing rule."""
    if not (at / "store.json").is_file():
        at.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(at, ignore_errors=True)
        subprocess.run(["cp", "-al", str(STORE), str(at)], check=True)
    return str(at)


def corrupt_store(at: Path, source: str) -> str:
    """A working copy with ONE byte flipped in a real object, the hardlink broken first so
    the original CAS is untouched. The most ordinary way a stored artifact goes wrong, and
    no switch inside the runtime (decisions #249)."""
    # Rebuilt EVERY time, never reused: TensorFS QUARANTINES an object it finds corrupt, so
    # the second lease over the same doctored store reports the object ABSENT rather than
    # corrupt. That is the store behaving correctly, and it would quietly turn this arm into
    # a different arm — so the plant is fresh on every run.
    shutil.rmtree(at, ignore_errors=True)
    subprocess.run(["cp", "-al", source, str(at)], check=True)
    target = max(
        (p for p in (at / "objects" / "sha256").rglob("*") if p.is_file()),
        key=lambda p: p.stat().st_size,
    )
    body = bytearray(target.read_bytes())
    body[len(body) // 2] ^= 0x40
    target.unlink()
    target.write_bytes(bytes(body))
    return str(at)


def plan_id_for(entrypoint: str, release: str = RELEASE) -> str:
    return "sha256:" + hashlib.sha256(f"ebp/{release}/{entrypoint}".encode()).hexdigest()


@dataclass
class Worker:
    """One live worker: the worker control process and everything staged for it."""

    home: Path
    root: Path
    process: subprocess.Popen[bytes]
    hub: Hub
    log: Path

    def stop(self, sig: int = signal.SIGTERM) -> int | None:
        if self.process.poll() is None:
            self.process.send_signal(sig)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        return self.process.returncode

    def tail(self, pattern: str = "", lines: int = 12) -> list[str]:
        text = self.log.read_text(errors="replace").splitlines()
        rows = [line for line in text if not pattern or re.search(pattern, line)]
        return rows[-lines:]

    def result_schema_digest(self, entrypoint: str) -> str:
        """The DECLARED result-surface digest, computed independently of the worker.

        In a SUBPROCESS on purpose: the RecordOwner has no business importing endpoint code
        either, and the import tripwire must stay a fact about the worker's processes.
        """
        code = (
            "import sys, json\n"
            f"sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'src')!r}]\n"
            "from pathlib import Path\n"
            "from cozy_runtime.internal import canonical, schema\n"
            "from cozy_runtime.internal.discovery import discover\n"
            f"found = discover(Path({str(ROOT / 'corpus' / 'endpoint')!r}))\n"
            f"surface = next(s for s in found.surfaces if s.name == {entrypoint!r})\n"
            "print(canonical.digest(schema.render(surface.result_type)))\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        return out.stdout.strip().splitlines()[-1]

    def executor_pid(self) -> int:
        for line in self.log.read_text(errors="replace").splitlines():
            match = re.search(r"incarnation (\d+) pid (\d+)", line)
            if match:
                pid = int(match.group(2))
        return pid if "pid" in dir() else 0


def stage(
    tmp: Path,
    entrypoints: tuple[str, ...],
    store: str = "",
    *,
    overrides: dict[str, Any] | None = None,
    release: str = RELEASE,
) -> dict[str, str]:
    """Write one binding-plan record per entrypoint, and install the artifact row they name.

    `overrides` is how a DIFFERENT fixture is driven through the same RecordOwner: the
    four-component pipeline is the same worker, the same protocol and the same records with
    another project and component set. Its ARTIFACT half (store, config, snapshots) goes
    into the index row rather than the record — #567e moved those facts to the resolver.
    """
    (tmp / "binding-plans").mkdir(parents=True, exist_ok=True)
    over = dict(overrides or {})
    facts = {
        "ref": over.pop("artifact_ref", release),
        "store": over.pop("store", "") or store,
        "config": over.pop("config", CONFIG),
        "snapshots": over.pop("snapshots", None),
    }
    over.pop("snapshot", None)
    over.pop("variant", None)
    over.pop("custody", None)
    over.pop("vram_floor_bytes", None)
    install_row(tmp, facts)
    ids = {}
    for entrypoint in entrypoints:
        plan_id = plan_id_for(entrypoint, release)
        record = binding_record(plan_id, entrypoint)
        record.update(over)
        record["entrypoint_binding_plan_id"] = plan_id
        record["entrypoint"] = entrypoint
        record["release"] = release
        record["artifact_ref"] = str(facts["ref"] or release)
        (tmp / "binding-plans" / f"{plan_id.split(':')[1]}.json").write_text(
            json.dumps(record, indent=2)
        )
        ids[entrypoint] = plan_id
    return ids


def start_worker(
    tmp: Path,
    *,
    entrypoints: tuple[str, ...] = ("denoise",),
    instance_id: str = "",
    hub: Hub | None = None,
    no_warm: bool = False,
    grace: float = 3.0,
    posture: pb.Posture = pb.Posture.POSTURE_ACCEPTING,
    extra: tuple[str, ...] = (),
    rlimit_fsize: int = 0,
    store: str = "",
    overrides: dict[str, Any] | None = None,
    release: str = RELEASE,
    connect: bool = True,
) -> Worker:
    """Spawn the worker (which SERVES the protocol, #436), then dial and claim it."""
    home = tmp / "home"
    root = tmp / "worker"
    home.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    ids = stage(home, entrypoints, store=store, overrides=overrides, release=release)
    project = Path(str((overrides or {}).get("project") or ROOT / "corpus" / "endpoint"))
    hub = hub or Hub()
    hub.directive = serving_directive(
        release, sorted(ids.values()), posture, home / "binding-plans"
    )
    hub.revision = 1
    # EVERY worker start waits for a quiet card. Without it a 4.78 GiB fill started while a
    # neighbour holds 5 GiB fails with a device shortfall that is true and uninformative.
    require_quiet_card()
    log = tmp / "worker.log"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "COZY_HOME": str(home),
        "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
    }
    (root / "control.addr").unlink(missing_ok=True)
    socket_path = str(tmp / "control.sock")
    Path(socket_path).unlink(missing_ok=True)
    argv = [
        "nice",
        "-n",
        "19",
        sys.executable,
        "-c",
        "import sys; from cozy_runtime.internal.worker.session import main; "
        "raise SystemExit(main(sys.argv[1:]))",
        "--listen",
        f"unix:{socket_path}",
        "--root",
        str(root),
        "--project",
        str(project),
        "--instance-id",
        instance_id or f"ins-{uuid.uuid4().hex[:12]}",
        "--release-id",
        release,
        "--grace",
        str(grace),
        *(("--no-warm",) if no_warm else ()),
        *extra,
    ]

    # stdout is a PIPE, not the log file, and this process copies it. RLIMIT_FSIZE (the
    # ENOSPC plant) applies to regular files and not to pipes, so the plant bites exactly
    # the journal — the one regular file the worker writes — instead of also killing
    # its ability to say what happened.
    def plant_disk_full() -> None:
        """HARNESS-SIDE fault injection: cap the child's file size so the journal's next
        append gets a REAL kernel error (EFBIG), the ENOSPC class. It runs in the harness's
        forked child before exec, so the runtime carries no plant switch of its own. stdout
        is a PIPE and pipes are not subject to RLIMIT_FSIZE, so the plant bites exactly the
        journal — the one regular file the worker writes."""
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (rlimit_fsize, rlimit_fsize))

    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=0,
        preexec_fn=plant_disk_full if rlimit_fsize else None,
    )
    threading.Thread(target=_pump, args=(process, log), daemon=True).start()
    hub.process, hub.log = process, log
    if connect:
        hub.connect(read_addr(root, process))
    return Worker(home, root, process, hub, log)


def _pump(process: subprocess.Popen[bytes], log: Path) -> None:
    with log.open("wb") as handle:
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()


# --------------------------------------------------------------------------- dispatch


#: The default payload every dispatch (and therefore every default spec) binds. The spec
#: carries the payload's identity INSIDE the digest (#439), so the two must agree.
DEFAULT_PAYLOAD = {"steps": 2, "latent": 64}


def stage_payload(tmp: Path, name: str, payload: dict[str, Any]) -> tuple[bytes, Path]:
    data = json.dumps(payload).encode()
    path = tmp / "grants" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data, path


def output_ids(outputs: int, output_id: str) -> list[str]:
    return [output_id if index == 0 else f"{output_id}-{index}" for index in range(outputs)]


def make_spec(
    plan_id: str,
    *,
    deadline_ms: int = 0,
    extra_key: str = "",
    release: str = RELEASE,
    payload: dict[str, Any] | None = None,
    payload_raw: bytes | None = None,
    outputs: int = 1,
    output_id: str = "image",
    max_bytes: int = 8 * MIB,
) -> tuple[bytes, bytes]:
    """The InvocationSpec document: input identities, output contracts and the deadline all
    INSIDE the digest (#439). Defaults mirror `dispatch`'s so a spec built here and a grant
    built there agree on the bound sets."""
    if payload_raw is not None:
        body = payload_raw
    else:
        body = json.dumps(payload or DEFAULT_PAYLOAD).encode()
    spec = pb.InvocationSpec(
        endpoint_release_id=release,
        environment_spec_digest="sha256:" + "11" * 32,
        config_digest="sha256:" + "22" * 32,
        payload_digest="sha256:" + hashlib.sha256(body).hexdigest(),
        inputs=[
            pb.InputBinding(
                input_id="payload",
                digest="sha256:" + hashlib.sha256(body).hexdigest(),
                length=len(body),
                kind_mime="application/json",
                order=0,
            )
        ],
        outputs=[
            pb.OutputBinding(output_id=name, max_bytes=max_bytes)
            for name in sorted(output_ids(outputs, output_id))
        ],
        deadline_unix_ms=deadline_ms,
    )
    spec.serving.entrypoint_binding_plan_id = plan_id
    spec.serving.attempt_binding_id = plan_id
    if extra_key:
        # The PLANT: a key the closed document has no slot for. It is written by hand
        # BECAUSE the schema cannot express it — which is the point of the arm.
        document = documents.document(spec)
        document[extra_key] = "interactive"
        from cozy_runtime.internal import canonical

        data = canonical.write(document)
        return data, hashlib.sha256(data).digest()
    return documents.identity(spec)


def dispatch(
    hub: Hub,
    tmp: Path,
    plan_id: str,
    *,
    request_id: str,
    attempt: int = 1,
    payload: dict[str, Any] | None = None,
    outputs: int = 1,
    deadline_ms: int = 0,
    extra_key: str = "",
    spec_override: tuple[bytes, bytes] | None = None,
    grant_expires: int = 0,
    output_id: str = "image",
    destination_url: str = "",
    release: str = RELEASE,
    max_bytes: int = 8 * MIB,
    placement_id: str = PLACEMENT,
    admission_generation: int = -1,
) -> tuple[bytes, bytes]:
    canonical_bytes, digest = spec_override or make_spec(
        plan_id,
        deadline_ms=deadline_ms,
        extra_key=extra_key,
        release=release,
        payload=payload,
        outputs=outputs,
        output_id=output_id,
        max_bytes=max_bytes,
    )
    body, path = stage_payload(tmp, request_id, payload or DEFAULT_PAYLOAD)
    del body
    out_dir = tmp / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    grant = pb.DeliveryGrant(
        invocation_spec_digest=digest,
        credential=pb.DeliveryAccessCredential(
            issuer="cozy-live",
            key_id="g1",
            epoch=1,
            expires_at_unix=int(time.time()) + 600,
            token=b"grant-token",
        ),
        file_base_url=f"file://{tmp / 'grants'}",
        expires_at_unix=grant_expires or int(time.time()) + 600,
        inputs=[pb.InputAccess(input_id="payload", url=f"file://{path}")],
        outputs=[
            pb.OutputAccess(
                output_id=name,
                url=destination_url or f"file://{out_dir / f'{request_id}-{index}.bin'}",
            )
            for index, name in enumerate(output_ids(outputs, output_id))
        ],
    )
    hub.send(
        pb.RecordOwnerFrame(
            attempt_offer=hub._env(
                pb.AttemptOffer(
                    request_id=request_id,
                    attempt_ordinal=attempt,
                    invocation_spec_digest=digest,
                    grant=grant,
                    invocation_spec_canonical_bytes=canonical_bytes,
                    placement_id=placement_id,
                    # The generation this owner OBSERVED when it dispatched (§6). -1 means
                    # "whatever the harness last saw"; an arm that wants a STALE echo passes
                    # an explicit number.
                    admission_generation=(
                        hub.admission_generation
                        if admission_generation < 0
                        else admission_generation
                    ),
                )
            )
        )
    )
    return canonical_bytes, digest


def terminal_body(terminal: pb.AttemptOutcome) -> dict[str, Any]:
    return documents.read(terminal.outcome_canonical_bytes, pb.AttemptOutcomeBody)


# --------------------------------------------------------------------------- sections


def section_fixtures(args: argparse.Namespace) -> None:
    """The copied bindings and this writer against worker-protocol's schema-rev-5 corpus."""
    head("the SCHEMA FENCE is DERIVED here, not declared (#530-A1/#536b)")
    derived = documents.wire_schema_document()
    derived_id = "sha256:" + documents.digest_of(derived).hex()
    frozen_schema = (FIXTURES / "canonical" / "wire_schema.json").read_bytes()
    check(
        "this binding's own descriptor digests to the constant regen.sh wrote",
        derived_id == SCHEMA_DIGEST,
        f"derived {derived_id[:30]}… == wire_identity.SCHEMA_DIGEST, schema rev "
        f"{WIRE_SCHEMA_REV} — a stale copy cannot spell it without BEING the schema",
    )
    check(
        "and reproduces the protocol's frozen WireSchema/1 document byte-for-byte",
        derived == frozen_schema,
        f"{len(derived)} B, normalized (source_code_info cleared, json_name stripped) with "
        f"the document-version map {documents.DOC_VERSION} inside it",
    )

    head("the vendored schema-rev-5 bindings decode the frozen fixture corpus")
    manifest = json.loads((FIXTURES / "MANIFEST.json").read_text())
    bad = []
    for name, row in manifest["positive"].items():
        data = (FIXTURES / "positive" / f"{name}.bin").read_bytes()
        cls = getattr(pb, row["type"].split(".")[-1])
        message = cls()
        message.ParseFromString(data)
        if message.SerializeToString(deterministic=True) != data:
            bad.append(name)
    check(
        "every positive fixture round-trips byte-identically through this copy",
        not bad,
        f"{len(manifest['positive'])} fixtures, {len(bad)} wrong",
    )

    head("canonical DOCUMENT identity: this writer reproduces the frozen bytes")
    wrong = []
    for name, row in manifest["canonical"].items():
        if row["type"] == "cozy.worker.v1.WireSchema":
            # NOT a message in worker.proto — a document ABOUT worker.proto. It has no
            # descriptor to parse a `.bin` into, and the arm above already proved it.
            continue
        frozen = (FIXTURES / "canonical" / f"{name}.json").read_bytes()
        cls = getattr(pb, row["type"].split(".")[-1])
        message = cls()
        message.ParseFromString((FIXTURES / "canonical" / f"{name}.bin").read_bytes())
        mine, digest = documents.identity(message)
        if mine != frozen or "sha256:" + digest.hex() != row["id"]:
            wrong.append(name)
        print(f"  {name:26} {len(mine):>5} B  sha256:{digest.hex()[:24]}…")
    check(
        "byte-identical canonical documents and ids",
        not wrong,
        f"{len(manifest['canonical']) - 1} message documents + the schema document, "
        f"{len(wrong)} divergent — the worker authors the same bytes worker-protocol froze, "
        "with no shared code",
    )

    head("the reading side refuses what the writer cannot have produced")
    from cozy_runtime.internal import canonical

    frozen = (FIXTURES / "canonical" / "invocation_spec_serving.json").read_bytes()

    def planted(data: bytes) -> bytes:
        """A service-class key, written CANONICALLY — sorted, no whitespace, its own digest.
        The refusal has to come from the closed key set, not from a formatting accident."""
        value = canonical.parse(data)
        value["service_class"] = "interactive"
        return canonical.write(value)

    for label, mutate in (
        ("a canonically-written service-class key", planted),
        ("a pretty-printed twin of the same document", lambda b: b.replace(b",", b", ")),
        (
            "a duplicate key",
            lambda b: b[:1] + b'"config_digest":"sha256:' + b"00" * 32 + b'",' + b[1:],
        ),
    ):
        try:
            documents.read(mutate(frozen), pb.InvocationSpec)
            check(label, False, "did NOT refuse")
        except documents.DocumentError as exc:
            check(label, True, f"refused [{exc.code}]")


def section_attempt(args: argparse.Namespace) -> dict[str, Any]:
    """One real attempt end to end: boot, directive, warm pass, dispatch, terminal, tail."""
    tmp = Path(args.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    idle = gpu_used_mib()
    boot_started = time.perf_counter()
    worker = start_worker(tmp, entrypoints=("denoise", "restyle"))
    hub = worker.hub
    plan_id = plan_id_for("denoise")
    try:
        head("boot: worker -> exec-first trampoline -> executor -> fill -> warm -> READY")
        hub.wait(
            lambda m: m.claim_ack if m.WhichOneof("msg") == "claim_ack" else None, 60, "ClaimAck"
        )
        register_at = time.perf_counter()
        report = hub.ready(plan_id, 400)
        status = hub.deployment(report)
        ready_at = time.perf_counter()
        check(
            "the worker reached DISPATCHABLE through the ordinary path",
            True,
            f"boot->ClaimAck {register_at - boot_started:.2f} s, "
            f"ClaimAck->DISPATCHABLE {ready_at - register_at:.2f} s "
            f"(executor generation {status.executor_generation}, admission generation "
            f"{report.admission_generation})",
        )
        check(
            "the TWO AXES say what one enum could not: materialization x serving",
            status.materialization == pb.MaterializationState.MATERIALIZATION_STATE_STAGED
            and status.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE,
            f"materialization={pb.MaterializationState.Name(status.materialization)}, "
            f"serving={pb.ServingState.Name(status.serving)}",
        )
        check(
            "DISPATCHABLE means first-request-servable, not connected — per placement",
            plan_id in status.dispatchable_plan_ids
            and report.available_attempt_slots > 0
            and report.admission_state == pb.AdmissionState.ADMISSION_STATE_OPEN,
            f"{len(status.dispatchable_plan_ids)} dispatchable plan(s), "
            f"{report.available_attempt_slots} attempt slot(s) in an "
            f"{pb.AdmissionState.Name(report.admission_state)} window",
        )
        check(
            "convergence is a SEPARATE fact from acceptance, and they agree here",
            report.converged_revision == report.accepted_desired_state_revision,
            f"accepted {report.accepted_desired_state_revision} == converged "
            f"{report.converged_revision} — a placement that stopped being DISPATCHABLE "
            "would pull converged back below it",
        )

        warm = (
            [json.loads(line.split("summary ", 1)[1]) for line in worker.tail("summary", 1)]
            if worker.tail("summary", 1)
            else []
        )
        del warm
        warm_rows = [line for line in worker.tail(r"warm:", 8)]
        check(
            "the boot warm pass ran through ORDINARY dispatch, pre-ready",
            bool(warm_rows),
            " | ".join(row.split("warm: ")[-1] for row in warm_rows),
        )
        skipped = [r for r in warm_rows if "restyle: skipped" in r]
        check(
            "a required VIDEO case reports `skipped`, never `warmed`",
            bool(skipped),
            "the synthesizer makes strings, numbers and images and says so when it cannot",
        )

        head("one real attempt: StartAttempt -> AttemptAccepted -> AttemptTerminal")
        started = time.perf_counter()
        _, digest = dispatch(
            hub,
            tmp,
            plan_id,
            request_id="req-1",
            payload={"steps": args.steps, "latent": 64, "seed": 1005},
        )
        accepted = hub.wait(
            lambda m: (
                m.attempt_accepted
                if m.WhichOneof("msg") == "attempt_accepted"
                and m.attempt_accepted.request_id == "req-1"
                else None
            ),
            120,
            "AttemptAccepted",
        )
        accepted_at = time.perf_counter()
        terminal = hub.wait(
            lambda m: (
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-1"
                else None
            ),
            300,
            "AttemptTerminal",
        )
        done_at = time.perf_counter()
        body = terminal_body(terminal)

        check(
            "acceptance carries the digests it JOURNALED",
            accepted.invocation_digest == digest
            and len(accepted.plan_digest) == 32
            and len(accepted.model_construction_digest) == 32,
            f"plan {accepted.plan_digest.hex()[:16]}… construction "
            f"{accepted.model_construction_digest.hex()[:16]}…",
        )
        check(
            "the plan summary is an OBSERVABLE projection, not just a hash",
            accepted.plan.delivery == "native" and accepted.plan.placement == "all_resident",
            f"{accepted.plan.delivery}/{accepted.plan.materialization}/"
            f"{accepted.plan.placement} {accepted.plan.compute_dtype}, "
            f"{accepted.plan.reserved_device_memory_bytes / GIB:.2f} GiB device reserved",
        )
        check(
            "the terminal SUCCEEDED",
            body["status"] == 1,
            f"cause {body.get('cause')} · {body.get('safe_message', '')[:60]}",
        )
        check(
            "terminal_digest is the sha256 of the carried bytes",
            hashlib.sha256(terminal.outcome_canonical_bytes).digest() == terminal.outcome_digest,
            f"{len(terminal.outcome_canonical_bytes)} canonical bytes, id {terminal.outcome_id}",
        )

        head("the output tail: written under the EXACT grant, manifest durable")
        manifest = body.get("output_manifest", {})
        entries = manifest.get("outputs", [])
        written = sorted((tmp / "outputs").glob("req-1-*.bin"))
        check(
            "a valid write produced a canonical manifest",
            len(entries) == 1 and bool(written),
            f"{manifest.get('manifest_id')}: {entries[0]['length'] if entries else 0} B "
            f"at {written[0].name if written else 'nothing'}",
        )
        if entries and written:
            check(
                "the manifest entry addresses the bytes that landed",
                entries[0]["digest"]
                == "sha256:" + hashlib.sha256(written[0].read_bytes()).hexdigest(),
                entries[0]["digest"][:24] + "…",
            )
            check(
                "the write produced no read authority and no publication",
                "grant_id" not in manifest and "url" not in str(entries[0]),
                "the manifest names output_id, digest, length, mime — no reader, no URL",
            )

        head("the ResultEnvelope: the typed function result, never the asset manifest")
        envelope = body.get("result", {})
        inline = envelope.get("inline_result")
        check(
            "a small result rides the terminal INLINE (<= 4 MiB)",
            bool(inline),
            f"{len(inline or '')} base64 chars, schema digest "
            f"{envelope.get('result_schema_digest', '')[:20]}…",
        )
        if inline:
            import base64

            from cozy_runtime.internal import canonical

            result = canonical.parse(base64.b64decode(inline))
            check(
                "the result is the real computation's typed return value",
                isinstance(result.get("std"), float) and result.get("warm") is False,
                f"steps={result.get('steps')} mean={result.get('mean')} "
                f"std={result.get('std')} digest={str(result.get('digest'))[:16]}…",
            )
            check(
                "`ctx.boot_warmup` is observably FALSE on a paying request",
                result.get("warm") is False,
                "true only during the boot warm pass",
            )

        head("metric attestation: a claim, clamped and named")
        metrics = body.get("metrics", {})
        window_ms = (done_at - accepted_at) * 1000
        check(
            "wall-clock is clamped to the worker-observed window",
            metrics.get("runtime_ms", 0) <= window_ms,
            f"runtime_ms {metrics.get('runtime_ms')} <= observed window {window_ms:.0f} ms "
            f"(handler {metrics.get('handler_ms')}, device lease "
            f"{metrics.get('device_lease_ms')}, finalization "
            f"{metrics.get('finalization_ms')})",
        )
        check(
            "structurally unverifiable fields are NAMED",
            metrics.get("unverified_fields")
            == ["input_tokens", "output_tokens", "peak_device_memory_bytes"],
            str(metrics.get("unverified_fields")),
        )

        head("the lossy progress lane")
        check(
            "progress arrived on its own RPC with worker-owned sequence numbers",
            len(hub.progress) > 0,
            f"{len(hub.progress)} AttemptProgress messages, seq 1..."
            f"{max((p.seq for p in hub.progress), default=0)}, "
            f"content_type={hub.progress[0].content_type if hub.progress else ''}",
        )

        row = {
            "boot_to_register_s": round(register_at - boot_started, 3),
            "register_to_ready_s": round(ready_at - register_at, 3),
            "dispatch_to_accept_ms": round((accepted_at - started) * 1000, 1),
            "accept_to_terminal_ms": round((done_at - accepted_at) * 1000, 1),
            "attempt_total_ms": round((done_at - started) * 1000, 1),
            "handler_ms": metrics.get("handler_ms"),
            "device_lease_ms": metrics.get("device_lease_ms"),
            "finalization_ms": metrics.get("finalization_ms"),
            "peak_device_memory_bytes": metrics.get("peak_device_memory_bytes"),
            "gpu_idle_mib": idle,
            "progress_messages": len(hub.progress),
        }
        return row
    finally:
        summary = _shutdown(worker)
        print(f"  worker summary: {json.dumps(summary)}", flush=True)
        _gpu_clean(idle)


def _shutdown(worker: Worker) -> dict[str, Any]:
    worker.stop()
    worker.hub.close()
    for line in worker.log.read_text(errors="replace").splitlines():
        if "summary " in line:
            return dict(json.loads(line.split("summary ", 1)[1]))
    return {}


def _gpu_clean(idle: int) -> None:
    for _ in range(30):
        used = gpu_used_mib()
        if used <= idle + 32:
            print(f"  GPU back to {used} MiB (idle baseline {idle} MiB)", flush=True)
            return
        time.sleep(1.0)
    print(f"  WARNING: GPU still at {gpu_used_mib()} MiB (idle baseline {idle} MiB)", flush=True)


def section_kill(args: argparse.Namespace) -> dict[str, Any]:
    """kill -9 the executor MID-ATTEMPT: a typed terminal, a fresh incarnation, then green."""
    tmp = Path(args.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    idle = gpu_used_mib()
    worker = start_worker(tmp, entrypoints=("denoise",), no_warm=True)
    hub = worker.hub
    plan_id = plan_id_for("denoise")
    row: dict[str, Any] = {}
    try:
        hub.ready(plan_id, 400)
        head("kill -9 the executor in the middle of a real attempt")
        pid = worker.executor_pid()
        check("the executor is a separate process", pid > 0, f"pid {pid}")
        dispatch(
            hub, tmp, plan_id, request_id="req-kill", payload={"steps": 40, "latent": 80, "seed": 7}
        )
        hub.wait(
            lambda m: m.attempt_accepted if m.WhichOneof("msg") == "attempt_accepted" else None,
            60,
            "accepted",
        )
        # Wait until the attempt is genuinely on the GPU before killing.
        hub.wait(lambda m: m if m.WhichOneof("msg") == "report" else None, 10, "report")
        time.sleep(args.kill_after)
        used_before = gpu_used_mib()
        mark = hub.mark()
        killed_at = time.perf_counter()
        os.kill(pid, signal.SIGKILL)
        terminal = hub.wait(
            lambda m: (
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-kill"
                else None
            ),
            120,
            "terminal after kill",
        )
        terminal_at = time.perf_counter()
        body = terminal_body(terminal)
        check(
            "the worker journaled a TYPED terminal for the lost attempt",
            body["status"] == 5 and body["cause"]["code"] == 16,
            f"ABANDONED/EXECUTOR_INVALIDATED in "
            f"{(terminal_at - killed_at) * 1000:.0f} ms — {body['safe_message'][:70]}",
        )
        check(
            "the terminal is never silence and never a fabricated result",
            "output_manifest" not in body and "result" not in body,
            "an accepted-but-incomplete attempt reports ABANDONED with no outputs",
        )

        report = hub.ready(plan_id, 400, since=mark)
        status = hub.deployment(report)
        recovered_at = time.perf_counter()
        check(
            "a fresh executor generation reloaded BEFORE reporting dispatchable",
            status.executor_generation >= 2 and report.admission_generation > 1,
            f"executor generation {status.executor_generation}, admission generation "
            f"{report.admission_generation} — `readiness_epoch` is DELETED (#486b), and "
            "the worker-level admission fence is the one counter that moved",
        )

        head("the next attempt succeeds on the new incarnation")
        dispatch(
            hub,
            tmp,
            plan_id,
            request_id="req-after",
            payload={"steps": 2, "latent": 64, "seed": 11},
        )
        after = hub.wait(
            lambda m: (
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-after"
                else None
            ),
            300,
            "terminal after recovery",
        )
        after_body = terminal_body(after)
        check(
            "SUCCEEDED after a kill -9 mid-attempt",
            after_body["status"] == 1,
            f"{after_body.get('metrics', {}).get('handler_ms')} ms of handler on the "
            "rebuilt executor",
        )
        row = {
            "kill_to_terminal_ms": round((terminal_at - killed_at) * 1000, 1),
            "kill_to_ready_s": round(recovered_at - killed_at, 2),
            "gpu_used_at_kill_mib": used_before,
            "executor_generation_after": status.executor_generation,
            "admission_generation_after": report.admission_generation,
        }
        return row
    finally:
        summary = _shutdown(worker)
        print(f"  worker summary: {json.dumps(summary)}", flush=True)
        _gpu_clean(idle)


def section_rollback(args: argparse.Namespace) -> dict[str, Any]:
    """A fill exception: the journal records it AND the device actually gets its bytes back."""
    tmp = Path(args.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    idle = gpu_used_mib()
    head("a real 4.78 GiB fill against a store with ONE byte flipped in a real object")
    worker = start_worker(
        tmp,
        entrypoints=("denoise",),
        no_warm=True,
        store=corrupt_store(WORK / "store-corrupt", working_store(WORK / "store")),
    )
    hub = worker.hub
    plan_id = plan_id_for("denoise")
    row: dict[str, Any] = {}
    try:
        report = hub.wait(
            lambda m: (
                m.observed_state
                if m.WhichOneof("msg") == "observed_state" and m.observed_state.faults
                else None
            ),
            400,
            "a fault report",
        )
        fault = report.faults[0]
        check(
            "the fill refused typed and the worker reported it",
            fault.reason in ("interrupted", "incomplete_fill") or bool(fault.reason),
            f"{pb.FaultKind.Name(fault.kind)} {fault.reason}: {fault.detail[:90]}",
        )
        line = worker.tail("prepare REFUSED", 1)
        check(
            "the refusal names what it RELEASED, not merely that it failed",
            bool(line),
            line[0].split("boot: ")[-1] if line else "",
        )

        head("since tfs-007 the refusal PRECEDES the reservation, which is the whole point")
        text = line[0] if line else ""
        peak_bytes = _number(text, r"peak allocator (\d+) B")
        reserved_bytes = _number(text, r"ledger (\d+) B was LIVE", default=-1)
        after_bytes = _number(text, r"allocator holds (\d+) B", default=-1)
        free_bytes = _number(text, r"has (\d+) B free")
        used = gpu_used_mib()
        check(
            "ZERO destinations were reserved before the artifact was refused",
            reserved_bytes == 0 and after_bytes == 0,
            f"{reserved_bytes} B of destinations were live, {after_bytes} B held afterwards "
            f"(the allocator's own peak was {peak_bytes} B — cr-006's device probe qualifies "
            "encodings before the lease, and that scratch is not a destination). TensorFS "
            "verifies the snapshot when the LEASE registers, so a corrupt object never "
            "reaches a fill. The interim reader could only find the same defect 85 % of the "
            "way through a 4.78 GiB stream, with 5,134,982,144 B of destinations already live",
        )
        check(
            "and the DEVICE agrees, from outside the process",
            used - idle < 512,
            f"nvidia-smi: {used} MiB used, {used - idle} MiB over this run's {idle} MiB "
            f"baseline — the poisoned executor is still alive and its CUDA context is what "
            f"that is; torch reports {free_bytes / GIB:.2f} GiB free. The comparison is "
            "against the BASELINE, not an absolute: this box is shared and a neighbour's "
            "residency is not this run's leak",
        )

        head("a fresh executor confirms the free memory from inside CUDA")
        dispatch(hub, tmp, plan_id, request_id="req-poisoned")
        terminal = hub.wait(
            lambda m: m.attempt_outcome if m.WhichOneof("msg") == "attempt_outcome" else None,
            120,
            "refusal terminal",
        )
        body = terminal_body(terminal)
        check(
            "an attempt against the failed generation REFUSES rather than cold-loading",
            body["status"] == 2,
            f"{body['safe_message'][:110]}",
        )
        row = {
            "peak_allocator_bytes": peak_bytes,
            "allocator_after_bytes": after_bytes,
            "device_free_after_bytes": free_bytes,
            "nvidia_smi_used_mib": used,
            "idle_mib": idle,
        }
        return row
    finally:
        _shutdown(worker)
        _gpu_clean(idle)


def section_nocuda(args: argparse.Namespace) -> dict[str, Any]:
    """torch stays OUT of the worker — proven against the RUNNING process."""
    tmp = Path(args.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    idle = gpu_used_mib()
    worker = start_worker(tmp, entrypoints=("denoise",), no_warm=True)
    hub = worker.hub
    plan_id = plan_id_for("denoise")
    row: dict[str, Any] = {}
    try:
        hub.ready(plan_id, 400)
        worker_pid = worker.process.pid
        executor_pid = worker.executor_pid()

        head("the worker's mapped libraries, while it is serving")
        maps = Path(f"/proc/{worker_pid}/maps").read_text(errors="replace")
        cuda_markers = (
            "libtorch",
            "libcuda",
            "libcudart",
            "libcublas",
            "libnv",
            "/torch/",
            "libtriton",
            "libc10",
        )
        offenders = sorted(
            {
                line.split()[-1]
                for line in maps.splitlines()
                if any(marker in line for marker in cuda_markers)
            }
        )
        check(
            "no torch/CUDA library is mapped into the worker",
            not offenders,
            f"{len(maps.splitlines())} mappings scanned for "
            f"{len(cuda_markers)} markers, {len(offenders)} hits"
            + (
                f" — {offenders[:3]}"
                if offenders
                else " — the worker is a bare "
                "interpreter, which is also what keeps OOM-victim ordering honest"
            ),
        )
        executor_maps = Path(f"/proc/{executor_pid}/maps").read_text(errors="replace")
        cuda_in_executor = sorted(
            {
                Path(line.split()[-1]).name
                for line in executor_maps.splitlines()
                if "libcuda" in line or "libtorch" in line
            }
        )[:4]
        check(
            "the executor IS the CUDA process", bool(cuda_in_executor), ", ".join(cuda_in_executor)
        )

        head("no CUDA context: the worker holds no device memory")
        contexts = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        pids = [int(line.split(",")[0]) for line in contexts.splitlines() if line.strip()]
        check(
            "the worker holds no CUDA context",
            worker_pid not in pids,
            f"compute apps on the device: {pids} (executor {executor_pid})",
        )

        head("process facts: OOM-victim ordering, privileges, parent death")
        sup_oom = Path(f"/proc/{worker_pid}/oom_score_adj").read_text().strip()
        exe_oom = Path(f"/proc/{executor_pid}/oom_score_adj").read_text().strip()
        check(
            "the executor raised its OWN oom_score_adj above the worker's",
            int(exe_oom) == 500 and int(exe_oom) > int(sup_oom),
            f"worker {sup_oom} (INHERITED — the runtime never writes its own; this "
            f"box's shells run at 200), executor {exe_oom} (written by the trampoline, "
            "before exec) — the kernel's victim ordering picks the disposable process",
        )
        status = Path(f"/proc/{executor_pid}/status").read_text()
        nnp = next(ln for ln in status.splitlines() if ln.startswith("NoNewPrivs"))
        check("no_new_privs holds in the executor", nnp.endswith("1"), nnp)
        check(
            "the executor runs in its own process group",
            os.getpgid(executor_pid) == executor_pid,
            f"pgid {os.getpgid(executor_pid)} — one killpg reaches every grandchild",
        )

        worker_rss = (
            int(
                next(
                    line
                    for line in Path(f"/proc/{worker_pid}/status").read_text().splitlines()
                    if line.startswith("VmRSS")
                ).split()[1]
            )
            * 1024
        )
        executor_rss = (
            int(
                next(
                    line
                    for line in Path(f"/proc/{executor_pid}/status").read_text().splitlines()
                    if line.startswith("VmRSS")
                ).split()[1]
            )
            * 1024
        )
        check(
            "both process footprints, measured while serving",
            True,
            f"worker RSS {worker_rss / MIB:.1f} MiB, executor RSS {executor_rss / MIB:.1f} MiB",
        )

        head("worker death is child death (pdeathsig), with no cooperation")
        worker.process.send_signal(signal.SIGKILL)
        worker.process.wait(timeout=20)
        gone_at = time.perf_counter()
        for _ in range(200):
            if not Path(f"/proc/{executor_pid}").exists():
                break
            time.sleep(0.01)
        check(
            "the executor died with its worker",
            not Path(f"/proc/{executor_pid}").exists(),
            f"within {(time.perf_counter() - gone_at) * 1000:.0f} ms of SIGKILL to the "
            "worker — pdeathsig, armed before exec",
        )
        row = {
            "worker_rss_bytes": worker_rss,
            "worker_oom_score_adj": int(sup_oom),
            "executor_oom_score_adj": int(exe_oom),
            "executor_rss_bytes": executor_rss,
        }
        return row
    finally:
        worker.hub.close()
        _gpu_clean(idle)


def _rss_of(log: Path) -> int:
    for line in log.read_text(errors="replace").splitlines():
        if "summary " in line:
            return int(json.loads(line.split("summary ", 1)[1]).get("rss_bytes", 0))
    return 0


def section_arms(args: argparse.Namespace) -> None:
    """The red arms: each planted for real, observed, and removed by the next section."""
    tmp = Path(args.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    idle = gpu_used_mib()
    worker = start_worker(tmp, entrypoints=("denoise", "stubborn"), no_warm=True, grace=2.0)
    hub = worker.hub
    plan_id = plan_id_for("denoise")
    stubborn_id = plan_id_for("stubborn")
    try:
        # A StartAttempt before the snapshot is acknowledged is dropped by the dispatch
        # gate (02 §6), which is correct and is not the arm being planted here — connect()
        # already completed the handshake.
        hub.wait(
            lambda m: m.claim_ack if m.WhichOneof("msg") == "claim_ack" else None, 60, "ClaimAck"
        )
        head("an attempt before the executor is Ready for that plan")
        # `stubborn` shares the construction, so both are prepared together; a plan the
        # directive never named is the honest non-Ready case.
        unknown = "sha256:" + "ab" * 32
        dispatch(hub, tmp, unknown, request_id="req-unready")
        body = terminal_body(
            hub.wait(
                lambda m: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-unready"
                    else None
                ),
                60,
                "refusal",
            )
        )
        check(
            "an unbound plan refuses REFUSED/LOCAL_SAFETY, and nothing cold-loads",
            body["status"] == 2,
            body["safe_message"][:100],
        )

        hub.ready(plan_id, 400)

        head("a planted service-class field below the dispatch boundary")
        dispatch(hub, tmp, plan_id, request_id="req-class", extra_key="service_class")
        body = terminal_body(
            hub.wait(
                lambda m: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-class"
                    else None
                ),
                60,
                "refusal",
            )
        )
        check(
            "service class is unrepresentable below dispatch",
            body["status"] == 2 and "unknown_field" in body["safe_message"],
            body["safe_message"][:120],
        )

        head("duplicate StartAttempt: the same triple executes exactly once")
        canonical_bytes, digest = dispatch(
            hub, tmp, plan_id, request_id="req-dup", payload={"steps": 2, "latent": 64}
        )
        hub.wait(
            lambda m: (
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-dup"
                else None
            ),
            300,
            "first",
        )
        dispatch(hub, tmp, plan_id, request_id="req-dup", spec_override=(canonical_bytes, digest))
        time.sleep(2.0)
        terminals = hub.terminals("req-dup")
        check(
            "two StartAttempts, one execution, one terminal body",
            len(terminals) == 2
            and terminals[0].outcome_canonical_bytes == terminals[1].outcome_canonical_bytes,
            f"{len(terminals)} terminals, byte-identical bodies, terminal_id stable "
            f"({terminals[0].outcome_id == terminals[1].outcome_id})",
        )
        replayed = worker.tail("executing NOTHING", 1)
        check(
            "the worker says it replayed rather than re-ran",
            bool(replayed),
            replayed[0].split("attempt: ")[-1][:100] if replayed else "",
        )

        head("same attempt, DIFFERENT digest")
        other, other_digest = make_spec(plan_id, deadline_ms=1)
        dispatch(hub, tmp, plan_id, request_id="req-dup", spec_override=(other, other_digest))
        body = terminal_body(
            hub.wait(
                lambda m: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-dup"
                    and m.attempt_outcome.invocation_digest == other_digest
                    else None
                ),
                60,
                "refusal",
            )
        )
        check(
            "a changed spec under a held attempt refuses and runs neither",
            body["status"] == 2 and body["cause"]["code"] == 4,
            body["safe_message"][:110],
        )

        head("a planted invocation_digest that does not hash the carried bytes")
        good, _ = make_spec(plan_id)
        dispatch(hub, tmp, plan_id, request_id="req-planted", spec_override=(good, bytes(32)))
        body = terminal_body(
            hub.wait(
                lambda m: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-planted"
                    else None
                ),
                60,
                "refusal",
            )
        )
        check(
            "the receiver RECOMPUTES over the resident bytes",
            "invocation_digest_mismatch" in body["safe_message"],
            body["safe_message"][:110],
        )

        head("a NEW ordinal arriving over a LIVE attempt")
        # `stubborn` never asks whether it was cancelled, and 900 real denoising steps at
        # 640 px keep it genuinely running for the whole of the next three arms.
        dispatch(hub, tmp, stubborn_id, request_id="req-live", payload={"steps": 900, "latent": 80})
        accepted = hub.wait(
            lambda m: (
                m.attempt_accepted
                if m.WhichOneof("msg") == "attempt_accepted"
                and m.attempt_accepted.request_id == "req-live"
                else None
            ),
            60,
            "accepted",
        )
        dispatch(hub, tmp, stubborn_id, request_id="req-live", attempt=2)
        body = terminal_body(
            hub.wait(
                lambda m: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-live"
                    and m.attempt_outcome.attempt_ordinal == 2
                    else None
                ),
                60,
                "refusal",
            )
        )
        check(
            "silent supersession is deleted from the protocol",
            body["status"] == 2 and "live_attempt_supersession" in body["safe_message"],
            body["safe_message"][:120],
        )
        check(
            "the held attempt is UNAFFECTED by its successor's arrival",
            not hub.terminals("req-live", 1),
            "attempt 1 is still running; no attempt is ever implicitly aborted",
        )

        head("a stale cancel digest cannot touch an executor")
        mark = hub.mark()
        hub.send(
            pb.RecordOwnerFrame(
                cancel_attempt=hub._env(
                    pb.CancelAttempt(
                        request_id="req-live",
                        attempt_ordinal=1,
                        reason=pb.CancelReason.CANCEL_REASON_CLIENT,
                        invocation_spec_digest=bytes(32),
                    )
                )
            )
        )
        time.sleep(1.5)
        dropped = worker.tail("cancel: dropped", 1)
        check(
            "a mismatched cancel is DROPPED, naming the digest fence",
            bool(dropped) and "digest" in dropped[0] and not hub.terminals("req-live", 1),
            (dropped[0].split("cancel: ")[-1][:110] if dropped else "no drop note")
            + " — and the attempt it named is still running",
        )

        head("a NON-COOPERATIVE handler: cooperative -> grace -> kill, in that order")
        cancel_at = time.perf_counter()
        hub.send(
            pb.RecordOwnerFrame(
                cancel_attempt=hub._env(
                    pb.CancelAttempt(
                        request_id="req-live",
                        attempt_ordinal=1,
                        reason=pb.CancelReason.CANCEL_REASON_CLIENT,
                        invocation_spec_digest=accepted.invocation_spec_digest,
                    )
                )
            )
        )
        terminal = hub.wait(
            lambda m: (
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-live"
                and m.attempt_outcome.attempt_ordinal == 1
                else None
            ),
            120,
            "CANCELED",
            mark,
        )
        canceled_at = time.perf_counter()
        body = terminal_body(terminal)
        check(
            "CANCELED is journaled only after the handler is provably gone",
            body["status"] == 4 and body["cause"]["code"] == 11,
            f"{(canceled_at - cancel_at) * 1000:.0f} ms cancel->terminal "
            f"(grace 2.0 s) — {body['safe_message'][:70]}",
        )
        forceful: list[str] = []
        for _ in range(50):  # the watchdog logs the kill facts on its own thread
            forceful = worker.tail("forceful:", 1)
            if forceful:
                break
            time.sleep(0.2)
        check(
            "the ordering is kill -> output abort -> epoch invalidation -> journal",
            bool(forceful),
            forceful[0].split("cancel: forceful: ")[-1] if forceful else "no kill facts",
        )

        head("a terminal ack that does not BIND the journaled terminal")
        # The forceful cancel above killed the executor; readiness is rebuilt AFTER, off the
        # cancellation critical path, so this waits for a FRESH ready report.
        hub.ready(plan_id, 400, since=hub.mark())
        hub.wrong_ack.add("req-ack#1")
        dispatch(hub, tmp, plan_id, request_id="req-ack", payload={"steps": 1, "latent": 48})
        hub.wait(
            lambda m: (
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-ack"
                else None
            ),
            300,
            "terminal",
        )
        time.sleep(2.0)
        not_an_ack = worker.tail("NOT AN ACK", 1)
        check(
            "a mismatched ack is NOT an ack; replay continues",
            bool(not_an_ack),
            not_an_ack[0].split("terminal: ")[-1][:110] if not_an_ack else "",
        )

        head("progress saturation cannot block a durable Control message")
        hub.ready(plan_id, 400, since=hub.mark())
        hub.block_progress.set()
        dispatch(hub, tmp, plan_id, request_id="req-sat", payload={"steps": 24, "latent": 48})
        terminal = hub.wait(
            lambda m: (
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-sat"
                else None
            ),
            300,
            "terminal under a stalled progress lane",
        )
        check(
            "the terminal arrived on Control while the lossy lane was held still",
            terminal_body(terminal)["status"] in (1, 2, 3),
            "two RPCs, two queues, two threads — no shared buffer to back up through",
        )
        hub.block_progress.clear()

        head("an output grant that names no destination")
        hub.ready(plan_id, 400, since=hub.mark())
        dispatch(
            hub, tmp, plan_id, request_id="req-grant", payload={"steps": 1, "latent": 48}, outputs=0
        )
        body = terminal_body(
            hub.wait(
                lambda m: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-grant"
                    else None
                ),
                300,
                "tail failure",
            )
        )
        check(
            "an output the RecordOwner did not grant is UNWRITABLE, however it was made",
            body["status"] == 3 and "destination_absent" in body["safe_message"],
            body["safe_message"][:130],
        )

        head("the hub-call authorization table is CLOSED")
        hub.ready(plan_id, 400, since=hub.mark())
        dispatch(
            hub,
            tmp,
            plan_id,
            request_id="req-offtable",
            payload={"steps": 1, "latent": 48},
            destination_url="https://exfiltrate.example/upload",
        )
        body = terminal_body(
            hub.wait(
                lambda m: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-offtable"
                    else None
                ),
                300,
                "refusal",
            )
        )
        check(
            "a destination outside the closed table is a SECURITY EVENT, not a 404",
            body["status"] == 3
            and (
                "hub_call_unauthorized" in body["safe_message"]
                or "grant_transport" in body["safe_message"]
            ),
            body["safe_message"][:150],
        )
    finally:
        summary = _shutdown(worker)
        print(f"  worker summary: {json.dumps(summary)}", flush=True)
        _gpu_clean(idle)


def section_journal(args: argparse.Namespace) -> None:
    """Journal honesty, executor honesty, and the worker restart."""
    tmp = Path(args.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    idle = gpu_used_mib()

    head("a journal that cannot durably commit acknowledges NOTHING and executes NOTHING")
    # 3,600 B leaves room for the worker's own boot `session` row (~200 B) under the
    # 4,096 B limit, and none for the acceptance row after it — which is the write the arm
    # is about. Grown too far, the plant bites at boot and the arm never reaches its subject.
    grown = prefill_journal(tmp / "worker", 3600)
    print(f"  the journal starts at {grown} B under a 4096 B planted file-size limit")
    worker = start_worker(tmp, entrypoints=("denoise",), no_warm=True, rlimit_fsize=4096)
    hub = worker.hub
    plan_id = plan_id_for("denoise")
    try:
        hub.ready(plan_id, 400)
        dispatch(hub, tmp, plan_id, request_id="req-enospc")
        time.sleep(5.0)
        accepted = hub.accepted("req-enospc")
        terminals = hub.terminals("req-enospc")
        stopped = worker.tail("journal:", 2)
        check(
            "no acceptance, no terminal, no execution",
            not accepted and not terminals and bool(stopped),
            (stopped[-1].split("journal: ")[-1][:150] if stopped else "no journal note"),
        )
        summary = _shutdown(worker)
        check(
            "and the attempt's host envelope was not left charged",
            bool(summary) and summary["ledger"]["host_ram"]["charged"] == 0,
            f"host ledger {summary.get('ledger', {}).get('host_ram')} — an acceptance "
            "that could not be journaled releases what it reserved",
        )
    finally:
        worker.hub.close()
        _gpu_clean(idle)

    head("a corrupted journal refuses to serve rather than guessing")
    tmp2 = Path(args.tmp + "-corrupt")
    shutil.rmtree(tmp2, ignore_errors=True)
    tmp2.mkdir(parents=True)
    worker = start_worker(tmp2, entrypoints=("denoise",), no_warm=True, instance_id="ins-fixed")
    try:
        worker.hub.ready(plan_id_for("denoise"), 400)
        dispatch(
            worker.hub,
            tmp2,
            plan_id_for("denoise"),
            request_id="req-j",
            payload={"steps": 1, "latent": 48},
        )
        worker.hub.wait(
            lambda m: m.attempt_outcome if m.WhichOneof("msg") == "attempt_outcome" else None,
            300,
            "terminal",
        )
        _shutdown(worker)
        worker.hub.close()
        journal = tmp2 / "worker" / "journal.log"
        data = bytearray(journal.read_bytes())
        target = len(data) // 2
        data[target] = data[target] ^ 0x20
        journal.write_bytes(bytes(data))
        restarted = start_worker(
            tmp2, entrypoints=("denoise",), no_warm=True, instance_id="ins-fixed", connect=False
        )
        code = restarted.process.wait(timeout=60)
        check(
            "a corrupted journal is a refusal to start, not a fresh start",
            code == 1,
            f"exit {code}: "
            f"{restarted.log.read_text(errors='replace').strip().splitlines()[-1][:120]}",
        )
        restarted.hub.close()
    finally:
        _gpu_clean(idle)

    head("worker restart: a NEW session that reports its recovered journal FIRST")
    tmp3 = Path(args.tmp + "-restart")
    shutil.rmtree(tmp3, ignore_errors=True)
    tmp3.mkdir(parents=True)
    worker = start_worker(tmp3, entrypoints=("denoise",), no_warm=True, instance_id="ins-stable")
    hub = worker.hub
    try:
        hub.ready(plan_id_for("denoise"), 400)
        hub.auto_ack = False
        dispatch(
            hub,
            tmp3,
            plan_id_for("denoise"),
            request_id="req-restart",
            payload={"steps": 1, "latent": 48},
        )
        terminal = hub.wait(
            lambda m: m.attempt_outcome if m.WhichOneof("msg") == "attempt_outcome" else None,
            300,
            "terminal",
        )
        first_bytes = terminal.outcome_canonical_bytes
        first_boot = hub.session()
        worker.process.send_signal(signal.SIGKILL)
        worker.process.wait(timeout=20)
        worker.hub.close()
        restarted = start_worker(
            tmp3, entrypoints=("denoise",), no_warm=True, instance_id="ins-stable"
        )
        hub2 = restarted.hub
        check(
            "the restart minted a NEW worker boot id, never an epoch bump",
            hub2.boot_id != first_boot and hub2.epoch == 1,
            f"{first_boot} -> {hub2.boot_id} (owner epoch still {hub2.epoch})",
        )
        check(
            "the recovered journal rides the SNAPSHOT, before any next ordinal (02 §6)",
            len(hub2.snapshot) == 1 and hub2.snapshot[0]["request_id"] == "req-restart",
            f"{len(hub2.snapshot)} recovered: state "
            + (pb.AttemptState.Name(hub2.snapshot[0]["state"]) if hub2.snapshot else "-"),
        )
        replay = hub2.wait(
            lambda m: m.attempt_outcome if m.WhichOneof("msg") == "attempt_outcome" else None,
            60,
            "replayed terminal",
        )
        check(
            "the SAME body is re-wrapped in a fresh envelope, byte for byte",
            replay.outcome_canonical_bytes == first_bytes and replay.worker_boot_id == hub2.boot_id,
            f"{len(first_bytes)} identical canonical bytes, id {replay.outcome_id}",
        )
        _shutdown(restarted)
        restarted.hub.close()
    finally:
        _gpu_clean(idle)


def section_seal(args: argparse.Namespace) -> None:
    """The late-imposition property, measured in a HARNESS-OWNED CUDA process.

    The runtime carries no plant switch for this (the red-arm discipline is
    plant -> observe -> REMOVE, and a permanent switch is decay). What is re-runnable is
    the PROPERTY and the CHECK that catches it, both exercised against real code: a child
    process snapshots the seal with the runtime's own `config.seal_snapshot`, initializes
    CUDA, imposes a value afterwards, and reports what the driver did and what the seal
    check says.
    """
    head("late environment imposition, after CUDA init")
    probe = Path(args.tmp) / "seal_probe.py"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(
        "import os, sys, json\n"
        f"sys.path[:0] = [{str(ROOT / 'src')!r}]\n"
        "from cozy_runtime.internal.config import seal_snapshot\n"
        "sealed = seal_snapshot()\n"
        "import torch\n"
        "torch.cuda.init()\n"
        "before = torch.cuda.device_count()\n"
        "os.environ['CUDA_VISIBLE_DEVICES'] = ''\n"
        "after = torch.cuda.device_count()\n"
        "print(json.dumps({'before': before, 'after': after,\n"
        "                  'intact': seal_snapshot() == sealed}))\n"
    )
    out = subprocess.run(
        ["nice", "-n", "19", sys.executable, str(probe)],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )
    facts = json.loads(out.stdout.strip().splitlines()[-1])
    check(
        "the imposition is a SILENT no-op for the running process",
        facts["before"] == facts["after"] == 1,
        f"device_count {facts['before']} -> {facts['after']} after "
        "CUDA_VISIBLE_DEVICES='' — the driver read its environment once, at init",
    )
    check(
        "the runtime's seal check is what turns that into a refusal",
        facts["intact"] is False,
        "config.seal_snapshot() no longer equals the snapshot CUDA initialized under, "
        "which is the `env_seal_broken` refusal the executor raises",
    )


def section_tripwire(args: argparse.Namespace) -> None:
    """STRUCTURAL: the worker never imports endpoint code, whatever the endpoint does.

    The endpoint module writes `<pid>.pid` into `corpus/endpoint/imports/` at MODULE SCOPE.
    Any process that imports it leaves a mark it cannot avoid leaving — a friendly endpoint
    whose torch imports are inside functions would prove nothing, so the tripwire fires on
    the import itself.
    """
    tmp = Path(args.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    idle = gpu_used_mib()
    marks = ROOT / "corpus" / "endpoint" / "imports"
    shutil.rmtree(marks, ignore_errors=True)
    worker = start_worker(tmp, entrypoints=("denoise", "restyle"))
    hub = worker.hub
    try:
        hub.ready(plan_id_for("denoise"), 400)
        head("who imported the endpoint module")
        worker_pid = worker.process.pid
        executor_pid = worker.executor_pid()
        seen = sorted(int(path.stem) for path in marks.glob("*.pid"))
        check(
            "the endpoint was imported by the EXECUTOR and by nobody else",
            seen == [executor_pid],
            f"import marks {seen}; executor {executor_pid}, worker {worker_pid} "
            "— the warm pass synthesizes and runs INSIDE the disposable process, so no "
            "endpoint module scope can ever contaminate the no-CUDA parent",
        )
        check(
            "and the warm pass still ran, through ordinary dispatch",
            bool(worker.tail(r"warm: denoise: warmed", 1)),
            " | ".join(row.split("warm: ")[-1] for row in worker.tail(r"warm:", 4)),
        )
    finally:
        _shutdown(worker)
        worker.hub.close()
        _gpu_clean(idle)


def section_acceptance(args: argparse.Namespace) -> None:
    """The acceptance boundary: nothing is accepted that cannot then run."""
    tmp = Path(args.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    idle = gpu_used_mib()
    worker = start_worker(tmp, entrypoints=("denoise",), no_warm=True)
    hub = worker.hub
    plan_id = plan_id_for("denoise")
    try:
        hub.ready(plan_id, 400)

        head("a granted payload input that is not there")
        canonical_bytes, digest = make_spec(plan_id, outputs=0)
        grant = pb.DeliveryGrant(
            invocation_spec_digest=digest,
            file_base_url=f"file://{tmp / 'grants'}",
            expires_at_unix=int(time.time()) + 600,
            inputs=[
                pb.InputAccess(
                    input_id="payload",
                    url=f"file://{tmp / 'grants' / 'absent.json'}",
                )
            ],
            outputs=[],
        )
        hub.send(
            pb.RecordOwnerFrame(
                attempt_offer=hub._env(
                    pb.AttemptOffer(
                        request_id="req-nopayload",
                        attempt_ordinal=1,
                        invocation_spec_digest=digest,
                        grant=grant,
                        invocation_spec_canonical_bytes=canonical_bytes,
                        placement_id=PLACEMENT,
                        admission_generation=hub.admission_generation,
                    )
                )
            )
        )
        body = terminal_body(
            hub.wait(
                lambda m: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-nopayload"
                    else None
                ),
                120,
                "refusal",
            )
        )
        check(
            "refused BEFORE acceptance, with no accept ever published",
            body["status"] == 2 and not hub.accepted("req-nopayload"),
            body["safe_message"][:130],
        )

        head("an EXPIRED delivery grant")
        dispatch(
            hub,
            tmp,
            plan_id,
            request_id="req-expired",
            payload={"steps": 1, "latent": 48},
            grant_expires=int(time.time()) - 60,
        )
        body = terminal_body(
            hub.wait(
                lambda m: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-expired"
                    else None
                ),
                120,
                "refusal",
            )
        )
        check(
            "an expired grant is a pre-accept refusal, never an accepted attempt",
            body["status"] == 2 and not hub.accepted("req-expired"),
            f"cause {body['cause']['code']} — {body['safe_message'][:100]}",
        )

        head("a malformed payload document")
        bad = tmp / "grants" / "malformed.json"
        bad.parent.mkdir(parents=True, exist_ok=True)
        raw = b"[1, 2, 3]"
        bad.write_bytes(raw)
        # The spec binds THESE bytes (the digest matches), so the refusal is about the
        # DOCUMENT SHAPE, not the digest — which is the arm.
        mal_bytes, mal_digest = make_spec(plan_id, outputs=0, payload_raw=raw)
        grant = pb.DeliveryGrant(
            invocation_spec_digest=mal_digest,
            file_base_url=f"file://{tmp / 'grants'}",
            expires_at_unix=int(time.time()) + 600,
            inputs=[pb.InputAccess(input_id="payload", url=f"file://{bad}")],
            outputs=[],
        )
        hub.send(
            pb.RecordOwnerFrame(
                attempt_offer=hub._env(
                    pb.AttemptOffer(
                        request_id="req-malformed",
                        attempt_ordinal=1,
                        invocation_spec_digest=mal_digest,
                        grant=grant,
                        invocation_spec_canonical_bytes=mal_bytes,
                        placement_id=PLACEMENT,
                        admission_generation=hub.admission_generation,
                    )
                )
            )
        )
        body = terminal_body(
            hub.wait(
                lambda m: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-malformed"
                    else None
                ),
                120,
                "refusal",
            )
        )
        check(
            "a payload that is not an object refuses typed",
            body["status"] == 2 and "payload_undecodable" in body["safe_message"],
            body["safe_message"][:120],
        )

        head("and a VALID request on the same worker still serves")
        dispatch(hub, tmp, plan_id, request_id="req-good", payload={"steps": 1, "latent": 48})
        body = terminal_body(
            hub.wait(
                lambda m: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-good"
                    else None
                ),
                300,
                "terminal",
            )
        )
        check(
            "three pre-accept refusals left the worker fully serving",
            body["status"] == 1,
            "the refusals never touched the executor",
        )
    finally:
        _shutdown(worker)
        worker.hub.close()
        _gpu_clean(idle)


def section_abandon(args: argparse.Namespace) -> None:
    """Worker death AFTER acceptance and BEFORE any terminal."""
    tmp = Path(args.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    idle = gpu_used_mib()
    worker = start_worker(
        tmp, entrypoints=("denoise", "stubborn"), no_warm=True, instance_id="ins-abandon"
    )
    hub = worker.hub
    stubborn_id = plan_id_for("stubborn")
    try:
        hub.ready(stubborn_id, 400)
        head("kill -9 the WORKER while an accepted attempt is running")
        dispatch(
            hub,
            tmp,
            stubborn_id,
            request_id="req-lost",
            payload={"steps": 900, "latent": 80},
            outputs=0,
        )
        hub.wait(
            lambda m: (
                m.attempt_accepted
                if m.WhichOneof("msg") == "attempt_accepted"
                and m.attempt_accepted.request_id == "req-lost"
                else None
            ),
            120,
            "accepted",
        )
        time.sleep(2.0)
        check(
            "the attempt is accepted with no terminal",
            not hub.terminals("req-lost"),
            "journaled ACCEPTED, running, nothing shipped",
        )
        worker.process.send_signal(signal.SIGKILL)
        worker.process.wait(timeout=20)
        worker.hub.close()

        head("the restarted worker mints the terminal the dead one owed")
        restarted = start_worker(
            tmp, entrypoints=("denoise", "stubborn"), no_warm=True, instance_id="ins-abandon"
        )
        hub2 = restarted.hub
        check(
            "the recovered attempt rides the SNAPSHOT as TERMINAL-PENDING-ACK",
            len(hub2.snapshot) == 1
            and hub2.snapshot[0]["state"] == pb.AttemptState.ATTEMPT_STATE_OUTCOME_PENDING_ACK,
            "recovery mints a durable terminal, never a live attempt with no plan",
        )
        terminal = hub2.wait(
            lambda m: m.attempt_outcome if m.WhichOneof("msg") == "attempt_outcome" else None,
            60,
            "ABANDONED",
        )
        body = terminal_body(terminal)
        check(
            "it is ABANDONED / EXECUTOR_INVALIDATED, journaled and replayable",
            body["status"] == 5 and body["cause"]["code"] == 16,
            body["safe_message"][:120],
        )

        head("a duplicate StartAttempt for that ordinal replays, and starts nothing")
        mark = hub2.mark()
        dispatch(
            hub2,
            tmp,
            stubborn_id,
            request_id="req-lost",
            payload={"steps": 900, "latent": 80},
            outputs=0,
        )
        # The restarted worker is preparing its executor on the reader thread, so the
        # duplicate waits behind that; it is queued, never lost.
        hub2.wait(
            lambda m: (
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-lost"
                else None
            ),
            400,
            "replayed terminal",
            mark,
        )
        replays = hub2.terminals("req-lost")
        check(
            "the recovered terminal replays byte-for-byte; no plan is ever demanded",
            len(replays) >= 2
            and replays[0].outcome_canonical_bytes == replays[-1].outcome_canonical_bytes,
            f"{len(replays)} terminals, identical bodies, id {replays[0].outcome_id}",
        )
        _shutdown(restarted)
        restarted.hub.close()
    finally:
        _gpu_clean(idle)


def section_results(args: argparse.Namespace) -> None:
    """Result truthfulness: serialized once, declared schema digest, ids not positions."""
    tmp = Path(args.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    idle = gpu_used_mib()
    worker = start_worker(tmp, entrypoints=("pair", "bloat"), no_warm=True)
    hub = worker.hub
    pair_id, bloat_id = plan_id_for("pair"), plan_id_for("bloat")
    try:
        hub.ready(pair_id, 400)
        head("two outputs, nested, granted in the WRONG order")
        out_dir = tmp / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        pair_payload = {"steps": 1, "latent": 48}
        body_bytes, payload_path = stage_payload(tmp, "req-pair", pair_payload)
        spec = pb.InvocationSpec(
            endpoint_release_id=RELEASE,
            environment_spec_digest="sha256:" + "11" * 32,
            config_digest="sha256:" + "22" * 32,
            payload_digest="sha256:" + hashlib.sha256(body_bytes).hexdigest(),
            inputs=[
                pb.InputBinding(
                    input_id="payload",
                    digest="sha256:" + hashlib.sha256(body_bytes).hexdigest(),
                    length=len(body_bytes),
                    kind_mime="application/json",
                    order=0,
                )
            ],
            outputs=[
                pb.OutputBinding(output_id="detail.thumb", max_bytes=8 * MIB),
                pb.OutputBinding(output_id="preview", max_bytes=8 * MIB),
            ],
        )
        spec.serving.entrypoint_binding_plan_id = pair_id
        spec.serving.attempt_binding_id = pair_id
        canonical_bytes, digest = documents.identity(spec)
        grant = pb.DeliveryGrant(
            invocation_spec_digest=digest,
            file_base_url=f"file://{tmp / 'grants'}",
            expires_at_unix=int(time.time()) + 600,
            inputs=[pb.InputAccess(input_id="payload", url=f"file://{payload_path}")],
            outputs=[
                # deliberately NOT in the order the handler produced them
                pb.OutputAccess(output_id="detail.thumb", url=f"file://{out_dir / 'thumb.png'}"),
                pb.OutputAccess(output_id="preview", url=f"file://{out_dir / 'preview.png'}"),
            ],
        )
        hub.send(
            pb.RecordOwnerFrame(
                attempt_offer=hub._env(
                    pb.AttemptOffer(
                        request_id="req-pair",
                        attempt_ordinal=1,
                        invocation_spec_digest=digest,
                        grant=grant,
                        invocation_spec_canonical_bytes=canonical_bytes,
                        placement_id=PLACEMENT,
                        admission_generation=hub.admission_generation,
                    )
                )
            )
        )
        body = terminal_body(
            hub.wait(
                lambda m: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-pair"
                    else None
                ),
                300,
                "terminal",
            )
        )
        entries = body.get("output_manifest", {}).get("outputs", [])
        ids = sorted(entry["output_id"] for entry in entries)
        check(
            "both outputs landed, addressed by their FIELD PATHS",
            ids == ["detail.thumb", "preview"],
            f"{ids} — never by list position",
        )
        landed = {
            "preview": (out_dir / "preview.png").read_bytes(),
            "detail.thumb": (out_dir / "thumb.png").read_bytes(),
        }
        by_id = {entry["output_id"]: entry for entry in entries}
        check(
            "each destination holds the bytes its own id names",
            all(
                by_id[name]["digest"] == "sha256:" + hashlib.sha256(data).hexdigest()
                for name, data in landed.items()
            )
            and len(landed["preview"]) != len(landed["detail.thumb"]),
            f"preview {len(landed['preview'])} B, thumb {len(landed['detail.thumb'])} B "
            "— a positional binding would have swapped them",
        )

        head("the typed result survived as TYPES, serialized exactly once")
        import base64 as b64

        from cozy_runtime.internal import canonical

        result = canonical.parse(b64.b64decode(body["result"]["inline_result"]))
        scores = result.get("detail", {}).get("scores")
        check(
            "a nested struct is a struct and its numbers are numbers",
            isinstance(scores, list)
            and all(isinstance(s, float) for s in scores)
            and isinstance(result["detail"]["note"], str),
            f"detail.scores = {scores} — no member was stringified on the way out",
        )

        head("the carried schema digest is the DECLARED result surface's")
        declared = worker.result_schema_digest("pair")
        check(
            "it equals the descriptor's annotated result-surface digest, exactly",
            body["result"]["result_schema_digest"] == declared,
            f"{declared[:32]}… — a digest of this VALUE would move every request",
        )

        head("a typed result past the inline door")
        hub.ready(bloat_id, 400, since=hub.mark())
        dispatch(
            hub,
            tmp,
            bloat_id,
            request_id="req-bloat",
            payload={"steps": 1, "latent": 48},
            outputs=0,
        )
        body = terminal_body(
            hub.wait(
                lambda m: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-bloat"
                    else None
                ),
                300,
                "refusal",
            )
        )
        check(
            "it REFUSES typed rather than minting a receipt for bytes nobody wrote",
            body["status"] == 3 and "result_too_large" in body["safe_message"],
            body["safe_message"][:150],
        )
    finally:
        _shutdown(worker)
        worker.hub.close()
        _gpu_clean(idle)


# --------------------------------------------------------------- the four-component fixture

#: cr-008b's real SDXL pipeline, driven through THIS RecordOwner so a resident worker can be
#: given more than one request. `pipeline-live.py` runs one request per process by design —
#: it measures cold-to-first-image — and the defect cl-003 found lives entirely in the SECOND
#: request on a worker that is already warm.
PIPE_WORK = Path("/tmp/cozy-sdxl4")
PIPE_RELEASE = "cozy/sdxl-pipeline@cr-008b"
PIPE_COMPONENTS = ("text_encoder", "text_encoder_2", "unet", "vae")


def pipeline_overrides() -> dict[str, Any]:
    """The binding record for the whole pipeline: four components, one snapshot each."""
    snapshots = dict(json.loads((PIPE_WORK / "snapshots.json").read_text()))
    return {
        "project": str(ROOT / "corpus" / "pipeline"),
        "model_class": "SdxlPipelineModel",
        "model_binding_path": "generate.models.model",
        "model_parameter_name": "model",
        "components": list(PIPE_COMPONENTS),
        "store": str(PIPE_WORK / "store"),
        "config": str(PIPE_WORK / "pipeline.config.json"),
        "snapshot": snapshots["unet"],
        "snapshots": dict(sorted(snapshots.items())),
        "variant": "sm89",
        "host_bytes": 2 * GIB,
        "pinned_bytes": 256 * MIB,
        "vram_floor_bytes": 1 * GIB,
        "tenancy": "local",
    }


def _ledger_rows(worker: Worker) -> list[dict[str, Any]]:
    """Every journaled ledger reconciliation, in order, read from the DURABLE journal."""
    out = []
    for line in (worker.root / "journal.log").read_text(errors="replace").splitlines():
        # Each journal line is `<sha256 of the canonical bytes> <canonical bytes>` — the
        # digest is the record's own integrity fence, so the body starts after the space.
        _, _, payload = line.partition(" ")
        try:
            row = json.loads(payload)
        except ValueError:
            continue
        if row.get("kind") == "ledger":
            out.append(dict(row.get("body", {})))
    return out


def _device_row(record: dict[str, Any]) -> dict[str, Any]:
    for row in record.get("classes", ()):
        if row.get("class") == "device_allocated":
            return dict(row)
    return {}


def section_resident(args: argparse.Namespace) -> dict[str, Any]:
    """N consecutive 1024px requests on ONE resident worker (cl-003's blocking defect).

    The defect: a `component_staged` attempt returned the generation with its components
    evicted, the ledger reported the drift as an unreconciled leak and then did nothing about
    it, and the NEXT request's chooser priced `all_resident` against a residency map holding
    one 167 MB VAE — so the 1024px decode ran out of memory inside the kernel and was
    reported as the endpoint's own exception.
    """
    tmp = Path(args.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    idle = gpu_used_mib()
    worker = start_worker(
        tmp,
        entrypoints=("generate",),
        overrides=pipeline_overrides(),
        release=PIPE_RELEASE,
        no_warm=args.no_warm,
    )
    hub = worker.hub
    plan_id = plan_id_for("generate", PIPE_RELEASE)
    row: dict[str, Any] = {}
    try:
        head(f"boot: the WHOLE four-component pipeline as one resident worker ({args.size}px)")
        boot = time.perf_counter()
        report = hub.ready(plan_id, 900)
        row["ready_s"] = round(time.perf_counter() - boot, 2)
        check(
            "the four-component pipeline reached DISPATCHABLE",
            plan_id in hub.deployment(report).dispatchable_plan_ids,
            f"{row['ready_s']:.1f} s, executor generation "
            f"{hub.deployment(report).executor_generation}",
        )

        head(f"{args.repeat} CONSECUTIVE {args.size}px requests on that one worker")
        served: list[dict[str, Any]] = []
        for index in range(1, args.repeat + 1):
            request_id = f"pipe-{index}"
            started = time.perf_counter()
            dispatch(
                hub,
                tmp,
                plan_id,
                request_id=request_id,
                payload={
                    "prompt": args.prompt,
                    "steps": args.steps,
                    "size": args.size,
                    "seed": 1005,
                },
                release=PIPE_RELEASE,
                max_bytes=16 * MIB,
            )
            accepted = hub.wait(
                lambda m, r=request_id: (
                    m.attempt_accepted
                    if m.WhichOneof("msg") == "attempt_accepted"
                    and m.attempt_accepted.request_id == r
                    else None
                ),
                300,
                "AttemptAccepted",
            )
            terminal = hub.wait(
                lambda m, r=request_id: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == r
                    else None
                ),
                900,
                "AttemptTerminal",
            )
            body = terminal_body(terminal)
            inline = body.get("result", {}).get("inline_result", "") or ""
            result = json.loads(base64.b64decode(inline)) if inline else {}
            outputs = body.get("output_manifest", {}).get("outputs") or [{}]
            served.append(
                {
                    "request": request_id,
                    "wall_ms": round((time.perf_counter() - started) * 1000),
                    "status": body["status"],
                    "cause": str(body.get("cause", {}).get("code", "")),
                    "message": str(body.get("safe_message", ""))[:200],
                    "placement": accepted.plan.placement,
                    "reserved_vram_bytes": int(accepted.plan.reserved_device_memory_bytes),
                    "choice": accepted.plan.decision_explanation,
                    "digest": str(result.get("digest", "")),
                    "png_bytes": int(outputs[0].get("length", 0)),
                }
            )
            last = served[-1]
            print(
                f"    {request_id}: {last['wall_ms']} ms status={last['status']} "
                f"{last['placement']} vram={last['reserved_vram_bytes']} "
                f"digest={last['digest'][:16]} png={last['png_bytes']} B",
                flush=True,
            )
            if last["status"] != 1:
                print(f"      {last['cause']}: {last['message']}", flush=True)
        row["served"] = served
        check(
            f"all {args.repeat} requests SUCCEEDED on one resident worker",
            all(s["status"] == 1 for s in served),
            "; ".join(
                f"{s['request']}={s['status']}"
                + (f" {s['cause']} {s['message'][:70]}" if s["status"] != 1 else "")
                for s in served
            ),
        )
        digests = {s["digest"] for s in served}
        check(
            "every one produced the SAME image digest — the pixels are a fence, not a hope",
            len(digests) == 1 and bool(next(iter(digests))),
            f"{next(iter(digests))[:32]}… on {len(served)} runs, {served[0]['png_bytes']} B PNG",
        )
        check(
            "and the plan reserved the GENERATION's weights every time, not what was resident",
            len({s["reserved_vram_bytes"] for s in served}) == 1
            and served[0]["reserved_vram_bytes"] > 6 * GIB,
            f"{served[0]['reserved_vram_bytes']} B on every attempt ({served[0]['placement']})",
        )

        head("the ledger reconciles to the same numbers after every attempt")
        records = _ledger_rows(worker)
        row["reconciliations"] = [
            {
                "attempt": rec.get("attempt"),
                "closed": rec.get("closed"),
                "declared_residency_delta": rec.get("declared_residency_delta"),
                "device": _device_row(rec),
                "leaks": rec.get("leaks"),
            }
            for rec in records
        ]
        for rec in row["reconciliations"]:
            device = rec["device"]
            print(
                f"    {rec['attempt']!s:<14} closed={rec['closed']} "
                f"device {device.get('before')} -> {device.get('after')} "
                f"delta {device.get('delta')} declared {rec['declared_residency_delta']} "
                f"residual {device.get('residual', 0)} [{device.get('verdict')}]",
                flush=True,
            )
        paying = [rec for rec in row["reconciliations"] if str(rec["attempt"]).startswith("pipe-")]
        check(
            "every attempt CLOSED — no unaccounted byte in any reconciled class",
            len(paying) == args.repeat and all(rec["closed"] for rec in paying),
            f"{len(paying)} reconciliation(s), leaks {[r['leaks'] for r in paying if r['leaks']]}",
        )
        # The two mechanisms COMPOSE, and the decomposition is the claim: the declared
        # transition comes off first, and whatever is left is either nothing (reconciled) or
        # the first-call library workspace this generation has not paid yet (`baseline_shift`
        # — only reachable on the first attempt, and only with the warm pass off). A `leak` is
        # what is left when neither explains it, and nothing here may be one.
        verdicts = [rec["device"].get("verdict") for rec in paying]
        check(
            "a staged rung's eviction is a DECLARED TRANSITION, reconciled, never a leak",
            bool(paying)
            and any(rec["declared_residency_delta"] for rec in paying)
            and "leak" not in verdicts
            and all(rec["device"].get("residual", 0) == 0 for rec in paying[1:]),
            "; ".join(
                f"{rec['attempt']} moved {rec['declared_residency_delta']:+d} B "
                f"[{rec['device'].get('verdict')}, residual "
                f"{rec['device'].get('residual', 0)} B]"
                for rec in paying
            ),
        )
        steady = paying[1:]
        check(
            "and the residency the ledger reports is STABLE across requests 2..N",
            bool(steady)
            and len({rec["device"].get("after") for rec in steady}) == 1
            and len({rec["device"].get("before") for rec in steady}) == 1,
            "device_allocated "
            + " · ".join(
                f"{rec['device'].get('before')} -> {rec['device'].get('after')}" for rec in steady
            ),
        )
        row["gpu_after"] = gpu_used_mib()
    finally:
        row["summary"] = _shutdown(worker)
        _gpu_clean(idle)
    return row


def section_drift(args: argparse.Namespace) -> dict[str, Any]:
    """RED ARM: a PLANTED residency leak refuses at CHOOSE time, not inside a decode.

    The plant is harness-side and it is the real mechanism rather than a switch in the
    runtime: a leak is planted by making the ENDPOINT hold a device allocation past its own
    terminal (`corpus/endpoint/sdxl_gpu.py`'s `leak_mib` request field), which is exactly
    what an executor-poisoning defect looks like from the ledger's side.
    """
    tmp = Path(args.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    idle = gpu_used_mib()
    # The warm pass ON, deliberately: it absorbs the lazily-allocated library workspace, so
    # the FIRST paying attempt already reconciles to zero and the only thing left for the
    # plant to be is what it is (slice-live's `ledger` section proves that separately).
    worker = start_worker(tmp, entrypoints=("denoise",), no_warm=False)
    hub = worker.hub
    plan_id = plan_id_for("denoise")
    row: dict[str, Any] = {}
    try:
        ready = hub.ready(plan_id, 400)
        head("a PLANTED persistent device allocation, held past the terminal")
        dispatch(
            hub,
            tmp,
            plan_id,
            request_id="req-leak",
            payload={"steps": 2, "latent": 64, "seed": 7, "leak_mib": 64},
        )
        hub.wait(
            lambda m: (
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-leak"
                else None
            ),
            300,
            "AttemptTerminal",
        )
        record = _ledger_rows(worker)[-1]
        device = _device_row(record)
        row["leaked"] = {"leaks": record.get("leaks"), "device": device}
        check(
            "the reconciliation NAMES it rather than absorbing it",
            not record.get("closed") and bool(record.get("leaks")),
            f"device_allocated {device.get('before')} -> {device.get('after')} "
            f"residual {device.get('residual', device.get('delta'))} B "
            f"[{device.get('verdict')}]",
        )
        reported = [
            fault
            for report in hub.reports()
            for fault in report.faults
            if fault.reason == "ledger_unreconciled"
        ]
        check(
            "an unreconciled ledger is LOUD at the RecordOwner seat, not a silent drift",
            bool(reported) and bool(record.get("leaks")),
            (
                f"{pb.FaultKind.Name(reported[-1].kind)} {reported[-1].reason}: "
                f"{reported[-1].detail[:120]}"
                if reported
                else "no ledger_unreconciled fault reached a Report"
            ),
        )

        head("the NEXT request is never priced against the drift: replaced, or refused typed")
        since = hub.mark()
        dispatch(
            hub,
            tmp,
            plan_id,
            request_id="req-after-leak",
            payload={"steps": 2, "latent": 64, "seed": 7},
        )
        after = hub.wait(
            lambda m: (
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-after-leak"
                else None
            ),
            300,
            "AttemptTerminal",
            since,
        )
        body = terminal_body(after)
        message = str(body.get("safe_message", ""))
        accepted = hub.accepted("req-after-leak")
        row["after"] = {"status": body["status"], "message": message[:300]}
        # WHAT THE NEXT REQUEST MUST NOT DO is be planned against the drift. There are two
        # honest ways not to: refuse typed at choose time, or run on a generation whose
        # ledger was re-established — which is what the invalidation buys, and which the
        # readiness fence makes the ordinary outcome (the request queues behind the rebuild).
        # Both are checked as ONE claim, because "either" is only weak if the alternative is
        # unspecified: it either refused, or it was priced against a ledger that reconciles.
        refused = body["status"] != 1 and "ledger_unreconciled" in message
        ready_generation = hub.deployment(ready).executor_generation
        reused = bool(accepted) and accepted[0].executor_generation <= ready_generation
        check(
            "the drifted generation is NEVER reused: refused typed, or replaced first",
            refused or (bool(accepted) and not reused),
            f"status {body['status']} · generation {ready_generation} -> "
            f"{accepted[0].executor_generation if accepted else '(never accepted)'} · "
            f"{message[:120]}",
        )
        closed = _ledger_rows(worker)[-1]
        check(
            "and whatever served next closed its own ledger — the drift did not survive",
            body["status"] != 1 or bool(closed.get("closed")),
            f"{closed.get('attempt')}: closed={closed.get('closed')} leaks={closed.get('leaks')}",
        )
        row["gpu_after"] = gpu_used_mib()
    finally:
        row["summary"] = _shutdown(worker)
        _gpu_clean(idle)
    return row


def section_readiness(args: argparse.Namespace) -> dict[str, Any]:
    """cr-007: a device process that cannot serve must stop advertising READY.

    cl-003 measured the shape this replaces: a forceful cancel killed the executor, the
    worker went on reporting `INTAKE_STATE_READY` until the rebuild started, and every
    attempt dispatched into that window died `EXECUTOR_FAULT` — one cancel spending a
    requeue budget of three in 1.8 seconds against a worker that served again 4 s later.
    """
    tmp = Path(args.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    idle = gpu_used_mib()
    # BOTH entrypoints, one construction: `stubborn` exists to be killed (it never asks
    # whether it was cancelled and it produces no output), and `denoise` is what the next
    # request needs — a handler that actually serves. Same store, same snapshot, same
    # component, so `prepare_executor` groups them into ONE generation and both plan ids go
    # ready together; the recovery gap measured below is therefore a real serve, not a boot.
    worker = start_worker(tmp, entrypoints=("stubborn", "denoise"), no_warm=True, grace=2.0)
    hub = worker.hub
    plan_id = plan_id_for("stubborn")
    serve_id = plan_id_for("denoise")
    row: dict[str, Any] = {}
    try:
        ready = hub.ready(plan_id, 400)
        head("a NON-COOPERATIVE handler, cancelled forcefully")
        check(
            "the worker starts DISPATCHABLE and says which plans it can serve",
            hub.deployment(ready).serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
            and plan_id in hub.deployment(ready).dispatchable_plan_ids,
            f"{len(hub.deployment(ready).dispatchable_plan_ids)} dispatchable plan(s), "
            f"executor generation {hub.deployment(ready).executor_generation}",
        )
        _, digest = dispatch(
            hub, tmp, plan_id, request_id="req-stub", payload={"steps": 400, "latent": 64}
        )
        hub.wait(
            lambda m: (
                m.attempt_accepted
                if m.WhichOneof("msg") == "attempt_accepted"
                and m.attempt_accepted.request_id == "req-stub"
                else None
            ),
            180,
            "AttemptAccepted",
        )
        time.sleep(args.kill_after)
        since = hub.mark()
        cancelled_at = time.perf_counter()
        hub.send(
            pb.RecordOwnerFrame(
                cancel_attempt=hub._env(
                    pb.CancelAttempt(
                        request_id="req-stub",
                        attempt_ordinal=1,
                        reason=pb.CancelReason.CANCEL_REASON_CLIENT,
                        invocation_spec_digest=digest,
                    )
                )
            )
        )
        dropped = hub.wait(
            lambda m: (
                m.observed_state
                if m.WhichOneof("msg") == "observed_state"
                and m.observed_state.admission_state != pb.AdmissionState.ADMISSION_STATE_OPEN
                else None
            ),
            60,
            "an ObservedWorkerState that stops claiming an OPEN window",
            since,
        )
        drop_ms = (time.perf_counter() - cancelled_at) * 1000
        row["capacity_drop_ms"] = round(drop_ms)
        check(
            "the forceful kill CLOSES admission at the RecordOwner seat",
            dropped.admission_state == pb.AdmissionState.ADMISSION_STATE_CLOSED
            and not list(hub.deployment(dropped).dispatchable_plan_ids),
            f"{pb.AdmissionState.Name(dropped.admission_state)} with "
            f"{len(hub.deployment(dropped).dispatchable_plan_ids)} dispatchable plan(s), "
            f"{drop_ms:.0f} ms after the cancel was sent (serving "
            f"{pb.ServingState.Name(hub.deployment(dropped).serving)}, admission generation "
            f"{dropped.admission_generation})",
        )
        # DISPATCHED INTO THE WINDOW, on purpose and immediately: the executor is dead, the
        # rebuild has not spawned its replacement yet, so the incarnation the RecordOwner
        # holds is still the current one and this StartAttempt is a live message aimed at a
        # worker that cannot serve. That window is the whole defect — it is where every
        # attempt used to die `executor_not_ready`/EXECUTOR_FAULT until a requeue budget was
        # gone. There is no rush to lose: the respawn is some 200 ms behind the Report that
        # said READY had ended, and this dispatch is one.
        head("the NEXT request QUEUES behind the rebuild instead of burning a requeue budget")
        queued_at = time.perf_counter()
        dispatch(hub, tmp, serve_id, request_id="req-next", payload={"steps": 1, "latent": 64})

        terminal = hub.wait(
            lambda m: (
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-stub"
                else None
            ),
            120,
            "the CANCELED terminal",
        )
        cancel_body = terminal_body(terminal)
        check(
            "the cancelled attempt still converges to a typed CANCELED terminal",
            cancel_body["status"] == 4,
            f"status {cancel_body['status']} cause {cancel_body.get('cause', {}).get('code')}",
        )
        next_terminal = hub.wait(
            lambda m: (
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-next"
                else None
            ),
            300,
            "the next attempt's terminal",
        )
        row["recovery_gap_ms"] = round((time.perf_counter() - queued_at) * 1000)
        next_body = terminal_body(next_terminal)
        accepted_next = hub.accepted("req-next")
        check(
            "it SERVED — one dispatch, one terminal, no dispatch/refuse/requeue loop",
            next_body["status"] == 1 and len(hub.terminals("req-next")) == 1,
            f"status {next_body['status']} after {row['recovery_gap_ms']} ms, "
            f"{len(hub.terminals('req-next'))} terminal(s), "
            f"{len(accepted_next)} acceptance(s)",
        )
        check(
            "on a genuinely re-spawned executor, not the dead one",
            bool(accepted_next)
            and accepted_next[0].executor_generation > hub.deployment(ready).executor_generation,
            f"generation {hub.deployment(ready).executor_generation} -> "
            f"{accepted_next[0].executor_generation if accepted_next else '?'}",
        )
        back = [
            report
            for report in hub.reports()
            if hub.deployment(report).serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
            and serve_id in hub.deployment(report).dispatchable_plan_ids
        ]
        check(
            "and DISPATCHABLE came back only once the executor was re-prepared",
            len(back) >= 2,
            f"{len(back)} READY report(s) carrying the plan; recovery gap "
            f"{row['recovery_gap_ms']} ms",
        )
        waited = list(worker.tail("reached the lane while the worker is rebuilding", 4))
        row["queue_note"] = waited[-1][-160:] if waited else ""
        check(
            "the worker WAITED for it rather than refusing it — one attempt, no refusal",
            bool(waited) and not any("refused" in line for line in worker.tail("req-next", 12)),
            row["queue_note"] or "no queue note in the worker's activity lane",
        )
        row["gpu_after"] = gpu_used_mib()
    finally:
        row["summary"] = _shutdown(worker)
        _gpu_clean(idle)
    return row


def section_bench(args: argparse.Namespace) -> None:
    """The numbers: boot, spawn-to-ready, protocol tax, recovery, RSS."""
    results = ROOT / "results" / "worker.jsonl"
    results.parent.mkdir(exist_ok=True)
    rows: dict[str, Any] = {}
    for name, fn in (
        ("attempt", section_attempt),
        ("kill", section_kill),
        ("nocuda", section_nocuda),
    ):
        rows[name] = fn(args)
    # The protocol tax used to be measured by shelling out to `fill-real.py serve` for a
    # bare forward pass. That script died with the interim reader it drove, and the
    # comparison no longer needs it: the attempt row already carries `handler_ms`, which IS
    # the bare computation, beside the accept-to-terminal window that wraps it. Subtracting
    # two numbers from one run beats comparing two runs of two programs.
    attempt = rows.get("attempt", {})
    rows["protocol_tax_ms"] = round(
        float(attempt.get("accept_to_terminal_ms", 0)) - float(attempt.get("handler_ms", 0)), 2
    )
    with results.open("a") as handle:
        handle.write(json.dumps({"at": int(time.time()), **rows}) + "\n")
    print(json.dumps(rows, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "section",
        choices=(
            "fixtures",
            "attempt",
            "kill",
            "rollback",
            "nocuda",
            "arms",
            "journal",
            "seal",
            "tripwire",
            "acceptance",
            "abandon",
            "results",
            "resident",
            "drift",
            "readiness",
            "bench",
        ),
    )
    parser.add_argument("--tmp", default="")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--kill-after", type=float, default=3.0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--no-warm", action="store_true")
    # cr-008b's banked prompt, so the 1024px pixel digest is comparable to the number this
    # repository already holds rather than to itself alone.
    parser.add_argument("--prompt", default="a photograph of an astronaut riding a horse")
    args = parser.parse_args()
    # One workspace per invocation. A worker whose harness went away keeps reconnecting
    # to its hub socket, so a shared path would let a previous run's worker register into
    # this run's RecordOwner — two workers, one hub, and unreadable evidence.
    args.tmp = args.tmp or f"/tmp/cozy-cr007/{args.section}-{os.getpid()}"
    sections = {
        "fixtures": section_fixtures,
        "attempt": section_attempt,
        "kill": section_kill,
        "rollback": section_rollback,
        "nocuda": section_nocuda,
        "arms": section_arms,
        "journal": section_journal,
        "seal": section_seal,
        "tripwire": section_tripwire,
        "acceptance": section_acceptance,
        "abandon": section_abandon,
        "results": section_results,
        "resident": section_resident,
        "drift": section_drift,
        "readiness": section_readiness,
        "bench": section_bench,
    }
    row = sections[args.section](args)
    if isinstance(row, dict):
        print("\n" + json.dumps(row, indent=2))
    print(f"\n{PASS} passed, {FAIL} failed", flush=True)
    # Hard exit: the RecordOwner's gRPC server and the hub's reader threads are not this
    # script's product, and a lingering harness holds a socket the next section wants.
    os._exit(1 if FAIL else 0)


if __name__ == "__main__":
    raise SystemExit(main())
