"""The `cozy.worker.v1` WORKER: the machine's one control process, hosting `WorkerControl`.

The worker is the gRPC SERVER; the record-plane owner dials it (worker-protocol/01,
decisions #436/#454 — the 2026-08-25 re-landing). Two RPCs run at launch and their
durability attributes are the whole design:

* **Control** — one durable bidi stream per accepted claim, fenced by the three-field
  envelope. Durable frames are never shed. Terminal authority lives only here.
* **WatchProgress** — bounded and LOSSY, owner-opened on its own connection. Saturating it
  can never block Control: a different RPC, a different queue, a different thread.

The conversation on every accepted stream is fixed (02): Claim -> ClaimAck -> snapshot
(Begin/Entry.../End) -> the owner's SnapshotAck -> dispatch opens. The ownership fence is
`(owner_epoch, control_generation, worker_boot_id)`, checked on every frame before any body
field is read. LAUNCH TIER (single-owner): the epoch is a provisioned constant — a claim
with a strictly higher epoch supersedes, an equal epoch from the same controller is a
reconnect, anything else refuses typed.

Threads, and why each exists: each accepted STREAM runs its conversation on the host's
thread and must never do slow work (a cancel has to arrive while an attempt runs); the
DEVICE LANE holds the single device lease and serializes attempts in arrival order — the
whole of worker-local arbitration; the REPORTER ticks the durable Report; the WATCHDOG owns
the deadline and the grace clock, because the deadline is worker-enforced and never merely
handed to author code.

This process never imports torch. `scripts/worker-live.py nocuda` proves that against the
RUNNING process's loaded modules and mapped libraries, not against the import graph.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hmac
import itertools
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cozy_runtime.internal import egress, hostfacts
from cozy_runtime.internal.config import RuntimeConfig, read_config
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.journal import Journal, JournalError
from cozy_runtime.internal.worker import child, grants, observe, triage
from cozy_runtime.internal.worker.attempts import (
    CAUSE,
    ORIGIN,
    STATUS,
    AttemptEngine,
    AttemptRecord,
    AttemptRefusal,
    UndurableOutcome,
)
from cozy_runtime.internal.worker.child import (
    GRACE_SECONDS,
    ExecutorGone,
    ExecutorSupervision,
    socket_path_refusal,
)
from cozy_runtime.internal.worker.control import (
    ControlHost,
    GrpcControlHost,
    WatchFanout,
)
from cozy_runtime.internal.worker.ledger import Ledger, read_vmrss
from cozy_runtime.internal.worker.plan import (
    DeclaredBinding,
    JobBinding,
    PlanChooser,
    PreparedGeneration,
)
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb

#: The lossy lane's per-watch depth. Oldest is shed first; a durable frame never enters it.
PROGRESS_QUEUE = 256

#: Report cadence. Report is DURABLE and carries the applied baseline, so the owner can
#: always see a directive it issued but the worker did not apply.
REPORT_SECONDS = 2.0

#: The RUN-ONCE RECYCLE exit (cr-009, v1 rc 75). A job worker that completed its one attempt
#: and had its terminal acknowledged asks to be RECLAIMED, and that is not a death.
RECYCLE_EXIT = 75

#: How often the rebuild queue LOOKS. A sampling cadence is the resolution of an
#: observation, never a bound on one (#434); it is the Report cadence because that is
#: already the rate at which this worker's non-attempt facts move.
REBUILD_SAMPLE_SECONDS = REPORT_SECONDS

#: The worker's accepted-attempt WINDOW behind `attempt_credits` (#441, #494f). ONE.
#:
#: It was 2, on the claim that attempt N+1's CPU phase overlaps attempt N's device phase.
#: That overlap does not exist: `device_lane()` is a single thread that admits, executes and
#: reconciles one attempt at a time, so a second admitted attempt could only WAIT — and it
#: waited in a state (`queued`) the wire cannot spell, invisible to the snapshot and to
#: `Report.active_attempts`. A worker holding work the owner cannot see breaks the
#: snapshot-is-complete law, so the false state is DELETED rather than surfaced: a dispatch
#: arriving while this worker already holds one is a typed REFUSED terminal the owner reads,
#: never a silent hold. cr-019 may re-open overlap as a resource-fit admission decision; it
#: does not re-open a hidden queue.
CREDIT_WINDOW = 1


@dataclass(frozen=True, slots=True)
class WorkerOptions:
    """One worker's settings, as a typed value rather than a parser's leftovers.

    The worker has two entrances — the long-lived `serve` process and the ephemeral local
    worker a one-shot `run` builds in memory (§8) — and only one of them has a command
    line. The owner is NOT here: who drives this worker is a transport
    (`worker/control.py`), not a setting.
    """

    root: Path
    worker_id: str = "wrk-local"
    instance_id: str = ""
    #: PROVENANCE, and every one of these is EMPTY when the launch did not supply it (#494g).
    #: The defaults used to name a real SDXL release and a `sha256:00…00` image, which rode
    #: real ClaimAcks and real BootFailures as if they had been measured. Absent is absent: an
    #: owner reading an empty `release_id` learns the truth, and an owner reading a plausible
    #: lie learns nothing it can act on.
    release_id: str = ""
    image_digest: str = ""
    git_commit: str = ""
    python: str = ""
    devices: str = "0"
    alloc_conf: str = "expandable_segments:True"
    threads: int = 4
    grace: float = GRACE_SECONDS
    host_budget: int = 8 << 30
    pinned_budget: int = 512 << 20
    no_warm: bool = False


@dataclass(slots=True)
class Deployment:
    """The ONE hosted deployment (launch enforces len(set) <= 1, #446). The id is the
    owner-minted routing + journal key; it is never invocation identity."""

    deployment_id: str
    endpoint_release_id: str
    plan_ids: tuple[str, ...]


@dataclass(slots=True)
class Fences:
    """The ownership fence and the directive fences, and who owns each (02 §0).

    `owner_epoch` is authority-granter-minted and durable; `control_generation` is
    WORKER-minted, incremented for every accepted stream; `worker_boot_id` is minted at
    worker boot and never reused. The executor generation is DEPLOYMENT truth
    (#454) and deliberately not here.
    """

    boot_id: str = ""
    owner_epoch: int = 0
    controller_id: str = ""
    #: TRUE once an (owner_epoch, controller_id) pair has been DURABLY recorded — this boot,
    #: or by a previous boot whose record recovery replayed (#494e). The claim fence reads
    #: this and not `generation`: authority survives a restart, so a pre-restart owner at a
    #: lower epoch must still be refused by a worker that has claimed nobody yet.
    owner_recorded: bool = False
    generation: int = 0  # worker-minted; 0 = never claimed
    snapshot_id: str = ""  # the snapshot the CURRENT stream must ack
    dispatch_open: bool = False
    revision: int = 0  # owner-owned
    applied_directive: bytes = b""
    posture: pb.Posture = pb.Posture.POSTURE_UNSPECIFIED
    intake: pb.IntakeState = pb.IntakeState.INTAKE_STATE_BOOTING
    wire_minor: int = 0
    drain_grace_ms: int = 0
    activity: list[pb.ActivityEvent] = field(default_factory=list)
    mode: str = ""  # the Directive `mode` oneof this worker APPLIED
    applied_set_digest: bytes = b""


#: Every envelope-bearing message type sets the same three fields; the stamp is central so
#: no constructor can forget one (the fence is checked before any body field is read).
def _stamped(message: Any, fences: Fences) -> Any:
    message.owner_epoch = fences.owner_epoch
    message.control_generation = fences.generation
    message.worker_boot_id = fences.boot_id
    return message


def desired_state(directive: pb.Directive) -> bytes:
    """The canonical bytes of a Directive's DESIRED STATE — the envelope excluded (#494a).

    A Directive carries two unrelated things: what the owner wants this worker to be
    (revision, posture, wire minor, drain grace, the mode's set or job), and the per-stream
    envelope that routes it (owner_epoch, control_generation, worker_boot_id). Only the first
    is the body §3's same-revision rule is about.

    Hashing the whole message made `control_generation` — which the WORKER mints afresh for
    every accepted stream, by design — part of the body, so the identical desired revision
    re-sent after an ordinary reconnect classified as `same_revision_changed_body` and took a
    permanent CONFIG_REFUSED fault. Same revision + same desired state is now the idempotent
    re-apply it always was; same revision + genuinely different desired state stays the
    protocol refusal, because that is the case the rule exists for.
    """
    body = pb.Directive()
    body.CopyFrom(directive)
    body.owner_epoch = 0
    body.control_generation = 0
    body.worker_boot_id = ""
    return documents.canonical_bytes(body)


class Worker:
    """The no-CUDA control parent — the machine's worker process (#455; formerly the
    `Supervisor`; "supervision" survives as the verb for its executor management)."""

    def __init__(self, config: RuntimeConfig, options: WorkerOptions, host: ControlHost) -> None:
        self.config = config
        self.options = options
        self.host = host
        self.root = Path(options.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fences = Fences(boot_id=f"boot-{uuid.uuid4().hex[:20]}")
        self.seq = itertools.count(1)  # WORKER-owned monotone counter
        self.progress_seq: dict[tuple[str, int], itertools.count[int]] = {}
        self.journal = Journal(self.root / "journal.log")
        self.ledger = Ledger(
            host_budget=options.host_budget,
            pinned_budget=options.pinned_budget,
            supervisor_pid=os.getpid(),
        )
        self.authorizer = grants.Authorizer()
        self.supervision = ExecutorSupervision(
            root=self.root,
            python=options.python or sys.executable,
            base_env=config.child_base_env,
            cozy_home=config.cozy_home,
        )
        self.chooser = PlanChooser(self.ledger)
        self.bindings: dict[str, DeclaredBinding] = {}
        self.generations: dict[str, PreparedGeneration] = {}
        self.jobs: dict[str, JobBinding] = {}
        self.deployment: Deployment | None = None
        self.boot = observe.BootRecord()
        self.monitor = observe.SilenceMonitor(self.boot.ring, busy=self.read_device_work)
        self.bundles = triage.BundleStore(self.root / "triage", self.journal)
        self.engine = AttemptEngine(
            journal=self.journal,
            ledger=self.ledger,
            chooser=self.chooser,
            supervision=self.supervision,
            authorizer=self.authorizer,
            bindings={},
            generations={},
            jobs={},
            spool_root=self.root / "spool",
            scratch_root=self.root.parent / "scratch",
            instance_id=options.instance_id,
            send=self.send,
            progress=self.emit_progress,
            bundles=self.bundles,
            monitor=self.monitor,
            posture=self.applied_posture,
            device_process=self.read_device_process,
        )
        self.engine.boot_id = self.fences.boot_id
        self.engine.deployment_id = lambda: (
            self.deployment.deployment_id if self.deployment is not None else ""
        )
        #: The durable lane's outbound queue, and it belongs to ONE accepted stream. A new
        #: claim installs a FRESH queue under `claim_lock`, so the superseded stream's sender
        #: keeps draining its own (now unread) queue and can never consume a frame authored
        #: for the live stream. A single process-wide queue made that theft routine: the dead
        #: generation's pump sits in `get(timeout=REPORT_SECONDS)`, and for up to that long
        #: after a reconnect it woke on the NEXT frame — a live AttemptTerminal — saw the
        #: generation had moved, and dropped it. The terminal was journaled, so it replayed on
        #: the following claim and the replay law held; live delivery on the stream that
        #: dispatched it did not.
        self.outbound: queue.Queue[pb.WorkerFrame | None] = queue.Queue()
        self.watches = WatchFanout(PROGRESS_QUEUE)
        self.work: queue.Queue[AttemptRecord | None] = queue.Queue()
        self.stop = threading.Event()
        self.settled = threading.Event()
        self.supervision.on_invalidate = self.capacity_dropped
        self.connected = threading.Event()
        self.exit_code: int = int(Exit.ok)
        self.boot_at = time.perf_counter()
        self.warm: list[dict[str, Any]] = []
        self.job_caps = pb.ResourceCaps()
        self.reclaim_on_terminal = False
        self.boot_fatal: pb.BootFailure | None = None
        #: The per-spawn bootstrap credential (#449/#463): the launcher that spawned this
        #: worker handed it through the environment — read by the ONE config authority,
        #: never here — and a Claim must present it as `proof`. The check is constant-time;
        #: the value never appears in a log. Where no launcher minted one (a hand-started
        #: serve), the fence is the socket's own filesystem authority and it is absent.
        self.bootstrap = config.bootstrap_credential
        #: serializes claim acceptance: one fence, whichever host thread observes the claim
        self.claim_lock = threading.Lock()

    # ------------------------------------------------------------------ plumbing

    def note(self, kind: str, step: str) -> None:
        """The NON-ATTEMPT activity lane: typed, sequence-bearing, on the DURABLE Report —
        never overloaded onto the lossy progress stream."""
        event = pb.ActivityEvent(
            seq=next(self.seq), kind=kind, step=step[:256], at_unix_ms=int(time.time() * 1000)
        )
        self.fences.activity.append(event)
        del self.fences.activity[:-64]
        print(f"[worker] {kind}: {step}", flush=True)

    def stamp(self, message: Any) -> Any:
        target = message
        if message.DESCRIPTOR.name in ("WorkerFrame", "OwnerFrame"):
            kind = message.WhichOneof("msg")
            if kind is None:
                return message
            target = getattr(message, kind)
        _stamped(target, self.fences)
        return message

    def send(self, frame: pb.WorkerFrame) -> None:
        """Durable. Queued, never shed; the inner message is stamped with the CURRENT fence
        so a frame authored before a re-claim still leaves under the live envelope, and an
        unacknowledged terminal is replayed after a reconnect.

        STAMP AND ENQUEUE ARE ONE STEP, under the same lock claim acceptance takes: the
        envelope a frame carries and the stream it is queued on are then the same generation
        by construction, with no window where a frame is stamped for one stream and handed to
        another's sender.
        """
        inner = getattr(frame, frame.WhichOneof("msg"))
        with self.claim_lock:
            self.stamp(inner)
            self.outbound.put(frame)

    def intake(self, state: pb.IntakeState) -> None:
        """Move the intake state and REPORT IT IMMEDIATELY — readiness is an EDGE."""
        if state == self.fences.intake:
            return
        self.fences.intake = state
        if state in (pb.IntakeState.INTAKE_STATE_READY, pb.IntakeState.INTAKE_STATE_ERROR):
            self.settled.set()
        else:
            self.settled.clear()
        if self.connected.is_set():
            self.send(pb.WorkerFrame(report=self.report()))

    def capacity_dropped(self, why: str) -> None:
        """The executor's state was invalidated: STOP ADVERTISING READY, right now."""
        if self.fences.intake != pb.IntakeState.INTAKE_STATE_READY:
            return
        self.note("readiness", f"capacity withdrawn: {why[:200]}")
        self.intake(pb.IntakeState.INTAKE_STATE_LOADING)

    def read_device_process(self) -> int:
        """NVML per-process device bytes for the EXECUTOR, read by the worker parent.
        -1 is UNREADABLE and never a zero."""
        pid = self.ledger.executor_pid
        if not pid:
            return -1
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return -1
        for line in out.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
                return int(parts[1]) * 1024 * 1024
        return -1

    def read_device_work(self) -> int:
        """How much work the EXECUTOR has done, read by the parent — THE meter."""
        return child.work(self.ledger.executor_pid)

    def applied_posture(self) -> dict[str, Any]:
        """The deployment state the worker was in when something happened (§3.6)."""
        return {
            "posture": pb.Posture.Name(self.fences.posture),
            "intake_state": pb.IntakeState.Name(self.fences.intake),
            "applied_revision": self.fences.revision,
            "owner_epoch": self.fences.owner_epoch,
            "control_generation": self.fences.generation,
            "applied_wire_minor": self.fences.wire_minor,
            "readiness_epoch": self.supervision.readiness_epoch,
            "executor_generation": self.supervision.incarnation,
            "deployment_id": self.deployment.deployment_id if self.deployment else "",
            "worker_id": self.options.worker_id,
            "release_id": self.options.release_id,
            "boot_steps": list(self.boot.steps),
        }

    def emit_progress(self, request_id: str, attempt: int, frame: dict[str, Any]) -> None:
        """The LOSSY lane. Bounded per watch, oldest-first shed, worker-owned sequence."""
        key = (request_id, attempt)
        position = frame.get("position")
        if isinstance(position, int):
            self.monitor.advance(f"{request_id}#{attempt}", position)
        if frame.get("kind") != "progress":
            self.engine.forward(request_id, attempt, frame)
        counter = self.progress_seq.setdefault(key, itertools.count(1))
        self.watches.publish(
            self.stamp(
                pb.AttemptProgress(
                    request_id=request_id,
                    attempt=attempt,
                    seq=next(counter),
                    content_type="application/x-cozy-event+json",
                    data=json.dumps(
                        {"type": frame.get("kind", "progress"), "payload": frame.get("value")}
                    ).encode()[:65536],
                    deployment_id=self.deployment.deployment_id if self.deployment else "",
                )
            )
        )

    # ------------------------------------------------------------------ the stream

    def serve_stream(
        self, inbound: Iterator[pb.OwnerFrame], send: Callable[[pb.WorkerFrame], None]
    ) -> None:
        """ONE control stream: Claim -> ClaimAck -> snapshot -> (SnapshotAck) -> frames.

        Runs on the host's thread. The first frame MUST be a Claim; acceptance fences any
        previous stream by minting the next control_generation, so the superseded stream's
        sender exits on its next frame and its watches end.
        """
        try:
            first = next(inbound)
        except (StopIteration, Exception):
            return
        claim = first.claim if first.WhichOneof("msg") == "claim" else None
        if claim is None:
            self.note("claim", "dropped a stream whose first frame is not a Claim")
            return
        generation, outbound = self.accept_claim(claim, send)
        if generation == 0:
            return  # refused; the rejection already went out and the stream ends
        if self.boot_fatal is not None:
            # The boot-fatal verdict INSTEAD of ClaimAck (01, #494c): the claim was
            # AUTHENTICATED and FENCED above — that is what makes the verdict deliverable to
            # this dialer — and then the worker says the one true thing it has to say. It
            # used to ack the claim, send a snapshot, and only then confess it never booted,
            # which is an owner told it holds a working worker and corrected afterwards.
            send(self.stamp(pb.WorkerFrame(boot_failure=self.boot_fatal)))
            self.note("boot", "boot failure delivered INSTEAD of ClaimAck; stopping")
            self.shutdown_from_stream()
            return
        self._send_claim_ack(claim, generation, send)

        sender = threading.Thread(
            target=self._pump_outbound,
            args=(generation, send, outbound),
            daemon=True,
            name=f"control-send-{generation}",
        )
        sender.start()
        try:
            for frame in inbound:
                if self.fences.generation != generation or self.stop.is_set():
                    self.note("fence", "a frame arrived on a superseded stream; ending it")
                    return
                self._dispatch_frame(frame, generation)
        except Exception as exc:
            self.note("session", f"the control stream ended: {type(exc).__name__}")
        finally:
            if self.fences.generation == generation:
                self.connected.clear()
                self.fences.dispatch_open = False

    def accept_claim(
        self, claim: pb.Claim, send: Callable[[pb.WorkerFrame], None]
    ) -> tuple[int, queue.Queue[pb.WorkerFrame | None]]:
        """The ownership fence (02 §4). AUTHENTICATE, fence, RECORD the authority durably,
        mint the generation. Returns `(generation, this stream's outbound queue)`, or
        `(0, …)` on refusal — the ClaimAck carrying the rejection has already gone out.

        Nothing is acknowledged here. The caller decides what the accepted claim is owed: a
        BootFailure when this worker never booted, a ClaimAck plus a snapshot otherwise.
        """

        def reject(reason: pb.ClaimRejection, why: str) -> tuple[int, queue.Queue[Any]]:
            self.note("claim", f"REFUSED ({pb.ClaimRejection.Name(reason)}): {why}")
            ack = pb.ClaimAck(accepted=False, rejection=reason, wire_minor=WIRE_MINOR)
            ack.owner_epoch = claim.owner_epoch
            ack.control_generation = self.fences.generation
            ack.worker_boot_id = self.fences.boot_id
            send(pb.WorkerFrame(claim_ack=ack))
            return 0, queue.Queue()

        if self.bootstrap and not hmac.compare_digest(
            claim.proof.decode(errors="replace"), self.bootstrap
        ):
            return reject(
                pb.ClaimRejection.CLAIM_REJECTION_UNAUTHENTICATED,
                "the bootstrap credential is absent or wrong",
            )
        if claim.worker_id and claim.worker_id != self.options.worker_id:
            return reject(
                pb.ClaimRejection.CLAIM_REJECTION_WORKER_ID_MISMATCH,
                f"this worker is {self.options.worker_id!r}",
            )
        with self.claim_lock:
            # `owner_recorded`, not `generation`: the high-water authority this worker
            # accepted is DURABLE (#494e), so a restart still refuses the owner it superseded
            # before the restart instead of welcoming it back as the first claim of a fresh
            # process.
            if self.fences.owner_recorded and claim.owner_epoch < self.fences.owner_epoch:
                return reject(
                    pb.ClaimRejection.CLAIM_REJECTION_STALE_OWNER_EPOCH,
                    f"epoch {claim.owner_epoch} is older than the accepted "
                    f"{self.fences.owner_epoch}",
                )
            if (
                self.fences.owner_recorded
                and claim.owner_epoch == self.fences.owner_epoch
                and claim.controller_id != self.fences.controller_id
            ):
                return reject(
                    pb.ClaimRejection.CLAIM_REJECTION_EPOCH_HELD,
                    "equal epoch, different controller: the epoch is held",
                )
            # RECORD BEFORE ACKNOWLEDGING. The ack is the moment this worker starts obeying
            # an owner, so the authority behind it has to outlive the process that obeys it —
            # exactly the order acceptance and terminals already follow. A journal that
            # cannot commit it makes this a boot-fatal verdict rather than an ack: accepting
            # an authority the worker will forget is how a superseded owner reclaims.
            if not self.record_ownership(claim):
                return 0, queue.Queue()
            superseded = self.fences.generation
            self.fences.owner_epoch = claim.owner_epoch
            self.fences.controller_id = claim.controller_id
            self.fences.owner_recorded = True
            self.fences.generation += 1
            self.fences.dispatch_open = False
            self.fences.wire_minor = min(WIRE_MINOR, claim.wire_minor)
            generation = self.fences.generation
            # A FRESH queue for this stream. The superseded sender keeps its own and drains
            # into a dead call; nothing it still holds was ever this stream's to deliver, and
            # everything the law owes regenerates — terminals replay from the journal after
            # SnapshotAck, reports re-tick.
            outbound: queue.Queue[pb.WorkerFrame | None] = queue.Queue()
            self.outbound = outbound
        if superseded:
            self.watches.end_generation(superseded)
            self.note(
                "claim",
                f"stream generation {superseded} fenced; generation {generation} claims "
                f"under epoch {claim.owner_epoch}",
            )
        return generation, outbound

    def record_ownership(self, claim: pb.Claim) -> bool:
        """Journal the authority this claim carries, BEFORE any ClaimAck (#494e).

        Called with `claim_lock` held. False means the record could not be committed, and the
        worker turns boot-fatal rather than obeying an owner it cannot remember obeying.
        """
        try:
            self.journal.append(
                "ownership",
                {
                    "worker_boot_id": self.fences.boot_id,
                    "instance_id": self.options.instance_id,
                    "owner_epoch": int(claim.owner_epoch),
                    "controller_id": claim.controller_id,
                },
            )
        except JournalError as exc:
            self.boot_fatal = self.boot_failure(
                pb.BootFailureReason.BOOT_FAILURE_REASON_DISK_SHAPE,
                f"the accepted ownership record could not be journaled ({exc.code}): a worker "
                "that cannot remember whose authority it accepted must not accept it",
            )
            self.exit_code = int(Exit.internal)
            self.note("claim", f"REFUSED: ownership is not durable ({exc.code})")
            return False
        return True

    def restore_ownership(self) -> None:
        """Replay the journalled ownership high-water mark at boot (#494e).

        The fence the owner relies on is "no epoch below the highest this worker ever
        accepted". Held only in memory it lasted exactly as long as the process, so a
        restart re-opened the worker to any epoch at all, including the stale owner a higher
        epoch had already superseded.
        """
        rows = self.journal.of_kind("ownership")
        if not rows:
            return
        best = max(rows, key=lambda record: int(record.body.get("owner_epoch", 0)))
        self.fences.owner_epoch = int(best.body.get("owner_epoch", 0))
        self.fences.controller_id = str(best.body.get("controller_id", ""))
        self.fences.owner_recorded = True
        self.note(
            "recovery",
            f"the durable ownership high-water mark is epoch {self.fences.owner_epoch} held "
            f"by {self.fences.controller_id or '(unnamed)'}; a lower epoch is refused",
        )

    def _send_claim_ack(
        self, claim: pb.Claim, generation: int, send: Callable[[pb.WorkerFrame], None]
    ) -> None:
        """ClaimAck + the recovery snapshot, for a claim this worker can actually serve."""
        ack = pb.ClaimAck(
            accepted=True,
            wire_minor=WIRE_MINOR,
            worker_id=self.options.worker_id,
            instance_id=self.options.instance_id,
            release_id=self.options.release_id,
            image_digest=self.options.image_digest,
            git_commit=self.options.git_commit,
            resources=self.resources(),
        )
        send(self.stamp(pb.WorkerFrame(claim_ack=ack)))
        self._send_snapshot(send)
        self.note(
            "session",
            f"claimed by {claim.controller_id or 'an unnamed owner'} at epoch "
            f"{claim.owner_epoch}, generation {generation} "
            f"(wire_minor effective {self.fences.wire_minor})",
        )

    def _send_snapshot(self, send: Callable[[pb.WorkerFrame], None]) -> None:
        """Begin -> one Entry per held/journaled attempt -> End. Dispatch stays closed
        until the owner's SnapshotAck names exactly this snapshot (02 §6)."""
        self.fences.snapshot_id = f"snp-{uuid.uuid4().hex[:16]}"
        entries = self.engine.snapshot_entries()
        begin = pb.SnapshotBegin(
            snapshot_id=self.fences.snapshot_id, journal_highwater=self.journal.appends
        )
        send(self.stamp(pb.WorkerFrame(snapshot_begin=begin)))
        for attempt in entries:
            entry = pb.SnapshotEntry(snapshot_id=self.fences.snapshot_id, attempt=attempt)
            send(self.stamp(pb.WorkerFrame(snapshot_entry=entry)))
        end = pb.SnapshotEnd(snapshot_id=self.fences.snapshot_id, entry_count=len(entries))
        send(self.stamp(pb.WorkerFrame(snapshot_end=end)))
        if entries:
            self.note(
                "recovery",
                f"snapshot {self.fences.snapshot_id} reports {len(entries)} attempt(s); "
                "the owner reconciles them before any dispatch",
            )

    def _pump_outbound(
        self,
        generation: int,
        send: Callable[[pb.WorkerFrame], None],
        outbound: queue.Queue[pb.WorkerFrame | None],
    ) -> None:
        """ONE stream's durable sender, draining ONE stream's queue.

        The queue is passed in rather than read off `self`: a superseded pump that keeps
        looping for a moment must drain the queue it was born with, never the live stream's.
        """
        while not self.stop.is_set() and self.fences.generation == generation:
            try:
                frame = outbound.get(timeout=REPORT_SECONDS)
            except queue.Empty:
                continue  # the generation check above is this thread's fence
            if frame is None:
                return
            try:
                send(frame)
            except Exception:
                return

    def fenced(self, message: Any) -> bool:
        """The three-field envelope, in its fixed order, BEFORE any body field (02 §0)."""
        if message.owner_epoch != self.fences.owner_epoch:
            self.note("fence", f"dropped: epoch {message.owner_epoch}")
            return True
        if message.control_generation != self.fences.generation:
            self.note("fence", f"dropped: superseded generation {message.control_generation}")
            return True
        if message.worker_boot_id != self.fences.boot_id:
            self.note("fence", f"dropped: wrong boot {message.worker_boot_id!r}")
            return True
        return False

    def _dispatch_frame(self, frame: pb.OwnerFrame, generation: int) -> None:
        kind = frame.WhichOneof("msg")
        if kind is None:
            return
        inner = getattr(frame, kind)
        if kind == "claim":
            self.note("claim", "PROTOCOL: a second Claim on an established stream; ignored")
            return
        if self.fenced(inner):
            return
        if kind == "snapshot_ack":
            if inner.snapshot_id != self.fences.snapshot_id:
                self.note("fence", f"SnapshotAck names {inner.snapshot_id!r}; not this snapshot")
                return
            self.fences.dispatch_open = True
            self.connected.set()
            self.note("session", f"snapshot {inner.snapshot_id} acknowledged; dispatch is open")
            for attempt in self.engine.pending_terminals():
                self.note(
                    "terminal",
                    f"replaying the journaled terminal for {attempt.request_id}#"
                    f"{attempt.attempt} in a fresh envelope ({attempt.terminal_id})",
                )
                self.send(pb.WorkerFrame(attempt_terminal=self.engine.wrap(attempt)))
            self.send(pb.WorkerFrame(report=self.report()))
            return
        if kind == "directive":
            try:
                self.apply_directive(inner)
            except (ExecutorGone, JournalError) as exc:
                self.note("directive", f"apply failed: {exc}")
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=f"revision {inner.revision}",
                        reason="directive_preparation_failed",
                        detail=f"{type(exc).__name__}: {exc}"[:1024],
                    )
                )
                self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
            return
        if not self.fences.dispatch_open:
            self.note("fence", f"dropped a {kind} before SnapshotAck: dispatch is closed")
            return
        if kind == "start_attempt":
            self.on_start(inner)
        elif kind == "cancel_attempt":
            self.note("cancel", self.engine.cancel(inner))
        elif kind == "terminal_ack":
            self.on_terminal_ack(inner)
        elif kind == "checkpoint_receipt":
            self.engine.absorb_receipt(inner)
            ack = pb.JobCheckpointAck(
                request_id=inner.request_id,
                attempt=inner.attempt,
                operation_key=inner.operation_key,
                receipt_id=inner.receipt_id,
                content_digest=inner.content_digest,
            )
            self.send(pb.WorkerFrame(checkpoint_ack=ack))

    def on_terminal_ack(self, acked: pb.TerminalAck) -> None:
        self.note("terminal", self.engine.close(acked))
        closed = self.engine.history.get((acked.request_id, acked.attempt))
        if self.reclaim_on_terminal and closed is not None and closed.kind == "job":
            self.note(
                "session",
                f"job attempt {acked.request_id}#{acked.attempt} is acknowledged and "
                f"reclaim_on_terminal is set: exiting {RECYCLE_EXIT} (recycle, not a death)",
            )
            self.exit_code = RECYCLE_EXIT
            self.shutdown_from_stream()

    def shutdown_from_stream(self) -> None:
        self.stop.set()
        self.outbound.put(None)
        self.work.put(None)
        self.watches.end_all()
        self.host.stop()

    def watch(self, opened: pb.ProgressOpen) -> Iterator[pb.AttemptProgress]:
        """One WatchProgress call: bound to the fenced control generation (#436)."""
        if self.fenced(opened):
            return iter(())
        return self.watches.open(opened.request_id, opened.control_generation).frames(self.watches)

    # ------------------------------------------------------------------ directive

    def apply_directive(self, directive: pb.Directive) -> None:
        """FULL REPLACE. A field absent means absent, never unchanged (§3).

        THE LAUNCH CLAMP RUNS FIRST (#446/#454): a deployment set longer than one refuses
        typed (FAULT_KIND_DEPLOYMENT_SET_UNSUPPORTED) with the directive UNAPPLIED —
        `applied_revision` does not advance, which is how the owner sees the refusal.
        """
        mode = directive.WhichOneof("mode") or ""
        if mode == "deployment_set":
            entries = list(directive.deployment_set.set.deployments)
            if len(entries) > 1:
                self.note(
                    "directive",
                    f"revision {directive.revision} REFUSED: {len(entries)} deployments; "
                    "launch enforces len(set) <= 1 and the directive is UNAPPLIED",
                )
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_DEPLOYMENT_SET_UNSUPPORTED,
                        subject=f"revision {directive.revision}",
                        reason="deployment_set_over_launch_clamp",
                        detail=f"the desired set names {len(entries)} deployments and this "
                        "worker hosts at most one until multi-deployment supervision lands",
                    )
                )
                return
        body = desired_state(directive)
        if directive.revision < self.fences.revision:
            self.note("directive", f"revision {directive.revision} ignored (older)")
            return
        if directive.revision == self.fences.revision:
            if body != self.fences.applied_directive:
                self.note(
                    "directive",
                    f"PROTOCOL ERROR: revision {directive.revision} arrived with a different "
                    "desired state; a changed body must carry a new revision",
                )
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        subject=f"revision {directive.revision}",
                        reason="same_revision_changed_body",
                        detail="a Directive body is a complete replacement; changing it "
                        "without a new revision is unrepresentable",
                    )
                )
            else:
                self.note("directive", f"revision {directive.revision} re-applied (idempotent)")
            return

        self.fences.revision = directive.revision
        self.fences.applied_directive = body
        self.fences.posture = directive.posture
        self.fences.wire_minor = directive.wire_minor
        self.fences.drain_grace_ms = directive.drain_grace_ms
        self.fences.mode = "serving" if mode == "deployment_set" else mode
        if mode == "job":
            self.apply_job_directive(directive.job)
            return
        if mode != "deployment_set":
            self.note("directive", "no mode: nothing to prepare")
            return

        ds = directive.deployment_set
        self.fences.applied_set_digest = ds.set_digest
        entries = list(ds.set.deployments)
        self.bindings = {}
        self.generations = {}
        self.jobs = {}
        self.engine.jobs = {}
        if not entries:
            self.deployment = None
            self.engine.bindings = {}
            self.engine.generations = {}
            self.note("directive", f"revision {directive.revision} applied: an EMPTY set")
            return
        entry = entries[0]
        self.deployment = Deployment(
            deployment_id=entry.deployment_id,
            endpoint_release_id=entry.endpoint_release_id,
            plan_ids=tuple(entry.entrypoint_binding_plan_ids),
        )
        for plan_id in entry.entrypoint_binding_plan_ids:
            record = self.config.cozy_home / "binding-plans" / f"{plan_id.split(':')[-1]}.json"
            if not record.is_file():
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=plan_id,
                        reason="unresolved_binding_plan",
                        detail=f"no local record for {plan_id}",
                    )
                )
                continue
            self.bindings[plan_id] = DeclaredBinding.read(json.loads(record.read_text()))
        self.engine.bindings = self.bindings
        hosts = sorted(
            {host for binding in self.bindings.values() for host in binding.egress_allowed_hosts}
        )
        self.engine.egress = egress.EgressPolicy(
            allowed_hosts=tuple(hosts),
            max_bytes=max((b.max_input_bytes for b in self.bindings.values()), default=64 << 20),
            allow_private=any(b.tenancy == "local" for b in self.bindings.values()),
        )
        self.note(
            "directive",
            f"revision {directive.revision} applied: deployment "
            f"{entry.deployment_id or '(unnamed)'} with {len(self.bindings)} binding plans, "
            f"posture={pb.Posture.Name(directive.posture)}",
        )
        self.prepare_executor()

    def apply_job_directive(self, job: pb.JobDirective) -> None:
        """The JOB lane's directive: resolve the descriptor, spawn a fresh executor, ready."""
        self.bindings = {}
        self.generations = {}
        self.engine.bindings = {}
        self.engine.generations = {}
        self.jobs = {}
        self.deployment = None
        plans = self.config.cozy_home / "job-plans"
        record = plans / f"{job.job_descriptor_id.split(':')[-1]}.json"
        if not record.is_file():
            self.engine.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                    subject=job.job_descriptor_id,
                    reason="unresolved_job_descriptor",
                    detail=f"no local record for {job.job_descriptor_id}",
                )
            )
            self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
            return
        self.jobs[job.job_descriptor_id] = JobBinding.read(json.loads(record.read_text()))
        self.engine.jobs = self.jobs
        self.job_caps = job.resource_caps
        self.reclaim_on_terminal = job.reclaim_on_terminal
        self.note(
            "directive",
            f"revision {self.fences.revision} applied in JOB mode: "
            f"{self.jobs[job.job_descriptor_id].job!r}, device floor {job.device_count}, "
            f"reclaim_on_terminal={job.reclaim_on_terminal}",
        )
        self.intake(pb.IntakeState.INTAKE_STATE_LOADING)
        executor = self.supervision.current
        if executor is None or not executor.alive():
            try:
                executor = self.supervision.spawn(imposed=self.imposed())
            except ExecutorGone as exc:
                self.note("boot", f"the job executor never dialled back: {exc}")
                self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
                return
            self.ledger.executor_pid = executor.pid
            self.note(
                "boot",
                f"job executor generation {executor.incarnation} pid {executor.pid} dialled "
                f"back in {(executor.ready_at - executor.spawned_at) * 1000:.0f} ms "
                f"(torch_loaded={executor.hello.get('torch_loaded')})",
            )
        self.intake(pb.IntakeState.INTAKE_STATE_READY)

    # ------------------------------------------------------------------ readiness

    def prepare_executor(self) -> None:
        """The ExecutorPreparer, at directive time: prepare BEFORE advertising ready."""
        if not self.bindings:
            return
        self.intake(pb.IntakeState.INTAKE_STATE_LOADING)
        executor = self.supervision.current
        if executor is None or not executor.alive():
            self.note("boot", "spawning the device executor (exec-first trampoline)")
            executor = self.supervision.spawn(imposed=self.imposed())
            self.ledger.executor_pid = executor.pid
            self.note(
                "boot",
                f"generation {executor.incarnation} pid {executor.pid} dialled back in "
                f"{(executor.ready_at - executor.spawned_at) * 1000:.0f} ms "
                f"(no_new_privs={executor.hello.get('no_new_privs')}, "
                f"oom_score_adj={executor.hello.get('oom_score_adj')}, "
                f"torch_loaded={executor.hello.get('torch_loaded')})",
            )
        by_construction: dict[str, list[DeclaredBinding]] = {}
        for binding in self.bindings.values():
            by_construction.setdefault(binding.construction_key(), []).append(binding)
        for group in by_construction.values():
            first = group[0]
            started = time.perf_counter()
            subject = f"prepare:{first.components[0] if first.components else first.entrypoint}"
            self.monitor.start(subject, "load")
            reply = executor.call(
                {
                    "cmd": "prepare",
                    "binding": {
                        "project": first.project,
                        "model_class": first.model_class,
                        "model_binding_path": first.model_binding_path,
                        "model_parameter_name": first.model_parameter_name,
                        "component": first.components[0] if first.components else "",
                        "components": list(first.components),
                        "snapshots": dict(first.snapshots),
                        "store": first.store,
                        "config": first.config,
                        "snapshot": first.reference_snapshot,
                        "release": first.release,
                        "variant": first.variant,
                        "custody": first.custody,
                        "strict_keys": first.strict_keys,
                        "probe_cache": str(self.config.cozy_home / "probe"),
                        "variants": [dataclasses.asdict(offer) for offer in first.offered],
                        "objective": first.objective,
                        "steps_basis": first.steps_basis,
                        "delivery_cache": str(self.config.cozy_home / "probe"),
                    },
                    "budgets": {
                        "vram": first.model_weight_bytes,
                        "pinned": first.pinned_bytes,
                        "resident": first.resident_budget_bytes,
                    },
                    "expect_construction_digest": first.expected_logical_model_digest,
                },
                timeout=None,
                on_progress=lambda frame: self.monitor.advance(
                    f"prepare:{frame.get('name', '')}", int(frame.get("position", 0) or 0)
                ),
            )
            took = (time.perf_counter() - started) * 1000
            if not reply.get("ok"):
                self.monitor.end(subject, str(reply.get("code")))
                self.boot.step(
                    "load",
                    "refused",
                    took,
                    component=first.components[0] if first.components else "",
                    code=str(reply.get("code")),
                )
                detail = str(reply.get("detail"))[:1024]
                shortfall = reply.get("shortfall") or {}
                if shortfall:
                    detail = (
                        f"{shortfall['resource']} shortfall at {shortfall['scope']}: "
                        f"needed {shortfall['needed_bytes']} B, "
                        f"{shortfall['available_bytes']} B free (measured) - {detail}"
                    )[:1024]
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=first.entrypoint_binding_plan_id,
                        reason=str(reply.get("code")),
                        detail=detail,
                    )
                )
                self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
                self.note(
                    "boot",
                    f"prepare REFUSED for {first.entrypoint_binding_plan_id[:20]}: "
                    f"{reply.get('code')} - "
                    f"peak allocator {reply.get('peak_allocator_bytes', 0)} B / ledger "
                    f"{reply.get('peak_ledger_vram_bytes', 0)} B was LIVE when it failed; "
                    f"after rollback the allocator holds "
                    f"{reply.get('allocator_bytes', -1)} B and the device has "
                    f"{reply.get('device_free_bytes', -1)} B free",
                )
                self.supervision.invalidate(f"prepare refused: {reply.get('code')}")
                return
            digest = reply["model_construction_digest"]
            delivery = reply.get("delivery") or {}
            resolved = PreparedGeneration(
                logical_model_digest=digest,
                constructed_generation_digest=str(reply["constructed_generation_digest"]),
                delivery_rung=str(delivery.get("rung", "")),
                delivery_variant=str(delivery.get("variant", "")),
                weight_bytes=int(reply["facts"]["filled_bytes"]),
            )
            for binding in group:
                if binding.expected_logical_model_digest and (
                    binding.expected_logical_model_digest != digest
                ):
                    self.note(
                        "boot",
                        f"{binding.entrypoint_binding_plan_id[:20]}… construction digest mismatch",
                    )
                    continue
                executor.ready_plans.add(binding.entrypoint_binding_plan_id)
                self.generations[binding.entrypoint_binding_plan_id] = resolved
            self.engine.bindings = self.bindings
            self.engine.generations = self.generations
            self.ledger.observe_construction(reply["facts"])
            self.monitor.end(subject, "filled")
            self.boot.step(
                "load",
                "filled",
                took,
                component=first.entrypoint,
                destinations=int(reply["facts"]["filled"]),
                bytes=int(reply["facts"]["filled_bytes"]),
                fill_ms=float(reply["facts"]["fill_ms"]),
                custody=str(reply["facts"].get("custody", "canonical")),
                ignored_extra_keys=", ".join(reply["facts"].get("ignored_extra_keys", ())),
                transcribed_markers=", ".join(reply["facts"].get("transcribed_markers", ())),
            )
            self.note(
                "boot",
                f"prepared {len(group)} binding plan(s) in {took:.0f} ms — "
                f"{reply['facts']['filled']} destinations, "
                f"{reply['facts']['filled_bytes'] / 2**30:.3f} GiB, fill leg "
                f"{reply['facts']['fill_ms'] / 1000:.3f} s, logical model {digest[:23]}… "
                f"generation {str(reply['constructed_generation_digest'])[:23]}… "
                f"resident={sorted(reply['facts'].get('resident', {}))} "
                f"parked={reply['facts'].get('parked', [])}",
            )
            if delivery:
                self.note(
                    "delivery",
                    f"{delivery['variant']!r} on rung {delivery['rung']} "
                    f"({delivery['delivery']}/{delivery['materialization'] or 'verbatim'}, "
                    f"{delivery['cost_class']} decode, {delivery['kernel']} kernel) chosen "
                    f"on the {delivery['objective']} axis"
                    + (f" — {delivery['confession']}" if delivery.get("confession") else ""),
                )
        self.run_warm_pass()
        self.intake(pb.IntakeState.INTAKE_STATE_READY)

    def run_warm_pass(self) -> None:
        """One attempted warm case per WARMABLE entrypoint, through ORDINARY dispatch."""
        if self.options.no_warm:
            return
        executor = self.supervision.current
        if executor is None:
            return
        for plan_id, binding in self.bindings.items():
            if plan_id not in executor.ready_plans:
                continue
            spool = self.root / "warm" / binding.entrypoint
            spool.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            try:
                reply = executor.call(
                    {
                        "cmd": "warm",
                        "request_id": f"warm-{binding.entrypoint}",
                        "entrypoint": binding.entrypoint,
                        # NO BOUND: warm is a BOOT STEP and boot is silence-bounded (§3.3).
                        "deadline_s": None,
                        "spool": str(spool),
                    },
                    timeout=None,
                )
                outcome: dict[str, Any] = {
                    "entrypoint": binding.entrypoint,
                    "outcome": str(reply.get("outcome", "failed")),
                    "reason": str(reply.get("reason", ""))[:200],
                    "boot_warmup_observed": bool(reply.get("boot_warmup_observed", False)),
                }
            except ExecutorGone as exc:
                outcome = {
                    "entrypoint": binding.entrypoint,
                    "outcome": "failed",
                    "reason": str(exc)[:200],
                    "boot_warmup_observed": False,
                }
            outcome["ms"] = round((time.perf_counter() - started) * 1000, 1)
            outcome["demand_evidence"] = False
            self.boot.step(
                "warm",
                str(outcome["outcome"]),
                float(outcome["ms"]),
                entrypoint=binding.entrypoint,
                benchmark_override=True,
                reason=str(outcome["reason"])[:200],
            )
            self.journal.append("warm", dict(outcome))
            self.warm.append(outcome)
            self.note(
                "warm",
                f"{binding.entrypoint}: {outcome['outcome']} ({outcome['ms']} ms) "
                f"{outcome['reason']}"[:200],
            )
            if outcome["outcome"] == "failed":
                self.engine.degraded[plan_id] = str(outcome["reason"])[:160]
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_DEGRADED,
                        subject=plan_id,
                        reason="serve_degrade",
                        detail=f"boot warm pass failed for {binding.entrypoint}: "
                        f"{outcome['reason']} - the worker still serves",
                    )
                )
        if self.warm and executor.alive():
            try:
                self.ledger.observe_probe(executor.call({"cmd": "probe"}, timeout=30.0))
            except ExecutorGone:
                return
            shift = self.ledger.rebaseline("boot warm pass")
            self.note("warm", f"the resident device baseline moved {shift:+d} B on the warm pass")

    def imposed(self) -> dict[str, str]:
        """The IMPOSE half of the seal. Every name here is in the reviewed allowlist."""
        return {
            "CUDA_VISIBLE_DEVICES": self.options.devices,
            "PYTORCH_CUDA_ALLOC_CONF": self.options.alloc_conf,
            "OMP_NUM_THREADS": str(self.options.threads),
        }

    # ------------------------------------------------------------------ dispatch

    def on_start(self, start: pb.StartAttempt) -> None:
        """STREAM THREAD: fence, record, enqueue. Nothing here waits, fetches or leases."""
        if self.fences.mode == "serving":
            wanted = self.deployment.deployment_id if self.deployment is not None else ""
            if start.deployment_id != wanted:
                self.note(
                    "fence",
                    f"unknown deployment {start.deployment_id!r} (this worker hosts {wanted!r})",
                )
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_UNKNOWN_DEPLOYMENT,
                        subject=start.deployment_id,
                        reason="unknown_deployment",
                        detail=f"the dispatch names deployment {start.deployment_id!r} and "
                        f"this worker hosts {wanted!r}",
                    )
                )
                self.refuse(
                    start,
                    "unknown_deployment",
                    f"no deployment {start.deployment_id!r} is hosted here",
                    CAUSE.CAUSE_CODE_PROTOCOL,
                )
                return
        if self.fences.posture == pb.Posture.POSTURE_DRAINING:
            self.note("intake", f"draining: refusing {start.request_id}")
            self.refuse(
                start,
                "draining",
                "the worker is draining and admits no new attempts",
                CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
            return
        try:
            attempt = self.engine.offer(start)
        except AttemptRefusal as exc:
            self.note("attempt", f"{start.request_id}#{start.attempt} refused: {exc.code}")
            self.refuse(start, exc.code, exc.detail, exc.cause, exc.origin)
            return
        if attempt.state in ("terminal", "closed"):
            self.note(
                "attempt",
                f"duplicate StartAttempt for {start.request_id}#{start.attempt}: replaying "
                f"the journaled terminal, executing NOTHING (executions={attempt.executions})",
            )
            self.send(pb.WorkerFrame(attempt_terminal=self.engine.wrap(attempt)))
            return
        if attempt.state == "queued":
            self.note(
                "attempt",
                "duplicate StartAttempt for the attempt currently crossing to the device "
                "lane: it is held, and its acceptance publishes when the lane admits it",
            )
            return
        if attempt.state in ("accepted", "running"):
            self.note(
                "attempt",
                "duplicate StartAttempt for a held attempt: re-accepting "
                "the SAME acceptance; nothing starts a second time",
            )
            self.send(pb.WorkerFrame(attempt_accepted=self.engine.accepted_message(attempt)))
            return
        # THE WINDOW IS ONE, AND OVER IT THE ANSWER IS A TERMINAL (#494f). The device lane is
        # a single serialized state machine, so a second held attempt could only wait — and
        # it would wait in a state the wire cannot spell, absent from the snapshot and from
        # `Report.active_attempts`. A refusal the owner can read and re-dispatch is strictly
        # more informative than a hold it cannot see.
        busy = [
            held
            for held in self.engine.live.values()
            if held is not attempt and held.state in ("queued", "accepted", "running")
        ]
        if busy:
            other = busy[0]
            self.note("attempt", f"{start.request_id}#{start.attempt} over the admission window")
            self.refuse(
                start,
                "no_admission_credit",
                f"this worker admits one attempt at a time and holds "
                f"{other.request_id}#{other.attempt} ({other.state}); the window is "
                f"{CREDIT_WINDOW} and `attempt_credits` reports it every tick",
                CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
            return
        self.engine.hold(attempt)
        self.work.put(attempt)

    def await_settled(self, attempt: AttemptRecord) -> None:
        """QUEUE behind a rebuild instead of refusing through one — for exactly as long as
        the rebuild is MAKING PROGRESS (#434: caller deadline or observed stall, never a
        wall-clock guess)."""
        if self.settled.is_set():
            return
        started = time.perf_counter()
        key = f"{attempt.request_id}#{attempt.attempt}"
        deadline_ms = int(attempt.spec.get("deadline_unix_ms", 0))
        self.note(
            "readiness",
            f"{key} reached the lane while the worker is rebuilding: waiting while the "
            "rebuild makes progress rather than refusing it",
        )
        still = 0
        position, burned = -1, -1
        why = ""
        while not (settled := self.settled.wait(REBUILD_SAMPLE_SECONDS)):
            if deadline_ms and time.time() * 1000 > deadline_ms:
                why = "the attempt's own deadline expired while it queued"
                break
            still += 1
            (position, before_position) = (self.boot.ring.admitted, position)
            (burned, before_burned) = (self.read_device_work(), burned)
            if position > before_position >= 0 or burned > before_burned >= 0:
                still = 0
            if still >= observe.GIVE_UP_AFTER:
                why = (
                    f"{still} consecutive observations found the rebuild at ring position "
                    f"{position} having burned {burned} ns: it is stalled, not slow"
                )
                break
        waited = (time.perf_counter() - started) * 1000
        self.note(
            "readiness",
            f"{key} waited {waited:.0f} ms; intake is "
            f"{pb.IntakeState.Name(self.fences.intake)}" + ("" if settled else f" ({why})"),
        )

    def refuse(
        self,
        start: pb.StartAttempt,
        code: str,
        detail: str,
        cause: pb.CauseCode,
        origin: pb.CauseOrigin = ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
    ) -> None:
        """A typed refusal is a TERMINAL: journaled before send, like any other."""
        held = self.engine.history.get((start.request_id, start.attempt))
        if held is not None and held.state in ("terminal", "closed"):
            held = None
        attempt = held or AttemptRecord(
            request_id=start.request_id,
            attempt=start.attempt,
            digest=start.invocation_digest,
            spec={},
        )
        try:
            terminal = self.engine.terminal(
                attempt,
                STATUS.TERMINAL_STATUS_REFUSED,
                cause,
                origin,
                f"{code}: {detail}",
            )
        except JournalError as exc:
            self.note("journal", f"cannot journal a refusal ({exc.code}); sending nothing")
            return
        self.engine.history[attempt.key()] = attempt
        self.send(pb.WorkerFrame(attempt_terminal=terminal))

    # ------------------------------------------------------------------ the device lane

    def device_lane(self) -> None:
        """ONE SERIALIZED PER-DEVICE STATE MACHINE: admit, execute, reconcile, rebuild."""
        while not self.stop.is_set():
            attempt = self.work.get()
            if attempt is None:
                return
            if self.admit(attempt):
                self.execute(attempt)
            # Rebuild on REFUSALS too (#458): a poisoned generation refuses every attempt,
            # so a rebuild gated on admission could never run again.
            self.rebuild_if_needed()

    def admit(self, attempt: AttemptRecord) -> bool:
        """FETCH -> HYDRATE -> CHOOSE -> ACCEPT, or a typed terminal. Never a silence."""
        key = f"{attempt.request_id}#{attempt.attempt}"
        self.await_settled(attempt)
        if attempt.canceling:
            self.note("cancel", f"{key} was cancelled before admission")
            self.terminate(attempt, self.engine.finish_cancel, attempt.canceling)
            return False
        deadline_ms = int(attempt.spec.get("deadline_unix_ms", 0))
        if deadline_ms and time.time() * 1000 > deadline_ms:
            self.note("cancel", f"{key} reached the lane past its deadline")
            self.terminate(attempt, self.engine.finish_cancel, "deadline")
            return False
        if self.fences.posture == pb.Posture.POSTURE_DRAINING:
            self.note("intake", f"draining: refusing queued {key}")
            self.terminate(
                attempt,
                self.engine.refuse_held,
                "draining",
                "the worker began draining while this attempt was queued",
                CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
            return False
        try:
            accepted = self.engine.prepare_and_accept(attempt)
        except AttemptRefusal as exc:
            self.note("attempt", f"{key} refused: {exc.code}")
            self.terminate(
                attempt, self.engine.refuse_held, exc.code, exc.detail, exc.cause, exc.origin
            )
            return False
        except JournalError as exc:
            self.note(
                "journal",
                f"{exc.code}: acknowledging nothing and executing nothing for {key}",
            )
            self.engine.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_LOCAL_SAFETY_REFUSAL,
                    subject=key,
                    reason=exc.code,
                    detail=exc.detail[:1024],
                )
            )
            self.intake(pb.IntakeState.INTAKE_STATE_ERROR)  # fault first, then the edge
            return False
        self.send(pb.WorkerFrame(attempt_accepted=accepted))
        return True

    def execute(self, attempt: AttemptRecord) -> None:
        """RUN -> TERMINAL -> RECONCILE. The ledger closes inside `terminal()`."""
        try:
            terminal = self.engine.execute(attempt)
        except JournalError as exc:
            self.note("journal", f"{exc.code}: only a last-resort terminal can be claimed")
            self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
            self.terminate(attempt, self.engine.last_resort, f"{exc.code}: {exc.detail}")
            return
        except Exception as exc:
            self.note("attempt", f"execution faulted: {type(exc).__name__}: {exc}")
            try:
                terminal = self.engine.abandon(
                    attempt, f"the worker faulted while running this attempt: {exc}"
                )
            except Exception as inner:
                self.note("attempt", f"even ABANDONED could not be built: {inner}")
                self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
                self.terminate(attempt, self.engine.last_resort, f"{type(inner).__name__}: {inner}")
                return
        self.send(pb.WorkerFrame(attempt_terminal=terminal))

    def terminate(
        self, attempt: AttemptRecord, build: Callable[..., pb.AttemptTerminal], *args: Any
    ) -> None:
        """Journal ONE terminal for a held attempt and send it.

        ACK-AFTER-JOURNAL IS THE LAW, INCLUDING HERE (#494b). An outcome that could not be
        journaled is never sent: the attempt stays HELD in a typed local-fault state and the
        worker goes INTAKE_STATE_ERROR with a fault naming it. Sending it anyway would hand
        the owner truth the worker cannot replay after a restart and cannot ack correctly if
        the owner tries — a durable claim about bytes nobody wrote down.
        """
        why = ""
        try:
            self.send(pb.WorkerFrame(attempt_terminal=build(attempt, *args)))
            return
        except UndurableOutcome as exc:
            self.note("journal", f"HOLDING {attempt.request_id}#{attempt.attempt}: {exc}")
            self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
            return
        except Exception as exc:
            why = f"{type(exc).__name__}: {exc}"
            self.note("attempt", f"the terminal could not be built ({why}); last resort")
        try:
            self.send(pb.WorkerFrame(attempt_terminal=self.engine.last_resort(attempt, why)))
        except UndurableOutcome as exc:
            self.note("journal", f"HOLDING {attempt.request_id}#{attempt.attempt}: {exc}")
            self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
        except Exception as inner:
            self.note("attempt", f"even the last-resort terminal failed: {inner}")
            self.intake(pb.IntakeState.INTAKE_STATE_ERROR)

    def rebuild_if_needed(self) -> None:
        """The queue starts behind the rebuild because BOTH live on this lane."""
        current = self.supervision.current
        if self.supervision.needs_rebuild:
            self.rebuild("readiness was invalidated during the attempt")
        elif current is None or not current.alive():
            self.rebuild("the executor is gone")
        elif current.poisoned:
            self.rebuild(f"poisoned: {current.poisoned}")

    def rebuild(self, why: str) -> None:
        """Replace the executor and RELOAD before reporting ready."""
        self.intake(pb.IntakeState.INTAKE_STATE_LOADING)
        started = time.perf_counter()
        self.note("recovery", f"rebuilding the executor: {why}")
        killed = self.supervision.sweep_orphans(
            exclude=self.supervision.current.pid if self.supervision.current else 0
        )
        if killed:
            self.note("recovery", f"fenced {len(killed)} orphan executor(s): {killed}")
        try:
            self.supervision.replace(imposed=self.imposed())
        except ExecutorGone as exc:
            self.note("recovery", f"respawn failed: {exc}")
            self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
            return
        self.prepare_executor()
        # Cleared LAST: an invalidation raised by another thread WHILE the rebuild ran is
        # about the executor this rebuild just replaced.
        self.supervision.needs_rebuild = False
        self.note(
            "recovery",
            f"ready again in {(time.perf_counter() - started) * 1000:.0f} ms at "
            f"generation {self.supervision.incarnation} / readiness epoch "
            f"{self.supervision.readiness_epoch}",
        )

    def watchdog(self) -> None:
        """The deadline and the grace clock — both the WORKER's, never author code's."""
        while not self.stop.wait(0.05):
            for attempt in list(self.engine.live.values()):
                if attempt.state != "running":
                    continue
                deadline_ms = int(attempt.spec.get("deadline_unix_ms", 0))
                if deadline_ms and time.time() * 1000 > deadline_ms and not attempt.canceling:
                    attempt.canceling = "deadline"
                    self.note("cancel", f"deadline expired for {attempt.request_id}")
                    if self.supervision.current is not None:
                        self.supervision.current.cooperative_cancel()
                    attempt.cancel_started = time.perf_counter()
                if attempt.canceling:
                    started = getattr(attempt, "cancel_started", 0.0) or time.perf_counter()
                    attempt.cancel_started = started
                    if time.perf_counter() - started > self.options.grace:
                        self.note(
                            "cancel",
                            f"grace of {self.options.grace:g} s expired: escalating to a "
                            "forceful kill (the handler never asked whether it was cancelled)",
                        )
                        facts = self.engine.escalate(attempt, self.options.grace)
                        self.note("cancel", f"forceful: {json.dumps(facts)}")
                        attempt.canceling = attempt.canceling or "client"
                        attempt.cancel_started = time.perf_counter() + 1e9

    def reporter(self) -> None:
        """The Report tick, and the liveness SAMPLE that rides it."""
        seen: dict[str, str] = {}
        while not self.stop.wait(REPORT_SECONDS):
            for subject in list(self.monitor.cursors):
                verdict = self.monitor.sample(subject)
                if verdict == "wedged" and seen.get(subject) != verdict:
                    self.note("liveness", f"{subject} is WEDGED by silence (no clock involved)")
                seen[subject] = verdict
            if self.connected.is_set():
                self.send(pb.WorkerFrame(report=self.report()))

    def report(self) -> pb.Report:
        """The Report tick. Serving truth is PER-DEPLOYMENT (#446); job mode reports its
        capacity credit. Before any directive it reports NEITHER — an honest 'no claim
        yet' rather than a fabricated zero."""
        report = pb.Report(
            applied_revision=self.fences.revision,
            intake_state=self.fences.intake,
            active_attempts=self.engine.active(),
            faults=self.engine.faults[-8:],
            activity=self.fences.activity[-8:],
            applied_wire_minor=self.fences.wire_minor,
        )
        if self.fences.mode == "job":
            report.job_capacity.CopyFrom(self.job_capacity())
        elif self.fences.mode == "serving":
            report.applied_set_digest = self.fences.applied_set_digest
            if self.deployment is not None:
                report.deployments.append(self.deployment_status())
        return report

    def deployment_status(self) -> pb.DeploymentStatus:
        """Per-deployment truth (#446): readiness epoch, executor generation, credits.

        The accelerator qualification is deliberately ABSENT: this torch-free process
        cannot measure capability facts, and absent IS the present-but-unqualified state
        (#454 — registration honesty is schema). The executor-measured qualification lands
        with cr-020/cr-021's lanes.
        """
        assert self.deployment is not None
        executor = self.supervision.current
        ready = sorted(executor.ready_plans) if executor is not None else []
        loadable = sorted(set(self.deployment.plan_ids) - set(ready))
        return pb.DeploymentStatus(
            deployment_id=self.deployment.deployment_id,
            intake_state=self.fences.intake,
            readiness_epoch=self.supervision.readiness_epoch,
            executor_generation=self.supervision.incarnation,
            attempt_credits=self.attempt_credits(),
            ready_entrypoint_binding_plan_ids=ready,
            loadable_entrypoint_binding_plan_ids=loadable,
            faults=[
                f
                for f in self.engine.faults[-8:]
                if f.kind
                in (
                    pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                    pb.FaultKind.FAULT_KIND_BINDING_DEGRADED,
                    pb.FaultKind.FAULT_KIND_EXECUTOR_POISONED,
                )
            ],
        )

    def attempt_credits(self) -> int:
        """Bounded admission credits at this readiness epoch (#441): the window less what
        this worker already holds. Zero while not READY — credits are a capacity claim."""
        if self.fences.intake != pb.IntakeState.INTAKE_STATE_READY:
            return 0
        held = sum(
            1
            for a in self.engine.live.values()
            if a.state in ("queued", "accepted", "running", "local_fault")
        )
        return max(0, CREDIT_WINDOW - held)

    def job_capacity(self) -> pb.JobCapacity:
        """How many JOB attempts this worker can take right now (cr-009)."""
        in_flight = sum(1 for a in self.engine.live.values() if a.job is not None)
        ready = bool(self.jobs) and self.fences.intake == pb.IntakeState.INTAKE_STATE_READY
        return pb.JobCapacity(
            jobs_in_flight=in_flight,
            jobs_available=max(0, (1 if ready else 0) - in_flight),
        )

    # ------------------------------------------------------------------ identity

    def resources(self) -> pb.WorkerResources:
        """Torch-free statics measured ONCE at claim time (#446): the worker NEVER
        fabricates capability facts — a discovered accelerator is present-but-unqualified
        until a deployment executor qualifies it."""
        facts = hostfacts.measure()
        memory_model = ""
        if facts.backend == "cuda":
            memory_model = "discrete"
        elif facts.backend == "mps":
            memory_model = "unified"
        return pb.WorkerResources(
            platform=facts.platform,
            backend=facts.backend,
            memory_model=memory_model,
            device_count=facts.gpu_count,
            device_name=facts.gpu_name,
            device_memory_total_bytes=facts.vram_total_bytes,
            driver_version=facts.driver_version,
            host_ram_total_bytes=facts.host_ram_total_bytes,
            vcpu_count=facts.vcpu_count,
            unreadable=list(facts.unreadable),
        )

    def disk_invoice(self) -> dict[str, int | str]:
        """The numbers a `DISK_SHAPE` claim must carry, read from the filesystem itself."""
        try:
            stat = os.statvfs(self.root)
        except OSError:
            return {"readable": "unreadable"}
        return {
            "readable": "measured",
            "total_bytes": stat.f_blocks * stat.f_frsize,
            "available_bytes": stat.f_bavail * stat.f_frsize,
            "free_inodes": stat.f_favail,
            "block_size": stat.f_frsize,
        }

    def boot_failure(self, reason: pb.BootFailureReason, detail: str) -> pb.BootFailure:
        """One boot verdict, classified ONCE, with its invoice attached. Delivered on the
        first claimed stream INSTEAD of ClaimAck (01) — the owner dials, so the verdict
        waits for it."""
        invoice = self.disk_invoice()
        self.boot.step("load", "boot_failure", 0.0, reason=pb.BootFailureReason.Name(reason))
        return pb.BootFailure(
            worker_id=self.options.worker_id,
            instance_id=self.options.instance_id,
            reason=reason,
            detail=f"{detail} | disk invoice: {json.dumps(invoice, sort_keys=True)}"[:1024],
            resources=self.resources(),
            release_id=self.options.release_id,
            image_digest=self.options.image_digest,
        )

    def run(self) -> int:
        """Serve until stopped. The worker outlives any one stream: an owner that drops
        the connection re-dials and re-claims the SAME boot; the in-memory host ending its
        one stream is the run ending."""
        try:
            self.journal.append(
                "session",
                {"worker_boot_id": self.fences.boot_id, "instance_id": self.options.instance_id},
            )
        except JournalError as exc:
            self.boot_fatal = self.boot_failure(
                pb.BootFailureReason.BOOT_FAILURE_REASON_DISK_SHAPE, exc.detail
            )
            self.exit_code = int(Exit.internal)
            self.note("boot", f"BOOT FAILURE (disk shape): {self.boot_fatal.detail}")
        if self.boot_fatal is None:
            unbindable = socket_path_refusal(self.supervision.socket_path)
            if unbindable:
                self.boot_fatal = self.boot_failure(
                    pb.BootFailureReason.BOOT_FAILURE_REASON_CONFIG_INVALID, unbindable
                )
                self.exit_code = int(Exit.structural)
                self.note("boot", f"BOOT FAILURE (config): {unbindable}")
        if self.boot_fatal is None:
            killed = self.supervision.sweep_orphans()
            if killed:
                self.note(
                    "recovery",
                    f"killed {len(killed)} orphan executor(s) from before this restart: {killed}",
                )
            # The accepted-authority high-water mark comes back BEFORE the first claim can
            # be fenced against it (#494e).
            self.restore_ownership()
            # Recovery replays the journal ONCE at boot; the snapshot on every later claim
            # reads the engine's own held state.
            recovered = self.engine.recover()
            if recovered:
                self.note(
                    "recovery",
                    f"the recovered journal holds {len(recovered)} attempt(s); they ride "
                    "the snapshot and the owner closes them before minting any next ordinal",
                )
            threading.Thread(target=self.device_lane, daemon=True, name="device-lane").start()
            threading.Thread(target=self.reporter, daemon=True, name="report").start()
            threading.Thread(target=self.watchdog, daemon=True, name="watchdog").start()
        self.host.serve(self)
        self.shutdown()
        return self.exit_code

    def shutdown(self) -> None:
        seam = (
            self.supervision.current.channel.accounting()
            if self.supervision.current is not None
            else {}
        )
        self.stop.set()
        self.work.put(None)
        self.outbound.put(None)
        self.watches.end_all()
        self.host.stop()
        self.supervision.close()
        self.journal.close()
        print(
            "[worker] summary "
            + json.dumps(
                {
                    "worker_boot_id": self.fences.boot_id,
                    "control_generation": self.fences.generation,
                    "executor_generations": self.supervision.spawns,
                    "readiness_epoch": self.supervision.readiness_epoch,
                    "executed": self.engine.executed,
                    "journal_appends": self.journal.appends,
                    "journal_fsync_ms": round(self.journal.fsync_ns / 1e6, 2),
                    "progress_sent": self.watches.sent,
                    "progress_shed": self.watches.shed,
                    "hub_calls_allowed": self.authorizer.allowed,
                    "hub_calls_refused": len(self.authorizer.refusals),
                    "seam": seam,
                    "ledger": self.ledger.rows(),
                    "warm": self.warm,
                    "rss_bytes": read_vmrss(os.getpid()),
                }
            ),
            flush=True,
        )


#: A provenance digest is `sha256:` plus 64 lowercase hex, or it is not one.
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

#: The loopback names a worker may serve plaintext, and without a bootstrap credential.
LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")


def instance_identity(root: Path) -> str:
    """THE instance identity for this journal root, minted once and reused (#494d).

    `instance_id` is the terminal-replay authorization key: recovery only replays journal
    rows whose instance matches. Minting a fresh `ins-<random>` on every boot therefore made
    a bare restart declare its own journal foreign — the rows were dropped, and until #494d
    they were dropped SILENTLY. The identity belongs to the durable root, so it lives beside
    the journal: the first boot writes it, every later boot reads it, and a launcher that
    genuinely rotates instances passes `--instance-id` and says so.
    """
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "instance-id"
    existing = ""
    with contextlib.suppress(OSError):
        existing = marker.read_text().strip()
    if existing:
        return existing
    minted = f"ins-{uuid.uuid4().hex[:16]}"
    marker.write_text(minted + "\n")
    return minted


def serve_worker(config: RuntimeConfig, argv: list[str] | None = None) -> int:
    """Parse the launch flags and run ONE worker hosting the protocol.

    Not an entrypoint: the config arrives from whoever already read it (one config per
    process, cl-004). `--listen` is the local binding this worker SERVES on — the owner
    dials it; the worker never dials anyone (#436).
    """
    parser = argparse.ArgumentParser(description="cozy-runtime worker (no CUDA)")
    parser.add_argument(
        "--listen", required=True, help="local binding, e.g. unix:/path/control.sock or 127.0.0.1:0"
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--worker-id", default="wrk-local")
    parser.add_argument("--instance-id", default="")
    parser.add_argument("--release-id", default="")
    parser.add_argument(
        "--image-digest",
        default="",
        help="the worker image's provenance digest, `sha256:<64 hex>`; absent stays absent",
    )
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--python", default="")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--alloc-conf", default="expandable_segments:True")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--grace", type=float, default=GRACE_SECONDS)
    parser.add_argument("--host-budget", type=int, default=8 << 30)
    parser.add_argument("--pinned-budget", type=int, default=512 << 20)
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument("--tls-cert", default="", help="PEM cert for a non-loopback listener")
    parser.add_argument("--tls-key", default="", help="PEM key for a non-loopback listener")
    args = parser.parse_args(argv)

    if args.image_digest and not _IMAGE_DIGEST.fullmatch(args.image_digest):
        print(
            f"[worker] --image-digest {args.image_digest!r} is not `sha256:<64 hex>`: a "
            "provenance digest is measured or it is absent, never approximated",
            file=sys.stderr,
            flush=True,
        )
        return int(Exit.structural)
    options = WorkerOptions(
        root=Path(args.root),
        worker_id=args.worker_id,
        instance_id=args.instance_id or instance_identity(Path(args.root)),
        release_id=args.release_id,
        image_digest=args.image_digest,
        git_commit=args.git_commit,
        python=args.python,
        devices=args.devices,
        alloc_conf=args.alloc_conf,
        threads=args.threads,
        grace=args.grace,
        host_budget=args.host_budget,
        pinned_budget=args.pinned_budget,
        no_warm=args.no_warm,
    )
    listen = args.listen
    if listen.startswith("unix:"):
        refusal = socket_path_refusal(listen[len("unix:") :])
        if refusal:
            print(f"[worker] {refusal}", file=sys.stderr, flush=True)
            return int(Exit.structural)
    tls_cert = Path(args.tls_cert) if args.tls_cert else None
    tls_key = Path(args.tls_key) if args.tls_key else None
    if (tls_cert is None) != (tls_key is None):
        print("[worker] --tls-cert and --tls-key come together or not at all", file=sys.stderr)
        return int(Exit.structural)
    host_part = listen.rsplit(":", 1)[0]
    remote = not listen.startswith("unix:") and host_part not in LOOPBACK
    if remote and tls_cert is None:
        # A worker never listens plaintext off-loopback: the remote leg is TLS + the
        # bearer owner token, or it does not exist (#445; cr-020 item 4 keeps loopback).
        print(
            f"[worker] refusing to bind {listen} without TLS: a non-loopback listener "
            "requires --tls-cert/--tls-key",
            file=sys.stderr,
            flush=True,
        )
        return int(Exit.structural)
    if remote and not config.bootstrap_credential:
        # THE OTHER HALF OF THE SAME REFUSAL (#493 item 10). One-way TLS authenticates the
        # WORKER to the dialer and says nothing about the dialer, and the Claim check is
        # conditional on a credential being configured — so a TLS listener with no credential
        # accepted whoever connected, and the first Claim owned the machine. Loopback and
        # unix sockets stay legal without one: there the fence is the socket's own filesystem
        # authority and the kernel's peer credentials, which is a real fence. Off-loopback
        # there is no such thing, and a worker with no way to tell its owner from a stranger
        # must not open the door at all.
        print(
            f"[worker] refusing to bind {listen} without a bootstrap credential: a "
            "non-loopback listener authenticates its owner (COZY_BOOTSTRAP_CREDENTIAL) or it "
            "does not serve — one-way TLS proves nothing about the dialer",
            file=sys.stderr,
            flush=True,
        )
        return int(Exit.structural)
    host = GrpcControlHost(
        listen, Path(args.root) / "control.addr", tls_cert=tls_cert, tls_key=tls_key
    )
    worker = Worker(config, options, host)

    def terminate(signum: int, frame: object) -> None:
        worker.stop.set()
        worker.outbound.put(None)
        worker.work.put(None)
        host.stop()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        return worker.run()
    except JournalError as exc:
        print(f"[worker] {exc}", file=sys.stderr, flush=True)
        return int(Exit.internal)


def main(argv: list[str] | None = None) -> int:
    """The PROCESS entrypoint: read the ONE config here, then serve."""
    return serve_worker(read_config(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
