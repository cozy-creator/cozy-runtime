"""The `cozy.worker.v1` WORKER: the machine's one control process, hosting `WorkerControl`.

The worker is the gRPC SERVER; the record-plane owner dials it (worker-protocol/01,
decisions #436/#454 — the 2026-08-25 re-landing). Two RPCs run at launch and their
durability attributes are the whole design:

* **Control** — one durable bidi stream per accepted claim, fenced by the three-field
  envelope. Durable frames are never shed. Terminal authority lives only here.
* **WatchProgress** — bounded and LOSSY, owner-opened on its own connection. Saturating it
  can never block Control: a different RPC, a different queue, a different thread.

The conversation on every accepted stream is fixed (02): Claim -> ClaimAck -> ONE
`WorkerSnapshot` -> the RecordOwner's `SnapshotAck` -> dispatch opens. The ownership fence is
`(record_owner_epoch, control_stream_generation, worker_boot_id)`, checked on every frame
before any body field is read. LAUNCH TIER (single-owner): the epoch is a provisioned
constant — a claim with a strictly higher epoch supersedes, an equal epoch from the same
RecordOwner is a reconnect, anything else refuses typed.

BEFORE the ownership fence comes the SCHEMA fence (#530-A1/#536): `Claim.wire_schema_digest`
is compared against this file set's own `SCHEMA_DIGEST` before ANY other body field of the
Claim is read, and the answering `ClaimAck` carries the worker's digest so the RecordOwner
can make the same check in the other direction. A mismatch — or an ABSENCE, which is what a
pre-rev-2 peer has to send — refuses `CLAIM_REJECTION_SCHEMA_DIGEST_MISMATCH` and the stream
closes, instead of a stale peer misparsing a renumbered field several frames later.

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
import base64
import contextlib
import dataclasses
import hashlib
import hmac
import itertools
import json
import os
import queue
import re
import signal
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cozy_runtime.internal import accel, child_env, config, egress, hostfacts, proctree
from cozy_runtime.internal.config import ConfigError, RuntimeConfig, read_config
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
    safe,
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
    PlanRefusal,
    PreparedModel,
    artifact_facts,
    held_refs,
)
from cozy_runtime.protocol import (
    SCHEMA_DIGEST,
    SCHEMA_DIGEST_RAW,
    WIRE_MINOR,
    WIRE_SCHEMA_REV,
    documents,
)
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

#: The worker's accepted-attempt WINDOW — the ONE shared acceptance window behind
#: `available_attempt_slots` (#472e/#494f). ONE.
#:
#: It was 2, on the claim that attempt N+1's CPU phase overlaps attempt N's device phase.
#: That overlap does not exist: `device_lane()` is a single thread that admits, executes and
#: reconciles one attempt at a time, so a second admitted attempt could only WAIT — and it
#: waited in a state (`queued`) the wire cannot spell, invisible to the snapshot and to the
#: held-attempt set. A worker holding work the RecordOwner cannot see breaks the
#: snapshot-is-complete law, so the false state is DELETED rather than surfaced: an offer
#: arriving while this worker already holds one is a JOURNALED REFUSED outcome the
#: RecordOwner reads, never a silent hold.
#:
#: rev-2 makes the SHAPE of the bound structural rather than advisory (§6): per-placement
#: `attempt_credits` are gone — N counters over ONE serialized device advertised N x the real
#: capacity — and the seats here are consumed by running attempts AND by outcomes awaiting
#: `AttemptOutcomeAck`, so a RecordOwner that stops acking starves its own admission instead
#: of growing this worker's held set.
ADMISSION_WINDOW = 1


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
    #: The defaults used to name a real SDXL release and a `sha256:00…00` artifact, which rode
    #: real ClaimAcks and real BootFailures as if they had been measured. Absent is absent: an
    #: owner reading an empty `release_id` learns the truth, and an owner reading a plausible
    #: lie learns nothing it can act on.
    release_id: str = ""
    artifact_digest: str = ""
    git_commit: str = ""
    python: str = ""
    devices: str = "0"
    grant_roots: tuple[str, ...] = ()
    """Every directory tree a `file://` DeliveryGrant may address on this machine (#506b).
    A launcher that knows its own root states it — cozy-creator passes its local root, a
    pod's provisioner passes the media subtree its co-resident server owns — and the
    Authorizer refuses a grant outside every one of them. Empty declares no fence."""
    project: Path | None = None
    """THIS MACHINE's copy of the endpoint release being served (#563f).

    A binding-plan record partitions into IDENTITY and RESOLUTION (#506a), and `project` is
    the RESOLUTION half: the path the machine that WROTE the record put the release's tree
    at. On that machine it is also this machine's path and the distinction never shows. On a
    RENTED POD it is a path that does not exist, and discovering an endpoint from it fails
    as `no_endpoint_toml` naming a directory on somebody else's disk — observed live.

    So a worker resolves against its own root and never the writer's. `--project` states it;
    absent, it is the process's working directory, which is what every launcher already sets
    (cozy-creator runs the worker in the generation's source, and a pod's provisioner runs it
    in the installed generation's source). The record's own `project` stays as recorded
    evidence of where the WRITER resolved it, and decides nothing here."""
    alloc_conf: str = "expandable_segments:True"
    threads: int = 4
    grace: float = GRACE_SECONDS
    host_budget: int = 8 << 30
    pinned_budget: int = 512 << 20
    no_warm: bool = False


@dataclass(slots=True)
class Placement:
    """The ONE hosted placement (launch enforces len(placements) <= 1, §2). The id is the
    RecordOwner-minted routing + journal key; it is never invocation identity.

    `spec_digest` is the placement's IDENTITY (§1): sha256 of the `PlacementSpec/1` document
    nested in the accepted set bytes. A single-placement replace is this SAME `placement_id`
    carrying a NEW spec digest, which is why routing survives an in-place release rollout.
    """

    placement_id: str
    spec_digest: bytes
    endpoint_release_id: str
    environment_spec_digest: bytes
    installed_environment_receipt_digest: bytes
    descriptor_digest: bytes
    plan_ids: tuple[str, ...]
    #: TWO INDEPENDENT AXES (§8, #473/#482), because one enum cannot say "staged on disk and
    #: offline" — the exact state an outgoing spec holds under fallback-retention.
    materialization: pb.MaterializationState = pb.MaterializationState.MATERIALIZATION_STATE_ABSENT
    serving: pb.ServingState = pb.ServingState.SERVING_STATE_OFFLINE
    #: #474/#485c: the predecessor kept for restore. EMPTY means replacement is PAUSED —
    #: fallback capability is a PRECONDITION of replacement, not a nicety.
    retained_fallback_spec_digest: bytes = b""


@dataclass(slots=True)
class OwnershipFence:
    """WHO may drive this worker, and on WHICH stream (02 §0/§4).

    Split out of the old single `Fences` bag (#484): ownership and accepted intent are two
    unrelated lifetimes — a reconnect mints a new `control_stream_generation` and touches no
    accepted revision at all, and #494a's reconnect defect was exactly that confusion made
    structural. `activity` lives in NEITHER of the two halves; it is the worker's own
    narration lane and now sits on the worker.

    `record_owner_epoch` is authority-granter-minted and durable; `control_stream_generation`
    is WORKER-minted, incremented for every accepted stream; `worker_boot_id` is minted at
    worker boot and never reused. The executor generation is PLACEMENT truth (#454) and
    deliberately not here.
    """

    worker_boot_id: str = ""
    record_owner_epoch: int = 0
    record_owner_id: str = ""
    #: TRUE once a (record_owner_epoch, record_owner_id) pair has been DURABLY recorded —
    #: this boot, or by a previous boot whose record recovery replayed (#494e). The claim
    #: fence reads this and not the generation: authority survives a restart, so a
    #: pre-restart owner at a lower epoch must still be refused by a worker that has claimed
    #: nobody yet.
    record_owner_recorded: bool = False
    control_stream_generation: int = 0  # worker-minted; 0 = never claimed
    snapshot_id: str = ""  # the snapshot the CURRENT stream must ack
    snapshot_digest: bytes = b""  # ...bound by BOTH; an ack naming either alone is not one
    dispatch_open: bool = False
    wire_minor: int = 0


@dataclass(slots=True)
class AcceptedDesiredState:
    """WHAT this worker was told to be, and the exact bytes it accepted (§4/§8).

    `accepted_desired_state_revision` advances on ACCEPTANCE — which is all it ever honestly
    meant — and `converged_revision` (computed, never stored here) advances only when the
    observed placement state satisfies it. `applied_revision` is retired as dishonest (#473):
    it advanced on acceptance while claiming to report reality.

    `placement_set_canonical_bytes` is the EXACT accepted bytes, journaled and re-quoted —
    never a re-serialization and never a structured projection (§4). The snapshot carries
    these bytes verbatim.
    """

    accepted_desired_state_revision: int = 0
    desired_state_body: bytes = b""  # the revision's own desired-state bytes (#494a)
    accepted_placement_set_digest: bytes = b""
    placement_set_canonical_bytes: bytes = b""
    posture: pb.Posture = pb.Posture.POSTURE_UNSPECIFIED
    drain_grace_ms: int = 0
    mode: str = ""  # the DesiredWorkerState `mode` oneof this worker APPLIED
    #: The revision whose placements are all materially STAGED and DISPATCHABLE. It may NEVER
    #: advance past a placement that is not — the field is a DERIVED fact, and a worker that
    #: advances it while observed state disagrees is in breach, not merely optimistic (§8).
    converged_revision: int = 0


#: Every envelope-bearing message type sets the same three fields; the stamp is central so
#: no constructor can forget one (the fence is checked before any body field is read).
def _stamped(message: Any, fence: OwnershipFence) -> Any:
    message.record_owner_epoch = fence.record_owner_epoch
    message.control_stream_generation = fence.control_stream_generation
    message.worker_boot_id = fence.worker_boot_id
    return message


def desired_state_body(desired: pb.DesiredWorkerState) -> bytes:
    """The canonical bytes of a DesiredWorkerState's DESIRED STATE — envelope excluded (#494a).

    A DesiredWorkerState carries two unrelated things: what the RecordOwner wants this worker
    to be (revision, posture, wire minor, drain grace, the mode's set or job), and the
    per-stream envelope that routes it (record_owner_epoch, control_stream_generation,
    worker_boot_id). Only the first is the body §3's same-revision rule is about.

    Hashing the whole message made `control_stream_generation` — which the WORKER mints
    afresh for every accepted stream, by design — part of the body, so the identical desired
    revision re-sent after an ordinary reconnect classified as `same_revision_changed_body`
    and took a permanent CONFIG_REFUSED fault. Same revision + same desired state is now the
    idempotent re-apply it always was; same revision + genuinely different desired state
    stays the protocol refusal, because that is the case the rule exists for.
    """
    body = pb.DesiredWorkerState()
    body.CopyFrom(desired)
    body.record_owner_epoch = 0
    body.control_stream_generation = 0
    body.worker_boot_id = ""
    return documents.canonical_bytes(body)


def _digest_field(spelled: Any) -> bytes:
    """A document spells a digest `sha256:<hex>`; the wire carries the 32 raw bytes. Absent
    is absent — an empty field is not a zero digest (#494g)."""
    return documents.raw(str(spelled)) if spelled else b""


def read_placement_set(desired_set: pb.DesiredPlacementSet) -> list[dict[str, Any]]:
    """§4's ORDER OF OPERATIONS, and it is not negotiable.

    RECOMPUTE `sha256(placement_set_canonical_bytes)` and compare BEFORE parsing a single
    field. The frozen wire "digest-addressed" the set by an ECHOED claim, which is a
    statement the sender makes about bytes it also sent; this is a fence the receiver
    computes. Only on a match is the document parsed, under unknown-field REFUSAL — at every
    depth, because a `PlacementSpec` with an unknown key is a `PlacementSpec/2` and there is
    no additive versioning inside a digest-fenced document.

    Raises `documents.DocumentError`; the caller turns that into the typed fault and leaves
    the previous set serving.
    """
    recomputed = documents.digest_of(desired_set.placement_set_canonical_bytes)
    if recomputed != desired_set.placement_set_digest:
        raise documents.DocumentError(
            "placement_set_digest_mismatch",
            f"the set claims {desired_set.placement_set_digest.hex()[:16]}… and its own "
            f"{len(desired_set.placement_set_canonical_bytes)} B hash to "
            f"{recomputed.hex()[:16]}… — nothing in it has been parsed",
        )
    document = documents.read(desired_set.placement_set_canonical_bytes, pb.PlacementSet)
    placements = document.get("placements", [])
    if not isinstance(placements, list):
        raise documents.DocumentError("wrong_type", "`placements` is a list of Placement")
    known_placement = {f.name for f in pb.Placement.DESCRIPTOR.fields}
    known_spec = {f.name for f in pb.PlacementSpec.DESCRIPTOR.fields}
    known_subject = {f.name for f in pb.ArtifactSubject.DESCRIPTOR.fields}
    for entry in placements:
        if not isinstance(entry, dict):
            raise documents.DocumentError("wrong_type", "a Placement is a JSON object")
        for key in entry:
            if key not in known_placement:
                raise documents.DocumentError("unknown_field", f"Placement: unknown key {key!r}")
        spec = entry.get("spec", {})
        if not isinstance(spec, dict):
            raise documents.DocumentError("wrong_type", "`spec` is a PlacementSpec object")
        for key in spec:
            if key not in known_spec:
                raise documents.DocumentError(
                    "unknown_field", f"PlacementSpec: unknown key {key!r} (a new key is /2)"
                )
        for subject in spec.get("binding_plans", []):
            if not isinstance(subject, dict):
                raise documents.DocumentError("wrong_type", "an ArtifactSubject is an object")
            for key in subject:
                if key not in known_subject:
                    raise documents.DocumentError(
                        "unknown_field", f"ArtifactSubject: unknown key {key!r}"
                    )
    return [entry for entry in placements if isinstance(entry, dict)]


HASH_ENTRY = re.compile(r"^sha256:[0-9a-f]{64}$")


class AccessTokenError(Exception):
    """A malformed COZY_ACCESS_TOKEN_SHA256 value, refused before anything binds."""


class AccessTokenSet:
    """The RENTED-POD claim credential (#495e/#560g): a set of sha256 hashes a Claim's
    `proof` is checked against by hashing it.

    The inversion is the whole design. A launcher that holds only hashes can provision a
    pod and can never authenticate to it — the renter minted the token and told nobody,
    so the hash the hub holds is not a credential and presenting it proves nothing. That
    is why this class only ever hashes what it was shown and never renders what it holds.

    Two value shapes, told apart by the value itself rather than by a flag:

      * `sha256:<hex>` entries (comma- or newline-separated) — the LITERAL set, frozen at
        launch. This is what a provider's create body carries into a pod's environment.
      * anything else — a PATH to a file of those same entries, re-stat'd on every claim.
        Rotation is only real if it CONVERGES (#560d), and an environment variable cannot
        rotate: a hash added after launch admits nobody and a hash removed after launch
        revokes nothing. A provisioner that intends to rotate delivers the path.

    Everything about the path form is FAIL-CLOSED, exactly as cozy-media's reader is: an
    unreadable file, a file holding no usable entry, or an mtime that moved BACKWARD
    leaves the loaded set in place and admits nobody until it reads clean again.
    """

    def __init__(self, entries: tuple[str, ...] = (), path: Path | None = None) -> None:
        self.path = path
        self.entries = entries
        self.size = -1
        self.stamp = -1.0
        self.lock = threading.Lock()
        self.unreadable = ""

    @classmethod
    def parse(cls, value: str) -> AccessTokenSet | None:
        """Build the set from the configured VALUE, or None when there is none."""
        raw = value.strip()
        if not raw:
            return None
        parts = [p.strip() for p in raw.replace(",", "\n").splitlines() if p.strip()]
        literal = [p for p in parts if p.startswith("sha256:")]
        if literal:
            bad = [p for p in literal if not HASH_ENTRY.match(p)]
            if bad or len(literal) != len(parts):
                raise AccessTokenError(
                    "COZY_ACCESS_TOKEN_SHA256 mixes hash entries with something that is not "
                    "one; every entry must be `sha256:<64 lowercase hex>`, or the whole "
                    "value must be a single path to a file of them"
                )
            return cls(entries=tuple(literal))
        if len(parts) != 1:
            raise AccessTokenError(
                "COZY_ACCESS_TOKEN_SHA256 holds several entries and none is a "
                "`sha256:<64 lowercase hex>` hash, so it is neither a hash set nor a path"
            )
        return cls(path=Path(parts[0]))

    def reload(self) -> None:
        """Re-read the hash file iff its (size, mtime) MOVED, refusing a move backward."""
        if self.path is None:
            return
        try:
            info = self.path.stat()
        except OSError as err:
            self.unreadable = f"{self.path} cannot be read: {err}"
            self.entries = ()
            return
        with self.lock:
            same = info.st_size == self.size and info.st_mtime == self.stamp
            older = self.stamp >= 0 and info.st_mtime < self.stamp
        if same:
            return
        if older:
            # The credential epoch is monotone. A file that went backward is a restore
            # or a clock, and either way it is not the rotation this worker last saw.
            self.unreadable = f"{self.path} has an mtime older than the loaded set"
            self.entries = ()
            return
        try:
            text = self.path.read_text()
        except OSError as err:
            self.unreadable = f"{self.path} cannot be read: {err}"
            self.entries = ()
            return
        found = tuple(
            line.strip()
            for line in text.replace(",", "\n").splitlines()
            if HASH_ENTRY.match(line.strip())
        )
        if not found:
            self.unreadable = f"{self.path} holds no `sha256:<64 hex>` entry"
            self.entries = ()
            return
        with self.lock:
            self.entries, self.size, self.stamp = found, info.st_size, info.st_mtime
        self.unreadable = ""

    def admits(self, presented: str) -> bool:
        """Hash what was presented and compare it into the live set, constant-time.

        One-directional on purpose: this answers whether a token is one of ours and can
        never produce one. The hub-side hash presented AS the proof therefore refuses —
        it is a hash, not a preimage, and `sha256("sha256:…")` is not in any set.
        """
        self.reload()
        with self.lock:
            entries = self.entries
        digest = "sha256:" + hashlib.sha256(presented.strip().encode()).hexdigest()
        admitted = False
        for entry in entries:
            if hmac.compare_digest(digest, entry):
                admitted = True
        return admitted

    def why_not(self) -> str:
        """The refusal's cause, naming the unreadable set without rendering any of it."""
        if self.unreadable:
            return (
                "this worker cannot read its own token-hash set, so it admits nobody: "
                + self.unreadable
            )
        return "the presented proof is not a token this worker was provisioned with"


class Worker:
    """The no-CUDA control parent — the machine's WORKER process (#455/#484; the name
    `Supervisor` is retired everywhere, and "supervision" survives only as the verb for its
    executor management)."""

    def __init__(self, config: RuntimeConfig, options: WorkerOptions, host: ControlHost) -> None:
        self.config = config
        self.options = options
        self.host = host
        self.root = Path(options.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fence = OwnershipFence(worker_boot_id=f"boot-{uuid.uuid4().hex[:20]}")
        self.accepted = AcceptedDesiredState()
        #: The non-attempt narration lane. In NEITHER fence half (#484): it is neither an
        #: authority fact nor accepted intent, it is what this worker has been doing.
        self.activity: list[pb.ActivityEvent] = []
        #: MACHINE lifecycle, pulled out of the placement enum (#482). BOOTING until the
        #: first accepted claim finds a worker that booted; FAILED is terminal for this boot.
        self.phase: pb.WorkerPhase = pb.WorkerPhase.WORKER_PHASE_BOOTING
        #: THE ONE WORKER-LEVEL ADMISSION FENCE (§6). It bumps whenever the MEANING of
        #: admission changes — executor respawn, window resize, phase change, cutover — and
        #: an `AttemptOffer` echoing anything else refuses DETERMINISTICALLY. Per-placement
        #: `attempt_credits` are deleted: N counters over ONE serialized device advertise
        #: N x the real capacity, which is an arithmetic defect, not a tuning problem.
        self.admission_generation = 1
        self.seq = itertools.count(1)  # WORKER-owned monotone counter
        self.progress_seq: dict[tuple[str, int], itertools.count[int]] = {}
        self.journal = Journal(self.root / "journal.log")
        self.ledger = Ledger(
            host_budget=options.host_budget,
            pinned_budget=options.pinned_budget,
            worker_pid=os.getpid(),
        )
        self.authorizer = grants.Authorizer(roots=tuple(options.grant_roots))
        self.supervision = ExecutorSupervision(
            root=self.root,
            python=options.python or sys.executable,
            base_env=config.child_base_env,
            cozy_home=config.cozy_home,
        )
        self.supervision.on_change = self.executor_changed
        # One accepted stream at a time may mutate control state. Claims take this through
        # their snapshot, so ownership cannot move in the middle of convergence and a frame
        # from the superseded stream cannot finish under its successor's generation.
        self.control_lock = threading.RLock()
        self.chooser = PlanChooser(self.ledger)
        self.bindings: dict[str, DeclaredBinding] = {}
        #: plan id -> why THIS binding could not be prepared (#572d). A faulted binding
        #: refuses its own entrypoints and denies its siblings nothing.
        self.failed_bindings: dict[str, str] = {}
        self.prepared_models: dict[str, PreparedModel] = {}
        self.jobs: dict[str, JobBinding] = {}
        self.placement: Placement | None = None
        #: Job mode has no placement, so it carries its own dispatchability edge (cr-009).
        self.job_ready = False
        #: An activation failure LATCHES for the accepted revision (§8): non-empty means the
        #: worker has stopped retrying and `converged_revision` stays behind on purpose.
        self.latched = ""
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
            prepared_models={},
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
        self.engine.boot_id = self.fence.worker_boot_id
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
        #: The OTHER launch mode (#560g): a rented pod, whose launcher was given the hash
        #: SET of a token the RENTER minted and that the launcher will never see. Config
        #: already refused a worker handed both, so at most one of these is live.
        self.access = AccessTokenSet.parse(config.access_token_hashes)
        #: The standing ACQUISITION authority this worker last accepted (§3). Echoed on
        #: every ObservedWorkerState so grant landing is observable with no ack lane.
        self.grant_id = ""
        self.grant_revision = 0
        #: serializes claim acceptance: one fence, whichever host thread observes the claim
        self.claim_lock = threading.Lock()

    # ------------------------------------------------------------------ plumbing

    def note(self, kind: str, step: str) -> None:
        """The NON-ATTEMPT activity lane: typed, sequence-bearing, on the DURABLE Report —
        never overloaded onto the lossy progress stream."""
        event = pb.ActivityEvent(
            seq=next(self.seq), kind=kind, step=step[:256], at_unix_ms=int(time.time() * 1000)
        )
        self.activity.append(event)
        del self.activity[:-64]
        print(f"[worker] {kind}: {step}", flush=True)

    def stamp(self, message: Any) -> Any:
        target = message
        if message.DESCRIPTOR.name in ("WorkerFrame", "RecordOwnerFrame"):
            kind = message.WhichOneof("msg")
            if kind is None:
                return message
            target = getattr(message, kind)
        _stamped(target, self.fence)
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

    def set_serving(self, state: pb.ServingState, why: str = "") -> None:
        """Move the placement's SERVING axis and REPORT IT IMMEDIATELY — dispatchability is
        an EDGE, and an ObservedWorkerState that arrives on the next tick is a stale one.

        `IntakeState` is retired with both of its uses (#482/#510b). What it conflated was
        machine lifecycle (now `worker_phase`), what is on disk (now the MATERIALIZATION
        axis) and what a placement will take (this one) — and a single enum could not say
        "staged on disk but offline", which is exactly the state an outgoing spec holds under
        fallback-retention.

        ADMISSION MEANING MOVES WITH IT, so `admission_generation` bumps here (§6): an offer
        minted against the old generation refuses deterministically rather than racing a
        placement that stopped being dispatchable while it was in flight.
        """
        if self.placement is None or state == self.placement.serving:
            return
        self.placement.serving = state
        self.bump_admission(why or f"serving -> {pb.ServingState.Name(state)}")
        self._settle()

    def set_job_ready(self, ready: bool) -> None:
        """Job mode has no placement, so its admission edge is its own (cr-009)."""
        if ready == self.job_ready:
            return
        self.job_ready = ready
        self.bump_admission(f"job_ready={ready}")
        self._settle()

    def _settle(self) -> None:
        """`settled` is the device lane's gate: has this worker STOPPED moving? A latched
        activation failure settles too — the lane must refuse through it, not wait forever."""
        if self.dispatchable() or self.latched:
            self.settled.set()
        else:
            self.settled.clear()
        if self.connected.is_set():
            self.send(pb.WorkerFrame(observed_state=self.observed_state()))

    def bump_admission(self, why: str) -> None:
        """The ONE worker-level admission fence moves (§6). Callers: executor respawn,
        window resize, phase change, cutover, and every serving-axis edge above."""
        self.admission_generation += 1
        self.note("admission", f"generation {self.admission_generation}: {why}"[:256])

    def dispatchable(self) -> bool:
        """Can this worker take an offer AT ALL right now? Serving asks the placement's
        axis; job mode asks whether its one build is ready. Neither answer is a SLOT count —
        `available_attempt_slots` is the other half, and #486c keeps them apart on purpose."""
        if self.accepted.mode == "job":
            return self.job_ready
        return (
            self.placement is not None
            and self.placement.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
        )

    def admission_state(self) -> pb.AdmissionState:
        """CLOSED is STRUCTURAL (#486c): before the snapshot barrier, while draining, while
        the placement is not dispatchable. OPEN with zero slots is TRANSIENT saturation, and
        a RecordOwner backs off differently for each — so the two never collapse into one
        counter here either."""
        if (
            not self.fence.dispatch_open
            or self.accepted.posture == pb.Posture.POSTURE_DRAINING
            or self.phase != pb.WorkerPhase.WORKER_PHASE_ONLINE
            or not self.dispatchable()
        ):
            return pb.AdmissionState.ADMISSION_STATE_CLOSED
        return pb.AdmissionState.ADMISSION_STATE_OPEN

    def available_attempt_slots(self) -> int:
        """Free seats in the ONE shared acceptance window, NOW.

        BOUNDEDNESS IS STRUCTURAL, NOT A PROMISE (§5): a seat is held by a running attempt
        AND by an outcome awaiting `AttemptOutcomeAck`, so a RecordOwner that stops acking
        runs itself out of admission instead of growing this worker's held set.
        """
        if self.admission_state() != pb.AdmissionState.ADMISSION_STATE_OPEN:
            return 0
        held = sum(1 for a in self.engine.history.values() if a.state in self.engine.HELD)
        return max(0, ADMISSION_WINDOW - held)

    def set_phase(self, phase: pb.WorkerPhase, why: str) -> None:
        if phase == self.phase:
            return
        self.phase = phase
        self.bump_admission(f"worker_phase -> {pb.WorkerPhase.Name(phase)}: {why}")
        self._settle()

    def latch(self, fault: pb.Fault) -> None:
        """An activation failure LATCHES for this desired revision (§8): the worker does not
        evict-and-retry forever, `converged_revision` stays behind, and the fault is typed on
        the placement. The previous spec keeps serving (fallback-retention, #474)."""
        self.latched = f"{pb.FaultKind.Name(fault.kind)}: {fault.reason}"[:200]
        self.engine.fault(fault)
        self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, "activation failed; LATCHED")
        self._settle()

    def fail_machine(self, why: str) -> None:
        """A MACHINE-scope failure: this worker can no longer honour journal-before-send.

        `WORKER_PHASE_FAILED` is where that fact lives now (#482) — machine lifecycle is out
        of the placement enum, because "the placement cannot serve" and "the worker cannot
        write" are different facts with different remedies, and one `INTAKE_STATE_ERROR`
        said both.
        """
        self.engine.fault(
            pb.Fault(
                kind=pb.FaultKind.FAULT_KIND_LOCAL_SAFETY_REFUSAL,
                subject=self.options.worker_id,
                reason=why[:1024],
                detail="the worker cannot durably record what it would claim, so it stops "
                "admitting rather than emitting an outcome it can never replay",
            )
        )
        self.set_phase(pb.WorkerPhase.WORKER_PHASE_FAILED, why)

    def capacity_dropped(self, why: str) -> None:
        """The executor's state was invalidated: STOP ADVERTISING DISPATCHABLE, right now."""
        if not self.dispatchable():
            return
        self.note("serving", f"capacity withdrawn: {why[:200]}")
        if self.accepted.mode == "job":
            self.set_job_ready(False)
        else:
            self.set_serving(pb.ServingState.SERVING_STATE_ACTIVATING)

    def executor_changed(self, executor: child.Executor | None) -> None:
        """Move the ledger at the same boundary that publishes executor ownership.

        A job generation has no construction report, so resetting only in the serving fill
        path lets it inherit the previous serving process's baseline and residency. The
        supervision callback covers every spawn and retirement before either can be
        observed through the session.
        """
        self.ledger.begin_generation(executor.pid if executor is not None else 0)

    def read_device_process(self) -> int:
        """NVML bytes for the current executor; -1 remains UNREADABLE/ABSENT here.

        The ledger's legacy scalar cannot express absence separately. Lifecycle reclaim uses
        AcceleratorOps' tri-state observation directly, where that distinction is authoritative.
        """
        executor = self.supervision.current
        if executor is None:
            return -1
        observed = accel.process_memory(executor.pid, accel.host_backend_family())
        return observed.bytes if observed.state == "present" else -1

    def read_device_work(self) -> int:
        """How much work the EXECUTOR has done, read by the parent — THE meter."""
        executor = self.supervision.current
        return child.work(executor.pid if executor is not None else 0)

    def applied_posture(self) -> dict[str, Any]:
        """The state the worker was in when something happened (§3.6). `readiness_epoch` is
        NOT here: it is DELETED, not renamed (#486b) — the serving axis says what the
        placement can serve and `admission_generation` says whether an offer is admissible,
        which is one fence instead of two to keep consistent."""
        return {
            "posture": pb.Posture.Name(self.accepted.posture),
            "worker_phase": pb.WorkerPhase.Name(self.phase),
            "materialization": pb.MaterializationState.Name(
                self.placement.materialization
                if self.placement
                else pb.MaterializationState.MATERIALIZATION_STATE_UNSPECIFIED
            ),
            "serving": pb.ServingState.Name(
                self.placement.serving
                if self.placement
                else pb.ServingState.SERVING_STATE_UNSPECIFIED
            ),
            "accepted_desired_state_revision": self.accepted.accepted_desired_state_revision,
            "converged_revision": self.converged_revision(),
            "record_owner_epoch": self.fence.record_owner_epoch,
            "control_stream_generation": self.fence.control_stream_generation,
            "applied_wire_minor": self.fence.wire_minor,
            "admission_generation": self.admission_generation,
            "executor_generation": self.supervision.incarnation,
            "placement_id": self.placement.placement_id if self.placement else "",
            "worker_id": self.options.worker_id,
            "worker_release_id": self.options.release_id,
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
                    attempt_ordinal=attempt,
                    seq=next(counter),
                    content_type="application/x-cozy-event+json",
                    data=json.dumps(
                        {"type": frame.get("kind", "progress"), "payload": frame.get("value")}
                    ).encode()[:65536],
                    placement_id=self.placement.placement_id if self.placement else "",
                )
            )
        )

    # ------------------------------------------------------------------ the stream

    def serve_stream(
        self, inbound: Iterator[pb.RecordOwnerFrame], send: Callable[[pb.WorkerFrame], None]
    ) -> None:
        """ONE control stream: Claim -> ClaimAck -> WorkerSnapshot -> (SnapshotAck) -> frames.

        Runs on the host's thread. The first frame MUST be a Claim; acceptance fences any
        previous stream by minting the next `control_stream_generation`, so the superseded
        stream's sender exits on its next frame and its watches end.
        """
        try:
            first = next(inbound)
        except (StopIteration, Exception):
            return
        claim = first.claim if first.WhichOneof("msg") == "claim" else None
        if claim is None:
            self.note("claim", "dropped a stream whose first frame is not a Claim")
            return
        # Acceptance and the snapshot are one control transition. In particular, the claim
        # cannot snapshot half of a desired-state replacement running on another stream.
        with self.control_lock:
            generation, outbound = self.accept_claim(claim, send)
            if generation == 0:
                return  # refused; the rejection already went out and the stream ends
            if self.boot_fatal is not None:
                # The boot-fatal verdict INSTEAD of ClaimAck (01, #494c): the claim was
                # AUTHENTICATED and FENCED above — that is what makes the verdict deliverable
                # to this dialer — and then the worker says the one true thing it has to say.
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
                if self.fence.control_stream_generation != generation or self.stop.is_set():
                    self.note("fence", "a frame arrived on a superseded stream; ending it")
                    return
                self._dispatch_frame(frame, generation)
        except Exception as exc:
            # THE MESSAGE, not the class name. A typed refusal carries the one sentence that
            # says what to do about it, and reducing it to `PlanRefusal` produced a log that
            # repeated forever and named nothing — the refusal had to be reproduced by hand
            # off the record to find out which key it was about.
            self.note("session", f"the control stream ended: {type(exc).__name__}: {exc}")
        finally:
            if self.fence.control_stream_generation == generation:
                self.connected.clear()
                self.fence.dispatch_open = False

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
            ack = pb.ClaimAck(
                accepted=False,
                rejection=reason,
                wire_minor=WIRE_MINOR,
                # A REFUSAL CARRIES THE DIGEST TOO. The peer has to be able to tell
                # "you are stale" from "I am stale", and it cannot do that from a
                # rejection code alone.
                wire_schema_digest=SCHEMA_DIGEST_RAW,
            )
            ack.record_owner_epoch = claim.record_owner_epoch
            ack.control_stream_generation = self.fence.control_stream_generation
            ack.worker_boot_id = self.fence.worker_boot_id
            send(pb.WorkerFrame(claim_ack=ack))
            return 0, queue.Queue()

        # THE SCHEMA FENCE, BEFORE ANY OTHER BODY FIELD (#530-A1/#536b). Not after the
        # credential and not after the worker id: those are checks whose MEANING depends on
        # the two ends agreeing what the fields are. A peer whose schema is not this one may
        # have written `proof` at a number this binding reads as something else, so reading
        # it first would be reading a field that does not exist.
        #
        # ABSENCE IS THE STALE SIGNAL. A pre-rev-2 RecordOwner cannot populate this field at
        # all, so an empty digest is not a peer that forgot — it is a peer that predates the
        # fence, and it refuses here rather than misparsing a renumbered field mid-stream.
        if claim.wire_schema_digest != SCHEMA_DIGEST_RAW:
            carried = claim.wire_schema_digest
            return reject(
                pb.ClaimRejection.CLAIM_REJECTION_SCHEMA_DIGEST_MISMATCH,
                (
                    "the claim carries NO wire_schema_digest, which is what a pre-rev-2 "
                    f"binding sends; this worker speaks schema rev {WIRE_SCHEMA_REV} "
                    f"{SCHEMA_DIGEST}"
                )
                if not carried
                else (
                    f"the claim speaks schema sha256:{carried.hex()} and this worker speaks "
                    f"rev {WIRE_SCHEMA_REV} {SCHEMA_DIGEST}; no later frame is worth parsing"
                ),
            )
        # THE CREDENTIAL CHECK, in whichever of the two launch modes this worker was
        # started in (#560g). A rented pod verifies BY HASH — it holds no copy of the
        # token, so neither does whoever provisioned it — and a self-spawned worker
        # compares against the credential its own launcher minted. Both are constant-time
        # and neither ever renders what it holds into a refusal.
        if self.access is not None:
            if not self.access.admits(claim.proof.decode(errors="replace")):
                return reject(
                    pb.ClaimRejection.CLAIM_REJECTION_UNAUTHENTICATED, self.access.why_not()
                )
        elif self.bootstrap and not hmac.compare_digest(
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
            # `record_owner_recorded`, not the generation: the high-water authority this worker
            # accepted is DURABLE (#494e), so a restart still refuses the owner it superseded
            # before the restart instead of welcoming it back as the first claim of a fresh
            # process.
            if (
                self.fence.record_owner_recorded
                and claim.record_owner_epoch < self.fence.record_owner_epoch
            ):
                return reject(
                    pb.ClaimRejection.CLAIM_REJECTION_STALE_RECORD_OWNER_EPOCH,
                    f"epoch {claim.record_owner_epoch} is older than the accepted "
                    f"{self.fence.record_owner_epoch}",
                )
            if (
                self.fence.record_owner_recorded
                and claim.record_owner_epoch == self.fence.record_owner_epoch
                and claim.record_owner_id != self.fence.record_owner_id
            ):
                return reject(
                    pb.ClaimRejection.CLAIM_REJECTION_EPOCH_HELD,
                    "equal epoch, different RecordOwner: the epoch is held",
                )
            # RECORD BEFORE ACKNOWLEDGING. The ack is the moment this worker starts obeying
            # an owner, so the authority behind it has to outlive the process that obeys it —
            # exactly the order acceptance and terminals already follow. A journal that
            # cannot commit it makes this a boot-fatal verdict rather than an ack: accepting
            # an authority the worker will forget is how a superseded owner reclaims.
            if not self.record_ownership(claim):
                return reject(
                    pb.ClaimRejection.CLAIM_REJECTION_UNDURABLE,
                    "the accepted ownership record could not be journaled, so this worker "
                    "cannot honour journal-before-send and must not accept the stream",
                )
            superseded = self.fence.control_stream_generation
            self.fence.record_owner_epoch = claim.record_owner_epoch
            self.fence.record_owner_id = claim.record_owner_id
            self.fence.record_owner_recorded = True
            self.fence.control_stream_generation += 1
            self.fence.dispatch_open = False
            self.fence.wire_minor = min(WIRE_MINOR, claim.wire_minor)
            generation = self.fence.control_stream_generation
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
                f"under epoch {claim.record_owner_epoch}",
            )
        return generation, outbound

    def record_ownership(self, claim: pb.Claim) -> bool:
        """Journal the authority this claim carries, BEFORE any ClaimAck (#494e).

        Called with `claim_lock` held. False means the record could not be committed, and the
        claim is REFUSED rather than accepted — a worker that cannot remember whose authority
        it accepted must not accept it.

        rev-2 gives that refusal its own name: `CLAIM_REJECTION_UNDURABLE` (#507d/#510d).
        The frozen wire could only route it to `BootFailure(DISK_SHAPE)`, which was
        representable and true but less direct — the failure is about this CLAIM's
        durability, not about a boot that never happened, and a RecordOwner reading
        DISK_SHAPE could not tell those apart.
        """
        try:
            self.journal.append(
                "ownership",
                {
                    "worker_boot_id": self.fence.worker_boot_id,
                    "instance_id": self.options.instance_id,
                    "record_owner_epoch": int(claim.record_owner_epoch),
                    "record_owner_id": claim.record_owner_id,
                },
            )
        except JournalError as exc:
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
        best = max(rows, key=lambda record: int(record.body.get("record_owner_epoch", 0)))
        self.fence.record_owner_epoch = int(best.body.get("record_owner_epoch", 0))
        self.fence.record_owner_id = str(best.body.get("record_owner_id", ""))
        self.fence.record_owner_recorded = True
        self.note(
            "recovery",
            f"the durable ownership high-water mark is epoch {self.fence.record_owner_epoch} held "
            f"by {self.fence.record_owner_id or '(unnamed)'}; a lower epoch is refused",
        )

    def _send_claim_ack(
        self, claim: pb.Claim, generation: int, send: Callable[[pb.WorkerFrame], None]
    ) -> None:
        """ClaimAck + the one recovery snapshot, for a claim this worker can actually serve.

        `worker_artifact_digest` is where SUBSTRATE identity rides and NOWHERE ELSE
        (#487/#489): it is evidence about a MACHINE, never identity of an ENVIRONMENT, which
        is why `EndpointEnvironmentSpec` has no substrate key at all.
        """
        ack = pb.ClaimAck(
            accepted=True,
            wire_minor=WIRE_MINOR,
            wire_schema_digest=SCHEMA_DIGEST_RAW,
            worker_id=self.options.worker_id,
            worker_instance_id=self.options.instance_id,
            worker_release_id=self.options.release_id,
            worker_artifact_digest=self.options.artifact_digest,
            git_commit=self.options.git_commit,
            resources=self.resources(),
        )
        send(self.stamp(pb.WorkerFrame(claim_ack=ack)))
        self._send_snapshot(send)
        self.note(
            "session",
            f"claimed by {claim.record_owner_id or 'an unnamed RecordOwner'} at epoch "
            f"{claim.record_owner_epoch}, generation {generation} "
            f"(wire schema rev {WIRE_SCHEMA_REV}, wire_minor effective "
            f"{self.fence.wire_minor})",
        )

    def snapshot_body(self) -> pb.WorkerSnapshotBody:
        """The whole of what this worker holds, in ONE document (§5).

        The three-message Begin/Entry/End form could only be bounded by a counted-entries
        check, and a count is a claim the sender makes about a stream the receiver is
        reassembling. A single canonical document is bounded by its own digest: a truncated
        or re-ordered snapshot cannot match, so there is no partial reconciliation to get
        wrong because there is no partial message.
        """
        body = pb.WorkerSnapshotBody(
            accepted_desired_state_revision=self.accepted.accepted_desired_state_revision,
            accepted_placement_set_digest=self.accepted.accepted_placement_set_digest,
            journal_highwater=self.journal.appends,
            worker_phase=self.phase,
            converged_revision=self.converged_revision(),
            admission_generation=self.admission_generation,
            admission_state=self.admission_state(),
            available_attempt_slots=self.available_attempt_slots(),
        )
        if self.placement is not None:
            body.placements.append(self.placement_status())
        body.held_attempts.extend(self.engine.snapshot_entries())
        return body

    def _send_snapshot(self, send: Callable[[pb.WorkerFrame], None]) -> None:
        """ONE `WorkerSnapshot`, digest-fenced. Dispatch stays CLOSED until the
        RecordOwner's `SnapshotAck` names exactly this `snapshot_id` AND this
        `snapshot_digest` — an ack naming a different one is NOT an ack (02 §6, §5)."""
        canonical_bytes, digest = documents.identity(self.snapshot_body())
        self.fence.snapshot_id = f"snp-{uuid.uuid4().hex[:16]}"
        self.fence.snapshot_digest = digest
        held = len(self.snapshot_body().held_attempts)
        send(
            self.stamp(
                pb.WorkerFrame(
                    snapshot=pb.WorkerSnapshot(
                        snapshot_id=self.fence.snapshot_id,
                        snapshot_digest=digest,
                        snapshot_canonical_bytes=canonical_bytes,
                        # THE EXACT ACCEPTED BYTES, as journaled — never a re-serialization
                        # and never a structured projection (§4). Its digest lives INSIDE
                        # the snapshot document, so the two cannot drift apart.
                        accepted_placement_set_canonical_bytes=(
                            self.accepted.placement_set_canonical_bytes
                        ),
                    )
                )
            )
        )
        self.note(
            "recovery",
            f"snapshot {self.fence.snapshot_id} ({len(canonical_bytes)} B, "
            f"{documents.spell(digest)[:23]}…) reports {held} held attempt(s); the "
            "RecordOwner reconciles them durably before any dispatch",
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
        while not self.stop.is_set() and self.fence.control_stream_generation == generation:
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
        if message.record_owner_epoch != self.fence.record_owner_epoch:
            self.note("fence", f"dropped: epoch {message.record_owner_epoch}")
            return True
        if message.control_stream_generation != self.fence.control_stream_generation:
            self.note(
                "fence",
                f"dropped: superseded generation {message.control_stream_generation}",
            )
            return True
        if message.worker_boot_id != self.fence.worker_boot_id:
            self.note("fence", f"dropped: wrong boot {message.worker_boot_id!r}")
            return True
        return False

    def _dispatch_frame(self, frame: pb.RecordOwnerFrame, generation: int) -> None:
        """Apply one frame while its stream still owns the worker.

        The stream generation is checked inside the same lock claims take. Trusting only the
        frame envelope left a check/use window in which a new claim could supersede this
        stream and the old handler could then mutate state and stamp replies as the new one.
        """
        with self.control_lock:
            if generation != self.fence.control_stream_generation:
                self.note("fence", f"dropped a frame from superseded stream {generation}")
                return
            self._dispatch_current_frame(frame)

    def _dispatch_current_frame(self, frame: pb.RecordOwnerFrame) -> None:
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
            # THE BARRIER IS BOUND BY BOTH (§5). An ack naming a different `snapshot_id` OR
            # a different `snapshot_digest` is NOT an ack: it would open dispatch against a
            # reconciliation of some other snapshot, which is the one thing the barrier
            # exists to prevent.
            if (
                inner.snapshot_id != self.fence.snapshot_id
                or inner.snapshot_digest != self.fence.snapshot_digest
            ):
                self.note(
                    "fence",
                    f"SnapshotAck names {inner.snapshot_id!r}/"
                    f"{inner.snapshot_digest.hex()[:12]} and this snapshot is "
                    f"{self.fence.snapshot_id!r}/{self.fence.snapshot_digest.hex()[:12]}: "
                    "not an ack, the barrier stays closed",
                )
                return
            self.fence.dispatch_open = True
            self.connected.set()
            self.bump_admission(f"snapshot {inner.snapshot_id} acknowledged; dispatch opens")
            self.note("session", f"snapshot {inner.snapshot_id} acknowledged; dispatch is open")
            for attempt in self.engine.pending_outcomes():
                self.note(
                    "outcome",
                    f"replaying the journaled outcome for {attempt.request_id}#"
                    f"{attempt.attempt} in a fresh envelope ({attempt.outcome_id})",
                )
                self.send(pb.WorkerFrame(attempt_outcome=self.engine.wrap(attempt)))
            self.send(pb.WorkerFrame(observed_state=self.observed_state()))
            return
        if kind == "desired_state":
            try:
                self._apply_desired_state(inner)
            except (ExecutorGone, JournalError) as exc:
                self.note("desired_state", f"apply failed: {exc}")
                self.latch(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=f"revision {inner.revision}",
                        reason="desired_state_preparation_failed",
                        detail=safe(f"{type(exc).__name__}: {exc}"),
                    )
                )
            return
        if kind == "artifact_grant_update":
            self.apply_artifact_grant(inner)
            return
        if not self.fence.dispatch_open:
            self.note("fence", f"dropped a {kind} before SnapshotAck: dispatch is closed")
            return
        if kind == "attempt_offer":
            self.on_offer(inner)
        elif kind == "cancel_attempt":
            self.note("cancel", self.engine.cancel(inner))
        elif kind == "outcome_ack":
            self.on_outcome_ack(inner)
        elif kind == "checkpoint_receipt":
            self.engine.absorb_receipt(inner)
            ack = pb.JobCheckpointAck(
                request_id=inner.request_id,
                attempt_ordinal=inner.attempt_ordinal,
                operation_key=inner.operation_key,
                receipt_id=inner.receipt_id,
                content_digest=inner.content_digest,
            )
            self.send(pb.WorkerFrame(checkpoint_ack=ack))

    def apply_artifact_grant(self, update: pb.ArtifactGrantUpdate) -> None:
        """Standing, refreshable ACQUISITION authority, on its OWN lane (§3, #480a).

        REFRESHING ACCESS NEVER CHANGES IDENTITY. This handler touches no revision, no set
        digest and no convergence fact — there is no field a refresh could substitute a
        placement identity into, because the identities live in the set document, which a
        grant cannot reach. `grant_revision` is monotonic: older is ignored, equal is
        idempotent.

        A grant MAY PRECEDE its set. Subjects not named by any desired spec are INERT, never
        a refusal — pre-authorizing an incoming release is the normal rollout order.

        WHAT THIS WORKER DOES WITH IT TODAY, said out loud rather than implied: it RECORDS
        the standing authority and reports it back on `applied_artifact_grant_id` /
        `applied_grant_revision`, so grant landing is observable without an ack lane. It
        FETCHES nothing, because this worker has no environment materializer — cr-023 owns
        that lane, and a worker that pretended to fetch would be inventing a materialization.
        """
        if update.grant_revision < self.grant_revision:
            self.note("grant", f"grant revision {update.grant_revision} ignored (older)")
            return
        self.grant_revision = update.grant_revision
        self.grant_id = update.grant.grant_id
        self.note(
            "grant",
            f"acquisition authority {update.grant.grant_id or '(unnamed)'} at revision "
            f"{update.grant_revision}: {len(update.grant.subjects)} authorized subject(s), "
            f"{len(update.grant.locations)} location(s) — recorded, and this worker fetches "
            "nothing until cr-023's materializer exists",
        )
        if self.connected.is_set():
            self.send(pb.WorkerFrame(observed_state=self.observed_state()))

    def on_outcome_ack(self, acked: pb.AttemptOutcomeAck) -> None:
        """A closed outcome FREES ITS ADMISSION SEAT — and that is a SLOT COUNT change, not
        a change in admission MEANING, so it does NOT bump `admission_generation` (§6).

        The distinction is load-bearing and it cost a live run to find. `admission_generation`
        moves when the meaning of an offer changes — executor respawn, window resize, phase
        change, cutover — and a RecordOwner dispatching against a stale one is refused
        deterministically. Freeing a seat changes neither: bumping here made EVERY
        steady-state dispatch refuse ADMISSION_GENERATION_STALE, because the ack that freed
        the seat invalidated the generation the RecordOwner was about to dispatch against.
        `available_attempt_slots` is the field that carries this, and it is recomputed on
        every tick.
        """
        self.note("outcome", self.engine.close(acked))
        if self.connected.is_set():
            self.send(pb.WorkerFrame(observed_state=self.observed_state()))
        closed = self.engine.history.get((acked.request_id, acked.attempt_ordinal))
        if self.reclaim_on_terminal and closed is not None and closed.kind == "job":
            self.note(
                "session",
                f"job attempt {acked.request_id}#{acked.attempt_ordinal} is acknowledged and "
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
        return self.watches.open(opened.request_id, opened.control_stream_generation).frames(
            self.watches
        )

    # ------------------------------------------------------------- the desired state

    def _apply_desired_state(self, desired: pb.DesiredWorkerState) -> None:
        """FULL REPLACE. A field absent means absent, never unchanged (§3).

        ORDER, and none of it is negotiable:

            recompute the set digest over the canonical bytes -> the launch clamp ->
            revision arithmetic -> JOURNAL the exact accepted bytes -> parse -> converge

        The BYTE FENCE runs before the clamp counts anything, because counting placements
        means parsing, and §4 forbids parsing a single field of a set whose bytes do not hash
        to what it claims. A mismatch is `FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH`, the
        desired state is UNAPPLIED, `accepted_desired_state_revision` does NOT advance, and
        the previous set keeps serving.
        """
        mode = desired.WhichOneof("mode") or ""
        placements: list[dict[str, Any]] = []
        if mode == "placement_set":
            try:
                placements = read_placement_set(desired.placement_set)
            except documents.DocumentError as exc:
                kind = (
                    pb.FaultKind.FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH
                    if exc.code == "placement_set_digest_mismatch"
                    else pb.FaultKind.FAULT_KIND_CONFIG_REFUSED
                )
                self.note(
                    "desired_state",
                    f"revision {desired.revision} REFUSED before any field was parsed: "
                    f"{exc.code} — UNAPPLIED, the previous set keeps serving",
                )
                self.engine.fault(
                    pb.Fault(
                        kind=kind,
                        subject=f"revision {desired.revision}",
                        reason=exc.code,
                        detail=safe(exc.detail),
                    )
                )
                return
            if len(placements) > 1:
                self.note(
                    "desired_state",
                    f"revision {desired.revision} REFUSED: {len(placements)} placements; "
                    "launch enforces len(placements) <= 1 and the desired state is UNAPPLIED",
                )
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_PLACEMENT_SET_UNSUPPORTED,
                        subject=f"revision {desired.revision}",
                        reason="placement_set_over_launch_clamp",
                        detail=f"the desired set names {len(placements)} placements and this "
                        "worker hosts at most one until cr-023's reconciler lands; #475 adds "
                        "that when the clamp lifts, multi-placement serving is CO-FITTING ONLY",
                    )
                )
                return
        body = desired_state_body(desired)
        if desired.revision < self.accepted.accepted_desired_state_revision:
            self.note("desired_state", f"revision {desired.revision} ignored (older)")
            return
        if desired.revision == self.accepted.accepted_desired_state_revision:
            if body != self.accepted.desired_state_body:
                self.note(
                    "desired_state",
                    f"PROTOCOL ERROR: revision {desired.revision} arrived with a different "
                    "desired state; a changed body must carry a new revision",
                )
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        subject=f"revision {desired.revision}",
                        reason="same_revision_changed_body",
                        detail="a DesiredWorkerState body is a complete replacement; changing "
                        "it without a new revision is unrepresentable",
                    )
                )
            else:
                self.note("desired_state", f"revision {desired.revision} re-applied (idempotent)")
            return

        self.accepted.accepted_desired_state_revision = desired.revision
        self.accepted.desired_state_body = body
        self.accepted.posture = desired.posture
        # DRAINING is machine lifecycle too: a worker under a draining posture is not a
        # placement that stopped serving, it is a machine on its way out.
        if self.phase != pb.WorkerPhase.WORKER_PHASE_FAILED:
            self.set_phase(
                pb.WorkerPhase.WORKER_PHASE_DRAINING
                if desired.posture == pb.Posture.POSTURE_DRAINING
                else pb.WorkerPhase.WORKER_PHASE_ONLINE,
                f"posture {pb.Posture.Name(desired.posture)}",
            )
        self.fence.wire_minor = desired.wire_minor
        self.accepted.drain_grace_ms = desired.drain_grace_ms
        self.accepted.mode = "serving" if mode == "placement_set" else mode
        self.latched = ""  # a NEW revision un-latches: the latch is per desired revision
        if mode == "job":
            self.apply_job_directive(desired.job)
            return
        if mode != "placement_set":
            self.note("desired_state", "no mode: nothing to converge to")
            executor = self.supervision.current
            if executor is not None:
                try:
                    self.supervision.retire_current(executor, "empty desired state")
                except ExecutorGone as exc:
                    self.latched = f"executor_reclaim_failed: {exc}"[:200]
                    self.note("recovery", self.latched)
            self.placement = None
            self.bindings = {}
            self.prepared_models = {}
            self.engine.bindings = {}
            self.engine.prepared_models = {}
            self.set_job_ready(False)
            return

        desired_set = desired.placement_set
        # JOURNAL THE EXACT ACCEPTED BYTES (§4) — never a re-serialization, never a
        # structured projection. Recovery, status attribution and the snapshot all quote
        # these, which is what makes "the set this worker accepted" a byte fact rather than
        # a reconstruction that is free to disagree with itself.
        self.journal.append(
            "placement_set",
            {
                "worker_boot_id": self.fence.worker_boot_id,
                "accepted_desired_state_revision": desired.revision,
                "placement_set_digest": documents.spell(desired_set.placement_set_digest),
                "placement_set_canonical_b64": base64.b64encode(
                    desired_set.placement_set_canonical_bytes
                ).decode(),
            },
        )
        self.accepted.accepted_placement_set_digest = desired_set.placement_set_digest
        self.accepted.placement_set_canonical_bytes = desired_set.placement_set_canonical_bytes

        previous = self.placement
        self.bindings = {}
        self.prepared_models = {}
        self.jobs = {}
        self.engine.jobs = {}
        if not placements:
            executor = self.supervision.current
            if executor is not None:
                try:
                    self.supervision.retire_current(executor, "accepted placement set is empty")
                except ExecutorGone as exc:
                    self.latched = f"executor_reclaim_failed: {exc}"[:200]
                    self.note("recovery", self.latched)
                    self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, self.latched)
                    self._settle()
                    return
            self.placement = None
            self.engine.bindings = {}
            self.engine.prepared_models = {}
            self.accepted.converged_revision = desired.revision
            self.bump_admission("the accepted set is EMPTY; nothing is dispatchable")
            self.note("desired_state", f"revision {desired.revision} accepted: an EMPTY set")
            return
        entry = placements[0]
        placement_id = str(entry.get("placement_id", ""))
        spec = dict(entry.get("spec", {}))
        subjects = [dict(row) for row in spec.get("binding_plans", [])]
        plan_ids = tuple(
            str(row.get("subject_id", "")) for row in subjects if row.get("subject_id")
        )
        self.placement = Placement(
            placement_id=placement_id,
            # THE PLACEMENT'S IDENTITY (§1): the digest of the nested PlacementSpec document,
            # reconstructed with its own format tag (#510g) from the bytes already held.
            spec_digest=documents.embedded_document_digest(spec, "cozy.worker.v1.PlacementSpec"),
            endpoint_release_id=str(spec.get("endpoint_release_id", "")),
            environment_spec_digest=_digest_field(spec.get("environment_spec_digest")),
            installed_environment_receipt_digest=_digest_field(
                spec.get("installed_environment_receipt_digest")
            ),
            descriptor_digest=_digest_field(spec.get("descriptor_digest")),
            plan_ids=plan_ids,
            # #474/#485c: the predecessor is the rollback pin. A REPLACEMENT (same
            # placement_id, new spec digest) retains the outgoing spec as the pin; a fresh
            # placement has none, and an empty pin means replacement is PAUSED — which is
            # why the pin is set from the PREVIOUS spec and never invented.
            retained_fallback_spec_digest=(
                previous.spec_digest
                if previous is not None and previous.placement_id == placement_id
                else b""
            ),
        )
        self.bump_admission(
            f"placement {self.placement.placement_id or '(unnamed)'} moved to spec "
            f"{documents.spell(self.placement.spec_digest)[:23]}…"
        )
        self.converge_placement(desired.revision)

    def converge_placement(self, revision: int) -> None:
        """Materialize what this worker CAN materialize, then activate — or refuse to lie.

        THE MATERIALIZATION AXIS HAS ONE HONEST SUBJECT HERE: the placement's binding-plan
        records, which are the only artifacts this worker stages today. All resolve => STAGED.
        Some missing => ABSENT with a typed fault, and the previous spec keeps serving
        (fallback-retention, #474).

        AND ONE HONEST REFUSAL. `MATERIALIZATION_STATE_STAGED` ASSERTS that the measured
        `InstalledEnvironmentReceipt` equalled `installed_environment_receipt_digest`; the
        proto says it is unspeakable otherwise. This worker has no environment materializer
        at all — cr-023 owns that lane — so when a spec NAMES a receipt digest, the worker
        cannot say STAGED and does not: it reports MATERIALIZING with a typed fault, holds
        `converged_revision` behind, and leaves the previous spec serving. That is a door,
        not a defect, and it is named rather than papered over with a fabricated STAGED.
        """
        assert self.placement is not None
        placement = self.placement
        if placement.installed_environment_receipt_digest:
            placement.materialization = pb.MaterializationState.MATERIALIZATION_STATE_MATERIALIZING
            self.latch(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                    subject=placement.placement_id,
                    reason="environment_materializer_unbuilt",
                    detail="the spec names installed_environment_receipt_digest "
                    f"{documents.spell(placement.installed_environment_receipt_digest)[:23]}… "
                    "and this worker cannot MEASURE an installed environment: STAGED asserts "
                    "the receipt matched and is unspeakable otherwise, so materialization "
                    "stays MATERIALIZING and converged_revision stays behind (cr-023 owns "
                    "the materializer)",
                )
            )
            return
        missing: list[str] = []
        for plan_id in placement.plan_ids:
            record = self.config.cozy_home / "binding-plans" / f"{plan_id.split(':')[-1]}.json"
            if not record.is_file():
                missing.append(plan_id)
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=plan_id,
                        reason="unresolved_binding_plan",
                        detail=f"no local record for {plan_id}",
                    )
                )
                continue
            # THE REF RESOLVES HERE, AGAINST THIS MACHINE'S OWN INDEX (#567e) — the same
            # rule `project` already follows (#563f), applied to `store`, which is its
            # twin. The record names `cozy/minimax-h3@se-001`; what that ref means on disk
            # is a question only the machine about to serve it can answer, and it is
            # answered before the placement can call itself STAGED. An absent artifact is a
            # typed BINDING_UNAVAILABLE naming what this index does hold, not a fill-time
            # surprise after a card is already rented and warm.
            try:
                declared = DeclaredBinding.read(json.loads(record.read_text()))
                self.bindings[plan_id] = declared.resolve(
                    artifact_facts(self.config.cozy_home, declared.artifact_ref)
                    if declared.artifact_ref
                    else None,
                    held=held_refs(self.config.cozy_home),
                )
            except PlanRefusal as refusal:
                missing.append(plan_id)
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=plan_id,
                        reason=refusal.code,
                        detail=safe(refusal.detail),
                    )
                )
                continue
        placement.materialization = (
            pb.MaterializationState.MATERIALIZATION_STATE_ABSENT
            if missing
            else pb.MaterializationState.MATERIALIZATION_STATE_STAGED
        )
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
            "desired_state",
            f"revision {revision} accepted: placement "
            f"{placement.placement_id or '(unnamed)'} at spec "
            f"{documents.spell(placement.spec_digest)[:23]}… with {len(self.bindings)} of "
            f"{len(placement.plan_ids)} binding plan(s) resolved, materialization="
            f"{pb.MaterializationState.Name(placement.materialization)}, "
            f"posture={pb.Posture.Name(self.accepted.posture)}",
        )
        self.prepare_executor()

    def apply_job_directive(self, job: pb.JobDirective) -> None:
        """The JOB lane: resolve the descriptor, spawn a fresh executor, become dispatchable.

        Job mode carries no placement, so its dispatchability edge is `job_ready` and its
        capacity is `JobCapacity` — the placement axes are unspeakable here and stay unset
        rather than being filled with a plausible OFFLINE.
        """
        self.bindings = {}
        self.prepared_models = {}
        self.engine.bindings = {}
        self.engine.prepared_models = {}
        self.jobs = {}
        self.placement = None
        self.set_job_ready(False)
        previous = self.supervision.current
        if previous is not None:
            try:
                self.supervision.retire_current(
                    previous, "entering job mode retires the serving generation"
                )
            except ExecutorGone as exc:
                self.note("boot", f"the serving executor could not be reclaimed: {exc}")
                self.latched = "job_executor_reclaim_failed"
                self._settle()
                return
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
            self.latched = "unresolved_job_descriptor"
            return
        self.jobs[job.job_descriptor_id] = JobBinding.read(json.loads(record.read_text()))
        self.engine.jobs = self.jobs
        self.job_caps = job.resource_caps
        self.reclaim_on_terminal = job.reclaim_on_terminal
        self.note(
            "desired_state",
            f"revision {self.accepted.accepted_desired_state_revision} accepted in JOB mode: "
            f"{self.jobs[job.job_descriptor_id].job!r}, device floor {job.device_count}, "
            f"reclaim_on_terminal={job.reclaim_on_terminal}",
        )
        try:
            executor = self.supervision.spawn(imposed=self.imposed())
        except ExecutorGone as exc:
            self.note("boot", f"the job executor never dialled back: {exc}")
            self.latched = "job_executor_absent"
            self._settle()
            return
        executor.purpose = "job"
        self.note(
            "boot",
            f"job executor generation {executor.incarnation} pid {executor.pid} dialled "
            f"back in {(executor.ready_at - executor.spawned_at) * 1000:.0f} ms "
            f"(torch_loaded={executor.hello.get('torch_loaded')})",
        )
        self.set_job_ready(True)

    # ------------------------------------------------------------- activation

    def prepare_executor(self) -> None:
        """ACTIVATION: prepare the executor BEFORE the placement becomes DISPATCHABLE.

        This is the SERVING axis moving OFFLINE -> ACTIVATING -> DISPATCHABLE. Serving never
        accepts-then-cold-loads, which is the whole reason the axis exists separately from
        materialization: bytes on disk and a process that will take work are two facts, and
        one enum could not hold both.
        """
        if not self.bindings:
            executor = self.supervision.current
            if executor is not None:
                try:
                    self.supervision.retire_current(executor, "placement has no bindings")
                except ExecutorGone as exc:
                    self.latched = f"executor_reclaim_failed: {exc}"[:200]
                    self.note("recovery", self.latched)
                    self._settle()
            self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, "no bindings")
            return
        # A fresh activation re-earns every verdict; a stale fault must not outlive the
        # prepare that produced it.
        self.failed_bindings.clear()
        self.set_serving(pb.ServingState.SERVING_STATE_ACTIVATING)
        by_construction: dict[str, list[DeclaredBinding]] = {}
        for binding in self.bindings.values():
            by_construction.setdefault(binding.construction_key(), []).append(binding)
        if len(by_construction) != 1:
            detail = (
                f"the launch worker received {len(by_construction)} distinct model "
                "constructions, but one executor can own exactly one prepared generation"
            )
            for binding in self.bindings.values():
                self.failed_bindings[binding.entrypoint_binding_plan_id] = detail
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=binding.entrypoint_binding_plan_id,
                        reason="multiple_constructions_unsupported",
                        detail=detail,
                    )
                )
            executor = self.supervision.current
            if executor is not None:
                try:
                    self.supervision.retire_current(executor, detail)
                except ExecutorGone as exc:
                    self.note("recovery", f"multi-construction reclaim failed: {exc}")
            self.latched = "multiple_constructions_unsupported"
            self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, self.latched)
            self._settle()
            return

        construction_key, group = next(iter(by_construction.items()))
        binding_set = frozenset(binding.entrypoint_binding_plan_id for binding in group)
        executor = self.supervision.current
        if (
            executor is not None
            and executor.alive()
            and not executor.poisoned
            and executor.construction_key == construction_key
            and executor.prepared_binding_set == binding_set
            and executor.prepared_model is not None
        ):
            executor.ready_plans.clear()
            for binding in group:
                if binding.expected_logical_model_digest and (
                    binding.expected_logical_model_digest
                    != executor.prepared_model.logical_model_digest
                ):
                    self.failed_bindings[binding.entrypoint_binding_plan_id] = (
                        "logical model digest mismatch"
                    )
                    continue
                executor.ready_plans.add(binding.entrypoint_binding_plan_id)
                self.prepared_models[binding.entrypoint_binding_plan_id] = executor.prepared_model
            self.engine.bindings = self.bindings
            self.engine.prepared_models = self.prepared_models
            self.note(
                "boot",
                f"reused generation {executor.incarnation} pid {executor.pid} for the same "
                f"construction ({len(executor.ready_plans)} binding plan(s))",
            )
            if executor.ready_plans:
                self.set_serving(pb.ServingState.SERVING_STATE_DISPATCHABLE)
                self.accepted.converged_revision = self.accepted.accepted_desired_state_revision
            else:
                self.latched = "logical model digest mismatch"
                self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, self.latched)
            return

        try:
            if executor is None:
                self.note("boot", "spawning the device executor (exec-first trampoline)")
                executor = self.supervision.spawn(imposed=self.imposed())
            elif (
                executor.alive()
                and not executor.poisoned
                and executor.purpose == "fresh"
                and not executor.ready_plans
                and executor.prepared_model is None
            ):
                self.note(
                    "boot",
                    f"preparing fresh generation {executor.incarnation} pid {executor.pid}",
                )
            else:
                self.note("boot", "replacing the device executor for a new construction")
                executor = self.supervision.replace(
                    executor,
                    imposed=self.imposed(),
                    why="desired construction changed",
                )
            self.note(
                "boot",
                f"generation {executor.incarnation} pid {executor.pid} dialled back in "
                f"{(executor.ready_at - executor.spawned_at) * 1000:.0f} ms "
                f"(no_new_privs={executor.hello.get('no_new_privs')}, "
                f"oom_score_adj={executor.hello.get('oom_score_adj')}, "
                f"torch_loaded={executor.hello.get('torch_loaded')})",
            )
        except ExecutorGone as exc:
            self.latched = f"executor_activation_failed: {exc}"[:200]
            self.note("boot", self.latched)
            self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, "executor absent")
            self._settle()
            return

        for group in by_construction.values():
            first = group[0]
            started = time.perf_counter()
            subject = f"prepare:{first.components[0] if first.components else first.entrypoint}"
            self.monitor.start(subject, "load")
            try:
                reply = executor.call(
                    {
                        "cmd": "prepare",
                        "binding": {
                            # THIS machine's root, not the record writer's (#563f). See
                            # WorkerOptions.project: the record's `project` is the resolution of
                            # whoever wrote it, and on a rented pod that path is not here.
                            "project": str(self.options.project or Path.cwd()),
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
                            # THE STAMP PLANE'S LAST HOP (#567d). The index row carries what the
                            # hub verified at publish; the executor hands it to `Artifact`, and
                            # a model class's keywords join it there. Dropping it here is what
                            # made `artifact.stamps` unconditionally `{}` and refused every
                            # stamped class on every checkpoint.
                            "stamps": {k: list(v) for k, v in first.stamps.items()},
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
            except ExecutorGone as exc:
                self.monitor.end(subject, "executor_gone")
                detail = f"executor disappeared during prepare: {exc}"[:1024]
                for member in group:
                    self.failed_bindings[member.entrypoint_binding_plan_id] = detail
                    self.engine.fault(
                        pb.Fault(
                            kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                            subject=member.entrypoint_binding_plan_id,
                            reason="executor_gone_during_prepare",
                            detail=safe(detail),
                        )
                    )
                try:
                    self.supervision.retire_current(executor, detail)
                except ExecutorGone as reclaim:
                    self.note("recovery", f"prepare-loss reclaim failed: {reclaim}")
                continue
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
                        # SANITIZED, like every other detail that crosses this wire (#574b).
                        # The document profile is printable ASCII; refusal prose in this
                        # codebase is written with em dashes, and one of them in a Fault
                        # made EVERY later claim ack raise `non_ascii_field` — the worker
                        # became unclaimable over a refusal it could not spell.
                        detail=safe(detail),
                    )
                )
                # AN ACTIVATION FAILURE LATCHES for this desired revision (§8): the worker
                # does not evict-and-retry forever, `converged_revision` stays behind, and
                # the fault is typed on the placement.
                # PREPARE IS PER-BINDING (#572d). This used to latch the whole placement
                # and `return`, so ONE binding's failure denied every sibling the card —
                # on H3 a hidden, unbuildable ref2va binding took the working T2VA one
                # down with it, and the working one never got its turn to try.
                #
                # It is #548d's refusal-vs-fault split applied to bindings: a placement can
                # hold a FAULTED binding and still serve the others. The fault is typed on
                # the binding that earned it, its entrypoints refuse naming it, and the
                # loop goes on to the next group.
                for member in group:
                    self.failed_bindings[member.entrypoint_binding_plan_id] = (
                        f"{reply.get('code')}: {detail}"
                    )
                # #567c's SECOND HALF: a refusal reduced to its class name is a refusal
                # that does not exist. This line carried the code and four memory numbers
                # and DROPPED the executor's own message — so an operator watching a boot
                # fail read "unhandled_exception" beside "the allocator holds -1 B", which
                # names neither what broke nor where. The detail leads; the memory state
                # follows it, because on a genuine shortfall those numbers are the answer
                # and on everything else they are context.
                self.note(
                    "boot",
                    f"prepare REFUSED for {first.entrypoint_binding_plan_id[:20]}: "
                    f"{reply.get('code')} - {detail} "
                    f"[peak allocator {reply.get('peak_allocator_bytes', 0)} B / ledger "
                    f"{reply.get('peak_ledger_vram_bytes', 0)} B was LIVE when it failed; "
                    f"after rollback the allocator holds "
                    f"{reply.get('allocator_bytes', -1)} B and the device has "
                    f"{reply.get('device_free_bytes', -1)} B free"
                    # A ROLLBACK THAT CANNOT PROVE ITS RETURN SAYS SO, HERE. The retained
                    # bytes are why the next attempt in this worker refuses before it
                    # begins, so they belong beside the refusal that produced them.
                    + (
                        ""
                        if (reply.get("rollback") or {}).get("complete", True)
                        else (
                            f"; ROLLBACK INCOMPLETE: claimed "
                            f"{(reply.get('rollback') or {}).get('claimed_bytes')} B, the "
                            f"allocator returned "
                            f"{(reply.get('rollback') or {}).get('returned_bytes')} B, "
                            f"{(reply.get('rollback') or {}).get('retained_bytes')} B are "
                            "still held and stay CHARGED on the ledger"
                        )
                    )
                    + "]",
                )
                # A failed construction owns no retry policy. Retire its possibly retained
                # allocations and remain offline; convergence decides whether to try again.
                try:
                    self.supervision.retire_current(
                        executor, f"prepare refused: {reply.get('code')}"
                    )
                except ExecutorGone as exc:
                    self.note("recovery", f"prepare refusal reclaim failed: {exc}")
                continue
            digest = reply["model_construction_digest"]
            delivery = reply.get("delivery") or {}
            resolved = PreparedModel(
                logical_model_digest=digest,
                constructed_model_digest=str(reply["constructed_model_digest"]),
                delivery_rung=str(delivery.get("rung", "")),
                delivery_variant=str(delivery.get("variant", "")),
                weight_bytes=int(reply["facts"]["filled_bytes"]),
            )
            executor.construction_key = construction_key
            executor.prepared_binding_set = binding_set
            executor.prepared_model = resolved
            executor.purpose = "serving"
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
                self.prepared_models[binding.entrypoint_binding_plan_id] = resolved
            self.engine.bindings = self.bindings
            self.engine.prepared_models = self.prepared_models
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
                f"generation {str(reply['constructed_model_digest'])[:23]}… "
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
        # A placement with NO working binding has nothing to serve and says so; one with
        # any working binding becomes dispatchable and refuses only what actually failed.
        if len(self.failed_bindings) >= len(self.bindings):
            self.latched = "prepare refused: every binding"
            self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, "prepare REFUSED")
            return
        if self.failed_bindings:
            self.note(
                "boot",
                f"{len(self.bindings) - len(self.failed_bindings)} of {len(self.bindings)} "
                f"binding(s) prepared; faulted: " + ", ".join(sorted(self.failed_bindings)),
            )
        if not self.run_warm_pass():
            self.latched = "executor_lost_during_warm"
            self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, self.latched)
            self._settle()
            return
        self.set_serving(pb.ServingState.SERVING_STATE_DISPATCHABLE)
        if not self.latched:
            self.accepted.converged_revision = self.accepted.accepted_desired_state_revision

    def run_warm_pass(self) -> bool:
        """One attempted warm case per WARMABLE entrypoint, through ORDINARY dispatch."""
        if self.options.no_warm:
            executor = self.supervision.current
            return executor is not None and self.supervision.owns(executor)
        executor = self.supervision.current
        if executor is None:
            return False
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
                self.supervision.invalidate(executor, "executor died during warmup")
                try:
                    self.supervision.retire_current(executor, "executor died during warmup")
                except ExecutorGone as reclaim:
                    self.note("recovery", f"warmup reclaim failed: {reclaim}")
                self.note("warm", f"{binding.entrypoint}: executor lost: {exc}"[:200])
                return False
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
                self.ledger.observe_probe(
                    executor.call({"cmd": "probe"}, timeout=30.0, total_timeout=30.0)
                )
            except ExecutorGone:
                self.supervision.invalidate(executor, "executor died during warmup probe")
                try:
                    self.supervision.retire_current(executor, "executor died during warmup probe")
                except ExecutorGone as reclaim:
                    self.note("recovery", f"warmup probe reclaim failed: {reclaim}")
                return False
            shift = self.ledger.rebaseline("boot warm pass")
            self.note("warm", f"the resident device baseline moved {shift:+d} B on the warm pass")
        return self.supervision.owns(executor)

    def imposed(self) -> dict[str, str]:
        """The IMPOSE half of the seal. Every name here is in the reviewed allowlist."""
        imposed = {
            "CUDA_VISIBLE_DEVICES": self.options.devices,
            "PYTORCH_CUDA_ALLOC_CONF": self.options.alloc_conf,
            "OMP_NUM_THREADS": str(self.options.threads),
        }
        # The FILL runs in the executor, so its capture directory has to cross the seal or
        # it instruments the one process that never reserves a destination. Projected only
        # when one is configured: an empty value is an imposition of nothing.
        where = config.forensics_dir()
        if where:
            imposed[child_env.FILL_FORENSICS_DIR] = where
        return imposed

    # ------------------------------------------------------------------ dispatch

    def on_offer(self, offered: pb.AttemptOffer) -> None:
        """STREAM THREAD: fence, record, enqueue. Nothing here waits, fetches or leases.

        EVERY OFFER GETS A JOURNALED OUTCOME (§7). The outcome-less third state is closed:
        an `AttemptOffer` receives either `AttemptAccepted` or a JOURNALED
        `AttemptOutcome(REFUSED)` — never silence, never an unjournaled soft decline. An
        unjournaled refusal is unreconcilable after a crash, because the RecordOwner cannot
        then distinguish "never landed" from "refused", which is precisely the distinction
        02 §6 is built on.

        The rename is load-bearing: receiving an OFFER begins VALIDATION, not execution, so
        refusing one is an ordinary outcome rather than an exception. A pre-execution refusal
        consumes the `attempt_ordinal` and ZERO billed execution budget — `execution_started`
        stays false — and the RecordOwner may immediately mint `attempt_ordinal + 1` and
        dispatch it elsewhere.
        """
        # THE ADMISSION FENCE FIRST (§6), because it is the cheapest deterministic verdict
        # and because a stale generation means the RecordOwner dispatched against a picture
        # of this worker that no longer holds.
        if offered.admission_generation != self.admission_generation:
            self.refuse(
                offered,
                "admission_generation_stale",
                f"the offer echoes admission generation {offered.admission_generation} and "
                f"this worker is at {self.admission_generation}; the verdict is the same on "
                "every replay of these bytes, and the ordinal is consumed with no budget draw",
                CAUSE.CAUSE_CODE_ADMISSION_GENERATION_STALE,
            )
            return
        if self.accepted.mode == "serving":
            wanted = self.placement.placement_id if self.placement is not None else ""
            if offered.placement_id != wanted:
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_UNKNOWN_PLACEMENT,
                        subject=offered.placement_id,
                        reason="unknown_placement",
                        detail=f"the offer names placement {offered.placement_id!r} and this "
                        f"worker hosts {wanted!r}",
                    )
                )
                self.refuse(
                    offered,
                    "unknown_placement",
                    f"no placement {offered.placement_id!r} is hosted here",
                    CAUSE.CAUSE_CODE_UNKNOWN_PLACEMENT,
                )
                return
            if not self.dispatchable():
                self.refuse(
                    offered,
                    "placement_not_dispatchable",
                    f"placement {wanted!r} is "
                    + (pb.ServingState.Name(self.placement.serving) if self.placement else "absent")
                    + (f" (LATCHED: {self.latched})" if self.latched else ""),
                    CAUSE.CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE,
                )
                return
        if self.accepted.posture == pb.Posture.POSTURE_DRAINING:
            self.refuse(
                offered,
                "draining",
                "the worker is draining and admits no new attempts",
                CAUSE.CAUSE_CODE_NO_CAPACITY,
            )
            return
        try:
            attempt = self.engine.offer(offered)
        except AttemptRefusal as exc:
            self.note(
                "attempt", f"{offered.request_id}#{offered.attempt_ordinal} refused: {exc.code}"
            )
            self.refuse(offered, exc.code, exc.detail, exc.cause, exc.origin)
            return
        if attempt.state in ("outcome", "closed"):
            self.note(
                "attempt",
                f"duplicate AttemptOffer for {offered.request_id}#{offered.attempt_ordinal}: "
                f"replaying the journaled outcome, executing NOTHING "
                f"(executions={attempt.executions})",
            )
            self.send(pb.WorkerFrame(attempt_outcome=self.engine.wrap(attempt)))
            return
        if attempt.state == "queued":
            self.note(
                "attempt",
                "duplicate AttemptOffer for the attempt currently crossing to the device "
                "lane: it is held, and its acceptance publishes when the lane admits it",
            )
            return
        if attempt.state in ("accepted", "running", "finalizing"):
            self.note(
                "attempt",
                "duplicate AttemptOffer for a held attempt: re-accepting "
                "the SAME acceptance; nothing starts a second time",
            )
            self.send(pb.WorkerFrame(attempt_accepted=self.engine.accepted_message(attempt)))
            return
        # THE WINDOW IS ONE, AND OVER IT THE ANSWER IS A JOURNALED OUTCOME (#494f/§7). The
        # device lane is a single serialized state machine, so a second held attempt could
        # only wait — and it would wait in a state the wire cannot spell, absent from the
        # snapshot and from the held-attempt set. A refusal the RecordOwner can read and
        # re-dispatch is strictly more informative than a hold it cannot see.
        if self.available_attempt_slots() < 1:
            busy = [
                held
                for held in self.engine.history.values()
                if held is not attempt and held.state in self.engine.HELD
            ]
            other = busy[0] if busy else None
            self.note(
                "attempt",
                f"{offered.request_id}#{offered.attempt_ordinal} over the admission window",
            )
            self.refuse(
                offered,
                "no_capacity",
                f"this worker admits {ADMISSION_WINDOW} attempt at a time and the window is "
                + (
                    f"held by {other.request_id}#{other.attempt} ({other.state})"
                    if other is not None
                    else "closed"
                )
                + "; a seat is held by a running attempt AND by an outcome awaiting "
                "AttemptOutcomeAck, so acking frees one",
                CAUSE.CAUSE_CODE_NO_CAPACITY,
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
            f"{key} waited {waited:.0f} ms; the placement is "
            f"{pb.ServingState.Name(self.placement.serving) if self.placement else 'absent'}"
            + ("" if settled else f" ({why})"),
        )

    def refuse(
        self,
        offered: pb.AttemptOffer,
        code: str,
        detail: str,
        cause: pb.CauseCode,
        origin: pb.CauseOrigin = ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
    ) -> None:
        """A typed refusal is an OUTCOME: journaled before send, like any other (§7).

        `execution_started` is FALSE here by construction — nothing ran — which is the
        BILLING FACT stated structurally rather than inferred from a cause-code allowlist
        (#480c). The RecordOwner reads it and knows the ordinal was consumed and the budget
        was not.
        """
        held = self.engine.history.get((offered.request_id, offered.attempt_ordinal))
        if held is not None and held.state in ("outcome", "closed"):
            held = None
        attempt = held or AttemptRecord(
            request_id=offered.request_id,
            attempt=offered.attempt_ordinal,
            digest=offered.invocation_spec_digest,
            spec={},
        )
        try:
            outcome = self.engine.outcome(
                attempt,
                STATUS.OUTCOME_STATUS_REFUSED,
                cause,
                origin,
                f"{code}: {detail}",
            )
        except JournalError as exc:
            # ACK-AFTER-JOURNAL HAS NO EXCEPTION FOR A REFUSAL (#494b). An outcome nobody
            # wrote down cannot be replayed and cannot be acked, so it is not sent.
            self.note("journal", f"cannot journal a refusal ({exc.code}); sending nothing")
            return
        self.engine.history[attempt.key()] = attempt
        self.send(pb.WorkerFrame(attempt_outcome=outcome))

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
        # Desired-state convergence and executor rebuilds take this same lock. Recheck the
        # offer's immutable routing facts under it, then hold it through the acceptance
        # journal: a queued offer cannot be accepted on the far side of a placement or
        # admission-generation change merely because it passed `on_offer()` earlier.
        with self.control_lock:
            if attempt.canceling:
                self.note("cancel", f"{key} was cancelled before admission")
                self.terminate(attempt, self.engine.finish_cancel, attempt.canceling)
                return False
            deadline_ms = int(attempt.spec.get("deadline_unix_ms", 0))
            if deadline_ms and time.time() * 1000 > deadline_ms:
                self.note("cancel", f"{key} reached the lane past its deadline")
                self.terminate(attempt, self.engine.finish_cancel, "deadline")
                return False
            if attempt.admission_generation != self.admission_generation:
                self.note(
                    "admission",
                    f"refusing queued {key}: offered generation "
                    f"{attempt.admission_generation}, current {self.admission_generation}",
                )
                self.terminate(
                    attempt,
                    self.engine.refuse_held,
                    "admission_generation_stale",
                    f"the queued offer echoes admission generation "
                    f"{attempt.admission_generation} and this worker is now at "
                    f"{self.admission_generation}",
                    CAUSE.CAUSE_CODE_ADMISSION_GENERATION_STALE,
                )
                return False
            if self.accepted.mode == "serving":
                wanted = self.placement.placement_id if self.placement is not None else ""
                if attempt.placement_id != wanted:
                    self.note(
                        "admission",
                        f"refusing queued {key}: placement {attempt.placement_id!r} is no "
                        f"longer {wanted!r}",
                    )
                    self.terminate(
                        attempt,
                        self.engine.refuse_held,
                        "unknown_placement",
                        f"the queued offer names placement {attempt.placement_id!r} and "
                        f"this worker now hosts {wanted!r}",
                        CAUSE.CAUSE_CODE_UNKNOWN_PLACEMENT,
                    )
                    return False
            if not self.dispatchable():
                self.note("admission", f"refusing queued {key}: worker is not dispatchable")
                self.terminate(
                    attempt,
                    self.engine.refuse_held,
                    "placement_not_dispatchable",
                    "the worker stopped being dispatchable while this offer was queued",
                    CAUSE.CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE,
                )
                return False
            if self.accepted.posture == pb.Posture.POSTURE_DRAINING:
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
                self.fail_machine(exc.code)  # fault first, then the edge
                return False
        self.send(pb.WorkerFrame(attempt_accepted=accepted))
        return True

    def execute(self, attempt: AttemptRecord) -> None:
        """RUN -> OUTCOME -> RECONCILE. The ledger closes inside `outcome()`."""
        try:
            outcome = self.engine.execute(attempt)
        except JournalError as exc:
            self.note("journal", f"{exc.code}: only a last-resort outcome can be claimed")
            self.fail_machine(exc.code)
            self.terminate(attempt, self.engine.last_resort, f"{exc.code}: {exc.detail}")
            return
        except Exception as exc:
            self.note("attempt", f"execution faulted: {type(exc).__name__}: {exc}")
            try:
                outcome = self.engine.abandon(
                    attempt, f"the worker faulted while running this attempt: {exc}"
                )
            except Exception as inner:
                self.note("attempt", f"even ABANDONED could not be built: {inner}")
                self.fail_machine(f"{type(inner).__name__}")
                self.terminate(attempt, self.engine.last_resort, f"{type(inner).__name__}: {inner}")
                return
        self.send(pb.WorkerFrame(attempt_outcome=outcome))

    def terminate(
        self, attempt: AttemptRecord, build: Callable[..., pb.AttemptOutcome], *args: Any
    ) -> None:
        """Journal ONE outcome for a held attempt and send it.

        ACK-AFTER-JOURNAL IS THE LAW, INCLUDING HERE (#494b). An outcome that could not be
        journaled is never sent: the attempt stays HELD, and rev-2 gives that hold its own
        wire state — `ATTEMPT_STATE_HELD_UNDURABLE` (#507d/#510d) — instead of the frozen
        wire's RUNNING-plus-a-LOCAL_SAFETY-fault compound. The worker takes the fault, goes
        `WORKER_PHASE_FAILED`, and the RecordOwner reads a worker that stopped rather than an
        outcome it can never close.
        """
        why = ""
        try:
            self.send(pb.WorkerFrame(attempt_outcome=build(attempt, *args)))
            return
        except UndurableOutcome as exc:
            self.note("journal", f"HOLDING {attempt.request_id}#{attempt.attempt}: {exc}")
            self.fail_machine("outcome_undurable")
            return
        except Exception as exc:
            why = f"{type(exc).__name__}: {exc}"
            self.note("attempt", f"the outcome could not be built ({why}); last resort")
        try:
            self.send(pb.WorkerFrame(attempt_outcome=self.engine.last_resort(attempt, why)))
        except UndurableOutcome as exc:
            self.note("journal", f"HOLDING {attempt.request_id}#{attempt.attempt}: {exc}")
            self.fail_machine("outcome_undurable")
        except Exception as inner:
            self.note("attempt", f"even the last-resort outcome failed: {inner}")
            self.fail_machine("last_resort_failed")

    def rebuild_if_needed(self) -> None:
        """The queue starts behind the rebuild because BOTH live on this lane."""
        with self.control_lock:
            current = self.supervision.current
            if current is None or not current.alive():
                self._rebuild_locked("the executor is gone")
            elif current.poisoned:
                self._rebuild_locked(f"poisoned: {current.poisoned}")

    def rebuild(self, why: str) -> None:
        """Replace the executor and RELOAD before the placement is dispatchable again."""
        with self.control_lock:
            self._rebuild_locked(why)

    def _rebuild_locked(self, why: str) -> None:
        """Rebuild while desired-state convergence is excluded."""
        self.set_serving(pb.ServingState.SERVING_STATE_ACTIVATING, "executor rebuild")
        started = time.perf_counter()
        self.note("recovery", f"rebuilding the executor: {why}")
        current = self.supervision.current
        try:
            if current is None:
                self.supervision.spawn(imposed=self.imposed())
            else:
                self.supervision.replace(current, imposed=self.imposed(), why=why)
        except ExecutorGone as exc:
            self.note("recovery", f"respawn failed: {exc}")
            self.latched = f"executor_respawn_failed: {exc}"[:200]
            self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, "respawn failed")
            self._settle()
            return
        self.bump_admission("executor respawned")
        self.prepare_executor()
        self.note(
            "recovery",
            f"dispatchable again in {(time.perf_counter() - started) * 1000:.0f} ms at "
            f"executor generation {self.supervision.incarnation} / admission generation "
            f"{self.admission_generation}",
        )

    def watchdog(self) -> None:
        """The deadline and the grace clock — both the WORKER's, never author code's."""
        while not self.stop.wait(0.05):
            for attempt in list(self.engine.live.values()):
                if attempt.state not in ("running", "reclaiming"):
                    continue
                deadline_ms = int(attempt.spec.get("deadline_unix_ms", 0))
                deadline_expired = False
                with attempt.transition_lock:
                    if (
                        attempt.state == "running"
                        and deadline_ms
                        and time.time() * 1000 > deadline_ms
                        and not attempt.canceling
                    ):
                        attempt.canceling = "deadline"
                        executor = attempt.executor
                        if executor is not None:
                            self.supervision.cooperative_cancel(
                                executor, f"{attempt.request_id}#{attempt.attempt}"
                            )
                        attempt.cancel_started = time.perf_counter()
                        deadline_expired = True
                if deadline_expired:
                    self.note("cancel", f"deadline expired for {attempt.request_id}")
                if attempt.canceling:
                    started = getattr(attempt, "cancel_started", 0.0) or time.perf_counter()
                    attempt.cancel_started = started
                    if time.perf_counter() - started > self.options.grace:
                        self.note(
                            "cancel",
                            f"grace of {self.options.grace:g} s expired: escalating to a "
                            "forceful kill (the handler never asked whether it was cancelled)",
                        )
                        try:
                            facts = self.engine.escalate(attempt, self.options.grace)
                        except Exception as exc:
                            # A failed reclaim is NOT a cancellation result. Keep the
                            # executor owned, withdraw machine admission, and leave this
                            # watchdog as the retrying owner. A one-shot exception here used
                            # to kill the watchdog thread and strand the executor forever.
                            self.note(
                                "cancel",
                                f"forceful reclaim failed; retrying after the grace window: "
                                f"{type(exc).__name__}: {exc}",
                            )
                            self.capacity_dropped("executor reclaim is not proved")
                            attempt.cancel_started = time.perf_counter()
                            continue
                        self.note("cancel", f"forceful: {json.dumps(facts)}")
                        attempt.canceling = attempt.canceling or "client"
                        attempt.cancel_started = time.perf_counter() + 1e9

    def reporter(self) -> None:
        """The ObservedWorkerState tick, and the liveness SAMPLE that rides it."""
        seen: dict[str, str] = {}
        while not self.stop.wait(REPORT_SECONDS):
            for subject in list(self.monitor.cursors):
                verdict = self.monitor.sample(subject)
                if verdict == "wedged" and seen.get(subject) != verdict:
                    self.note("liveness", f"{subject} is WEDGED by silence (no clock involved)")
                seen[subject] = verdict
            if self.connected.is_set():
                self.send(pb.WorkerFrame(observed_state=self.observed_state()))

    def observed_state(self) -> pb.ObservedWorkerState:
        """The observed half of the pair (§8; was `Report`).

        ACCEPTANCE AND CONVERGENCE ARE TWO SEPARATE, CHECKABLE FACTS.
        `accepted_desired_state_revision` advances when this worker durably accepts intent —
        which is all `applied_revision` ever honestly meant, while claiming to report
        reality. `converged_revision` advances ONLY when observed state satisfies the
        accepted set, and `converged < accepted` is the normal, visible state of a
        convergence in progress or a latched failure. Never an error; always readable.

        Serving truth is PER PLACEMENT; job mode reports its capacity credit. Before any
        desired state it reports NEITHER — an honest "no claim yet" rather than a fabricated
        zero.
        """
        observed = pb.ObservedWorkerState(
            accepted_desired_state_revision=self.accepted.accepted_desired_state_revision,
            converged_revision=self.converged_revision(),
            worker_phase=self.phase,
            held_attempts=self.engine.active(),
            faults=self.engine.faults[-8:],
            activity=self.activity[-8:],
            applied_wire_minor=self.fence.wire_minor,
            # THE ONE WORKER-LEVEL ADMISSION FENCE (§6). Not per placement: N counters over
            # ONE serialized device advertise N x the real capacity.
            admission_generation=self.admission_generation,
            admission_state=self.admission_state(),
            available_attempt_slots=self.available_attempt_slots(),
            # The grant lane is observable without an ack lane of its own (§3).
            applied_artifact_grant_id=self.grant_id,
            applied_grant_revision=self.grant_revision,
        )
        if self.accepted.mode == "job":
            observed.job_capacity.CopyFrom(self.job_capacity())
        elif self.accepted.mode == "serving":
            observed.accepted_placement_set_digest = self.accepted.accepted_placement_set_digest
            if self.placement is not None:
                observed.placements.append(self.placement_status())
        return observed

    def converged_revision(self) -> int:
        """A DERIVED fact, recomputed on every read — never a stored optimism.

        It may NEVER advance past a placement whose serving state is not DISPATCHABLE for a
        desired spec (§8): a worker that advances it while observed state disagrees is in
        breach, not merely optimistic. So the stored value is the high-water mark
        `prepare_executor` reached, and this read CLAMPS it back the moment reality moves —
        a placement that stops being dispatchable un-converges immediately, which is exactly
        what a RecordOwner watching a cutover needs to see.
        """
        if self.accepted.mode == "serving" and not self.dispatchable():
            return min(
                self.accepted.converged_revision,
                max(0, self.accepted.accepted_desired_state_revision - 1),
            )
        return self.accepted.converged_revision

    def placement_status(self) -> pb.PlacementStatus:
        """Per-placement truth on TWO AXES (#473/#482), and `readiness_epoch` is not among
        them: it is DELETED, not renamed (#486b), because the serving axis plus
        `dispatchable_plan_ids` already carry what a placement can serve and the worker-level
        `admission_generation` carries whether an offer is admissible. `attempt_credits` is
        gone for the same reason, one level up.

        The accelerator qualification is deliberately ABSENT: this torch-free process cannot
        measure capability facts, and absent IS the present-but-unqualified state (#454 —
        registration honesty is schema). The executor-measured qualification lands with
        cr-020/cr-021's lanes.
        """
        assert self.placement is not None
        executor = self.supervision.current
        dispatchable = sorted(executor.ready_plans) if executor is not None else []
        # DISJOINT by construction (the proto says so): a plan is either servable now or
        # still to be materialized, never counted in both.
        materializable = sorted(set(self.placement.plan_ids) - set(dispatchable))
        return pb.PlacementStatus(
            placement_id=self.placement.placement_id,
            placement_spec_digest=self.placement.spec_digest,
            materialization=self.placement.materialization,
            serving=self.placement.serving,
            retained_fallback_spec_digest=self.placement.retained_fallback_spec_digest,
            executor_generation=self.supervision.incarnation,
            dispatchable_plan_ids=dispatchable,
            materializable_plan_ids=materializable,
            faults=[
                f
                for f in self.engine.faults[-8:]
                if f.kind
                in (
                    pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                    pb.FaultKind.FAULT_KIND_BINDING_DEGRADED,
                    pb.FaultKind.FAULT_KIND_EXECUTOR_POISONED,
                    pb.FaultKind.FAULT_KIND_ENVIRONMENT_RECEIPT_MISMATCH,
                    pb.FaultKind.FAULT_KIND_FALLBACK_PIN_MISSING,
                )
            ],
        )

    def job_capacity(self) -> pb.JobCapacity:
        """How many JOB attempts this worker can take right now (cr-009)."""
        in_flight = sum(1 for a in self.engine.live.values() if a.job is not None)
        ready = bool(self.jobs) and self.job_ready
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
            worker_instance_id=self.options.instance_id,
            reason=reason,
            detail=f"{detail} | disk invoice: {json.dumps(invoice, sort_keys=True)}"[:1024],
            resources=self.resources(),
            worker_release_id=self.options.release_id,
            worker_artifact_digest=self.options.artifact_digest,
        )

    def run(self) -> int:
        """Serve until stopped. The worker outlives any one stream: an owner that drops
        the connection re-dials and re-claims the SAME boot; the in-memory host ending its
        one stream is the run ending."""
        # Executor grandchildren orphan here, so retirement can reap every group member
        # and prove its exact PID absent before publishing a successor.
        proctree.arm_subreaper()
        try:
            self.journal.append(
                "session",
                {
                    "worker_boot_id": self.fence.worker_boot_id,
                    "instance_id": self.options.instance_id,
                },
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
            # THE MACHINE IS UP (#482). `worker_phase` is machine lifecycle, pulled out of
            # the placement enum: BOOTING is true only until the journal, the socket, the
            # orphan sweep, the recovered ownership and the lanes are all real. It moves
            # here and not at the first claim, because a worker is online whether or not a
            # RecordOwner has dialled it — and admission is CLOSED until the barrier anyway.
            self.set_phase(pb.WorkerPhase.WORKER_PHASE_ONLINE, "boot completed")
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
        try:
            self.supervision.close()
        except ExecutorGone as exc:
            self.exit_code = int(Exit.internal)
            self.note("shutdown", f"executor reclaim failed: {exc}")
        finally:
            self.journal.close()
        print(
            "[worker] summary "
            + json.dumps(
                {
                    "worker_boot_id": self.fence.worker_boot_id,
                    "control_generation": self.fence.control_stream_generation,
                    "executor_generations": self.supervision.spawns,
                    "admission_generation": self.admission_generation,
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
_ARTIFACT_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

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
        "--artifact-digest",
        default="",
        help="the worker ARTIFACT's provenance digest, `sha256:<64 hex>`; absent stays "
        "absent. Was --image-digest: a native worker has no OCI image (#481), and this is "
        "the one field substrate identity rides (#487/#489)",
    )
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--python", default="")
    parser.add_argument("--devices", default="0")
    parser.add_argument(
        "--grant-root",
        default="",
        help="comma-separated directory trees a file:// grant may address (#506b)",
    )
    parser.add_argument(
        "--project",
        default="",
        help="THIS machine's copy of the endpoint release tree; defaults to the working "
        "directory. A binding-plan record's own `project` is the WRITER's resolution and is "
        "not a path on a rented pod (#563f)",
    )
    parser.add_argument("--alloc-conf", default="expandable_segments:True")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--grace", type=float, default=GRACE_SECONDS)
    parser.add_argument("--host-budget", type=int, default=8 << 30)
    parser.add_argument("--pinned-budget", type=int, default=512 << 20)
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument("--tls-cert", default="", help="PEM cert for a non-loopback listener")
    parser.add_argument("--tls-key", default="", help="PEM key for a non-loopback listener")
    args = parser.parse_args(argv)

    if args.artifact_digest and not _ARTIFACT_DIGEST.fullmatch(args.artifact_digest):
        print(
            f"[worker] --artifact-digest {args.artifact_digest!r} is not `sha256:<64 hex>`: a "
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
        artifact_digest=args.artifact_digest,
        git_commit=args.git_commit,
        python=args.python,
        devices=args.devices,
        grant_roots=tuple(r for r in args.grant_root.split(",") if r),
        project=Path(args.project).resolve() if args.project else Path.cwd(),
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
    # THE HASH SET, VALIDATED BEFORE ANYTHING BINDS (#560g). A malformed value is a
    # provisioning mistake, and a worker that binds with an unusable credential set would
    # discover it one claim later — at which point a port is open.
    try:
        access = AccessTokenSet.parse(config.access_token_hashes)
    except AccessTokenError as exc:
        print(f"[worker] refusing to bind {listen}: {exc}", file=sys.stderr, flush=True)
        return int(Exit.structural)
    if remote and access is not None and access.path is not None:
        # The path form, proven READABLE at launch rather than at the first claim: a pod
        # whose hash file never arrived admits nobody, and saying so now beats answering
        # UNAUTHENTICATED to the renter who just paid for it.
        access.reload()
        if access.unreadable:
            print(
                f"[worker] refusing to bind {listen}: {access.why_not()}",
                file=sys.stderr,
                flush=True,
            )
            return int(Exit.structural)
    if remote and not config.bootstrap_credential and access is None:
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
            "non-loopback listener authenticates its owner (COZY_BOOTSTRAP_CREDENTIAL when "
            "the launcher minted the credential, COZY_ACCESS_TOKEN_SHA256 when a renter did) "
            "or it does not serve — one-way TLS proves nothing about the dialer",
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
    """The PROCESS entrypoint: read the ONE config here, then serve.

    A configuration refusal is the worker's own verdict, printed the way every other
    pre-bind refusal is: a launcher reading this process's output must see WHY nothing
    bound, and a traceback is not that.
    """
    try:
        config = read_config()
    except ConfigError as err:
        print(f"[worker] {err.message}", file=sys.stderr, flush=True)
        print(f"[worker] remedy: {err.remedy}", file=sys.stderr, flush=True)
        return int(err.code)
    return serve_worker(config, argv)


if __name__ == "__main__":
    raise SystemExit(main())
