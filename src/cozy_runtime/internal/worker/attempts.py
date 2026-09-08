"""The attempt engine: offer -> prepare -> accept -> execute -> terminal.

Everything the worker may CLAIM about an attempt passes through here, in the one order
worker-protocol/02 fixes:

    VALIDATING  fences, digest recomputation over the resident bytes, local safety
    PREPARING   the PlanChooser fixes ONE plan; the ExecutorPreparer applies it
    ACCEPTED    acceptance recorded (spec + plan + model-construction digests) FIRST
    RELEASED    the executor answered: device metrics, ledger probe/close and every
                executor-side consequence taken on the lane thread; the device is free
    TERMINAL    the post phase (encode, bound, write under the grant) on the lane's post
                thread; the canonical TerminalBody recorded BEFORE it is sent, replayed
                byte-for-byte until a TerminalAck echoes its id AND its digest
    CLOSED      closure recorded

Two things are deliberately NOT here, because they are not the worker's: retryability (a
RecordOwner projection over status/cause/origin) and attempt-selection order beyond this
worker (dispatch is the RecordOwner's; below that boundary service class does not exist and
there is no field to carry it). Two attempts assigned to this worker SERIALIZE on the single
device lease, in arrival order, and that is the whole of local arbitration.

Metric attestation lives here too: what the executor reports is a CLAIM. Wall-clock is
clamped to the worker-observed dispatch->result window, and the fields the worker
structurally cannot verify are NAMED in `unverified_fields` rather than passed off as
measurements.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from dataclasses import replace as dataclasses_replace
from pathlib import Path
from typing import Any, cast

from cozy_runtime.author import _codec
from cozy_runtime.author._errors import DEVICE_OOM, OutputError
from cozy_runtime.author._observations import EventRing, Observation
from cozy_runtime.author._services import MAX_OUTPUT_BYTES
from cozy_runtime.internal import canonical, fill, package_interface, seam
from cozy_runtime.internal import weights_sink as weights_bridge
from cozy_runtime.internal.pathkey import opaque_key
from cozy_runtime.internal.worker import grants, observe, refusal, triage
from cozy_runtime.internal.worker.calls import Calls
from cozy_runtime.internal.worker.child import Executor, ExecutorGone, ExecutorSupervision
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.plan import (
    AttemptPlan,
    DeclaredBinding,
    JobBinding,
    PlanRefusal,
    PreparedModel,
    PreparedRequest,
)
from cozy_runtime.internal.worker.records import WorkerRecords
from cozy_runtime.internal.worker.weights import WeightsExchange
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import documents, weights_limits
from cozy_runtime.protocol import worker_pb2 as pb

#: `safe_message` and `cause.detail` are bounded and SANITIZED (01 §3): printable ASCII
#: only, no secrets, no local paths, no signed URLs. The document profile is integer-only
#: and printable-ASCII, so an unsanitized message is not merely impolite — it cannot be
#: canonicalized at all, and the terminal that carries it could not be recorded.
_PATHY = re.compile(r"(file|https?)://\S+|(?<![\w.])/[\w./-]{4,}")


def safe(text: str, cap: int = 1024) -> str:
    """Bound and sanitize a message for the wire."""
    stripped = _PATHY.sub("<redacted>", text)
    ascii_only = "".join(c if 0x20 <= ord(c) <= 0x7E else "." for c in stripped)
    return ascii_only[:cap]


def _coded(outcome: Mapping[str, Any]) -> str:
    """`<code>: <message>` — the AUTHOR's typed code in front of its own sentence.

    The wire carries a small neutral `CauseCode` on purpose (a RecordOwner classifies, it
    does not read author vocabulary), so the typed code has to ride the message or it does
    not survive the boundary at all. Every other refusal on this path already renders this
    way; author outcomes were the exception.
    """
    code = str(outcome.get("code") or "")
    message = str(outcome.get("message") or "")
    if not message:
        return code
    return message if not code or message.startswith(f"{code}:") else f"{code}: {message}"


#: What a request DOCUMENT may weigh, checked at the acceptance boundary. The seam admits
#: one 64 KiB frame; the dispatch command wraps the payload in ids, paths, tree grants and
#: budget facts, so the payload's own share is the frame less a generous envelope. It is a
#: PRE-ACCEPT condition (ev-003, decisions #302): over it, the old path accepted the attempt
#: and then hung on a worker-side SeamError with no terminal ever produced.
PAYLOAD_MAX_BYTES = seam.MAX_FRAME - 12 * 1024

#: Fields the worker structurally CANNOT verify, by their WIRE names. Named on every
#: terminal that carries them, sorted, because a claim printed as a measurement is worse
#: than no number.
UNVERIFIABLE = ("input_tokens", "output_tokens", "peak_device_memory_bytes")


def weights_exchange_bytes(spool: Path, output_slot: str, kind: str, expected_length: int) -> bytes:
    """Read one worker-chosen immutable spool file; the child never supplies a path."""
    cap = (
        weights_limits.MAX_WEIGHTS_DECLARATION_BYTES
        if kind == "declaration"
        else weights_limits.MAX_WEIGHTS_RECEIPT_BYTES
    )
    if not 0 < expected_length <= cap:
        raise ValueError(f"weights {kind} length is outside 1..={cap}")
    path = weights_bridge.exchange_file(spool, output_slot, kind)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_length:
            raise ValueError(f"weights {kind} spool file is not an exact regular file")
        chunks: list[bytes] = []
        remaining = expected_length + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 << 10))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(raw) != expected_length or (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise ValueError(f"weights {kind} spool file changed while being read")
        return raw
    finally:
        os.close(descriptor)


#: The fault tail this worker keeps. Every reader takes the last 8.
FAULTS_HELD = 64

CAUSE = pb.CauseCode
ORIGIN = pb.CauseOrigin
STATUS = pb.OutcomeStatus

#: The author-surface outcome vocabulary, mapped onto the wire's NEUTRAL one. The mapping is
#: total and closed: a new author terminal without a wire meaning is a compile-time hole.
#: The executor's own faults under a group lane (cr-068): a follower died, diverged or
#: could not cross an argument. RUNTIME origin already; the cause is the executor's, not the
#: author's, and the executor poisons itself so the worker rebuilds the whole group.
GROUP_FAULT_CODES = frozenset(
    {
        "group_broken",
        "group_unformed",
        "rank_divergence",
        "uncrossable_argument",
        "ungated_sharded_forward",
        "context_parallel_unavailable",
    }
)

#: The device-capacity codes a FAILED attempt can carry (cr-097). Their peak is a
#: truncated lower bound and may never calibrate; the picture they died under is recorded
#: instead, as a refusal-grade fact. `device_shortfall` is the runtime's own pre-mutation
#: refusal and `device_out_of_memory` is the allocator's surprise past it — both say the
#: same thing about the next attempt under the same picture.
DEVICE_SHORTFALL_CODES = frozenset({"device_out_of_memory", "device_shortfall"})

_TERMINAL = {
    "succeeded": (STATUS.OUTCOME_STATUS_SUCCEEDED, CAUSE.CAUSE_CODE_UNSPECIFIED),
    "refused": (STATUS.OUTCOME_STATUS_REFUSED, CAUSE.CAUSE_CODE_INVALID_REQUEST),
    "failed": (STATUS.OUTCOME_STATUS_FAILED, CAUSE.CAUSE_CODE_AUTHOR_EXCEPTION),
    "canceled": (STATUS.OUTCOME_STATUS_CANCELED, CAUSE.CAUSE_CODE_CLIENT_CANCEL),
}
#: the worker's own attempt-kind word, mapped onto the wire's enum.
_KIND = {
    "serving": pb.AttemptKind.ATTEMPT_KIND_SERVING,
    "job": pb.AttemptKind.ATTEMPT_KIND_JOB,
}
_ORIGIN = {
    "author": ORIGIN.CAUSE_ORIGIN_AUTHOR,
    "request": ORIGIN.CAUSE_ORIGIN_CLIENT,
    "runtime": ORIGIN.CAUSE_ORIGIN_RUNTIME,
}


@dataclass(frozen=True, slots=True)
class InlineResult:
    """The typed result as the EXECUTOR serialized it, with what the wire carries beside it."""

    canonical: bytes
    schema_digest: str
    adjustments: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class Released:
    """ONE attempt off the device (cr-079): the executor's reply and what the post phase
    needs to turn it into a terminal. Built on the lane thread at device release, consumed
    on the lane's post thread; nothing in it reaches the executor again."""

    attempt: AttemptRecord
    reply: dict[str, Any]
    spool: Path
    key: str


class AttemptRefusal(Exception):
    """A typed refusal that becomes a REFUSED terminal (or a dropped message, when the
    fence says the message was never ours to read)."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        cause: pb.CauseCode,
        drop: bool = False,
        origin: pb.CauseOrigin = ORIGIN.CAUSE_ORIGIN_WORKER,
    ) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.cause = cause
        self.drop = drop
        #: WHOSE fault, decided by the origin of the address that missed (v1 REF_ORIGIN).
        #: A caller-supplied address is the caller's typed error; a platform-produced one
        #: stays platform-fatal, and the two must never render as the same terminal.
        self.origin = origin


@dataclass(frozen=True, slots=True)
class JobModel:
    """One descriptor parameter joined to one held TensorFS Manifest for this attempt."""

    parameter: str
    class_name: str
    manifest: str
    length: int


@dataclass(frozen=True, slots=True)
class AttemptSlot:
    """The PLACEMENT SLOT an attempt was admitted against, captured once (cr-066).

    One worker hosts several placements on several lanes, and lanes run concurrently. The
    engine used to hold ONE `supervision`/`bindings`/`ledger` that the session re-pointed
    at whichever placement the current attempt named — correct while one thread admitted
    one attempt at a time, and a race the moment two lanes admit at once. Everything an
    attempt reads about its placement is therefore bound HERE, under the session's lock,
    and read from the record afterwards. Never a cross-placement lookup.
    """

    supervision: ExecutorSupervision
    ledger: Ledger
    chooser: Any
    bindings: Mapping[str, DeclaredBinding]
    prepared_models: Mapping[str, PreparedModel]
    failed_bindings: Mapping[str, str]
    lane_id: str
    #: what the executor on this slot was sealed to see (`CUDA_VISIBLE_DEVICES`)
    devices: str = ""
    job_device_count: int = 0
    orchestration: bool = False


@dataclass(slots=True)
class AttemptRecord:
    """ONE attempt, as this worker holds it. Named for what it is: the worker's RECORD
    of an attempt, distinct from `author._services.Attempt`, which is the handler's own
    per-attempt surface — two different things under one name is how a reader loses which
    side of the seam they are on."""

    request_id: str
    attempt: int
    digest: bytes
    spec: dict[str, Any]
    grant: grants.BoundGrant = dataclasses_field(default_factory=grants.BoundGrant)
    plan: AttemptPlan | None = None
    declared: DeclaredBinding | None = None
    #: what THIS worker's prepare resolved for that binding — never what it declared
    prepared_model: PreparedModel | None = None
    executor: Executor | None = None
    """The exact process epoch this attempt was accepted against."""
    slot: AttemptSlot | None = None
    """The placement slot — supervision, ledger, bindings, lane — bound at admission."""
    lane_id: str = ""
    """The lane whose seat this attempt holds, fixed when the offer was queued."""
    executor_epoch: int = 0
    executor_pid: int = 0
    placement_id: str = ""
    """The immutable routing identity this attempt was accepted against."""
    admission_epoch: int = 0
    """The worker admission epoch echoed by the offer that created this record."""
    kind: str = "serving"
    """`serving` or `job`. It SURVIVES compaction, because what an attempt was is still a
    fact about it after its heavy state is gone — the run-once recycle disposition and the
    Report's attempt kind both read it, and both used to infer it from `job`, which
    compaction clears."""
    job: JobBinding | None = None
    """Set on a JOB attempt instead of `binding`. Exactly one of the two is ever set: the
    ExecutionSpec names one mode, and a spec naming neither refuses before anything runs."""
    trees: dict[str, tuple[Path, str]] = dataclasses_field(default_factory=dict)
    """ref -> (materialized root, declared digest) for the input trees the grant carries."""
    inputs: dict[str, Any] = dataclasses_field(default_factory=dict)
    """input_id -> `GrantedInput` for every input ASSET this attempt was granted (cr-012).
    Fetched, length- and digest-checked, sniffed and spooled BEFORE acceptance, so the
    executor is handed verified files and never an address."""
    scratch: Path | None = None
    """The RUN's scratch tree, when the grant names one (`tmp/<request-id>/` on the owner's
    side; owner ruling 2026-09-02). Keyed on the run and NOT on the attempt, which is the
    whole of resumability: a retried attempt of the same run reads what the killed one
    left. None until the grant carries it — the Scratch service then refuses typed."""
    state: str = "validating"
    spool: Path | None = None
    outcome_id: str = ""
    outcome_digest: bytes = b""
    outcome_bytes: bytes = b""
    queued_at: float = 0.0
    """When the READER recorded this attempt and handed it to the device lane. The queue a
    caller waits in starts here, not at acceptance — acceptance is now something the lane
    does, so measuring from it would report every wait as zero."""
    accepted_at: float = 0.0
    dispatched_at: float = 0.0
    resulted_at: float = 0.0
    replays: int = 0
    payload: dict[str, Any] = dataclasses_field(default_factory=dict)
    canceling: str = ""
    cancel_started: float = 0.0
    transition_lock: threading.Lock = dataclasses_field(default_factory=threading.Lock)
    """Arbitrates natural finalization against forceful executor reclaim."""
    reclaim_done: threading.Event = dataclasses_field(default_factory=threading.Event)
    """Set only after forceful executor reclaim has external proof."""
    executions: int = 0
    #: every ledger class as it stood BEFORE this attempt's envelope was granted. The
    #: reconciliation at close compares against this line, never against the mid-attempt
    #: state, because "the attempt gave back what it took" is a claim about the whole
    #: envelope — the grant included.
    opened: dict[str, int] = dataclasses_field(default_factory=dict)
    #: cr-011's bounded observation record for THIS attempt. Worker-side, merged with
    #: whatever the executor's own ring sent back, and capped the same way on both sides.
    ring: EventRing = dataclasses_field(default_factory=EventRing)
    #: where the attempt's time went (executor-reported, worker-attested)
    attribution: dict[str, Any] = dataclasses_field(default_factory=dict)
    #: typed confessions: every serve differing from the plain reading of the request
    confessions: list[dict[str, Any]] = dataclasses_field(default_factory=list)
    #: the ledger reconciliation, folded in at close so the bundle is self-contained
    reconciliation: dict[str, Any] = dataclasses_field(default_factory=dict)
    #: what BOTH rings held and shed. A lost tail is a number, never a silence.
    caps: dict[str, Any] = dataclasses_field(default_factory=dict)
    #: the highest EXECUTOR-side sequence already held here, so the reply's ring fills the
    #: gap the lossy lane left instead of duplicating what it delivered.
    forwarded_seq: int = 0
    #: the opaque triage handle. Minted at the terminal, and the ONLY way to read the bundle.
    subject_id: str = ""
    #: The executor's formatted exception behind a FAILED terminal (bundle-only, cl-101).
    traceback: str = ""
    #: what the component-use plane actually did: stages, evictions, leases, resident set
    residency: dict[str, Any] = dataclasses_field(default_factory=dict)
    #: the lane's execution contract as ATTESTED by this attempt (cr-109): calls of the
    #: contracted entry against the pinned sites x forwards, and the profiled kernel symbol
    execution_contract: dict[str, Any] = dataclasses_field(default_factory=dict)
    execution_observation: dict[str, Any] = dataclasses_field(default_factory=dict)
    capture_result: dict[str, Any] = dataclasses_field(default_factory=dict)

    #: the record the plan was PRICED against, resolved once by the executor's kernel
    prepared: PreparedRequest | None = None
    #: cr-009's typed publication record, once the output transaction committed one
    publication: dict[str, Any] = dataclasses_field(default_factory=dict)
    #: Host-ACKed WeightsReceipt/1 refs, one per interface-declared output slot.
    weights_receipts: dict[str, pb.WeightsReceiptRef] = dataclasses_field(default_factory=dict)
    weights_work_fingerprint: str = ""
    job_models: dict[str, JobModel] = dataclasses_field(default_factory=dict)
    model_leases: list[Any] = dataclasses_field(default_factory=list)

    def begin_reclaim(self) -> bool:
        """Win the completion race, or report that natural finalization won first."""
        with self.transition_lock:
            if self.state in ("finalizing", "released", "outcome", "closed"):
                return False
            self.state = "reclaiming"
            return True

    def await_reclaim_before_finalizing(self) -> None:
        """A buffered reply cannot finalize across an unproved forceful reclaim."""
        with self.transition_lock:
            reclaiming = self.state == "reclaiming"
            if not reclaiming:
                self.state = "finalizing"
                return
        self.reclaim_done.wait()
        with self.transition_lock:
            self.state = "finalizing"

    def key(self) -> tuple[str, int]:
        return (self.request_id, self.attempt)

    def compact(self) -> None:
        """Keep the REPLAY IDENTITY; drop everything a closed attempt cannot need again.

        A closed attempt is answered by exactly two questions: the (ordinal, digest) fence,
        and a duplicate StartAttempt that must replay its terminal. Everything else — the
        spec, the payload, the grant, the plan, the hydrated inputs, the observation ring,
        the reconciliation — was retained for the life of the process because nothing ever
        released it. The terminal BYTES go too: they can carry a 4 MiB inline result and the
        process records already hold them, so `wrap` re-reads if a replay comes,
        adopted 2026-08-25).
        """
        self.spec = {}
        self.payload = {}
        self.opened = {}
        self.grant = grants.BoundGrant()
        self.plan = None
        self.prepared = None
        self.declared = None
        self.prepared_model = None
        self.executor = None
        self.slot = None
        self.job = None
        self.trees = {}
        self.inputs = {}
        self.outcome_bytes = b""
        self.ring = EventRing()
        self.attribution = {}
        self.confessions = []
        self.reconciliation = {}
        self.caps = {}
        self.residency = {}
        self.execution_contract = {}
        self.publication = {}
        self.weights_receipts = {}
        self.weights_work_fingerprint = ""
        self.job_models = {}
        self.model_leases = []

    def confess(self, kind: str, quantified: str, *, benchmark: bool = False) -> None:
        """Say out loud that this request was not served by its plain reading (§3.6)."""
        row = observe.Confession(kind, f"{self.request_id}#{self.attempt}", quantified, benchmark)
        self.confessions.append(
            {
                "kind": row.kind,
                "name": row.kind,
                "value": row.quantified,
                "benchmark_override": row.benchmark_override,
            }
        )
        row.emit_into(self.ring)


# --------------------------------------------------------------------------- the engine


class AttemptEngine:
    """One worker's attempts. Single device lease, so one execution at a time."""

    def __init__(
        self,
        *,
        records: WorkerRecords,
        authorizer: grants.Authorizer,
        jobs: Mapping[str, JobBinding],
        spool_root: Path,
        instance_id: str,
        send: Callable[[pb.WorkerFrame], None],
        progress: Callable[[str, int, dict[str, Any]], None],
        bundles: triage.BundleStore,
        monitor: observe.SilenceMonitor,
        posture: Callable[[], dict[str, Any]],
        device_process: Callable[[int], int],
        weights: WeightsExchange | None = None,
        owner_scope: Callable[[], str] = lambda: "",
        tensorfs_root: Path | None = None,
        work_fingerprint: Callable[[AttemptRecord], str] | None = None,
    ) -> None:
        self.records = records
        self.authorizer = authorizer
        self.jobs = dict(jobs)
        self.calls: Calls | None = None
        self.job_device_count = 0
        self.spool_root = spool_root
        self.instance_id = instance_id
        self.send = send
        self.progress = progress
        self.live: dict[str, AttemptRecord] = {}  # request_id -> the attempt held right now
        self.history: dict[tuple[str, int], AttemptRecord] = {}
        self.boot_id = ""
        self.executed = 0
        #: the worker's fault tail. BOUNDED: the Report and the triage bundle each read the
        #: last 8, and an unbounded list of the other 65,528 is memory nobody reads.
        self.faults: list[pb.Fault] = []
        self.bundles = bundles
        self.monitor = monitor
        #: the worker's applied posture, read at bundle time — a triage bundle that cannot
        #: say which directive was applied when it ran explains half of what went wrong.
        self.posture = posture
        #: the NVML per-process reader BY PID, so the out-of-allocator meter has a real
        #: producer for whichever lane's executor an attempt ran on
        self.device_process = device_process
        self.weights = weights
        self.owner_scope = owner_scope
        self.tensorfs_root = tensorfs_root
        self.work_fingerprint = work_fingerprint
        #: the demand falsifier. It ships first, it counts, and it decides nothing — the
        #: `timing-decides-nothing` fence proves the module it lives in cannot reach the
        #: planner, the ledger or the fill plane.
        self.demand = observe.DemandLedger()
        #: cr-009's DURABLE checkpoint ledger: (request_id, operation_key, logical_key) ->
        #: the recorded receipt. Identity replays; the same key with different bytes is a
        #: CONFLICT and never a replacement (§2).
        self.receipts: dict[tuple[str, str, str], dict[str, Any]] = {}

    def slot_of(self, attempt: AttemptRecord) -> AttemptSlot:
        """The placement slot bound at admission. An attempt with none is a defect: the
        session binds one before anything below `hold` runs, and nothing reads a slot
        after compaction."""
        if attempt.slot is None:
            raise RuntimeError(
                f"{attempt.request_id}#{attempt.attempt} has no placement slot bound"
            )
        return attempt.slot

    def supervision_for(self, attempt: AttemptRecord) -> ExecutorSupervision:
        """Return the exact placement slot captured before this attempt was accepted."""

        return self.slot_of(attempt).supervision

    def ledger_for(self, attempt: AttemptRecord) -> Ledger:
        return self.slot_of(attempt).ledger

    def fault(self, row: pb.Fault) -> None:
        """Record ONE fault, bounded. Readers take the last 8; the tail past that is memory
        nobody reads, and an unbounded one is how a long-lived worker grows without a leak
        anywhere (codex audit, adopted 2026-08-25)."""
        self.faults.append(row)
        del self.faults[:-FAULTS_HELD]

    # ------------------------------------------------------------------ offer

    def offer(self, offered: pb.AttemptOffer) -> AttemptRecord:
        """VALIDATING: fences and local safety, before any body field is trusted."""
        recomputed = documents.digest_of(offered.invocation_spec_canonical_bytes)
        if recomputed != offered.invocation_spec_digest:
            raise AttemptRefusal(
                "invocation_digest_mismatch",
                f"carried {offered.invocation_spec_digest.hex()[:16]}… but the resident bytes hash "
                f"to {recomputed.hex()[:16]}… — a digest never bypasses the lower check",
                cause=CAUSE.CAUSE_CODE_PROTOCOL,
            )
        try:
            spec = documents.read(offered.invocation_spec_canonical_bytes, pb.InvocationSpec)
        except documents.DocumentError as exc:
            raise AttemptRefusal(
                f"invocation_{exc.code}",
                f"the InvocationSpec document refuses: {exc.detail}",
                cause=CAUSE.CAUSE_CODE_PROTOCOL,
            ) from exc

        if spec.get("capture") is not None and spec.get("job") is not None:
            raise AttemptRefusal(
                "capture_mode",
                "activation capture requires an ordinary serving invocation",
                cause=CAUSE.CAUSE_CODE_INVALID_REQUEST,
            )

        # The (attempt, digest) fence binds the ORDINAL, not the current state: an attempt
        # is one execution of one spec whether it is running, recorded or already closed.
        known = self.history.get((offered.request_id, offered.attempt_ordinal))
        if known is not None and known.digest != offered.invocation_spec_digest:
            raise AttemptRefusal(
                "attempt_digest_conflict",
                f"attempt {offered.attempt_ordinal} of {offered.request_id} is bound to "
                f"{known.digest.hex()[:16]} (state {known.state}); a different spec under "
                "the same ordinal is unrepresentable, so neither spec runs",
                cause=CAUSE.CAUSE_CODE_PROTOCOL,
            )

        held = self.live.get(offered.request_id)
        if held is not None:
            if held.attempt == offered.attempt_ordinal:
                # a refresh swaps ACCESS; identity never moves (#439's structural half:
                # the bind refuses a grant whose subject or id sets disagree)
                held.grant = self._bind(spec, offered)
                return held  # idempotent: re-send AttemptAccepted, run once
            if held.state in ("queued", "accepted", "running", "finalizing"):
                raise AttemptRefusal(
                    "live_attempt_supersession",
                    f"attempt {offered.attempt_ordinal} arrived while attempt "
                    f"{held.attempt} is live with no recorded outcome: supersession is "
                    "explicit (CancelAttempt(SUPERSEDED) -> recorded outcome -> ordinal+1), never "
                    "the arrival of a successor",
                    cause=CAUSE.CAUSE_CODE_PROTOCOL,
                )
        previous = self.history.get((offered.request_id, offered.attempt_ordinal))
        if previous is not None and previous.state in ("outcome", "closed"):
            previous.replays += 1
            return previous  # duplicate StartAttempt for a finished attempt: replay, no run

        return AttemptRecord(
            request_id=offered.request_id,
            attempt=offered.attempt_ordinal,
            digest=offered.invocation_spec_digest,
            spec=spec,
            grant=self._bind(spec, offered),
            # The stream fence already verified this routing identity before `offer()`.
            # Capture the offered value itself; rereading the worker's live placement during
            # admission can observe a concurrently converging successor placement.
            placement_id=offered.placement_id,
            admission_epoch=offered.admission_epoch,
        )

    def _bind(self, spec: Mapping[str, Any], offered: pb.AttemptOffer) -> grants.BoundGrant:
        try:
            return grants.bind(spec, offered.grant, offered.invocation_spec_digest)
        except grants.GrantRefusal as exc:
            raise AttemptRefusal(exc.code, exc.detail, cause=CAUSE.CAUSE_CODE_PROTOCOL) from exc

    def hold(self, attempt: AttemptRecord) -> None:
        """RECORD the attempt on the READER, before the device lane can reach it.

        Registering here and not at acceptance is what keeps the idempotency fences honest
        once acceptance moved off this thread: a second StartAttempt arriving while the
        first is still queued must find it, and a successor ordinal must see a live attempt
        with no recorded terminal and refuse supersession.
        """
        attempt.state = "queued"
        attempt.queued_at = time.perf_counter()
        self.live[attempt.request_id] = attempt
        self.history[attempt.key()] = attempt

    def refuse_held(
        self,
        attempt: AttemptRecord,
        code: str,
        detail: str,
        cause: pb.CauseCode,
        origin: pb.CauseOrigin = ORIGIN.CAUSE_ORIGIN_WORKER,
    ) -> pb.AttemptOutcome:
        """A REFUSED terminal for an attempt this worker is already holding."""
        return self.outcome(
            attempt,
            STATUS.OUTCOME_STATUS_REFUSED,
            cause,
            origin,
            f"{code}: {detail}",
        )

    # ------------------------------------------------------------------ prepare/accept

    def _fetch_payload(self, attempt: AttemptRecord) -> None:
        """Fetch, verify and decode the granted payload. Shared by BOTH modes.

        It happens at ACCEPTANCE and not at execution time on purpose: a missing, expired or
        malformed payload must be a refusal the RecordOwner can read, not an accepted attempt
        that later dies on a thread with no terminal to show for it.
        """
        try:
            attempt.payload = self._payload(attempt)
        except grants.GrantRefusal as exc:
            raise AttemptRefusal(
                exc.code,
                exc.detail,
                cause=CAUSE.CAUSE_CODE_GRANT_EXPIRED
                if exc.code in ("grant_expired", "input_absent")
                else CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            ) from exc
        except OSError as exc:
            # The ADDRESS was minted by the RecordOwner, so a miss here is PLATFORM-FATAL,
            # not a bad request (v1 REF_ORIGIN). This used to fall into the branch below and
            # report a missing file as "not a JSON object" — the wrong system, the wrong
            # person, and a sentence with no remedy in it.
            classified = refusal.classify(
                "payload_unreadable",
                f"the granted payload location could not be read ({type(exc).__name__}); the "
                "RecordOwner minted this address, so this worker cannot substitute another",
                origin=refusal.PLATFORM,
            )
            code, cause_origin = classified.cause()
            raise AttemptRefusal(
                f"{classified.code}[{classified.severity()}]",
                classified.sentence(),
                cause=code,
                origin=cause_origin,
            ) from exc
        except ValueError as exc:
            # The BYTES are the caller's, so this one is the caller's typed error.
            classified = refusal.classify(
                "payload_undecodable",
                f"the granted payload input is not a JSON object: {exc}",
                origin=refusal.CALLER,
            )
            code, cause_origin = classified.cause()
            raise AttemptRefusal(
                f"{classified.code}[{classified.severity()}]",
                classified.sentence(),
                cause=code,
                origin=cause_origin,
            ) from exc
        self._check_control_size(attempt)

    @staticmethod
    def _check_control_size(attempt: AttemptRecord) -> None:
        """The payload must FIT THE CONTROL SEAM, checked BEFORE durable acceptance.

        The seam carries control, not data: one frame is capped at 64 KiB precisely so that
        "a caller trying to move tensors or media through here fails immediately". It did not
        fail immediately. Measured by ev-003 (decisions #302): a 73,721 B request is accepted,
        the dispatch frame then raises a worker-side `SeamError` AFTER acceptance, and the
        run HANGS — `LocalRecordOwner` only deadlines a run that never dispatched. A 42,719 B
        request on the same path succeeds.

        That violates the acceptance-boundary law directly: everything before acceptance
        either completes or raises a typed PRE-ACCEPT refusal. So the size is a condition of
        acceptance, it is the CALLER's typed error (the bytes are theirs), and the refusal
        names both the cap and the remedy — media rides as ASSETS, hydrated into the attempt
        spool, which is the shape the author surface already has.
        """
        size = len(canonical.write(dict(attempt.payload)))
        if size <= PAYLOAD_MAX_BYTES:
            return
        raise AttemptRefusal(
            "payload_over_control_cap",
            f"the request document canonicalizes to {size} B and the worker/executor "
            f"control seam admits {PAYLOAD_MAX_BYTES} B per payload (one {seam.MAX_FRAME} B "
            "frame, less this attempt's own command envelope). The seam carries CONTROL: "
            "media and tensors ride as typed Assets, which the runtime hydrates into the "
            "attempt spool and the handler reads from there",
            cause=CAUSE.CAUSE_CODE_INVALID_REQUEST,
            origin=ORIGIN.CAUSE_ORIGIN_CLIENT,
        )

    def _prepare_request(
        self, attempt: AttemptRecord, executor: Any, declared: DeclaredBinding
    ) -> PreparedRequest:
        """Resolve the request in the executor, and hold what a plan can be priced against.

        This is the pre-admission half of the invocation kernel, run across the seam before
        anything is accepted. The worker cannot run it itself — resolving needs the
        package's declared types and this process never imports package code — and it must
        not be skipped: a plan priced against a projection of the wire document is a plan
        chosen against a request nobody will run (codex audit, adopted 2026-08-25).

        A refusal here is PRE-ENTRY in the strongest sense. Nothing was accepted, no device
        was touched, and the executor is exactly as it was.
        """
        try:
            reply = executor.call(
                {
                    "cmd": "prepare_request",
                    "request_id": f"{attempt.request_id}#{attempt.attempt}",
                    "entrypoint": declared.entrypoint,
                    "payload": attempt.payload,
                    **({"capture": attempt.spec["capture"]} if "capture" in attempt.spec else {}),
                    "input_metadata": {
                        name: {
                            "input_id": name,
                            "media_type": item.kind_mime,
                            "digest": documents.spell(item.digest),
                            "length": item.length,
                            "order": item.order,
                        }
                        for name, item in attempt.grant.inputs.items()
                        if name != "payload"
                    },
                },
                timeout=None,
            )
        except ExecutorGone as exc:
            raise AttemptRefusal(
                "executor_absent",
                f"the executor died while resolving this request: {exc}",
                cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
            ) from exc
        if not reply.get("ok"):
            code = str(reply.get("code", "request_unresolvable"))
            if code == "poisoned_generation":
                # The poison can live only executor-side (a cancel's outcome poisons the
                # CHILD; the cancel-completion path does not absorb replies). The refusal
                # that names it is the supervision's one signal — invalidate HERE so the
                # lane's rebuild pass replaces the generation instead of refusing forever
                # (lifecycle-live §4 caught the wedge).
                self.supervision_for(attempt).invalidate(
                    executor, str(reply.get("detail", "poisoned"))[:256]
                )
            raise AttemptRefusal(
                code,
                str(reply.get("detail", ""))[:1024],
                cause=CAUSE.CAUSE_CODE_INVALID_REQUEST
                if str(reply.get("origin")) in ("request", "author")
                else CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                origin=_ORIGIN.get(str(reply.get("origin")), ORIGIN.CAUSE_ORIGIN_EXECUTOR),
            )
        return PreparedRequest(
            request_id=f"{attempt.request_id}#{attempt.attempt}",
            entrypoint=declared.entrypoint,
            features={str(k): int(v) for k, v in (reply.get("features") or {}).items()},
            features_digest=str(reply.get("features_digest", "")),
            overlay_digest=str(reply.get("overlay_digest", "")),
            facts_digest=str(reply.get("facts_digest", "")),
            adjustments=tuple(dict(row) for row in reply.get("adjustments", ())),
        )

    def _hydrate_inputs(self, attempt: AttemptRecord) -> None:
        """Materialize every granted input asset into THIS attempt's spool (cr-012).

        The spool is created here rather than in `_run` because that is what "before
        acceptance" means for bytes: the attempt must own a directory before anything is
        written into it, and an attempt refused during hydration leaves a spool with
        nothing in it rather than an accepted attempt with nowhere to put its inputs.
        """
        spool = self.spool_root / opaque_key(
            "attempt-input-spool", attempt.request_id, attempt.attempt
        )
        spool.mkdir(parents=True, exist_ok=True)
        attempt.spool = spool
        binding = attempt.declared
        try:
            attempt.inputs = grants.hydrate_inputs(
                attempt.grant,
                self.authorizer,
                spool=spool,
                max_bytes=binding.max_input_bytes if binding else 64 << 20,
                max_total_bytes=binding.max_inputs_total_bytes if binding else 256 << 20,
                workspace=Workspace(self.tensorfs_root) if self.tensorfs_root is not None else None,
                owner=self.owner_scope(),
            )
            supervision = self.supervision_for(attempt)
            if getattr(supervision, "isolated", False):
                supervision.grant_directory(spool, attempt.executor_epoch)
        except grants.GrantRefusal as exc:
            # WHOSE fault decides the terminal. A cap, a media-type disagreement or an
            # absent grant entry are facts about the REQUEST; an expired grant or an
            # unreachable RecordOwner-minted address are facts about the platform, and
            # rendering the two the same way sends the wrong person to the wrong question.
            caller = exc.code in (
                "input_over_cap",
                "inputs_over_total_cap",
                "input_media_type",
                "input_digest_mismatch",
                "stream_over_declared_size",
                "stream_under_declared_size",
            )
            raise AttemptRefusal(
                exc.code,
                exc.detail,
                cause=CAUSE.CAUSE_CODE_INVALID_REQUEST
                if caller
                else CAUSE.CAUSE_CODE_GRANT_EXPIRED
                if exc.code == "grant_expired"
                else CAUSE.CAUSE_CODE_LOCAL_SAFETY,
                origin=ORIGIN.CAUSE_ORIGIN_CLIENT if caller else ORIGIN.CAUSE_ORIGIN_WORKER,
            ) from exc

    def _job_declaration(self, binding: JobBinding, descriptor_id: str) -> Mapping[str, Any]:
        """Read the exact installed package interface without importing package code."""
        try:
            interface_path = Path(binding.package_interface)
            interface_stat = interface_path.lstat()
            if (
                not stat.S_ISREG(interface_stat.st_mode)
                or not 0 < interface_stat.st_size <= canonical.DOC_MAX_BYTES
            ):
                raise ValueError("installed package interface is not one bounded regular file")
            raw = interface_path.read_bytes()
            body = package_interface.read_bytes(raw, "installed job package interface")
            if package_interface.job_descriptor_id(body, binding.job) != descriptor_id:
                raise ValueError("job_descriptor_id does not bind the installed callable")
            if str(body.get("application", "")) != binding.application:
                raise ValueError("installed package-interface application changed")
            jobs = body.get("jobs")
            rows = (
                [row for row in jobs if isinstance(row, Mapping) and row.get("name") == binding.job]
                if isinstance(jobs, list)
                else []
            )
            if len(rows) != 1:
                raise ValueError("installed package interface has no unique selected job")
            return rows[0]
        except AttemptRefusal:
            raise
        except Exception as exc:
            raise AttemptRefusal(
                "job_model_interface_invalid",
                f"the installed job interface cannot bind Model parameters ({type(exc).__name__})",
                cause=CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            ) from exc

    def _job_model_declarations(self, binding: JobBinding, descriptor_id: str) -> dict[str, str]:
        try:
            models = self._job_declaration(binding, descriptor_id).get("models", [])
            if not isinstance(models, list):
                raise ValueError("selected job models are not an array")
            declared: dict[str, str] = {}
            prefix = f"{binding.job}.models."
            for row in models:
                if not isinstance(row, Mapping):
                    raise ValueError("selected job model row is not an object")
                model_path = str(row.get("path", ""))
                parameter = model_path.removeprefix(prefix)
                class_name = str(row.get("class", ""))
                if (
                    not model_path.startswith(prefix)
                    or not parameter.isidentifier()
                    or parameter in declared
                    or not class_name
                ):
                    raise ValueError("selected job model parameter/class is malformed")
                declared[parameter] = class_name
            return declared
        except AttemptRefusal:
            raise
        except Exception as exc:
            raise AttemptRefusal(
                "job_model_interface_invalid",
                f"the installed job interface cannot bind Model parameters ({type(exc).__name__})",
                cause=CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            ) from exc

    def _hold_job_models(
        self, attempt: AttemptRecord, binding: JobBinding, descriptor_id: str
    ) -> None:
        """Join interface params to grants and hold each CozyTensors closure until terminal."""
        declared = self._job_model_declarations(binding, descriptor_id)
        selected = grants.model_inputs(attempt.grant)
        if set(selected) != set(declared):
            raise AttemptRefusal(
                "job_model_binding_mismatch",
                f"job interface declares Model parameters {sorted(declared) or 'none'} and "
                f"InvocationSpec carries {sorted(selected) or 'none'}; the sets must match exactly",
                cause=CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
        if not declared:
            return
        if self.tensorfs_root is None or not self.tensorfs_root.is_dir():
            raise AttemptRefusal(
                "job_model_store_unavailable",
                "this job declares Model parameters and the worker has no existing TensorFS store",
                cause=CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
        leases: dict[str, Any] = {}
        models: dict[str, JobModel] = {}
        try:
            tensorfs = fill.tensorfs_module()
            store = tensorfs.Store.open(str(self.tensorfs_root))
            for parameter in sorted(declared):
                entry = selected[parameter]
                manifest = documents.spell(entry.digest)
                held = store.manifest(manifest)
                manifest_bytes = bytes(held["manifest"])
                if len(manifest_bytes) != entry.length:
                    raise ValueError(
                        f"Model parameter {parameter!r} declares {entry.length} B but the "
                        f"local Manifest is {len(manifest_bytes)} B"
                    )
                if manifest not in leases:
                    leases[manifest] = store.acquire_cozytensors(manifest)
                models[parameter] = JobModel(
                    parameter=parameter,
                    class_name=declared[parameter],
                    manifest=manifest,
                    length=entry.length,
                )
        except Exception as exc:
            for lease in leases.values():
                lease.release()
            raise AttemptRefusal(
                "job_model_manifest_unavailable",
                "a selected job Model Manifest is absent, stale, or invalid "
                f"({type(exc).__name__})",
                cause=CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            ) from exc
        attempt.job_models = models
        attempt.model_leases = list(leases.values())

    def _release_job_models(self, attempt: AttemptRecord) -> None:
        leases, attempt.model_leases = attempt.model_leases, []
        for lease in leases:
            try:
                lease.release()
            except Exception as exc:
                attempt.ring.emit(
                    "fault",
                    "job Model Manifest lease release failed",
                    type(exc).__name__,
                )

    def _accept_job(self, attempt: AttemptRecord) -> pb.AttemptAccepted:
        """BOUNDED ACCEPT-THEN-PREPARE — the ONE mode difference from serving (cr-009).

        Serving never accepts-then-cold-loads: a warm generation is the whole point of a
        serving worker, so an attempt is accepted only against a Ready executor. A job is the
        opposite shape — one immutable build, one bounded attempt, terminal, reclaim — and
        there is nothing to be warm FOR. Preparing before acceptance would only hold a rented
        pod idle while the RecordOwner waits to hear whether its work was taken.

        What does NOT change: acceptance is recorded BEFORE anything runs, the payload is
        fetched and decoded here, the input trees are resolved under the grant here, and the
        envelope is reserved here. The preparation this admits is bounded by the attempt's own
        deadline, which the worker owns.
        """
        job = attempt.spec["job"]
        descriptor_id = str(job["job_descriptor_id"])
        binding = self.jobs.get(descriptor_id)
        if binding is None:
            raise AttemptRefusal(
                "unknown_job_descriptor",
                f"no job descriptor {descriptor_id[:24]}… is bound on this worker (it has "
                f"{len(self.jobs)} job plan(s))",
                cause=CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
        attempt.job = binding
        attempt.kind = "job"
        attempt.state = "preparing"
        slot = self.slot_of(attempt)
        supervision = slot.supervision
        executor = supervision.current
        if executor is None or not executor.alive():
            raise AttemptRefusal(
                "executor_absent",
                "no live executor: a job is accepted only against the fresh executor "
                "created for its directive",
                cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
            )
        attempt.executor = executor
        attempt.executor_epoch = executor.epoch
        attempt.executor_pid = executor.pid
        attempt.placement_id = ""  # job attempts are not placement-routed
        self._fetch_payload(attempt)
        self._hydrate_inputs(attempt)
        try:
            attempt.trees = grants.read_trees(
                attempt.grant,
                self.authorizer,
                workspace=Workspace(self.tensorfs_root) if self.tensorfs_root is not None else None,
                owner=self.owner_scope(),
                spool=attempt.spool,
            )
        except grants.GrantRefusal as exc:
            raise AttemptRefusal(
                exc.code,
                exc.detail,
                cause=CAUSE.CAUSE_CODE_GRANT_EXPIRED
                if exc.code == "grant_expired"
                else CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            ) from exc

        caps = attempt.spec.get("resource_caps") or {}
        plan = AttemptPlan.for_job(
            binding,
            {
                "vram": int(caps.get("max_vram_bytes", 0)),
                "rss": int(caps.get("max_rss_bytes", 0)),
                "disk": int(caps.get("max_disk_bytes", 0)),
            },
            device_count=slot.job_device_count,
        )
        attempt.plan = plan
        # A job's scratch is the RUN's, granted by the owner as `tmp/<request-id>/` and
        # removed by the owner when the request settles; the worker owns no scratch tree of
        # its own. Until a grant names one, `attempt.scratch` stays None and the Scratch
        # service refuses typed (`scratch_unavailable`).
        attempt.scratch = None

        opened = slot.ledger.snapshot()
        self._hold_job_models(attempt, binding, descriptor_id)
        attempt.opened = opened

        plan_digest = plan.digest()
        try:
            with supervision.hold(executor) as owned:
                if not owned:
                    raise AttemptRefusal(
                        "executor_replaced_before_acceptance",
                        "the job executor changed before its acceptance could be committed",
                        cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                    )
                self.records.append(
                    "accepted",
                    {
                        "request_id": attempt.request_id,
                        "attempt": attempt.attempt,
                        "worker_boot_id": self.boot_id,
                        "instance_id": self.instance_id,
                        "attempt_kind": "job",
                        "invocation_spec_digest": documents.spell(attempt.digest),
                        "plan_digest": plan_digest,
                        "job_descriptor_id": descriptor_id,
                        "build_id": binding.build_id,
                        "grant_epoch": attempt.grant.credential_epoch,
                        "executor_epoch": attempt.executor_epoch,
                        "executor_pid": attempt.executor_pid,
                        "placement_id": attempt.placement_id,
                        "input_trees": ",".join(sorted(attempt.trees)),
                    },
                )
        except Exception:
            self._release_job_models(attempt)
            raise
        return self._accepted(attempt)

    def prepare_and_accept(self, attempt: AttemptRecord) -> pb.AttemptAccepted:
        """The ATOMIC ACCEPTANCE BOUNDARY, in exactly this order:

            resolve binding -> fetch/verify/decode the payload -> derive REAL features from
            it -> choose the plan -> prepare the executor -> reserve the envelope -> RECORD
            -> publish AttemptAccepted.

        Everything before the process record either completes or raises a typed pre-accept
        refusal. The payload is fetched here and not at execution time: a missing, expired
        or malformed payload
        must be a refusal the RecordOwner can read, not an accepted attempt that later dies
        on a thread with no terminal to show for it.

        Features are derived from the DECODED PAYLOAD. Deriving them from the plan id would
        be inventing the very input the plan is supposed to be chosen against.
        """
        if attempt.spec.get("job") is not None:
            return self._accept_job(attempt)
        serving = attempt.spec.get("serving")
        if serving is None:
            raise AttemptRefusal(
                "execution_mode_unspecified",
                "an ExecutionSpec names exactly ONE mode — serving or job — and this one "
                "names neither; a worker that guessed would run the wrong lifecycle",
                cause=CAUSE.CAUSE_CODE_PROTOCOL,
            )
        plan_id = serving["entrypoint_binding_digest"]
        slot = self.slot_of(attempt)
        declared = slot.bindings.get(plan_id)
        prepared_model = slot.prepared_models.get(plan_id)
        if declared is None or prepared_model is None:
            raise AttemptRefusal(
                "unknown_binding",
                f"no entrypoint binding {plan_id[:24]}… is selected on this worker",
                cause=CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
        if declared.development and attempt.spec.get("environment_digest"):
            raise AttemptRefusal(
                "development_environment_present",
                "development execution has no published Environment identity",
                cause=CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
        if not declared.development and (
            attempt.spec.get("environment_digest") != declared.environment_digest
        ):
            raise AttemptRefusal(
                "environment_mismatch",
                "InvocationSpec does not name the selected Environment",
                cause=CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
        attempt.declared = declared
        attempt.prepared_model = prepared_model
        attempt.state = "preparing"

        self._fetch_payload(attempt)

        # The ExecutorPreparer: when the selected construction is not already ready, prepare
        # it BEFORE durable acceptance. Serving never accepts-then-cold-loads. It is checked
        # HERE, ahead of the request preparation below, because that preparation runs in the
        # executor and there is no sense resolving a request against one that cannot serve.
        supervision = slot.supervision
        executor = supervision.current
        if executor is None or not executor.alive():
            raise AttemptRefusal(
                "executor_absent",
                "no live executor: an online attempt is accepted only against a Ready one",
                cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
            )
        if plan_id not in executor.ready_bindings:
            # A binding that FAILED to prepare says so in its own words (#572d). "the
            # executor is ready for nothing" is true of a faulted binding and tells the
            # caller nothing about why THIS one cannot serve, while its siblings can.
            fault = slot.failed_bindings.get(plan_id)
            if fault is not None:
                raise AttemptRefusal(
                    "binding_faulted",
                    f"{plan_id[:24]}… could not be prepared and this placement serves its "
                    f"other bindings without it: {fault}",
                    cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                )
            raise AttemptRefusal(
                "executor_not_ready",
                f"the executor is ready for {sorted(executor.ready_bindings) or 'nothing'}, "
                f"not {plan_id[:24]}… — the directive prepares an executor before an "
                "attempt may be accepted against it, and serving never accepts-then-"
                "cold-loads",
                cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
            )
        if executor.poisoned:
            raise AttemptRefusal(
                "poisoned_generation",
                f"the executor generation is poisoned ({executor.poisoned}) and has no "
                "recovery edge to Ready",
                cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
            )
        attempt.executor = executor
        attempt.executor_epoch = executor.epoch
        attempt.executor_pid = executor.pid

        prepared = self._prepare_request(attempt, executor, declared)
        attempt.prepared = prepared
        try:
            plan = slot.chooser.choose(declared, prepared_model, prepared)
        except PlanRefusal as exc:
            raise AttemptRefusal(
                exc.code, exc.detail, cause=CAUSE.CAUSE_CODE_CONSTRAINT_INFEASIBLE
            ) from exc
        attempt.plan = plan

        # HYDRATE, after the plan and before acceptance (cr-012, §1.3's launch contract:
        # validate -> plan -> hydrate -> prepare -> accept -> invoke). After the plan,
        # because a request that has no plan should download nothing; before acceptance,
        # because an oversized, ungranted, mislabelled or unreachable input must be a
        # refusal the RecordOwner can read rather than an accepted attempt that dies on a
        # thread. And before the lease, which is the ordering that matters most: a slow URL
        # fetched under a lease would pin staged weights for its whole duration.
        self._hydrate_inputs(attempt)

        attempt.opened = slot.ledger.snapshot()

        plan_digest = plan.digest()
        with supervision.hold(executor) as owned:
            if not owned:
                raise AttemptRefusal(
                    "executor_replaced_before_acceptance",
                    "the prepared executor changed before acceptance could be committed",
                    cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                )
            self.records.append(
                "accepted",
                {
                    "request_id": attempt.request_id,
                    "attempt": attempt.attempt,
                    "worker_boot_id": self.boot_id,
                    "instance_id": self.instance_id,
                    "invocation_spec_digest": documents.spell(attempt.digest),
                    "plan_digest": plan_digest,
                    "constructed_model_digest": plan.constructed_model_digest,
                    "request_features_digest": plan.request_features_digest,
                    "preflight_facts_digest": plan.preflight_facts_digest,
                    "grant_epoch": attempt.grant.credential_epoch,
                    "placement_id": attempt.placement_id,
                    "executor_epoch": attempt.executor_epoch,
                    "executor_pid": attempt.executor_pid,
                    "entrypoint_binding_digest": plan_id,
                },
            )
        return self._accepted(attempt)

    # ------------------------------------------------------------------ execute

    def execute(self, attempt: AttemptRecord) -> pb.AttemptOutcome | Released:
        """Run ONE accepted attempt through its DEVICE PHASE.

        Answers a terminal when nothing reached the device or the executor died under it,
        and a `Released` record otherwise: the device is free the moment this returns, and
        `finish` — the post phase, on the lane's post thread — turns the record into the
        terminal. The ledger closes HERE, at release, while the executor that ran the
        attempt is still the one on the lane; the triage bundle the terminal REFERENCES
        carries that reconciliation."""
        if attempt.canceling:
            # Cancelled between arrival and dispatch — while it queued, or while admission
            # was fetching and hydrating it. Nothing reached the device, so the caller's
            # cancel is complete the moment this terminal is recorded, and spending the
            # lease on an answer nobody is waiting for would be the opposite of honouring it.
            return self.finish_cancel(attempt, attempt.canceling)
        return self._run(attempt)

    def reconcile(self, attempt: AttemptRecord) -> dict[str, Any]:
        """Every byte class, before and after, with the executor's after-state re-read.

        The device numbers come from the process that owns the CUDA context, so a dead
        executor makes them UNREADABLE rather than zero (§3.6). That distinction is the
        whole point on the kill path: "0 B allocated" would be a claim about a device this
        worker can no longer see, while "unreadable, the context died with the process" is
        what actually happened — and the bytes really are gone, because the kernel reclaims
        a dead process's device context, which `nvidia-smi` confirms from outside.
        """
        key = f"{attempt.request_id}#{attempt.attempt}"
        executor = attempt.executor
        if executor is None:
            return self._unavailable_reconciliation(attempt, key)
        # Keep the generation stable across the probe AND close. Checking ownership and then
        # mutating the ledger without this hold lets convergence publish E2 between those two
        # operations; E1's terminal would then mark E2 unreadable or reconcile against E2's
        # freshly installed baseline.
        supervision = self.supervision_for(attempt)
        ledger = self.ledger_for(attempt)
        with supervision.hold(executor) as owned:
            if not owned:
                return self._unavailable_reconciliation(attempt, key)
            if not executor.poisoned and attempt.spool:
                try:
                    with executor.watched("probe"):
                        ledger.observe_probe(executor.call({"cmd": "probe"}, timeout=None))
                    ledger.device_process = self.device_process(executor.pid)
                except ExecutorGone:
                    ledger.device_unreadable("the executor died before the ledger was read")
            else:
                ledger.device_unreadable("the executor died before the ledger was read")
            record = ledger.close_attempt(key, attempt.opened)
            attempt.reconciliation = record
            self.records.append("ledger", record)
        if record["leaks"]:
            self.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_LOCAL_SAFETY_REFUSAL,
                    subject=key,
                    reason="ledger_unreconciled",
                    detail=safe(
                        "the attempt reached a terminal without returning every class to "
                        "its opening value, beyond the residency transition it declared: "
                        + "; ".join(record["leaks"])
                    ),
                )
            )
            # AND THE GENERATION GOES. An unreconciled ledger is a worker whose model of its
            # own device is provably wrong, and §3.2 already says what re-establishes one:
            # a new executor generation, never a repair of the old numbers. Until the rebuild
            # lands the chooser refuses typed rather than pricing a rung against drift —
            # which is the difference between a typed answer before acceptance and an OOM
            # halfway through somebody's VAE decode (cl-003).
            supervision.invalidate(executor, f"ledger_unreconciled: {record['leaks'][0][:160]}")
        return record

    def _unavailable_reconciliation(self, attempt: AttemptRecord, key: str) -> dict[str, Any]:
        """Record an old generation's unavailable close without touching its successor.

        Host/pinned grants were already released by `outcome()`. Device facts, baselines,
        unreadable markers and reconciliation counters are generation-scoped, so once this
        attempt no longer owns `current` they must remain exactly as the successor reported
        them. The attempt still gets a durable, explicit reconciliation saying why no after
        measurement exists.
        """
        reason = (
            f"executor epoch {attempt.executor_epoch} pid {attempt.executor_pid} "
            "was replaced before reconciliation; successor telemetry was left untouched"
        )
        record = {
            "attempt": key,
            "executor_epoch": attempt.executor_epoch,
            "executor_pid": attempt.executor_pid,
            "classes": [
                {
                    "class": name,
                    "bytes": -1,
                    "kind": "unreadable",
                    "method": reason,
                    "before": before,
                    "after": -1,
                    "reconciled": False,
                    "verdict": "executor_epoch_replaced",
                }
                for name, before in attempt.opened.items()
            ],
            "leaks": [],
            "baseline_shifts": [],
            "declared_residency_delta": int(attempt.residency.get("allocator_delta_bytes", 0)),
            "unreadable": [reason],
            "closed": False,
        }
        attempt.reconciliation = record
        self.records.append("ledger", record)
        return record

    # ------------------------------------------------------------------ the job lane

    def checkpoint_exchange(
        self, attempt: AttemptRecord, frame: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Answer one checkpoint declaration request from the executor.

        Identity is (run, attempt, operation key, logical key, content digest). The row is
        recorded before this
        returns.

        WHAT IS DURABLE HERE IS THE DECLARATION, NOT THE BYTES (#553a). This handler is where
        the false promise was measurable: it records a metadata row and sends a
        `JobCheckpointRequest` whose `artifact` field it never fills, so no component on
        either side ever copies the checkpoint. `Checkpoints.save()` used to report success
        on this answer; it now refuses `checkpoint_sink_unbuilt`, and `Checkpoints.declare()`
        is what reaches here. The row keeps landing because it is TRUE — an attempt declared
        a checkpoint with this digest and this length — and it is what the WeightsSink will
        attach real object references to when #552's transaction lands.

        Three answers and no fourth:
          * a NEW identity is recorded and gets a fresh declaration id;
          * the SAME identity replays it and writes nothing a second time;
          * the same key with DIFFERENT bytes is a CONFLICT, never a replacement.
        """
        if str(frame.get("kind")) != "checkpoint":
            return {
                "ok": False,
                "code": "unknown_durable_request",
                "detail": f"{frame.get('kind')!r} is not a durable exchange this worker answers",
            }
        if attempt.job is None:
            return {
                "ok": False,
                "code": pb.CheckpointFaultCode.Name(
                    pb.CheckpointFaultCode.CHECKPOINT_FAULT_CODE_NOT_JOB_MODE
                ),
                "detail": "checkpoints are a JOB service; this attempt is not a job",
            }
        operation_key = str(frame.get("operation_key", ""))
        logical_key = str(frame.get("logical_key", ""))
        content_digest = str(frame.get("content_digest", ""))
        key = (attempt.request_id, operation_key, logical_key)
        held = self.receipts.get(key)
        if held is not None:
            if held["content_digest"] != content_digest:
                return {
                    "ok": False,
                    "code": pb.CheckpointFaultCode.Name(
                        pb.CheckpointFaultCode.CHECKPOINT_FAULT_CODE_IDENTITY_CONFLICT
                    ),
                    "detail": (
                        f"{logical_key}: operation key {operation_key!r} is already durable at "
                        f"{held['content_digest'][:23]}… and these bytes digest to "
                        f"{content_digest[:23]}… — one identity is one content, and a "
                        "checkpoint is never replaced"
                    ),
                    "recorded_content_digest": held["content_digest"],
                }
            return {"ok": True, "receipt_id": held["receipt_id"], "replayed": True}

        receipt_id = f"ckpt-{uuid.uuid4().hex[:24]}"
        record = {
            "request_id": attempt.request_id,
            "attempt": attempt.attempt,
            "worker_boot_id": self.boot_id,
            "operation_key": operation_key,
            "logical_key": logical_key,
            "content_digest": content_digest,
            "length": int(frame.get("length", 0)),
            "receipt_id": receipt_id,
        }
        self.records.append("job_checkpoint", record)
        self.receipts[key] = record
        # The RecordOwner's copy rides the DURABLE control lane, after the worker's own
        # record, at-least-once like every other control message. `artifact` stays
        # UNSET, and that is the honest state of this frame: no WeightsSink exists to fill
        # it, so the RecordOwner is being told an attempt DECLARED a checkpoint and nothing
        # more (#552/#553a).
        # The envelope is stamped centrally by the worker's `send` (#454: three fields, one
        # stamp — no constructor can forget one).
        self.send(
            pb.WorkerFrame(
                checkpoint_request=pb.JobCheckpointRequest(
                    request_id=attempt.request_id,
                    attempt_ordinal=attempt.attempt,
                    operation_key=operation_key,
                    logical_key=logical_key,
                    content_digest=documents.raw(content_digest),
                    seq=len(self.receipts),
                )
            )
        )
        attempt.ring.emit(
            "log",
            f"checkpoint {logical_key!r} DECLARED (bytes not stored: no weights sink)",
            "info",
            declaration=receipt_id,
            bytes=int(frame.get("length", 0)),
        )
        return {"ok": True, "receipt_id": receipt_id, "replayed": False}

    def durable_exchange(self, attempt: AttemptRecord, frame: Mapping[str, Any]) -> dict[str, Any]:
        """Dispatch the existing executor control request lane by typed operation."""
        kind = str(frame.get("kind", ""))
        if kind in ("child_call", "child_poll", "child_cancel"):
            if self.calls is None:
                return {
                    "ok": False,
                    "code": "child_broker_absent",
                    "detail": "no RecordOwner call lane",
                }
            return self.calls.handle(attempt, frame)
        if kind == "checkpoint":
            return self.checkpoint_exchange(attempt, frame)
        if self.weights is None or attempt.spool is None:
            return {
                "ok": False,
                "code": "weights_host_unavailable",
                "detail": "this worker has no WeightsSink host",
            }
        try:
            if kind == "weights_writer":
                return self.weights.writer_broker.handle(attempt, frame)
            output_slot = str(frame.get("output_slot", ""))
            if kind == "weights_intent":
                if attempt.job is None:
                    raise ValueError("a weights intent requires the current job")
                declared = self._job_declaration(attempt.job, attempt.job.job_descriptor_id)
                outputs = {
                    str(row["output_id"]): row for row in declared.get("weights_outputs", [])
                }
                bound = attempt.grant.outputs.get(output_slot)
                if (
                    output_slot not in outputs
                    or bound is None
                    or (
                        bound.max_bytes != outputs[output_slot]["max_bytes"]
                        or bound.mime_type != outputs[output_slot]["mime_type"]
                    )
                ):
                    raise ValueError(
                        "weights intent is outside the package and invocation output contract"
                    )
                length = int(frame.get("tensorfs_declaration_length", 0))
                raw = weights_exchange_bytes(attempt.spool, output_slot, "declaration", length)
                return self.weights.intent(
                    attempt,
                    output_slot=output_slot,
                    transaction_id=str(frame.get("weights_transaction_id", "")),
                    declaration_digest=documents.raw(
                        str(frame.get("tensorfs_declaration_digest", ""))
                    ),
                    declaration=raw,
                    requested_writer_epoch=int(frame.get("requested_writer_epoch", 0)),
                )
            if kind == "weights_receipt":
                length = int(frame.get("weights_receipt_length", 0))
                raw = weights_exchange_bytes(attempt.spool, output_slot, "receipt", length)
                return self.weights.receipt(
                    attempt,
                    output_slot=output_slot,
                    transaction_id=str(frame.get("weights_transaction_id", "")),
                    receipt_digest=documents.raw(str(frame.get("weights_receipt_digest", ""))),
                    canonical_receipt=raw,
                )
            return {
                "ok": False,
                "code": "unknown_durable_request",
                "detail": f"{kind!r} is not a durable exchange this worker answers",
            }
        except Exception as exc:
            return {
                "ok": False,
                "code": str(getattr(exc, "code", "weights_host_refused")),
                "detail": str(getattr(exc, "detail", exc))[:1024],
            }

    def absorb_receipt(self, receipt: pb.JobCheckpointReceipt) -> None:
        """Record the RecordOwner's verdict on a checkpoint this worker already recorded.

        It is a second observer, not a second authority: the declaration is already in this
        worker's process records. A RecordOwner conflict is recorded as a fault,
        because it means the run's identity is contested upstream — but it cannot
        retroactively un-write a declaration that is durable here.
        """
        if receipt.outcome == pb.CheckpointOutcome.CHECKPOINT_OUTCOME_CONFLICT:
            self.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_LOCAL_SAFETY_REFUSAL,
                    subject=f"{receipt.request_id}#{receipt.attempt_ordinal}",
                    reason="checkpoint_conflict_upstream",
                    detail=safe(
                        f"the RecordOwner reports {receipt.logical_key!r} under operation key "
                        f"{receipt.operation_key!r} conflicts: {receipt.fault.detail}"
                    ),
                )
            )

    def _run(self, attempt: AttemptRecord) -> pb.AttemptOutcome | Released:
        """ONE attempt: dispatch, freeze point, release.

        ONE pipeline for both lanes. They differ in the COMMAND they send and in nothing
        after the executor answers, and keeping two copies of everything after it had
        already let them drift — the job lane never checked the env seal it reports, never
        classified a device OOM, and its oversize refusal named no remedy (codex audit,
        adopted 2026-08-25). What a job actually has that a serving attempt does not is a
        durable mid-attempt exchange, and that is one argument to one call.
        """
        job = attempt.job is not None
        executor = attempt.executor
        supervision = self.supervision_for(attempt)
        if executor is None or not supervision.owns(executor):
            return self.abandon(attempt, "its accepted executor vanished before dispatch")
        # The spool already exists — cr-012's hydration made it before acceptance. This
        # stays for the recovered-attempt path, whose acceptance was another process's.
        spool = attempt.spool or self.spool_root / opaque_key(
            "attempt-input-spool", attempt.request_id, attempt.attempt
        )
        spool.mkdir(parents=True, exist_ok=True)
        attempt.spool = spool
        if getattr(supervision, "isolated", False):
            supervision.grant_directory(spool, attempt.executor_epoch)
        attempt.state = "running"
        attempt.executions += 1
        self.executed += 1

        # WHAT THE HANDLER IS GIVEN, and it is the caller's own number or nothing at all.
        # A 0.5 s FLOOR under an already-expired deadline is not a floor, it is 0.5 s of
        # extension the caller never granted — and `admit` cancels a past-deadline attempt
        # before the lane, so the only thing the floor ever did was hide the race. An
        # attempt with NO deadline has no bound to derive one from: 3600 s for a job and
        # 600 s for a serving attempt were ceilings on what an attempt may BE, invented
        # here and contradicting the worker's own watchdog, which cancels on a deadline
        # only when the caller minted one (cr-009). NO BOUND is spelled `None` and never 0:
        # a duration of zero is a real remaining time — the instant a deadline lapses — and
        # collapsing the two would silently hand an expired attempt an unbounded one.
        deadline_ms = int(attempt.spec.get("deadline_unix_ms", 0))
        seconds = deadline_ms / 1000 - time.time() if deadline_ms else None
        attempt.dispatched_at = time.perf_counter()
        key = f"{attempt.request_id}#{attempt.attempt}"
        # The WORKER's own rows. They are the half of the narrative that cannot be lost
        # to the executor's death, because the process that writes them is the one that
        # outlives it — which is exactly what a killed-executor postmortem is made of.
        self.monitor.start(key, "invoke")
        attempt.ring.emit(
            "log",
            "dispatched to the executor",
            "info",
            subject=self._subject(attempt),
            epoch=attempt.executor_epoch,
            deadline_s=round(seconds, 3) if seconds is not None else None,
        )
        try:
            reply = executor.call(
                self._command(attempt, key, spool, seconds),
                timeout=None,
                on_progress=lambda frame: self.progress(attempt.request_id, attempt.attempt, frame),
                on_request=(lambda frame: self.durable_exchange(attempt, frame)) if job else None,
            )
        except ExecutorGone as exc:
            attempt.resulted_at = time.perf_counter()
            attempt.ring.emit(
                "fault",
                "the executor died during this attempt",
                str(exc)[:400],
                epoch=attempt.executor_epoch,
                canceling=attempt.canceling or "no",
            )
            self.monitor.end(key, "executor_died")
            # WHICHEVER THREAD SEES THE DEATH MARKS IT: `waitpid` answers 0 for a few ms
            # on a dying process, so invalidating at the OBSERVATION removes the race with
            # the watchdog's kill instead of narrowing it (cr-007).
            supervision.invalidate(executor, "the executor died during an attempt")
            # A cancel that ESCALATED killed this executor on purpose: the attempt's
            # terminal is CANCELED only after the watchdog proves reclaim AND aborts the
            # spool. Any other death is an invalidation the attempt did not ask for.
            if attempt.canceling:
                # The cgroup can be empty before the driver proves every captured birth
                # absent. Only the watchdog owns that proof/retry; EOF is not permission to
                # record CANCELED ahead of it.
                attempt.reclaim_done.wait()
                return self.finish_cancel(attempt, attempt.canceling)
            # The attempt SPOOL is aborted on both lanes, so nothing half-written can
            # reach a granted destination.
            grants.abort_outputs(spool)
            return self.abandon(attempt, str(exc))
        finally:
            if self.weights is not None:
                self.weights.writer_broker.close_attempt(attempt)
        return self.release(attempt, reply, spool, key)

    @staticmethod
    def _subject(attempt: AttemptRecord) -> str:
        if attempt.job is not None:
            return attempt.job.job
        return attempt.declared.entrypoint if attempt.declared else ""

    def _command(
        self, attempt: AttemptRecord, key: str, spool: Path, seconds: float | None
    ) -> dict[str, Any]:
        """The ONE thing the two lanes genuinely differ in: what they ask the executor."""
        inputs = {
            input_id: {
                "local": str(granted.local),
                "media_type": granted.media_type,
                "digest": granted.digest,
                "length": granted.length,
                "order": granted.order,
            }
            for input_id, granted in attempt.inputs.items()
        }
        if attempt.job is not None:
            if not attempt.weights_work_fingerprint and self.work_fingerprint is not None:
                attempt.weights_work_fingerprint = self.work_fingerprint(attempt)
            return {
                "cmd": "run_job",
                "request_id": key,
                "application": attempt.job.application,
                "package_interface": attempt.job.package_interface,
                "job": attempt.job.job,
                "call_interfaces": self.calls.bindings(attempt) if self.calls is not None else [],
                "payload": attempt.payload,
                "deadline_s": seconds,
                "spool": str(spool),
                "scratch": str(attempt.scratch) if attempt.scratch else "",
                "trees": {ref: [str(r), d] for ref, (r, d) in attempt.trees.items()},
                "inputs": inputs,
                "models": {
                    parameter: {
                        "class": model.class_name,
                        "manifest": model.manifest,
                        "length": model.length,
                    }
                    for parameter, model in sorted(attempt.job_models.items())
                },
                "max_input_bytes": 64 << 20,
                "tensorfs_root": str(self.tensorfs_root) if self.tensorfs_root else "",
                "weights_owner_scope": self.owner_scope(),
                "weights_request_id": attempt.request_id,
                "invocation_spec_digest": documents.spell(attempt.digest),
                "weights_work_fingerprint": attempt.weights_work_fingerprint,
                "budget": {
                    "gpu_rate_micro_usd_per_hour": attempt.job.gpu_rate_micro_usd_per_hour,
                    "gpu_count": self.slot_of(attempt).job_device_count,
                    "cap_micro_usd": attempt.job.cap_micro_usd,
                },
            }
        assert attempt.declared is not None and attempt.plan is not None
        return {
            "cmd": "invoke",
            **({"capture": attempt.spec["capture"]} if "capture" in attempt.spec else {}),
            "request_id": key,
            "entrypoint": attempt.declared.entrypoint,
            "deadline_s": seconds,
            "spool": str(spool),
            "placement": attempt.plan.placement,
            "headroom_bytes": attempt.plan.headroom_bytes,
            "scope_headroom_bytes": dict(attempt.plan.scope_headroom_bytes),
            # cr-012: the VERIFIED inputs, by their stable identity. Paths into this
            # attempt's own spool and nothing else — no URL, no credential and no fetch
            # reaches the executor. What the executor CAN still do is dial one of its own,
            # which is why it installs `sandbox.refuse_network` (cr-042); that fence bounds
            # a buggy package at the Python layer and is not a kernel bound.
            "inputs": inputs,
            "max_input_bytes": attempt.declared.max_input_bytes,
        }

    def release(
        self, attempt: AttemptRecord, reply: dict[str, Any], spool: Path, key: str
    ) -> Released:
        """DEVICE RELEASE (cr-079 §3): the executor answered, and this is everything that
        must happen while it is still the executor that ran the attempt — on the lane
        thread, under the device. The reply's telemetry is absorbed and applied to the
        ledger, every executor-side consequence of the reply is taken (unproven quiescence,
        a broken env seal, a poisoned generation), and the ledger row is probed and closed.
        Nothing after this touches the executor: the post phase is immune to what
        residency does to it next."""
        # A buffered reply and the watchdog's forceful escalation can arrive together.
        # Exactly one transition wins: finalization either closes the signal window first,
        # or waits until cgroup, process and driver reclaim have all been proved.
        attempt.await_reclaim_before_finalizing()
        attempt.resulted_at = time.perf_counter()
        self.monitor.end(key, "returned")
        attempt.execution_contract = dict(reply.get("execution_contract") or {})
        attempt.execution_observation = dict(reply.get("execution_observation") or {})
        attempt.capture_result = dict(reply.get("capture") or {})
        if attempt.execution_contract:
            # WHAT ATTESTED, as a typed row every reader of confessions keys on (cr-109): the
            # throughput table and the gate read the lane by what ran, never by the contract's
            # promise. The line is the attestation's own BOUNDED rendering (cr-122) — digest,
            # entry, counts, kernel symbol — because an observation is a bounded record and
            # the scheme prose this used to inline blew the 400-character cap, faulting the
            # executor on its own telemetry. The prose is in `plan.execution`.
            attempt.confess(
                "execution_contract", str(attempt.execution_contract.get("confession", ""))
            )
        # What this attempt's kernel allocated ON TOP of the resident weights. Recorded from
        # whatever came back, success or fault: a failed attempt's activation peak is
        # exactly the number an operator needs, and dropping it on the failure path would
        # leave the ledger blind in the one case that matters.
        # THIS attempt's transitions, not the generation's history. Reading the cumulative
        # counters made every attempt after the first staged one claim its activation basis
        # had moved, on attempts that never touched a component.
        cell = attempt.plan.shape_cell if attempt.plan else ""
        executor = attempt.executor
        if executor is None:
            self.absorb(attempt, reply, apply_ledger=False)
        else:
            # A reply can race desired-state replacement during finalization. Preserve its
            # attempt-local observations either way, but apply residency, activation and
            # demand facts only while this is still the exact generation they describe.
            supervision = self.supervision_for(attempt)
            with supervision.hold(executor) as owned:
                self.absorb(attempt, reply, apply_ledger=owned)
                if owned:
                    moved = bool(
                        attempt.residency.get("attempt_stages")
                        or attempt.residency.get("attempt_evictions")
                    )
                    terminal = str((reply.get("outcome") or {}).get("terminal", ""))
                    failed_code = str((reply.get("outcome") or {}).get("code", ""))
                    ledger = self.ledger_for(attempt)
                    ledger.observe_attempt(
                        reply.get("metrics", {}),
                        resident_moved=moved,
                        cell=cell,
                        succeeded=terminal == "succeeded",
                    )
                    if terminal != "succeeded" and failed_code in DEVICE_SHORTFALL_CODES:
                        # cr-097: the attempt died climbing, so its peak is a truncated
                        # lower bound and may not calibrate. What IS usable is the picture
                        # it died under — recorded so the next attempt for this shape can
                        # refuse typed instead of spending the handler to rediscover it.
                        ledger.observe_device_shortfall(cell, ledger.device_free)
                    self.bank_demand(attempt, reply.get("metrics", {}), moved)
                else:
                    attempt.confess(
                        "unavailable",
                        f"executor epoch {attempt.executor_epoch} was replaced "
                        "before its terminal telemetry could be applied; successor telemetry "
                        "was left untouched",
                    )
        if reply.get("ok") and executor is not None:
            supervision = self.supervision_for(attempt)
            if not reply.get("quiescent"):
                # No terminal is recorded while author computation might still mutate GPU
                # or output state. Unproven quiescence is an executor fault, not a success.
                supervision.invalidate(executor, "unproven quiescence")
            elif not reply.get("env_intact", True):
                supervision.invalidate(executor, "env seal broken")
            elif reply.get("poisoned"):
                supervision.invalidate(executor, str(reply["poisoned"]))
        # ONE ENVELOPE, ONE CLOSE, at release: the probe reads the executor's after-state
        # for THIS attempt, before the lane hands the device to the next one.
        if attempt.opened:
            self.reconcile(attempt)
            attempt.opened = {}
        attempt.state = "released"
        return Released(attempt, reply, spool, key)

    def finish(self, released: Released) -> pb.AttemptOutcome:
        """THE POST PHASE: everything after device release, on the lane's post thread.

        Identical for both lanes, and once. The executor is not consulted — every fact it
        had to give was taken at release — so a failure here is the WORKER's, and the
        executor is never poisoned by it."""
        attempt, reply, spool, key = released.attempt, released.reply, released.spool, released.key
        self.monitor.start(f"{key}/post", "post")
        try:
            return self._terminal(attempt, reply, spool, key)
        finally:
            self.monitor.end(f"{key}/post", attempt.state)

    def _terminal(
        self, attempt: AttemptRecord, reply: dict[str, Any], spool: Path, key: str
    ) -> pb.AttemptOutcome:
        if attempt.canceling:
            return self.finish_cancel(attempt, attempt.canceling)
        if not reply.get("ok"):
            grants.abort_outputs(spool)
            return self.outcome(
                attempt,
                STATUS.OUTCOME_STATUS_FAILED,
                CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                ORIGIN.CAUSE_ORIGIN_EXECUTOR,
                f"{reply.get('code')}: {reply.get('detail', '')}",
                shortfall=reply.get("shortfall") or None,
            )
        if not reply.get("quiescent"):
            grants.abort_outputs(spool)
            return self.outcome(
                attempt,
                STATUS.OUTCOME_STATUS_FAILED,
                CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                ORIGIN.CAUSE_ORIGIN_WORKER,
                "the executor could not prove quiescence after the handler returned; a "
                "terminal recorded under unproven quiescence would be a claim about state "
                "that may still be moving",
            )
        if not reply.get("env_intact", True):
            grants.abort_outputs(spool)
            changed = ", ".join(str(name) for name in (reply.get("env_changed") or ()))
            names = f"; changed sealed names: {changed}" if changed else ""
            return self.outcome(
                attempt,
                STATUS.OUTCOME_STATUS_FAILED,
                CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                ORIGIN.CAUSE_ORIGIN_WORKER,
                "env_seal_broken: the executor's environment changed after CUDA init — the "
                f"allocator read it once, so the imposition was a silent no-op{names}",
            )
        if "state_seq" in reply:
            # Monotone wire counters belong to the WORKER. A disposable process that
            # restarts a counter at 1 must never become the worker's truth.
            self.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_LOCAL_SAFETY_REFUSAL,
                    subject=key,
                    reason="executor_owned_counter",
                    detail=f"the executor claimed state_seq={reply['state_seq']}; monotone "
                    "wire counters are the worker's and this claim is dropped",
                )
            )

        outcome = reply["outcome"]
        status, cause = _TERMINAL[outcome["terminal"]]
        attempt.traceback = str(outcome.get("traceback") or "")[-16384:]
        breach = outcome["code"] == DEVICE_OOM
        if breach:
            # THE ACCEPTED ENVELOPE WAS BREACHED. `_TERMINAL["failed"]` defaults to
            # AUTHOR_EXCEPTION because most failures are the handler's, and a device that ran
            # out is the one class where the package provably had no say — but it is not
            # CONSTRAINT_INFEASIBLE either. That code means "no plan fits", which is a
            # PRE-ENTRY answer this worker gives before it accepts anything; rendering a
            # post-accept OOM as one makes a planner or envelope defect read as a legitimate
            # capacity result, and nobody goes to look at the planner. The runtime priced
            # this attempt and admitted it, so the fault is the runtime's.
            cause = CAUSE.CAUSE_CODE_EXECUTOR_FAULT
        if outcome["terminal"] == "failed" and outcome["code"] in GROUP_FAULT_CODES:
            cause = CAUSE.CAUSE_CODE_EXECUTOR_FAULT
        if outcome["terminal"] == "failed" and outcome["code"] == "execution_contract_unobserved":
            # The kernel the lane promises was not what ran (cr-109): a capability the
            # contract named was unavailable to this attempt, from the runtime's side.
            cause = CAUSE.CAUSE_CODE_CAPABILITY_UNAVAILABLE
        if outcome["terminal"] == "refused" and outcome["code"] in (
            "unsupported_input",
            "tree_ungranted",
        ):
            cause = CAUSE.CAUSE_CODE_UNSUPPORTED_INPUT
        if outcome["terminal"] == "canceled":
            cause = (
                CAUSE.CAUSE_CODE_DEADLINE_EXPIRED
                if outcome["code"] == "deadline"
                else CAUSE.CAUSE_CODE_CLIENT_CANCEL
            )
        oversize = int(reply.get("oversize_result_bytes", 0))
        if oversize:
            grants.abort_outputs(spool)
            return self.outcome(
                attempt,
                STATUS.OUTCOME_STATUS_FAILED,
                CAUSE.CAUSE_CODE_CAPABILITY_UNAVAILABLE,
                ORIGIN.CAUSE_ORIGIN_RUNTIME,
                f"result_too_large: the typed result canonicalizes to {oversize} B, over "
                "the inline door; the blob-receipt branch needs a real write-and-receipt "
                "transaction and this runtime will not mint a receipt for bytes nobody "
                "wrote (cr-017 owns that transaction)",
            )
        # The output transaction belongs to a SUCCEEDING attempt. Running the tail over a
        # refused one finds the grant unsatisfied and reports `destination_absent`, which
        # replaces the author's real cause with a transaction nobody owed (cr-012).
        manifest: pb.OutputManifest | None = None
        inline: InlineResult | None = None
        if status == STATUS.OUTCOME_STATUS_SUCCEEDED:
            # The result document is read and held to its declared identity BEFORE the output
            # transaction: a result that cannot be verified must not leave written outputs at
            # granted destinations behind it.
            inline, result_error = self._inline_result(attempt, reply)
            if result_error is not None:
                grants.abort_outputs(spool)
                return self.outcome(
                    attempt,
                    STATUS.OUTCOME_STATUS_FAILED,
                    CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                    ORIGIN.CAUSE_ORIGIN_EXECUTOR,
                    result_error,
                )
            manifest, tail_error = self._tail(attempt, reply, key)
            if tail_error is not None:
                # The device phase DID run and its numbers are real; a post-phase fault
                # keeps them, so a failed encode still banks its device_lease_ms.
                return self.outcome(
                    attempt,
                    STATUS.OUTCOME_STATUS_FAILED,
                    CAUSE.CAUSE_CODE_CAPABILITY_UNAVAILABLE,
                    ORIGIN.CAUSE_ORIGIN_WORKER,
                    tail_error,
                    metrics=reply.get("metrics", {}),
                )
            if inline is not None and manifest is not None:
                from .byte_outputs import settle_result

                try:
                    inline = dataclasses_replace(
                        inline,
                        canonical=settle_result(
                            canonical.parse(inline.canonical), list(manifest.outputs)
                        ),
                    )
                except ValueError as exc:
                    return self.outcome(
                        attempt,
                        STATUS.OUTCOME_STATUS_FAILED,
                        CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                        ORIGIN.CAUSE_ORIGIN_WORKER,
                        f"native_output_result: {exc}",
                        metrics=reply.get("metrics", {}),
                    )
        else:
            grants.abort_outputs(spool)
        origin = _ORIGIN[outcome["origin"]]
        shortfall = reply.get("shortfall") or None
        if shortfall:
            # A capacity fact reached the handler as an exception, so the author-error
            # classifier saw an author error. It is not one: the runtime could not admit a
            # declared component set, which is CONSTRAINT_INFEASIBLE from the RUNTIME, and
            # the two numbers belong in the terminal's structured shortfall slot rather
            # than only in a sentence.
            cause = CAUSE.CAUSE_CODE_CONSTRAINT_INFEASIBLE
            origin = ORIGIN.CAUSE_ORIGIN_RUNTIME
            breach = False
        return self.outcome(
            attempt,
            status,
            cause,
            origin,
            # The AUTHOR-SURFACE code in front of its own sentence. The wire's `CauseCode`
            # is a small NEUTRAL class, so without this the typed code the author raised
            # exists only inside prose and nothing downstream can key on it (cr-012).
            f"accepted_envelope_breach: {_coded(outcome)}" if breach else _coded(outcome),
            manifest=manifest,
            metrics=reply.get("metrics", {}),
            result=inline,
            shortfall=shortfall,
        )

    def forward(self, request_id: str, number: int, frame: Mapping[str, Any]) -> None:
        """Hold ONE live observation the executor just emitted, before it can be lost."""
        attempt = self.live.get(request_id)
        if attempt is None or attempt.attempt != number:
            return
        attempt.ring.admit(self._row(frame))
        attempt.forwarded_seq = max(attempt.forwarded_seq, int(frame.get("seq", 0)))

    @staticmethod
    def _row(frame: Mapping[str, Any]) -> Observation:
        """One executor-side row, re-bounded on this side. A disposable process is not
        trusted to bound the worker's memory, so every cap is re-applied here."""
        fields = frame.get("fields") or {}
        return Observation(
            cast("Any", frame.get("kind", "log")),
            str(frame.get("name", ""))[:200],
            cast("Any", frame.get("value")),
            {str(k)[:48]: v for k, v in list(fields.items())[:8]},
            int(frame.get("at_unix_ms", 0)),
        )

    def absorb(
        self,
        attempt: AttemptRecord,
        reply: Mapping[str, Any],
        *,
        apply_ledger: bool = True,
    ) -> None:
        """Take the executor's BOUNDED observation tail into the worker's own ring.

        The executor's ring already applied every cap and every credential check at the emit
        boundary, so this side re-applies the CAPS (a disposable process is not trusted to
        bound the worker's memory) and keeps the rows as they came. Whatever the ring
        already shed is carried through as a number, never inferred.
        """
        caps = dict(reply.get("observation_caps") or {})
        attempt.attribution = dict(reply.get("attribution") or {})
        rows = list(reply.get("observations") or ())
        kept = 0
        for row in rows[-attempt.ring.max_events :]:
            # Anything the lossy lane already delivered is skipped by the EXECUTOR's own
            # sequence. Both sides shed oldest first, so this fills the gap after the last
            # forwarded row and can never double-count one.
            if not isinstance(row, dict) or int(row.get("seq", 0)) <= attempt.forwarded_seq:
                continue
            attempt.ring.admit(self._row(row))
            kept += 1
        residency = reply.get("residency") or {}
        if residency:
            attempt.residency = dict(residency)
            # THE DECLARED TRANSITION, into the ledger BEFORE it reconciles. Under a staged
            # rung the attempt moved gigabytes on purpose; the reconciliation subtracts
            # exactly this and holds the remainder to zero.
            if apply_ledger:
                self.ledger_for(attempt).observe_residency(attempt.residency)
            if residency.get("attempt_stages"):
                attempt.confess(
                    "degraded",
                    f"served on the {residency.get('placement')} rung: "
                    f"{residency['attempt_stages']} component stage(s) costing "
                    f"{residency.get('staged_ms', 0)} ms and "
                    f"{residency.get('attempt_evictions', 0)} eviction(s) costing "
                    f"{residency.get('evicted_ms', 0)} ms - the weights did not all fit "
                    "at once and the runtime moved them rather than refusing",
                )
        shortfall = reply.get("shortfall")
        if shortfall:
            attempt.confess(
                "degraded",
                f"{shortfall['resource']} shortfall at {shortfall['scope']}: needed "
                f"{shortfall['needed_bytes']} B, {shortfall['available_bytes']} B available, "
                f"short by {shortfall['needed_bytes'] - shortfall['available_bytes']} B",
            )
        caps["executor_rows"] = len(rows)
        caps["forwarded_live"] = attempt.forwarded_seq
        caps["absorbed_from_reply"] = kept
        caps.update(attempt.ring.caps())
        attempt.caps = caps
        # Every caller-visible divergence is ALSO an operator-visible confession. The
        # envelope tells the caller what changed; the confession tells the operator why,
        # with the numbers, in the record that outlives the process.
        for row in reply.get("adjustments", ()):
            attempt.confess(
                "substituted" if str(row.get("reason", "")).startswith("adjusted") else "degraded",
                f"{row.get('field')}: requested {row.get('requested')}, served "
                f"{row.get('applied')} - {row.get('reason')}",
            )

    def bank_demand(self, attempt: AttemptRecord, metrics: Mapping[str, Any], moved: bool) -> None:
        """Every serve evaluates DECLARED versus ACTUAL and banks the pair (§3.2).

        The actual is the attempt's own peak above what it started holding — a number that
        survives staging, unlike a figure measured against a construction baseline the
        attempt then moved out from under itself. A miss is COUNTED and CONFESSED; it
        selects nothing and refuses nothing.
        """
        plan = attempt.plan
        if plan is None:
            return
        peak = int(metrics.get("peak_vram_bytes", 0))
        started = int(metrics.get("allocated_at_start_bytes", -1))
        actual = peak - started if peak and started >= 0 else -1
        if moved:
            # The resident set moved under the attempt, so `peak - start` is not the
            # attempt's own demand. Named absent rather than banked as a number.
            actual = -1
        # THE DESIGNED KEY (§3.2): construction variant x accelerator x exact plan x
        # normalized shape cell. Banking per entrypoint x placement folded every request
        # shape a deployment serves into one lane, so a 512px serve and a 1024px one moved
        # the same number and neither described either.
        lane = " x ".join(
            (
                plan.delivery_variant or "default",
                self.ledger_for(attempt).accelerator or "unknown-accelerator",
                f"{plan.delivery}/{plan.materialization or 'verbatim'}/{plan.placement}",
                plan.shape_cell or "-",
            )
        )
        # THE DESCRIPTION, not the enforcement figure (cr-100). An attempt admitted against
        # 0 B — because the only observation of this shape was made on a roomier card — has
        # still been described by that observation, and banking the description is what
        # makes the instrument able to falsify it.
        confession = self.demand.bank(
            lane, plan.activation_envelope_bytes or plan.headroom_bytes, actual
        )
        if confession is not None:
            attempt.confess(confession.kind, confession.quantified)

    def _payload(self, attempt: AttemptRecord) -> dict[str, Any]:
        """The payload travels as a grant INPUT, fetched worker-side under the grant."""
        data = grants.read_input(
            attempt.grant,
            "payload",
            self.authorizer,
            cap=PAYLOAD_MAX_BYTES,
        )
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError(f"a payload document is an object, not {type(value).__name__}")
        return value

    def _tail(
        self, attempt: AttemptRecord, reply: dict[str, Any], key: str = ""
    ) -> tuple[pb.OutputManifest | None, str | None]:
        """§3.4 steps 4-5, worker-side: the device lease is already released, the raw
        snapshots are already immutable in the attempt spool, and this is where they are
        ENCODED (cr-079: the codec runs here, never in the handler), bounded on the encoded
        bytes, written under the exact output grant and become a durable manifest.

        The tail runs on the lane's POST THREAD, overlapping the next attempt's device
        phase; it takes no device lock and consults no executor.

        Outputs bind to granted destinations by STABLE OUTPUT ID — the result's own field
        path (`image`, `pair.thumb`, `frames.0`) — with EXACT SET EQUALITY against the
        media grant subset. Declared weights outputs finish through exact host-acknowledged
        native receipts before they leave that subset. Never by list index: outputs must still
        land in their own destinations, and a position cannot express that. A produced
        output the RecordOwner did not grant, or a granted destination the attempt did not
        fill, refuses the tail WHOLE — an output transaction is not partially successful.
        """
        rows = list(reply.get("outputs", ()))
        weights_outputs: set[str] = set()
        if attempt.job is not None:
            declared = self._job_declaration(attempt.job, attempt.job.job_descriptor_id)
            weights_outputs = {str(row["output_id"]) for row in declared.get("weights_outputs", [])}
        if set(attempt.weights_receipts) != weights_outputs:
            return (
                None,
                "weights_receipt_set_mismatch: every weights output requires "
                "a host-acknowledged receipt",
            )
        granted = set(attempt.grant.outputs) - weights_outputs
        if not weights_outputs <= set(attempt.grant.outputs):
            return None, "weights_receipt_set_mismatch: a weights output has no granted destination"
        spool = attempt.spool
        if not rows and not granted and not reply.get("capture"):
            return None, None
        if spool is None:
            return None, "the attempt has no spool to read outputs from"
        produced = {str(row["output_id"]): row for row in rows}
        if set(produced) != granted:
            grants.abort_outputs(spool)
            return None, (
                f"destination_absent: the attempt produced {sorted(produced) or 'nothing'} "
                f"and the grant names {sorted(granted) or 'nothing'}; an output the "
                "RecordOwner did not grant is unwritable however it was produced, and a "
                "granted destination the attempt never filled is not a success"
            )
        encode_error = self._encode_frames(attempt, spool, reply, key)
        if encode_error is not None:
            grants.abort_outputs(spool)
            return None, encode_error
        entries = []
        native = (
            self.tensorfs_root is not None and self.calls is not None and self.calls.minor() >= 43
        )
        try:
            if len(rows) + bool(reply.get("capture")) > 32:
                raise grants.GrantRefusal(
                    "output_count", "native outputs and capture exceed 32 slots"
                )
            if grants.expired(attempt.grant):
                raise grants.GrantRefusal(
                    "grant_expired", "output grant expired before post processing"
                )
            for output_id in sorted(produced):
                row = produced[output_id]
                if row.get("kind") == "tree":
                    if not native:
                        raise grants.GrantRefusal(
                            "native_tree_unsupported", "tree outputs require wire43 native custody"
                        )
                    from . import byte_outputs
                    from .workspace import Workspace
                    from .workspace_byte_outputs import TREE_MIME

                    root = self._blob_path(spool, str(row["asset_ref"]))
                    if root is None or self.tensorfs_root is None:
                        raise grants.GrantRefusal(
                            "output_spool_io", "tree output has no owned spool"
                        )
                    destination = attempt.grant.outputs[output_id]
                    media_type = grants._output_media_type(
                        output_id, destination.mime_type, TREE_MIME
                    )
                    entries.append(
                        byte_outputs.commit(
                            Workspace(self.tensorfs_root),
                            self.owner_scope(),
                            attempt.request_id,
                            attempt.attempt,
                            attempt.digest,
                            output_id,
                            root,
                            tree=True,
                            max_bytes=destination.max_bytes or (256 << 20),
                            media_type=media_type,
                        )
                    )
                    continue
                blob = self._blob(spool, str(row["asset_ref"]))
                if blob is None:
                    return None, (f"output {output_id} has no bytes in the attempt spool")
                entries.append(
                    grants.write_output(
                        attempt.grant,
                        output_id,
                        blob,
                        self.authorizer,
                        # The AUTHOR's observed media type. write_output joins it to the
                        # invocation's bound type before spending the destination; when the
                        # invocation leaves MIME open, this remains the honest carried fact.
                        media_type=str(row.get("media_type") or ""),
                    )
                )
                if native:
                    from . import byte_outputs
                    from .workspace import Workspace

                    assert self.tensorfs_root is not None
                    committed = byte_outputs.commit(
                        Workspace(self.tensorfs_root),
                        self.owner_scope(),
                        attempt.request_id,
                        attempt.attempt,
                        attempt.digest,
                        output_id,
                        blob,
                        tree=False,
                        max_bytes=attempt.grant.outputs[output_id].max_bytes or (256 << 20),
                        media_type=entries[-1].mime_type,
                    )
                    if (
                        committed.digest != entries[-1].digest
                        or committed.length != entries[-1].length
                    ):
                        raise grants.GrantRefusal(
                            "output_changed",
                            "output changed between granted write and native import",
                        )
                    entries[-1].native_tree.CopyFrom(committed.native_tree)
            if capture := reply.get("capture"):
                if not native or not attempt.spec.get("capture") or "runtime.capture" in produced:
                    raise grants.GrantRefusal(
                        "capture_ungranted", "capture output has no exact requested slot"
                    )
                from . import byte_outputs
                from .workspace import Workspace
                from .workspace_byte_outputs import TREE_MIME

                assert self.tensorfs_root is not None
                root = spool / "runtime.capture"
                if (
                    capture.get("root") != str(root)
                    or capture.get("output_id") != "runtime.capture"
                ):
                    raise grants.GrantRefusal(
                        "capture_output", "capture output changed its owned spool"
                    )
                entry = byte_outputs.commit(
                    Workspace(self.tensorfs_root),
                    self.owner_scope(),
                    attempt.request_id,
                    attempt.attempt,
                    attempt.digest,
                    "runtime.capture",
                    root,
                    tree=True,
                    max_bytes=128 << 20,
                    media_type=TREE_MIME,
                )
                if entry.native_tree.content_bytes != capture.get("length"):
                    raise grants.GrantRefusal(
                        "capture_output", "capture output changed its measured length"
                    )
                content = hashlib.sha256()
                for name in ("capture.json", "sketches.f32"):
                    with (root / name).open("rb") as member:
                        while block := member.read(1 << 20):
                            content.update(block)
                if documents.spell(content.digest()) != capture.get("content_digest"):
                    raise grants.GrantRefusal(
                        "capture_output", "capture semantic digest differs from retained bytes"
                    )
                entries.append(entry)
        except grants.GrantRefusal as exc:
            grants.abort_outputs(spool)
            return None, f"{exc.code}: {exc.detail}"
        except (OSError, ValueError) as exc:
            grants.abort_outputs(spool)
            return None, f"native_output_refused: {exc}"
        receipt = self._publication(attempt, entries)
        self.records.append("publication", receipt.document())
        attempt.publication = receipt.document()
        return pb.OutputManifest(publication_receipt_digest=receipt.digest(), outputs=entries), None

    def _publication(
        self, attempt: AttemptRecord, entries: list[pb.OutputEntry]
    ) -> grants.PublicationReceipt:
        """The typed subject of ONE publication, from what this attempt was actually given."""
        job = attempt.spec.get("job") or {}
        contract = job.get("publication_contract") or {}
        return grants.PublicationReceipt(
            mode="job" if attempt.job is not None else "serving",
            grant_id=str(contract.get("grant_id", "")),
            build_id=(
                attempt.job.build_id
                if attempt.job is not None
                else (attempt.declared.release if attempt.declared else "")
            ),
            subject_id=(
                attempt.job.job_descriptor_id
                if attempt.job is not None
                else (attempt.declared.entrypoint_binding_digest if attempt.declared else "")
            ),
            request_id=attempt.request_id,
            attempt=attempt.attempt,
            outputs=tuple(entries),
        )

    def _encode_frames(
        self, attempt: AttemptRecord, spool: Path, reply: Mapping[str, Any], key: str
    ) -> str | None:
        """Encode every host frame the handler REGISTERED (cr-079 §5): first-party codecs,
        the aggregate ceiling re-checked on the ENCODED bytes, each blob taking its
        handle's spool name so the write below finds it as it always did. A fault here is
        typed and the worker's own; the executor was not involved."""
        frames = list(reply.get("frames") or ())
        if not frames:
            return None
        budget = int(reply.get("max_output_bytes") or MAX_OUTPUT_BYTES)
        total = sum(
            int(row.get("size_bytes") or 0)
            for row in reply.get("outputs", ())
            if str(row.get("asset_ref")) not in {str(f.get("handle")) for f in frames}
        )
        for index, frame in enumerate(frames):
            handle = str(frame.get("handle", ""))
            name = str(frame.get("raw", ""))
            blob = self._blob_path(spool, handle)
            if blob is None or not name or name != Path(name).name:
                return f"output_frame_invalid: frame {handle!r} names no spool blob"
            raw = spool / name
            try:
                data = raw.read_bytes()
                if len(data) != int(frame.get("raw_bytes", -1)):
                    return (
                        f"output_frame_invalid: {handle} registered {frame.get('raw_bytes')} "
                        f"raw bytes and the spool holds {len(data)}"
                    )
                started = time.perf_counter()
                encoded = _codec.encode_frame(
                    str(frame.get("codec", "")), dict(frame.get("facts") or {}), data
                )
                del data
                total += len(encoded)
                if total > budget:
                    return (
                        f"output_too_large: {handle} encodes to {len(encoded)} B, bringing "
                        f"the attempt to {total} B over its {budget} B aggregate ceiling"
                    )
                blob.write_bytes(encoded)
                raw.unlink(missing_ok=True)
            except OutputError as exc:
                return f"{exc.code}: {exc}"
            except OSError as exc:
                return f"output_encode_io: {handle}: {type(exc).__name__}: {exc}"
            except Exception as exc:  # a codec's own fault is typed, never a dead post thread
                return f"output_encode: {handle}: {type(exc).__name__}: {exc}"
            self.monitor.advance(f"{key}/post", index + 1)
            attempt.ring.emit(
                "log",
                f"encoded {handle} as {frame.get('codec')}",
                "info",
                raw_bytes=int(frame.get("raw_bytes", 0)),
                encoded_bytes=len(encoded),
                encode_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        return None

    @staticmethod
    def _blob_path(spool: Path, ref: str) -> Path | None:
        tail = ref.rsplit("/", 2)
        if len(tail) < 3 or not tail[-1] or tail[-1] != Path(tail[-1]).name:
            return None
        return spool / f"{tail[-2]}-{tail[-1]}"

    @classmethod
    def _blob(cls, spool: Path, ref: str) -> Path | None:
        candidate = cls._blob_path(spool, ref)
        return candidate if candidate is not None and candidate.is_file() else None

    # ------------------------------------------------------------------ terminals

    def outcome(
        self,
        attempt: AttemptRecord,
        status: pb.OutcomeStatus,
        cause: pb.CauseCode,
        origin: pb.CauseOrigin,
        detail: str,
        *,
        manifest: pb.OutputManifest | None = None,
        metrics: Mapping[str, Any] | None = None,
        result: InlineResult | None = None,
        shortfall: Mapping[str, Any] | None = None,
    ) -> pb.AttemptOutcome:
        """CLOSE the ledger, commit the triage bundle, build, RECORD, then return to send.

        The order is the whole point and none of it is negotiable:

            release the envelope -> reconcile every byte class -> WRITE and fsync the
            bundle -> record its write receipt -> record the terminal that names it

        A terminal may only reference a bundle that is already durable, or it would be a
        claim about bytes nobody wrote. And the bundle is committed BEFORE the terminal so
        that a crash between the two leaves an unreferenced bundle (harmless, readable)
        rather than a terminal pointing at nothing.
        """
        if attempt.opened:
            self.reconcile(attempt)
            # ONE ENVELOPE, ONE CLOSE. The opening line is spent here; a second terminal for
            # the same attempt has nothing left to reconcile and re-reading this snapshot
            # would report the first close's own baseline move as a leak.
            attempt.opened = {}
        body = pb.AttemptOutcomeBody(
            request_id=attempt.request_id,
            attempt_ordinal=attempt.attempt,
            invocation_spec_digest=documents.spell(attempt.digest),
            status=status,
            safe_message=safe(detail, 4096),
            cause=pb.OutcomeCause(code=cause, origin=origin, detail=safe(detail)),
            # THE BILLING FACT, STRUCTURAL (#480c/§7). Not inferred from a cause-code
            # allowlist, and not "did we intend to run" — it is `executions`, the counter
            # `_run` bumps at the moment the handler is actually dispatched. False means the
            # author never ran, zero execution budget was billed, and the RecordOwner may
            # mint `attempt_ordinal + 1` and send it elsewhere with no cooldown.
            execution_started=attempt.executions > 0,
        )
        if attempt.execution_observation:
            from cozy_runtime.internal.capture_observation import environment

            body.observation.environment.CopyFrom(
                environment(attempt.execution_observation, boot_id=self.boot_id)
            )
        if manifest is not None:
            body.output_manifest.CopyFrom(manifest)
            if attempt.capture_result and any(
                row.output_id == "runtime.capture" for row in manifest.outputs
            ):
                body.observation.capture.CopyFrom(
                    pb.ActivationCaptureResult(
                        output_id="runtime.capture",
                        content_digest=documents.raw(attempt.capture_result["content_digest"]),
                    )
                )
        if shortfall is not None:
            body.cause.shortfall.CopyFrom(
                pb.ResourceShortfall(
                    resource=str(shortfall["resource"]),
                    scope=str(shortfall["scope"]),
                    needed_bytes=int(shortfall["needed_bytes"]),
                    available_bytes=int(shortfall["available_bytes"]),
                    evidence_class=str(shortfall.get("evidence_class", "measured")),
                )
            )
        if metrics:
            body.metrics.CopyFrom(self.attest(attempt, metrics, manifest))
        if result is not None:
            body.result.CopyFrom(self._envelope(result))
        for output_slot in sorted(attempt.weights_receipts):
            body.weights_receipts.append(attempt.weights_receipts[output_slot])
        receipt = self.commit_bundle(attempt, status, cause, origin, detail)
        if receipt is not None:
            body.triage_bundle.CopyFrom(
                pb.TriageBundleRef(
                    subject_id=receipt.subject_id,
                    write_receipt_digest=documents.raw(receipt.bundle_digest),
                    length=receipt.length,
                )
            )

        canonical_bytes, digest = documents.identity(body)
        attempt.outcome_id = f"out-{uuid.uuid4().hex[:24]}"
        attempt.outcome_digest = digest
        attempt.outcome_bytes = canonical_bytes
        self.records.append(
            "outcome",
            {
                "request_id": attempt.request_id,
                "attempt": attempt.attempt,
                "worker_boot_id": self.boot_id,
                "instance_id": self.instance_id,
                "invocation_spec_digest": documents.spell(attempt.digest),
                "placement_id": attempt.placement_id,
                "outcome_id": attempt.outcome_id,
                "outcome_digest": documents.spell(digest),
                "outcome_canonical_b64": base64.b64encode(canonical_bytes).decode(),
            },
        )
        attempt.state = "outcome"
        self._release_job_models(attempt)
        return self.wrap(attempt)

    # ------------------------------------------------------------------ triage

    def commit_bundle(
        self,
        attempt: AttemptRecord,
        status: pb.OutcomeStatus,
        cause: pb.CauseCode,
        origin: pb.CauseOrigin,
        detail: str,
    ) -> triage.Receipt | None:
        """Assemble and durably commit ONE bundle for this terminal attempt.

        Observation NEVER costs a terminal (worker-protocol/01: a triage ref can replace no
        terminal field). A bundle that cannot be built or written is a FAULT the operator
        sees and a terminal that ships without a reference — never a lost result.
        """
        attempt.subject_id = attempt.subject_id or triage.mint_subject()
        key = f"{attempt.request_id}#{attempt.attempt}"
        plan = attempt.plan
        plan_document = plan.document() if plan is not None else {}
        contract = attempt.prepared_model.execution_contract if attempt.prepared_model else {}
        if contract or attempt.execution_contract:
            # `plan.execution` (cr-109): the contract as applied by the prepare beside what
            # this attempt attested. Outside `AttemptPlan.document()`, which is an identity.
            plan_document["execution"] = {**contract, **attempt.execution_contract}
        assembly = triage.Assembly(
            subject_id=attempt.subject_id,
            attempt={
                "request_id": attempt.request_id,
                "attempt": attempt.attempt,
                "worker_boot_id": self.boot_id,
                "instance_id": self.instance_id,
                "executor_epoch": attempt.executor_epoch,
                "invocation_spec_digest": documents.spell(attempt.digest),
                "entrypoint": attempt.declared.entrypoint if attempt.declared else "",
                "executions": attempt.executions,
                "replays": attempt.replays,
            },
            plan=plan_document,
            posture=self.posture(),
            terminal={
                "status": pb.OutcomeStatus.Name(status),
                "cause_code": pb.CauseCode.Name(cause),
                "cause_origin": pb.CauseOrigin.Name(origin),
                "safe_message": safe(detail, 4096),
                "outcome_id": attempt.outcome_id,
                # The frames behind a FAILED terminal, as the executor formatted them. Not
                # `safe()`d: a traceback is quoted source and file paths, and the whole
                # point of keeping it is reading it whole after the pod is gone.
                "traceback": attempt.traceback,
            },
            faults=[
                {
                    "kind": pb.FaultKind.Name(f.kind),
                    "subject": f.subject,
                    "reason": f.reason,
                    "detail": f.detail[:512],
                }
                for f in self.faults[-8:]
            ],
            measurements={
                "residency": attempt.residency,
                "demand": self.demand.document(),
                "ledger": attempt.reconciliation
                or {
                    "classes": [
                        c.document()
                        for c in (
                            attempt.slot.ledger.counters() if attempt.slot is not None else ()
                        )
                    ]
                },
                "attribution": attempt.attribution,
                "windows_ms": {
                    "queue_ms": round(
                        max(attempt.dispatched_at - (attempt.queued_at or attempt.accepted_at), 0.0)
                        * 1000,
                        2,
                    ),
                    "admission_ms": round(
                        max(attempt.accepted_at - attempt.queued_at, 0.0) * 1000, 2
                    ),
                    "dispatch_to_result_ms": round(
                        max(attempt.resulted_at - attempt.dispatched_at, 0.0) * 1000, 2
                    ),
                },
            },
            confessions=attempt.confessions,
            liveness=[row for row in self.monitor.document() if str(row["subject"]).endswith(key)]
            or self.monitor.document(),
            events=attempt.ring.rows(),
            caps=attempt.caps or attempt.ring.caps(),
        )
        try:
            return self.bundles.commit(assembly)
        except (triage.TriageError, OSError) as exc:
            self.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_LOCAL_SAFETY_REFUSAL,
                    subject=key,
                    reason="triage_bundle_uncommitted",
                    detail=safe(
                        f"the triage bundle for {attempt.subject_id} could not be committed "
                        f"({exc}); the terminal ships WITHOUT a reference, because a triage "
                        "ref can replace no terminal field and must never cost one"
                    ),
                )
            )
            return None

    def _accepted(self, attempt: AttemptRecord) -> pb.AttemptAccepted:
        """HOLD the accepted attempt and publish it. The one acceptance both lanes make."""
        attempt.state = "accepted"
        attempt.accepted_at = time.perf_counter()
        self.live[attempt.request_id] = attempt
        self.history[attempt.key()] = attempt
        return self.accepted_message(attempt)

    def accepted_message(self, attempt: AttemptRecord) -> pb.AttemptAccepted:
        """THE acceptance message — one constructor, for a first send and for a re-send.

        Acceptance is queue ADMISSION (minor 23, proto-026): journaled, hydrated, queued on
        a lane. It binds neither a plan nor an executor epoch — both bind at device entry
        and ride HeldAttempt from RUNNING on — so a re-send can never move anything.
        """
        return pb.AttemptAccepted(
            request_id=attempt.request_id,
            attempt_ordinal=attempt.attempt,
            invocation_spec_digest=attempt.digest,
            placement_id=attempt.placement_id,
            lane_id=attempt.lane_id,
        )

    def wrap(self, attempt: AttemptRecord) -> pb.AttemptOutcome:
        """Wrap the SAME recorded body in the current session envelope. The body and accepted
        placement are byte facts; only ownership-envelope fields move across a reconnect.

        A closed attempt no longer holds those bytes on the attempt object; process records
        do. This is the only path that needs them again.
        """
        if not attempt.outcome_bytes and attempt.outcome_id:
            attempt.outcome_bytes = self.records.outcome_bytes(attempt.key())
        return pb.AttemptOutcome(
            request_id=attempt.request_id,
            attempt_ordinal=attempt.attempt,
            invocation_spec_digest=attempt.digest,
            outcome_id=attempt.outcome_id,
            outcome_digest=attempt.outcome_digest,
            outcome_canonical_bytes=attempt.outcome_bytes,
            placement_id=attempt.placement_id,
        )

    def _inline_result(
        self, attempt: AttemptRecord, reply: Mapping[str, Any]
    ) -> tuple[InlineResult | None, str | None]:
        """Read the result bytes the executor spooled, and hold them to their declared identity.

        The seam carries CONTROL. The result document takes the same road the output blobs
        take — the attempt spool the worker brokered — and only its digest and length
        cross the frame, so a result between the 64 KiB frame cap and the 4 MiB public door
        is served instead of becoming an executor fault (reproduced live at 71,712 B).

        The worker names the file. Nothing here is read from a path the child chose.
        """
        ref = reply.get("result_ref")
        if not isinstance(ref, Mapping):
            return None, None
        if attempt.spool is None:
            return None, "the attempt has no spool to read its result document from"
        path = attempt.spool / seam.RESULT_DOCUMENT
        try:
            data = path.read_bytes()
        except OSError as exc:
            return None, f"result_document_unreadable: {type(exc).__name__}"
        if len(data) != int(ref.get("length", -1)):
            return None, (
                f"result_document_length: the executor declared {ref.get('length')} B and "
                f"the spool holds {len(data)} B"
            )
        spelled = "sha256:" + hashlib.sha256(data).hexdigest()
        if spelled != str(ref.get("digest", "")):
            return None, (
                f"result_document_digest: the executor declared {str(ref.get('digest'))[:23]}… "
                f"and the spooled bytes hash to {spelled[:23]}…"
            )
        return (
            InlineResult(
                canonical=data,
                schema_digest=str(reply.get("result_schema_digest") or spelled),
                adjustments=tuple(dict(row) for row in reply.get("adjustments", ())),
            ),
            None,
        )

    @staticmethod
    def _envelope(result: InlineResult) -> pb.ResultEnvelope:
        """The TYPED function result — never the asset manifest under another name.

        Small results ride the terminal INLINE (<= 4 MiB), skipping the file-store round
        trip; a larger one takes the blob-receipt door. The choice is kept here explicitly
        rather than lost: it is the whole reason the field is a XOR pair.
        """
        envelope = pb.ResultEnvelope(
            # The DECLARED annotated result surface's identity, computed by the executor from
            # the package's own schema — never a digest of this particular value, which
            # would move every request and fence nothing.
            result_schema_digest=documents.raw(result.schema_digest),
            adjustments=[
                pb.AdjustmentRow(
                    field=row["field"],
                    requested=row["requested"],
                    applied=row["applied"],
                    reason=row["reason"],
                )
                for row in result.adjustments
            ],
        )
        # The EXECUTOR's exact serialized bytes, not a re-serialization of a decoded copy:
        # the envelope's whole claim is that the bytes a RecordOwner holds and the value the
        # handler returned are one thing.
        envelope.inline_result = result.canonical
        return envelope

    def attest(
        self,
        attempt: AttemptRecord,
        claim: Mapping[str, Any],
        manifest: pb.OutputManifest | None,
    ) -> pb.AttemptMetrics:
        """The executor's numbers are a CLAIM; this is the worker's attestation.

        Wall-clock is clamped to the window the worker itself observed (dispatch ->
        result). Anything it cannot observe is NAMED, never silently promoted.
        """
        window_ms = (attempt.resulted_at - attempt.dispatched_at) * 1000
        queue_ms = (attempt.dispatched_at - (attempt.queued_at or attempt.accepted_at)) * 1000
        return pb.AttemptMetrics(
            runtime_ms=int(min(float(claim.get("handler_ms", 0.0)), window_ms)),
            queue_ms=int(max(queue_ms, 0.0)),
            peak_device_memory_bytes=int(claim.get("peak_vram_bytes", 0)),
            rss_at_end_bytes=int(claim.get("rss_at_end_bytes", 0)),
            output_count=len(manifest.outputs) if manifest is not None else 0,
            device_lease_ms=int(min(float(claim.get("device_lease_ms", 0.0)), window_ms)),
            device_count=int(claim.get("gpu_count", 0)),
            handler_ms=int(min(float(claim.get("handler_ms", 0.0)), window_ms)),
            finalization_ms=int(max((time.perf_counter() - attempt.resulted_at) * 1000, 0)),
            unverified_fields=list(UNVERIFIABLE),
        )

    # ------------------------------------------------------------------ cancel / abandon

    def cancel(self, message: pb.CancelAttempt) -> str:
        """Cooperative, then forceful. Returns what was done, for the activity lane."""
        held = self.live.get(message.request_id)
        if held is None or held.attempt != message.attempt_ordinal:
            return "dropped: no such held attempt"
        if held.digest != message.invocation_spec_digest:
            return "dropped: cancel digest does not match the recorded acceptance"
        with held.transition_lock:
            if held.state in ("finalizing", "released", "outcome", "closed"):
                return "ignored: the attempt already left the device"
            held.canceling = {
                pb.CancelReason.CANCEL_REASON_CLIENT: "client",
                pb.CancelReason.CANCEL_REASON_DRAIN: "drain",
                pb.CancelReason.CANCEL_REASON_SUPERSEDED: "superseded",
                pb.CancelReason.CANCEL_REASON_POLICY: "policy",
                pb.CancelReason.CANCEL_REASON_DEADLINE: "deadline",
            }.get(message.reason, "client")
            if held.state == "reclaiming":
                return "forceful executor reclaim is already in progress"
            if held.state != "running":
                # NOTHING OF THIS ATTEMPT IS ON THE DEVICE. The mark is enough: the device
                # lane reads it when it reaches this attempt and terminates without dispatch.
                return f"marked before dispatch ({held.canceling}); nothing signalled"
            executor = held.executor
            if executor is None or not self.supervision_for(held).cooperative_cancel(
                executor, f"{held.request_id}#{held.attempt}"
            ):
                return "executor already gone or cooperative marker unavailable"
            return f"cooperative cancel marked ({held.canceling})"

    def escalate(self, attempt: AttemptRecord, grace: float) -> dict[str, Any]:
        """The FORCEFUL half, and the exact ordering §5 requires: forced kill -> output
        abort -> epoch invalidation -> record CANCELED. Rebuilding readiness happens
        AFTER, off the cancellation critical path."""
        executor = attempt.executor
        started = time.perf_counter()
        evidence = None
        if not attempt.begin_reclaim():
            return {
                "grace_s": grace,
                "skipped": "natural finalization already owns completion",
                "executor_epoch": attempt.executor_epoch,
            }
        if executor is not None:
            # Reclaim failure is not a cancellation result.  Let the watchdog keep the
            # worker non-dispatchable and retry; swallowing this error would disable the
            # only actor still able to finish the owned process.
            evidence = self.supervision_for(attempt).retire_current(executor, "forceful cancel")
        gone_at = time.perf_counter()
        aborted = grants.abort_outputs(attempt.spool) if attempt.spool else 0
        attempt.reclaim_done.set()
        return {
            "grace_s": grace,
            "killed": int(evidence is not None),
            "exit_status": evidence.exit_status if evidence is not None else None,
            "kill_to_gone_ms": round((gone_at - started) * 1000, 2),
            "aborted_outputs": aborted,
            "executor_epoch": attempt.executor_epoch,
            "device_reclaim": evidence.device.state if evidence is not None else "stale",
        }

    def finish_cancel(self, attempt: AttemptRecord, reason: str) -> pb.AttemptOutcome:
        cause = {
            "client": CAUSE.CAUSE_CODE_CLIENT_CANCEL,
            "drain": CAUSE.CAUSE_CODE_DRAIN_CANCEL,
            "superseded": CAUSE.CAUSE_CODE_SUPERSEDED_CANCEL,
            "policy": CAUSE.CAUSE_CODE_POLICY_CANCEL,
            "deadline": CAUSE.CAUSE_CODE_DEADLINE_EXPIRED,
        }[reason]
        return self.outcome(
            attempt,
            STATUS.OUTCOME_STATUS_CANCELED,
            cause,
            ORIGIN.CAUSE_ORIGIN_RECORD_OWNER
            if reason in ("drain", "policy", "superseded")
            else ORIGIN.CAUSE_ORIGIN_CLIENT,
            f"canceled ({reason}); "
            + (
                "the executor was dead and uncommitted outputs aborted before this terminal "
                "was recorded"
                if attempt.executions
                else "nothing was dispatched, so there is nothing to unwind"
            ),
        )

    def last_resort(self, attempt: AttemptRecord, why: str) -> pb.AttemptOutcome:
        """The SMALLEST terminal this worker can make, for when the ordinary one could not.

        `terminal()` builds a rich body — a reconciliation, a triage bundle, attested
        metrics, an inline result — and every one of those can fail on its own. When one
        does, the attempt used to end in SILENCE: the lane returned, the RecordOwner held an
        accepted attempt with no terminal. Progress observation cannot honestly invent that
        terminal (job-001). An accepted attempt ALWAYS produces a terminal; this is the body
        that has nothing left to fail on.

        It is recorded before it is sent so same-process reconnects replay identical bytes.
        The caller records the frame before exposing it and owns restart recovery.
        """
        body = pb.AttemptOutcomeBody(
            request_id=attempt.request_id,
            attempt_ordinal=attempt.attempt,
            invocation_spec_digest=documents.spell(attempt.digest),
            status=STATUS.OUTCOME_STATUS_ABANDONED,
            safe_message=safe(f"outcome_construction_failed: {why}", 4096),
            cause=pb.OutcomeCause(
                code=CAUSE.CAUSE_CODE_EXECUTOR_INVALIDATED,
                origin=ORIGIN.CAUSE_ORIGIN_WORKER,
                detail=safe(f"outcome_construction_failed: {why}"),
            ),
            execution_started=attempt.executions > 0,
        )
        for output_slot in sorted(attempt.weights_receipts):
            body.weights_receipts.append(attempt.weights_receipts[output_slot])
        canonical_bytes, digest = documents.identity(body)
        attempt.outcome_id = f"out-{uuid.uuid4().hex[:24]}"
        attempt.outcome_digest = digest
        attempt.outcome_bytes = canonical_bytes
        self.records.append(
            "outcome",
            {
                "request_id": attempt.request_id,
                "attempt": attempt.attempt,
                "worker_boot_id": self.boot_id,
                "instance_id": self.instance_id,
                "invocation_spec_digest": documents.spell(attempt.digest),
                "placement_id": attempt.placement_id,
                "outcome_id": attempt.outcome_id,
                "outcome_digest": documents.spell(digest),
                "outcome_canonical_b64": base64.b64encode(canonical_bytes).decode(),
            },
        )
        attempt.state = "outcome"
        self._release_job_models(attempt)
        return self.wrap(attempt)

    def abandon(self, attempt: AttemptRecord, why: str) -> pb.AttemptOutcome:
        """An accepted-but-incomplete attempt ALWAYS produces a terminal, never silence."""
        if attempt.spool is not None:
            grants.abort_outputs(attempt.spool)
        if attempt.executor is not None:
            self.supervision_for(attempt).invalidate(attempt.executor, "executor invalidated")
        return self.outcome(
            attempt,
            STATUS.OUTCOME_STATUS_ABANDONED,
            CAUSE.CAUSE_CODE_EXECUTOR_INVALIDATED,
            ORIGIN.CAUSE_ORIGIN_EXECUTOR,
            f"executor invalidated: {why}",
        )

    def close(self, ack: pb.AttemptOutcomeAck) -> str:
        """Closure records ONLY on an ack bound by terminal_id AND terminal_digest."""
        attempt = self.history.get((ack.request_id, ack.attempt_ordinal))
        if attempt is None or attempt.state != "outcome":
            return "dropped: no recorded terminal for that attempt"
        if ack.outcome_id != attempt.outcome_id or ack.outcome_digest != attempt.outcome_digest:
            return (
                f"NOT AN ACK: id {ack.outcome_id!r}/digest {ack.outcome_digest.hex()[:12]} "
                f"does not bind the recorded terminal — replay continues"
            )
        self.records.append(
            "closed",
            {
                "request_id": attempt.request_id,
                "attempt": attempt.attempt,
                "outcome_id": attempt.outcome_id,
                "outcome_digest": documents.spell(attempt.outcome_digest),
            },
        )
        attempt.state = "closed"
        # INPUT RETENTION ENDS HERE (cr-012), and it is its own bounded story: an input
        # asset takes no CAS lease, joins no eviction ladder and is not content-addressed
        # storage — it is one attempt's spool, and a closed attempt has no reader left.
        # Deliberately at CLOSE and not at terminal: a terminal that is never acknowledged
        # replays, and a replay whose inputs were already deleted would refuse on bytes it
        # had verified once.
        if attempt.spool is not None:
            dropped = grants.release_inputs(attempt.spool)
            if dropped:
                attempt.ring.emit("log", f"released {dropped} hydrated inputs", "info")
        # Only the attempt that IS held clears the hold. A refusal terminal for ordinal N+1
        # closing while ordinal N runs must not evict the running attempt from `live`.
        if self.live.get(attempt.request_id) is attempt:
            self.live.pop(attempt.request_id, None)
        attempt.compact()
        return "closed"

    def _active_row(
        self, attempt: AttemptRecord, record: Mapping[str, Any] | None = None
    ) -> pb.HeldAttempt:
        """One placement-aware `HeldAttempt` (#481: the set holds unacked OUTCOMES too, and
        those are not active — which is what the old `ActiveAttempt` name got wrong).

        Every held attempt names the immutable placement and executor epoch captured by
        acceptance.
        """
        row = record or {}
        pending = attempt.state in ("outcome", "closed")
        held = pb.HeldAttempt(
            request_id=attempt.request_id,
            attempt_ordinal=attempt.attempt,
            kind=_KIND[attempt.kind],
            state=pb.AttemptState.ATTEMPT_STATE_OUTCOME_PENDING_ACK
            if pending
            else pb.AttemptState.ATTEMPT_STATE_RUNNING,
            invocation_spec_digest=attempt.digest,
            placement_id=str(row.get("placement_id", "")) or attempt.placement_id,
            executor_epoch=int(row.get("executor_epoch", 0)) or attempt.executor_epoch,
        )
        if pending:
            # Set IFF the state is OUTCOME_PENDING_ACK, so the RecordOwner can reconcile the
            # exact outcome it owes an ack for without a second round trip.
            held.outcome_id = attempt.outcome_id
            held.outcome_digest = attempt.outcome_digest
        return held

    def pending_outcomes(self) -> list[AttemptRecord]:
        return [a for a in self.history.values() if a.state == "outcome"]

    #: Every state in which this worker still holds an attempt and owes the owner something.
    HELD = ("accepted", "running", "reclaiming", "finalizing", "released", "outcome")
    #: The states that HOLD THE LANE'S DEVICE (cr-079 D3): from the offer's hold to device
    #: release. A queued offer is the device's next occupant and counts.
    ON_DEVICE = ("queued", "accepted", "running", "reclaiming", "finalizing")
    #: Off the device and not yet acknowledged: the post phase and the unacked outcome.
    RELEASED = ("released", "outcome")

    def snapshot_entries(self) -> list[pb.HeldAttempt]:
        """Every attempt this worker still holds — accepted, running, or outcome-pending-ack.

        A queued one is deliberately absent
        (the wire has no state for held-not-admitted; the owner's DISPATCHING-absent rule
        resends it, and the admission window is one, so there is never a QUEUE of them)."""
        return [self._active_row(a) for a in self.history.values() if a.state in self.HELD]

    def active(self) -> list[pb.HeldAttempt]:
        """The attempts this worker HOLDS, for the ObservedWorkerState tick.

        Running attempts, released ones in their post phase, AND outcomes pending ack — the
        same set the snapshot carries; `available_attempt_slots` reads the same states by
        phase (cr-079 D3). Reading only `live` under-reported exactly those: `close()`
        drops the attempt from `live`, but an outcome nobody acked is still owed.

        A queued one is deliberately absent: acceptance is what makes an attempt a claim,
        and the wire has no state for "held, not yet admitted".
        """
        return [self._active_row(a) for a in self.history.values() if a.state in self.HELD]

    def in_flight(self) -> int:
        """How many attempts this worker holds — `active()`'s set, counted without building
        its rows. It is what the pod supervisor's release decision reads, every tick."""
        return sum(1 for a in self.history.values() if a.state in self.HELD)
