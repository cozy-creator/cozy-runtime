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
import logging
import re
import stat
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from dataclasses import replace as dataclasses_replace
from pathlib import Path
from typing import Any, cast

import msgspec

from cozy_runtime.author import _codec
from cozy_runtime.author._assets import InputMetadata
from cozy_runtime.author._errors import DEVICE_OOM, OutputError
from cozy_runtime.author._executor_requests import (
    Answer,
    Checkpoint,
    CheckpointReceipt,
    DeviceRoom,
    Handler,
    Publish,
    Published,
    Reply,
    Request,
    TreeMember,
    WriterAdopt,
    WriterOutput,
    WriterRequest,
    WriterSource,
    refuse,
)
from cozy_runtime.author._observations import EventRing, Observation
from cozy_runtime.author._observations import Kind as ObservationKind
from cozy_runtime.internal import (
    canonical,
    execution_evidence,
    executor_replies,
    fill,
    liveness,
    output_budget,
    package_interface,
    readiness,
    seam,
)
from cozy_runtime.internal.executor_commands import (
    InputFile,
    Invoke,
    JobBudget,
    ModelManifest,
    PrepareRequest,
    Probe,
    RunJob,
)
from cozy_runtime.internal.executor_replies import (
    Adjustment,
    AttemptReply,
    CaptureWritten,
    Metrics,
    ObservationRow,
    Outcome,
    PreparedReply,
    Shortfall,
)
from cozy_runtime.internal.pathkey import opaque_key
from cozy_runtime.internal.worker import grants, machine_byte_results, observe, refusal, triage
from cozy_runtime.internal.worker.calls import Calls
from cozy_runtime.internal.worker.child import Executor, ExecutorGone, ExecutorSupervision
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.machine_publication import output_destination
from cozy_runtime.internal.worker.plan import (
    AttemptPlan,
    DeclaredBinding,
    JobBinding,
    PlanRefusal,
    PreparedModel,
    PreparedRequest,
)
from cozy_runtime.internal.worker.products import Products
from cozy_runtime.internal.worker.records import WorkerRecords
from cozy_runtime.internal.worker.tree_members import project as project_tree_member
from cozy_runtime.internal.worker.weights import WeightsExchange
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

_LOG = logging.getLogger(__name__)

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


def _coded(outcome: Outcome) -> str:
    """`<code>: <message>` — the AUTHOR's typed code in front of its own sentence.

    The wire carries a small neutral `CauseCode` on purpose (a RecordOwner classifies, it
    does not read author vocabulary), so the typed code has to ride the message or it does
    not survive the boundary at all. Every other refusal on this path already renders this
    way; author outcomes were the exception.
    """
    code, message = outcome.code, outcome.message
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
UNVERIFIABLE = (
    "input_tokens",
    "output_tokens",
    "peak_device_memory_bytes",
    "working_peak_device_bytes",
)


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
        "gpu_divergence",
        # an older executor's name for it: drop once no package locks a Runtime that says it
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
    adjustments: tuple[Adjustment, ...]


@dataclass(frozen=True, slots=True)
class Released:
    """ONE attempt off the device (cr-079): the executor's reply and what the post phase
    needs to turn it into a terminal. Built on the lane thread at device release, consumed
    on the lane's post thread; nothing in it reaches the executor again."""

    attempt: AttemptRecord
    reply: AttemptReply
    spool: Path
    key: str


class _Checkpointed(msgspec.Struct, frozen=True, kw_only=True):
    """One journaled checkpoint declaration."""

    request_id: str
    attempt: int
    worker_boot_id: str
    operation_key: str
    logical_key: str
    content_digest: str
    length: int
    receipt_id: str


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
    failed_bindings: Mapping[str, str]
    lane_id: str
    #: what the executor on this slot was sealed to see (`CUDA_VISIBLE_DEVICES`)
    devices: str = ""
    job_device_count: int = 0
    orchestration: bool = False
    job_binding: JobBinding | None = None
    job_resource_caps: Mapping[str, int] = dataclasses_field(default_factory=dict)
    bindings_digest: str = ""


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
    #: the memory manager made room for this call's measured need on every device
    room: bool = True
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
    #: why a "failed" cancel ends this attempt FAILED rather than CANCELED
    failure: str = ""
    #: the caller's absolute deadline (`InvocationSpec.deadline_unix_ms`), 0 for none
    deadline_ms: int = 0
    #: wakes this attempt's GUARD: a cancel, dispatch, and leaving the device
    poke: Callable[[], None] = lambda: None
    #: a guard is running for this attempt (one per attempt: escalation is single-flight)
    guarded: bool = False
    #: monotonic instant the cancel was marked; `math.inf` once escalation has finished
    cancel_started: float = 0.0
    #: the executor's frames for this attempt, counted: the meter a cancel is judged by
    pace: liveness.Pace = dataclasses_field(default_factory=liveness.Pace)
    moves: int = 0
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
    execution_observation: dict[str, Any] = dataclasses_field(default_factory=dict)
    #: the executor's `execution` record: degree, per-GPU rows, executor and construction
    execution: dict[str, Any] = dataclasses_field(default_factory=dict)
    attention_applied: bool = False
    capture_result: CaptureWritten | None = None

    #: the record the plan was PRICED against, resolved once by the executor's kernel
    prepared: PreparedRequest | None = None
    #: cr-009's typed publication record, once the output transaction committed one
    publication: dict[str, Any] = dataclasses_field(default_factory=dict)
    #: Host-ACKed WeightsReceipt/1 refs, one per interface-declared output slot.
    weights_receipts: dict[str, pb.WeightsReceiptRef] = dataclasses_field(default_factory=dict)
    weights_work_fingerprint: str = ""
    #: The Host refused custody of one of this attempt's weights transactions: its detail.
    weights_refused: str = ""
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

    def moved(self) -> None:
        """One executor frame for this attempt: measured movement."""
        self.moves += 1
        self.pace.observe(self.moves)

    def cancel_stall(self, floor: float, now: float | None = None) -> dict[str, float] | None:
        """The measurement a cancelled attempt that has not stopped is killed on, or None.

        Stillness counts from its last frame or the cancel, whichever is later, and is
        judged by `liveness.Pace`: STILL_FACTOR times the longest gap between frames this
        executor has shown (this attempt or an earlier one), never below `floor`.
        """
        at = time.monotonic() if now is None else now
        still = at - max(self.pace.moved_at, self.cancel_started)
        patience = self.pace.patience(floor)
        if still <= patience:
            return None
        return {
            "still_s": round(still, 3),
            "worst_gap_s": round(self.pace.worst_pause, 3),
            "frames": self.moves,
            "patience_s": round(patience, 3),
        }

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
        self.publication = {}
        self.weights_receipts = {}
        self.weights_work_fingerprint = ""
        self.weights_refused = ""
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
        progress: Callable[[str, int, dict[str, Any]], None],
        dispatched: Callable[[AttemptRecord], None] = lambda attempt: None,
        bundles: triage.BundleStore,
        monitor: observe.SilenceMonitor,
        posture: Callable[[], dict[str, Any]],
        device_process: Callable[[int], int],
        weights: WeightsExchange | None = None,
        owner_scope: Callable[[], str] = lambda: "",
        tensorfs_root: Path | None = None,
        work_fingerprint: Callable[[AttemptRecord], str] | None = None,
        room: Callable[[AttemptRecord], Handler] | None = None,
        inventory: Sequence[readiness.RuntimeGPU] = (),
    ) -> None:
        self.records = records
        self.authorizer = authorizer
        self.jobs = dict(jobs)
        self.calls: Calls | None = None
        self.job_device_count = 0
        self.spool_root = spool_root
        self.instance_id = instance_id
        self.progress = progress
        #: journals that execution STARTED, immediately before the executor is handed the
        #: command: the one boundary between work the machine never got and work it ran
        self.dispatched = dispatched
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
        #: a serving attempt's answer to its executor asking for device room mid-call
        self.room = room
        #: the GPUs nvidia-smi listed (`readiness.RuntimeGPU`): names each record's GPU
        self.inventory = tuple(inventory)
        #: the run output log: `Outputs.publish` mid-run and the returned result at post
        self.products: Products | None = None
        #: the demand falsifier. It ships first, it counts, and it decides nothing — the
        #: `timing-decides-nothing` fence proves the module it lives in cannot reach the
        #: planner, the ledger or the fill plane.
        self.demand = observe.DemandLedger()
        #: cr-009's DURABLE checkpoint ledger: (request_id, operation_key, logical_key) ->
        #: the recorded receipt. Identity replays; the same key with different bytes is a
        #: CONFLICT and never a replacement (§2).
        self.receipts: dict[tuple[str, str, str], _Checkpointed] = {}

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
                return held  # idempotent: run once
            if held.state in ("queued", "staged", "entering", "accepted", "running", "finalizing"):
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
            deadline_ms=int(spec.get("deadline_unix_ms", 0)),
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
            raw = executor.call(
                PrepareRequest(
                    construction=declared.construction_key(),
                    request_id=f"{attempt.request_id}#{attempt.attempt}",
                    entrypoint=declared.entrypoint,
                    payload=attempt.payload,
                    capture=attempt.spec.get("capture"),
                    attention_kernel=str(attempt.spec.get("attention_kernel") or ""),
                    input_metadata={
                        name: InputMetadata(
                            input_id=name,
                            media_type=item.kind_mime,
                            digest=documents.spell(item.digest),
                            length=item.length,
                            order=item.order,
                        )
                        for name, item in attempt.grant.inputs.items()
                        if name != "payload"
                    },
                ),
                timeout=None,
            )
        except ExecutorGone as exc:
            raise AttemptRefusal(
                "executor_absent",
                f"the executor died while resolving this request: {exc}",
                cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
            ) from exc
        reply = executor_replies.decode(raw, PreparedReply)
        if not reply.ok:
            code = reply.code
            if code == "poisoned_generation":
                # The poison lives executor-side (a failed load, a broken env seal or a
                # frozen attention refusal; request refusals and cooperative cancels no
                # longer poison). The refusal that names it is the supervision's one
                # signal — invalidate HERE so the lane's rebuild pass replaces the
                # generation instead of refusing forever (lifecycle-live §4).
                self.supervision_for(attempt).invalidate(
                    executor, (reply.detail or "poisoned")[:256]
                )
            raise AttemptRefusal(
                code,
                reply.detail[:1024],
                cause=CAUSE.CAUSE_CODE_INVALID_REQUEST
                if reply.origin in ("request", "author")
                else CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                origin=_ORIGIN.get(reply.origin, ORIGIN.CAUSE_ORIGIN_EXECUTOR),
            )
        return PreparedRequest(
            request_id=f"{attempt.request_id}#{attempt.attempt}",
            entrypoint=declared.entrypoint,
            features=reply.features,
            features_digest=reply.features_digest,
            overlay_digest=reply.overlay_digest,
            facts_digest=reply.facts_digest,
            adjustments=reply.adjustments,
        )

    def _hydrate_inputs(self, attempt: AttemptRecord) -> None:
        """Verify input files; the spool only materializes native retained inputs."""
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
            attempt.trees = grants.read_trees(
                attempt.grant,
                self.authorizer,
                workspace=Workspace(self.tensorfs_root) if self.tensorfs_root is not None else None,
                owner=self.owner_scope(),
                spool=spool,
                max_bytes=binding.max_inputs_total_bytes if binding else 256 << 20,
            )
            # A staged serving attempt is bound to no executor yet; `_run` grants the spool
            # to the executor it enters on.
            supervision = self.supervision_for(attempt)
            if getattr(supervision, "isolated", False) and attempt.executor_epoch:
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
            store = fill.store(self.tensorfs_root)
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
            if getattr(exc, "code", "") == "OBJECT_ABSENT":
                # No accepted/executed record exists yet. Use the existing bounded
                # pre-execution requeue path so the owner re-enters TensorFS ensure;
                # every later attempt reacquires its own native read leases.
                raise AttemptRefusal(
                    "model_materialization_required",
                    "a selected job model needs TensorFS materialization before admission",
                    cause=CAUSE.CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE,
                ) from exc
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

    def _accept_job(self, attempt: AttemptRecord) -> None:
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
        binding = self.slot_of(attempt).job_binding or self.jobs.get(descriptor_id)
        if binding is None or binding.installation_id != job.get("installation_id"):
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

        caps = slot.job_resource_caps
        plan = AttemptPlan.for_job(
            binding,
            {
                "vram": int(caps.get("max_device_memory_bytes", 0)),
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
                        "installation_id": binding.installation_id,
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
        self._accepted(attempt)

    def prepare_and_accept(
        self, attempt: AttemptRecord, *, before_plan: Callable[[], None] | None = None
    ) -> None:
        """Stage and enter in one step: the path of an attempt whose lane has no queue ahead
        of it (a job, or a caller that holds the device already)."""
        if attempt.spec.get("job") is not None:
            self._accept_job(attempt)
            return
        self.stage(attempt)
        if not self.claim_staged(attempt):
            raise AttemptRefusal(
                "attempt_left_queue",
                "the staged attempt was cancelled before it could enter the device",
                cause=CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
        self.enter(attempt, before_plan=before_plan)

    def stage(self, attempt: AttemptRecord) -> None:
        """QUEUE ADMISSION (proto-026, proto-061 G): validate -> fetch -> hydrate -> journal.

        Everything an attempt needs that is not the device happens here, while the lane's
        device may be running the attempt ahead of it: the binding is resolved against the
        slot, the payload fetched and decoded, the inputs hydrated into the spool, and the
        acceptance journaled. Neither a plan nor an executor epoch binds here; both bind at
        DEVICE ENTRY (`enter`), so a staged attempt survives an executor respawn. Everything
        before the journal record either completes or raises a typed pre-accept refusal.
        """
        serving = attempt.spec.get("serving")
        if serving is None:
            raise AttemptRefusal(
                "execution_mode_unspecified",
                "an ExecutionSpec names exactly ONE mode — serving or job — and this one "
                "names neither; a worker that guessed would run the wrong lifecycle",
                cause=CAUSE.CAUSE_CODE_PROTOCOL,
            )
        attempt.declared = self._resolve_declared_binding(attempt)
        attempt.state = "preparing"
        self._fetch_payload(attempt)
        # HYDRATE before acceptance (cr-012): an oversized, ungranted, mislabelled or
        # unreachable input is a refusal the RecordOwner can read, never an accepted attempt
        # that dies on a thread. It happens off the device, beside the running attempt.
        self._hydrate_inputs(attempt)
        self.records.append(
            "accepted",
            {
                "request_id": attempt.request_id,
                "attempt": attempt.attempt,
                "worker_boot_id": self.boot_id,
                "instance_id": self.instance_id,
                "invocation_spec_digest": documents.spell(attempt.digest),
                "grant_epoch": attempt.grant.credential_epoch,
                "placement_id": attempt.placement_id,
                "lane_id": attempt.lane_id,
                "entrypoint_binding_digest": serving["entrypoint_binding_digest"],
            },
        )
        self._accepted(attempt, "staged")

    def _resolve_declared_binding(self, attempt: AttemptRecord) -> DeclaredBinding:
        """Validate queued work against its declaration, before any model is loaded.

        The executor imports the package at startup and loads a construction only when
        its first queued attempt enters the device. Queue admission must therefore need
        only the immutable declaration; requiring PreparedModel here prevents that first
        attempt from ever reaching the load.
        """
        serving = attempt.spec["serving"]
        plan_id = serving["entrypoint_binding_digest"]
        slot = self.slot_of(attempt)
        declared = slot.bindings.get(plan_id)
        if not slot.bindings_digest or serving.get("bindings_digest") != slot.bindings_digest:
            raise AttemptRefusal(
                "model_bindings_changed",
                "serving invocation differs from the exact activated model bindings",
                cause=CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
        if declared is None:
            raise AttemptRefusal(
                "unknown_binding",
                f"no entrypoint binding {plan_id[:24]}… is selected on this worker",
                cause=CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
        if fault := slot.failed_bindings.get(plan_id):
            raise AttemptRefusal(
                "binding_faulted",
                f"{plan_id[:24]}… could not be loaded: {fault}",
                cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
            )
        if attempt.spec.get("installation_id") != declared.installation_id:
            raise AttemptRefusal(
                "environment_mismatch",
                "InvocationSpec does not name the selected Environment",
                cause=CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
        return declared

    def claim_staged(self, attempt: AttemptRecord) -> bool:
        """Take a STAGED attempt out of the queue — for device entry or for a cancel. Exactly
        one caller wins; the loser finds it gone."""
        with attempt.transition_lock:
            if attempt.state != "staged":
                return False
            attempt.state = "entering"
            return True

    def enter(
        self, attempt: AttemptRecord, *, before_plan: Callable[[], None] | None = None
    ) -> None:
        """DEVICE ENTRY for a claimed staged attempt, under the lane's device lock: bind the
        executor, resolve the request in it, choose the plan, journal the binding.

        The slot was re-bound for entry, so the binding is re-resolved against the
        placement as it stands now. A refusal here is typed and nothing reached the device.
        """
        slot = self.slot_of(attempt)
        declared = attempt.declared = self._resolve_declared_binding(attempt)
        plan_id = declared.entrypoint_binding_digest
        supervision = slot.supervision
        executor = supervision.current
        loaded = executor.loaded.get(declared.construction_key()) if executor else None
        if loaded is None or plan_id not in loaded.bindings:
            raise AttemptRefusal(
                "construction_not_loaded",
                f"entrypoint {declared.entrypoint!r} is selected but its construction "
                "has not loaded on this executor",
                cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
            )
        attempt.prepared_model = loaded.prepared_model
        if executor is None or not executor.alive():
            raise AttemptRefusal(
                "executor_absent",
                "no live executor: an online attempt enters the device only on a Ready one",
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
                f"not {plan_id[:24]}… — serving never enters an attempt on an executor that "
                "has not prepared its binding",
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
        if before_plan is not None:
            before_plan()
        try:
            plan = slot.chooser.choose(declared, loaded.prepared_model, prepared, fits=attempt.room)
        except PlanRefusal as exc:
            raise AttemptRefusal(
                exc.code, exc.detail, cause=CAUSE.CAUSE_CODE_CONSTRAINT_INFEASIBLE
            ) from exc
        attempt.plan = plan
        attempt.opened = slot.ledger.snapshot()

        with supervision.hold(executor) as owned:
            if not owned:
                raise AttemptRefusal(
                    "executor_replaced_before_entry",
                    "the prepared executor changed before device entry could be committed",
                    cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                )
            self.records.append(
                "entered",
                {
                    "request_id": attempt.request_id,
                    "attempt": attempt.attempt,
                    "plan_digest": plan.digest(),
                    "constructed_model_digest": plan.constructed_model_digest,
                    "request_features_digest": plan.request_features_digest,
                    "preflight_facts_digest": plan.preflight_facts_digest,
                    "executor_epoch": attempt.executor_epoch,
                    "executor_pid": attempt.executor_pid,
                },
            )
        attempt.state = "accepted"

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
        processes: list[dict[str, Any]] = []
        with supervision.hold(executor) as owned:
            if not owned:
                return self._unavailable_reconciliation(attempt, key)
            if not executor.poisoned and attempt.spool:
                try:
                    with executor.watched("probe"):
                        probe = executor.call(Probe(), timeout=None)
                        ledger.observe_probe(probe)
                        processes = list(probe.get("processes") or [])
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
        reuse_miss_reason = executor.poisoned or ("" if executor.alive() else "executor_gone")
        self.progress(
            attempt.request_id,
            attempt.attempt,
            {
                "kind": "log",
                "name": "executor release",
                "value": "info",
                "at_unix_ms": int(time.time() * 1000),
                "fields": {
                    "epoch": executor.epoch,
                    "pid": executor.pid,
                    "started_ticks": executor.process.started_ticks,
                    "ranks": processes,
                    "reusable": not bool(reuse_miss_reason),
                    "reuse_miss_reason": reuse_miss_reason,
                },
            },
        )
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

    def checkpoint_exchange(self, attempt: AttemptRecord, request: Checkpoint) -> Answer:
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
        a checkpoint with this digest and this length — the native writer will
        attach real object references to when #552's transaction lands.

        Three answers and no fourth:
          * a NEW identity is recorded and gets a fresh declaration id;
          * the SAME identity replays it and writes nothing a second time;
          * the same key with DIFFERENT bytes is a CONFLICT, never a replacement.
        """
        if attempt.job is None:
            return refuse(
                pb.CheckpointFaultCode.Name(
                    pb.CheckpointFaultCode.CHECKPOINT_FAULT_CODE_NOT_JOB_MODE
                ),
                "checkpoints are a JOB service; this attempt is not a job",
            )
        operation_key, logical_key = request.operation_key, request.logical_key
        content_digest = request.content_digest
        key = (attempt.request_id, operation_key, logical_key)
        held = self.receipts.get(key)
        if held is not None:
            if held.content_digest != content_digest:
                return refuse(
                    pb.CheckpointFaultCode.Name(
                        pb.CheckpointFaultCode.CHECKPOINT_FAULT_CODE_IDENTITY_CONFLICT
                    ),
                    f"{logical_key}: operation key {operation_key!r} is already durable at "
                    f"{held.content_digest[:23]}… and these bytes digest to "
                    f"{content_digest[:23]}… — one identity is one content, and a "
                    "checkpoint is never replaced",
                )
            return CheckpointReceipt(ok=True, receipt_id=held.receipt_id, replayed=True)

        record = _Checkpointed(
            request_id=attempt.request_id,
            attempt=attempt.attempt,
            worker_boot_id=self.boot_id,
            operation_key=operation_key,
            logical_key=logical_key,
            content_digest=content_digest,
            length=request.length,
            receipt_id=f"ckpt-{uuid.uuid4().hex[:24]}",
        )
        self.records.append("job_checkpoint", msgspec.to_builtins(record))
        self.receipts[key] = record
        attempt.ring.emit(
            "log",
            f"checkpoint {logical_key!r} DECLARED (bytes not stored: no weights sink)",
            "info",
            declaration=record.receipt_id,
            bytes=request.length,
        )
        return CheckpointReceipt(ok=True, receipt_id=record.receipt_id)

    def durable_exchange(self, attempt: AttemptRecord, request: Request) -> Reply:
        """Answer one durable request of a job attempt's executor."""
        match request:
            case Publish():
                return self.publish(attempt, request)
            case Checkpoint():
                return self.checkpoint_exchange(attempt, request)
            case DeviceRoom():
                return refuse("no_durable_exchange", "a job's executor is granted no device room")
            case TreeMember():
                if self.calls is None:
                    return refuse("child_broker_absent", "no RecordOwner call lane")
                return project_tree_member(self.calls, attempt, request)
            case WriterSource() | WriterOutput() | WriterAdopt():
                return self._writer(attempt, request)
            case _:
                if self.calls is None:
                    return refuse("child_broker_absent", "no RecordOwner call lane")
                return self.calls.handle(attempt, request)

    def publish(self, attempt: AttemptRecord, request: Publish) -> Reply:
        """One product onto the run's output log; nothing to show without a journal."""
        if self.products is None:
            return Published(ok=True)
        return self.products.publish(self.owner_scope(), attempt, request)

    def _serving_request(self, attempt: AttemptRecord) -> Handler | None:
        """A serving attempt's mid-call requests: products, and device room when granted."""
        room = self.room(attempt) if self.room else None

        def answer(request: Request) -> Reply:
            attempt.moved()
            if isinstance(request, Publish):
                return self.publish(attempt, request)
            if room is None:
                return refuse("no_durable_exchange", "a serving attempt asks only for room")
            return room(request)

        return answer

    def _writer(self, attempt: AttemptRecord, request: WriterRequest) -> Reply:
        if self.weights is None or attempt.spool is None:
            return refuse(
                "weights_host_unavailable", "this worker has no native TensorFS writer host"
            )
        if isinstance(request, WriterOutput):
            if attempt.job is None:
                return refuse("weights_host_refused", "a weights output requires the current job")
            try:
                declared = self._job_declaration(attempt.job, attempt.job.job_descriptor_id)
            except AttemptRefusal as exc:
                return refuse(exc.code, exc.detail)
            outputs = {str(row["output_id"]): row for row in declared.get("weights_outputs", [])}
            slot = request.output_slot
            bound = attempt.grant.outputs.get(slot)
            if (
                slot not in outputs
                or bound is None
                or bound.max_bytes != outputs[slot]["max_bytes"]
                or bound.mime_type != outputs[slot]["mime_type"]
            ):
                return refuse(
                    "weights_host_refused",
                    "weights intent is outside the package and invocation output contract",
                )
        return self.weights.writer_broker.handle(attempt, request)

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
        seconds = attempt.deadline_ms / 1000 - time.time() if attempt.deadline_ms else None
        key = f"{attempt.request_id}#{attempt.attempt}"
        command = self._command(attempt, key, spool, seconds)
        # DISPATCH is one decision against a cancel: either the cancel's mark is seen here
        # and nothing runs, or execution starts (journaled first) and the cancel signals it.
        with attempt.transition_lock:
            if not attempt.canceling:
                self.dispatched(attempt)
                # The pace opens at dispatch, seeded with the gaps this process already showed.
                attempt.pace = liveness.Pace(worst_pause=executor.worst_pause)
                attempt.pace.observe(0)
                attempt.state = "running"
                attempt.executions += 1
        if attempt.state != "running":
            return self.finish_cancel(attempt, attempt.canceling)
        self.executed += 1
        attempt.poke()
        attempt.dispatched_at = time.perf_counter()
        self.progress(
            attempt.request_id,
            attempt.attempt,
            {
                "kind": "log",
                "name": "executor invoke",
                "value": "info",
                "at_unix_ms": int(time.time() * 1000),
                "fields": {
                    "epoch": executor.epoch,
                    "pid": executor.pid,
                    "started_ticks": executor.process.started_ticks,
                },
            },
        )
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

        def on_progress(frame: dict[str, Any]) -> None:
            if frame.get("kind") != "execution_activity":
                attempt.moved()
            self.progress(attempt.request_id, attempt.attempt, frame)

        def on_request(request: Request) -> Reply:
            attempt.moved()
            return self.durable_exchange(attempt, request)

        try:
            reply = executor_replies.decode(
                executor.call(
                    command,
                    timeout=None,
                    on_progress=on_progress,
                    on_request=on_request if job else self._serving_request(attempt),
                ),
                AttemptReply,
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
            executor.worst_pause = max(executor.worst_pause, attempt.pace.worst_pause)
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
    ) -> RunJob | Invoke:
        """The ONE thing the two lanes genuinely differ in: what they ask the executor."""
        inputs = {
            input_id: InputFile(
                local=str(granted.local),
                file_state=granted.file_state,
                media_type=granted.media_type,
                digest=granted.digest,
                length=granted.length,
                order=granted.order,
            )
            for input_id, granted in attempt.inputs.items()
        }
        trees = {ref: (str(root), digest) for ref, (root, digest) in attempt.trees.items()}
        max_output_bytes = output_budget.intermediate(attempt.spec)
        if attempt.job is not None:
            if not attempt.weights_work_fingerprint and self.work_fingerprint is not None:
                attempt.weights_work_fingerprint = self.work_fingerprint(attempt)
            return RunJob(
                request_id=key,
                application=attempt.job.application,
                package_interface=attempt.job.package_interface,
                job=attempt.job.job,
                call_interfaces=self.calls.bindings(attempt) if self.calls is not None else (),
                publish_to=output_destination(
                    output.url for output in attempt.grant.outputs.values()
                ),
                payload=attempt.payload,
                deadline_s=seconds,
                spool=str(spool),
                scratch=str(attempt.scratch) if attempt.scratch else "",
                trees=trees,
                inputs=inputs,
                models={
                    parameter: ModelManifest(
                        class_key=model.class_name, manifest=model.manifest, length=model.length
                    )
                    for parameter, model in sorted(attempt.job_models.items())
                },
                max_input_bytes=64 << 20,
                max_output_bytes=max_output_bytes,
                budget=JobBudget(
                    gpu_rate_micro_usd_per_hour=attempt.job.gpu_rate_micro_usd_per_hour,
                    gpu_count=self.slot_of(attempt).job_device_count,
                    cap_micro_usd=attempt.job.cap_micro_usd,
                ),
            )
        assert attempt.declared is not None and attempt.plan is not None
        return Invoke(
            construction=attempt.declared.construction_key(),
            capture=attempt.spec.get("capture"),
            attention_kernel=str(attempt.spec.get("attention_kernel") or ""),
            request_id=key,
            entrypoint=attempt.declared.entrypoint,
            deadline_s=seconds,
            spool=str(spool),
            placement=attempt.plan.placement,
            headroom_bytes=attempt.plan.headroom_bytes,
            scope_headroom_bytes=dict(attempt.plan.scope_headroom_bytes),
            measured_scopes=tuple(attempt.plan.measured_scopes),
            # cr-012: the VERIFIED inputs, by their stable identity. Paths into this
            # attempt's own spool and nothing else — no URL, no credential and no fetch
            # reaches the executor. What the executor CAN still do is dial one of its own,
            # which is why it installs `sandbox.refuse_network` (cr-042); that fence bounds
            # a buggy package at the Python layer and is not a kernel bound.
            inputs=inputs,
            trees=trees,
            max_input_bytes=attempt.declared.max_input_bytes,
            max_output_bytes=max_output_bytes,
        )

    def release(
        self, attempt: AttemptRecord, reply: AttemptReply, spool: Path, key: str
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
        attempt.execution_observation = dict(reply.execution_observation)
        attempt.execution = dict(reply.execution)
        if isinstance(attempt.execution.get("ranks"), list):
            attempt.execution["gpus"] = execution_evidence.by_gpu(
                attempt.execution["ranks"], self.inventory
            )
        attempt.attention_applied = reply.attention_applied
        attempt.capture_result = reply.capture
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
                    terminal = reply.outcome.terminal if reply.outcome else ""
                    ledger = self.ledger_for(attempt)
                    ledger.observe_attempt(
                        reply.metrics,
                        resident_moved=moved,
                        cell=cell,
                        succeeded=terminal == "succeeded",
                    )
                    terminal = reply.outcome.terminal if reply.outcome else ""
                else:
                    attempt.confess(
                        "unavailable",
                        f"executor epoch {attempt.executor_epoch} was replaced "
                        "before its terminal telemetry could be applied; successor telemetry "
                        "was left untouched",
                    )
        if reply.ok and executor is not None:
            supervision = self.supervision_for(attempt)
            if not reply.quiescent:
                # No terminal is recorded while author computation might still mutate GPU
                # or output state. Unproven quiescence is an executor fault, not a success.
                supervision.invalidate(executor, "unproven quiescence")
            elif reply.poisoned:
                supervision.invalidate(executor, reply.poisoned)
        # ONE ENVELOPE, ONE CLOSE, at release: the probe reads the executor's after-state
        # for THIS attempt, before the lane hands the device to the next one.
        if attempt.opened:
            self.reconcile(attempt)
            attempt.opened = {}
        attempt.state = "released"
        attempt.poke()
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
        self, attempt: AttemptRecord, reply: AttemptReply, spool: Path, key: str
    ) -> pb.AttemptOutcome:
        if attempt.canceling:
            return self.finish_cancel(attempt, attempt.canceling)
        if attempt.weights_refused:
            # The Host holds no custody for one of its outputs: whatever the handler did,
            # this attempt cannot deliver it. Only this attempt fails.
            grants.abort_outputs(spool)
            return self.outcome(
                attempt,
                STATUS.OUTCOME_STATUS_FAILED,
                CAUSE.CAUSE_CODE_LOCAL_SAFETY,
                ORIGIN.CAUSE_ORIGIN_WORKER,
                f"weights_custody_refused: {attempt.weights_refused}",
            )
        if not reply.ok or reply.outcome is None:
            grants.abort_outputs(spool)
            return self.outcome(
                attempt,
                STATUS.OUTCOME_STATUS_FAILED,
                CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                ORIGIN.CAUSE_ORIGIN_EXECUTOR,
                f"{reply.code or 'executor_outcome_absent'}: {reply.detail}",
                shortfall=reply.shortfall,
            )
        if not reply.quiescent:
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
        outcome = reply.outcome
        status, cause = _TERMINAL[outcome.terminal]
        attempt.traceback = outcome.traceback[-16384:]
        breach = outcome.code == DEVICE_OOM
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
        if outcome.terminal == "failed" and outcome.code in GROUP_FAULT_CODES:
            cause = CAUSE.CAUSE_CODE_EXECUTOR_FAULT
        if outcome.terminal == "refused" and outcome.code in (
            "unsupported_input",
            "tree_ungranted",
        ):
            cause = CAUSE.CAUSE_CODE_UNSUPPORTED_INPUT
        if outcome.terminal == "canceled":
            cause = (
                CAUSE.CAUSE_CODE_DEADLINE_EXPIRED
                if outcome.code == "deadline"
                else CAUSE.CAUSE_CODE_CLIENT_CANCEL
            )
        oversize = reply.oversize_result_bytes
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
                    metrics=reply.metrics,
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
                        metrics=reply.metrics,
                    )
        else:
            grants.abort_outputs(spool)
        origin = _ORIGIN[outcome.origin]
        shortfall = reply.shortfall
        if shortfall is not None:
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
            metrics=reply.metrics,
            result=inline,
            shortfall=shortfall,
        )

    def forward(self, request_id: str, number: int, frame: Mapping[str, Any]) -> None:
        """Hold ONE live observation the executor just emitted, before it can be lost. A
        `load` frame is a fill's MiB position for liveness, not an observation: thousands of
        them per staged attempt would shed every real row from the bounded ring."""
        attempt = self.live.get(request_id)
        if attempt is None or attempt.attempt != number or frame.get("kind") == "load":
            return
        try:
            row = msgspec.convert(frame, ObservationRow)
        except msgspec.ValidationError:
            return  # the lossy lane sheds a row it cannot read
        attempt.ring.admit(self._row(row))
        attempt.forwarded_seq = max(attempt.forwarded_seq, row.seq)

    @staticmethod
    def _row(row: ObservationRow) -> Observation:
        """One executor-side row, re-bounded on this side. A disposable process is not
        trusted to bound the worker's memory, so every cap is re-applied here."""
        return Observation(
            cast(ObservationKind, row.kind),
            row.name[:200],
            row.value,
            {key[:48]: value for key, value in list(row.fields.items())[:8]},
            row.at_unix_ms,
        )

    def absorb(
        self,
        attempt: AttemptRecord,
        reply: AttemptReply,
        *,
        apply_ledger: bool = True,
    ) -> None:
        """Take the executor's BOUNDED observation tail into the worker's own ring.

        The executor's ring already applied every cap and every credential check at the emit
        boundary, so this side re-applies the CAPS (a disposable process is not trusted to
        bound the worker's memory) and keeps the rows as they came. Whatever the ring
        already shed is carried through as a number, never inferred.
        """
        caps = dict(reply.observation_caps)
        attempt.attribution = dict(reply.attribution)
        rows: list[ObservationRow] = []
        unreadable: list[str] = []
        for frame in reply.observations[-attempt.ring.max_events :]:
            try:
                rows.append(msgspec.convert(frame, ObservationRow))
            except msgspec.ValidationError as exc:
                name = frame.get("name") if isinstance(frame, dict) else None
                unreadable.append(f"{name!r}: {exc}")
        if unreadable:
            # Telemetry never fails the attempt it observes: the row goes, the result stays.
            _LOG.warning(
                "%s#%d: dropped %d executor observation(s) this worker cannot read; first %s",
                attempt.request_id,
                attempt.attempt,
                len(unreadable),
                unreadable[0],
            )
        kept = 0
        for row in rows:
            # Anything the lossy lane already delivered is skipped by the EXECUTOR's own
            # sequence. Both sides shed oldest first, so this fills the gap after the last
            # forwarded row and can never double-count one.
            if row.seq <= attempt.forwarded_seq:
                continue
            attempt.ring.admit(self._row(row))
            kept += 1
        residency = reply.residency
        if residency:
            attempt.residency = dict(residency)
            # THE DECLARED TRANSITION, into the ledger BEFORE it reconciles. Under a staged
            # rung the attempt moved gigabytes on purpose; the reconciliation subtracts
            # exactly this and holds the remainder to zero.
            if apply_ledger:
                self.ledger_for(attempt).observe_residency(attempt.residency)
            if residency.get("attempt_stages"):
                # ATTEMPT-SCOPED THROUGHOUT (cr-134). `staged_ms`/`evicted_ms` are the
                # generation's whole history across every request it has served; only the
                # `attempt_` counters were zeroed for this one. Reading a count from one
                # scope beside a duration from the other reported 6 stages costing 31220.3
                # ms on a run whose six stages cost 5542.85 ms, and that number became the
                # case for changing residency policy.
                attempt.confess(
                    "degraded",
                    f"served on the {residency.get('placement')} rung: "
                    f"{residency['attempt_stages']} component stage(s) costing "
                    f"{residency.get('attempt_staged_ms', 0)} ms and "
                    f"{residency.get('attempt_evictions', 0)} eviction(s) costing "
                    f"{residency.get('attempt_evicted_ms', 0)} ms - the weights did not all "
                    "fit at once and the runtime moved them rather than refusing",
                )
        shortfall = reply.shortfall
        if shortfall is not None:
            attempt.confess(
                "degraded",
                f"{shortfall.resource} shortfall at {shortfall.scope}: needed "
                f"{shortfall.needed_bytes} B, {shortfall.available_bytes} B available, "
                f"short by {shortfall.needed_bytes - shortfall.available_bytes} B",
            )
        caps["executor_rows"] = len(reply.observations)
        caps["unreadable_rows"] = len(unreadable)
        caps["forwarded_live"] = attempt.forwarded_seq
        caps["absorbed_from_reply"] = kept
        caps.update(attempt.ring.caps())
        attempt.caps = caps
        # Every caller-visible divergence is ALSO an operator-visible confession. The
        # envelope tells the caller what changed; the confession tells the operator why,
        # with the numbers, in the record that outlives the process.
        for adjusted in reply.adjustments:
            attempt.confess(
                "substituted" if adjusted.reason.startswith("adjusted") else "degraded",
                f"{adjusted.field}: requested {adjusted.requested}, served "
                f"{adjusted.applied} - {adjusted.reason}",
            )

    def bank_demand(self, attempt: AttemptRecord, metrics: Metrics, moved: bool) -> None:
        """Every serve evaluates DECLARED versus ACTUAL and banks the pair (§3.2).

        The actual is the attempt's own peak above what it started holding — a number that
        survives staging, unlike a figure measured against a construction baseline the
        attempt then moved out from under itself. A miss is COUNTED and CONFESSED; it
        selects nothing and refuses nothing.
        """
        plan = attempt.plan
        if plan is None:
            return
        peak, started = metrics.peak_vram_bytes, metrics.allocated_at_start_bytes
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
        confession = self.demand.bank(lane, plan.headroom_bytes, actual)
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
        self, attempt: AttemptRecord, reply: AttemptReply, key: str = ""
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
        rows = reply.outputs
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
        if attempt.spec.get("capture"):
            if reply.capture is None:
                return None, "capture_output_absent: requested activation capture was not produced"
            # Runtime writes this reserved native tree after normal author outputs.
            granted.discard("runtime.capture")
        if not weights_outputs <= set(attempt.grant.outputs):
            return None, "weights_receipt_set_mismatch: a weights output has no granted destination"
        spool = attempt.spool
        if not rows and not granted and reply.capture is None:
            return None, None
        if spool is None:
            return None, "the attempt has no spool to read outputs from"
        produced = {row.output_id: row for row in rows}
        # A list output's one `x.*` slot grants every produced item `x.0`, `x.1`, ...
        slot = {
            output_id: machine_byte_results.slot_of(output_id, granted) for output_id in produced
        }
        if None in slot.values() or {s for s in granted if not s.endswith(".*")} - set(produced):
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
        native = self.tensorfs_root is not None and self.calls is not None
        try:
            if len(rows) + (reply.capture is not None) > 32:
                raise grants.GrantRefusal(
                    "output_count", "native outputs and capture exceed 32 slots"
                )
            if grants.expired(attempt.grant):
                raise grants.GrantRefusal(
                    "grant_expired", "output grant expired before post processing"
                )
            for output_id in sorted(produced):
                row = produced[output_id]
                received = (
                    self.calls.received_output(attempt, row)
                    if native and self.calls is not None and row.asset_ref.startswith("sha256:")
                    else None
                )
                tree = row.kind == "tree"
                if tree or (native and not attempt.grant.outputs[slot[output_id] or output_id].url):
                    if not native:
                        raise grants.GrantRefusal(
                            "native_tree_unsupported", "tree outputs require wire43 native custody"
                        )
                    from . import byte_outputs
                    from .workspace import Workspace
                    from .workspace_byte_outputs import TREE_MIME

                    root = (
                        received[0]
                        if received
                        else self._blob_path(spool, row.asset_ref)
                        if tree
                        else self._blob(spool, row.asset_ref)
                    )
                    if root is None or self.tensorfs_root is None:
                        raise grants.GrantRefusal(
                            "output_spool_io", "tree output has no owned spool"
                        )
                    destination = attempt.grant.outputs[slot[output_id] or output_id]
                    media_type = grants._output_media_type(
                        output_id,
                        destination.mime_type,
                        TREE_MIME if tree else row.media_type,
                    )
                    if received:
                        entry = byte_outputs.reexport(
                            Workspace(self.tensorfs_root),
                            self.owner_scope(),
                            attempt.request_id,
                            attempt.attempt,
                            attempt.digest,
                            output_id,
                            received[1],
                            tree=tree,
                            max_bytes=destination.max_bytes or (256 << 20),
                            media_type=media_type,
                        )
                    else:
                        entry = byte_outputs.commit(
                            Workspace(self.tensorfs_root),
                            self.owner_scope(),
                            attempt.request_id,
                            attempt.attempt,
                            attempt.digest,
                            output_id,
                            root,
                            tree=tree,
                            max_bytes=destination.max_bytes or (256 << 20),
                            media_type=media_type,
                        )
                    entries.append(entry)
                    continue
                blob = received[0] if received else self._blob(spool, row.asset_ref)
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
                        media_type=row.media_type,
                    )
                )
                if native:
                    from . import byte_outputs
                    from .workspace import Workspace

                    assert self.tensorfs_root is not None
                    if received:
                        committed = byte_outputs.reexport(
                            Workspace(self.tensorfs_root),
                            self.owner_scope(),
                            attempt.request_id,
                            attempt.attempt,
                            attempt.digest,
                            output_id,
                            received[1],
                            tree=False,
                            max_bytes=attempt.grant.outputs[slot[output_id] or output_id].max_bytes
                            or (256 << 20),
                            media_type=entries[-1].mime_type,
                        )
                    else:
                        committed = byte_outputs.commit(
                            Workspace(self.tensorfs_root),
                            self.owner_scope(),
                            attempt.request_id,
                            attempt.attempt,
                            attempt.digest,
                            output_id,
                            blob,
                            tree=False,
                            max_bytes=attempt.grant.outputs[slot[output_id] or output_id].max_bytes
                            or (256 << 20),
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
            if (capture := reply.capture) is not None:
                if not native or not attempt.spec.get("capture") or "runtime.capture" in produced:
                    raise grants.GrantRefusal(
                        "capture_ungranted", "capture output has no exact requested slot"
                    )
                from . import byte_outputs
                from .workspace import Workspace
                from .workspace_byte_outputs import TREE_MIME

                assert self.tensorfs_root is not None
                root = spool / "runtime.capture"
                if capture.root != str(root) or capture.output_id != "runtime.capture":
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
                if entry.native_tree.content_bytes != capture.length:
                    raise grants.GrantRefusal(
                        "capture_output", "capture output changed its measured length"
                    )
                content = hashlib.sha256()
                for name in ("capture.json", "sketches.f32"):
                    with (root / name).open("rb") as member:
                        while block := member.read(1 << 20):
                            content.update(block)
                if documents.spell(content.digest()) != capture.content_digest:
                    raise grants.GrantRefusal(
                        "capture_output", "capture semantic digest differs from retained bytes"
                    )
                entries.append(entry)
            if sum(
                entry.native_tree.content_bytes if entry.HasField("native_tree") else entry.length
                for entry in entries
            ) > output_budget.effective(attempt.spec, executor_replies.budget_claim(reply)):
                raise grants.GrantRefusal(
                    "output_too_large", "returned bytes exceed the output budget"
                )
            if self.products is not None and (
                changed := self.products.finish(self.owner_scope(), attempt, entries)
            ):
                grants.abort_outputs(spool)
                return None, changed
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
            installation_id=(
                attempt.job.installation_id
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
        self, attempt: AttemptRecord, spool: Path, reply: AttemptReply, key: str
    ) -> str | None:
        """Encode every host frame the handler REGISTERED (cr-079 §5): first-party codecs,
        the aggregate ceiling re-checked on the ENCODED bytes, each blob taking its
        handle's spool name so the write below finds it as it always did. A fault here is
        typed and the worker's own; the executor was not involved."""
        frames = reply.frames
        if not frames:
            return None
        try:
            budget = output_budget.effective(attempt.spec, executor_replies.budget_claim(reply))
        except ValueError as exc:
            return f"output_budget_invalid: {exc}"
        handles = {frame.handle for frame in frames}
        total = sum(row.size_bytes or 0 for row in reply.outputs if row.asset_ref not in handles)
        for index, frame in enumerate(frames):
            handle, name = frame.handle, frame.raw
            blob = self._blob_path(spool, handle)
            if blob is None or not name or name != Path(name).name:
                return f"output_frame_invalid: frame {handle!r} names no spool blob"
            raw = spool / name
            try:
                data = raw.read_bytes()
                if len(data) != frame.raw_bytes:
                    return (
                        f"output_frame_invalid: {handle} registered {frame.raw_bytes} "
                        f"raw bytes and the spool holds {len(data)}"
                    )
                started = time.perf_counter()
                encoded = _codec.encode_frame(frame.codec, dict(frame.facts), data)
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
                f"encoded {handle} as {frame.codec}",
                "info",
                raw_bytes=frame.raw_bytes,
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
        metrics: Metrics | None = None,
        result: InlineResult | None = None,
        shortfall: Shortfall | None = None,
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
            if attempt.capture_result is not None and any(
                row.output_id == "runtime.capture" for row in manifest.outputs
            ):
                body.observation.capture.CopyFrom(
                    pb.ActivationCaptureResult(
                        output_id="runtime.capture",
                        content_digest=documents.raw(attempt.capture_result.content_digest),
                    )
                )
        if shortfall is not None:
            body.cause.shortfall.CopyFrom(
                pb.ResourceShortfall(
                    resource=shortfall.resource,
                    scope=shortfall.scope,
                    needed_bytes=shortfall.needed_bytes,
                    available_bytes=shortfall.available_bytes,
                    evidence_class=shortfall.evidence_class,
                )
            )
        if metrics is not None:
            body.metrics.CopyFrom(self.attest(attempt, metrics, manifest))
        if result is not None:
            body.result.CopyFrom(self._envelope(result))
            if (
                status == pb.OUTCOME_STATUS_SUCCEEDED
                and attempt.job is not None
                and self.tensorfs_root is not None
            ):
                from cozy_runtime.internal.worker import machine_models
                from cozy_runtime.internal.worker.workspace import Workspace

                workspace = Workspace(self.tensorfs_root)
                owner = self.owner_scope()
                with workspace.locked() as db:
                    managed = (
                        db.execute(
                            "SELECT 1 FROM executions WHERE owner=? AND request=?",
                            (owner, attempt.request_id),
                        ).fetchone()
                        is not None
                    )
                if managed:
                    try:
                        schema = self._job_declaration(attempt.job, attempt.job.job_descriptor_id)[
                            "result"
                        ]
                        records = machine_models.result_records(
                            workspace,
                            owner,
                            attempt.request_id,
                            canonical.parse_canonical(result.canonical),
                            schema,
                        )
                        for pointer, artifact, retention in records:
                            body.result.retained_models.append(
                                pb.RetainedModelResult(
                                    result_pointer=pointer,
                                    model_artifact_canonical_bytes=artifact,
                                    retention=retention,
                                )
                            )
                    except Exception as exc:
                        status, cause, origin = (
                            pb.OUTCOME_STATUS_FAILED,
                            pb.CAUSE_CODE_LOCAL_SAFETY,
                            pb.CAUSE_ORIGIN_WORKER,
                        )
                        detail = "result_custody_unavailable: " + safe(str(exc), 1024)
                        body.status, body.safe_message = status, detail
                        body.cause.CopyFrom(
                            pb.OutcomeCause(code=cause, origin=origin, detail=detail)
                        )
                        body.ClearField("result")
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
        attempt.poke()
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
        pinned = attempt.prepared_model.attention if attempt.prepared_model else None
        if pinned:
            # `plan.attention` (cr-124): what the prepare pinned, uncapped, for triage. The
            # boot note carries the one-line version; this is where the full map lives.
            plan_document["attention"] = pinned
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
                "execution": attempt.execution,
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
            streams=attempt.ring.stream_rows(),
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

    def _accepted(self, attempt: AttemptRecord, state: str = "accepted") -> None:
        """HOLD the accepted attempt. The one acceptance both lanes make: a job is bound at
        acceptance ("accepted"), a serving attempt queues ("staged")."""
        attempt.state = state
        attempt.accepted_at = time.perf_counter()
        self.live[attempt.request_id] = attempt
        self.history[attempt.key()] = attempt

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
        self, attempt: AttemptRecord, reply: AttemptReply
    ) -> tuple[InlineResult | None, str | None]:
        """Read the result bytes the executor spooled, and hold them to their declared identity.

        The seam carries CONTROL. The result document takes the same road the output blobs
        take — the attempt spool the worker brokered — and only its digest and length
        cross the frame, so a result between the 64 KiB frame cap and the 4 MiB public door
        is served instead of becoming an executor fault (reproduced live at 71,712 B).

        The worker names the file. Nothing here is read from a path the child chose.
        """
        ref = reply.result_ref
        if ref is None:
            return None, None
        if attempt.spool is None:
            return None, "the attempt has no spool to read its result document from"
        path = attempt.spool / seam.RESULT_DOCUMENT
        try:
            data = path.read_bytes()
        except OSError as exc:
            return None, f"result_document_unreadable: {type(exc).__name__}"
        if len(data) != ref.length:
            return None, (
                f"result_document_length: the executor declared {ref.length} B and "
                f"the spool holds {len(data)} B"
            )
        spelled = "sha256:" + hashlib.sha256(data).hexdigest()
        if spelled != ref.digest:
            return None, (
                f"result_document_digest: the executor declared {ref.digest[:23]}… "
                f"and the spooled bytes hash to {spelled[:23]}…"
            )
        return (
            InlineResult(
                canonical=data,
                schema_digest=reply.result_schema_digest or spelled,
                adjustments=reply.adjustments,
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
                    field=row.field, requested=row.requested, applied=row.applied, reason=row.reason
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
        claim: Metrics,
        manifest: pb.OutputManifest | None,
    ) -> pb.AttemptMetrics:
        """The executor's numbers are a CLAIM; this is the worker's attestation.

        Wall-clock is clamped to the window the worker itself observed (dispatch ->
        result). Anything it cannot observe is NAMED, never silently promoted.
        """
        window_ms = (attempt.resulted_at - attempt.dispatched_at) * 1000
        queue_ms = (attempt.dispatched_at - (attempt.queued_at or attempt.accepted_at)) * 1000
        return pb.AttemptMetrics(
            runtime_ms=int(min(claim.handler_ms, window_ms)),
            queue_ms=int(max(queue_ms, 0.0)),
            peak_device_memory_bytes=claim.peak_vram_bytes,
            rss_at_end_bytes=claim.rss_at_end_bytes,
            output_count=len(manifest.outputs) if manifest is not None else 0,
            device_lease_ms=int(min(claim.device_lease_ms, window_ms)),
            device_count=claim.gpu_count,
            handler_ms=int(min(claim.handler_ms, window_ms)),
            finalization_ms=int(max((time.perf_counter() - attempt.resulted_at) * 1000, 0)),
            unverified_fields=list(UNVERIFIABLE),
            working_peak_device_bytes=claim.working_peak_vram_bytes,
            # The worker's own plan names the cell it priced; the claim only fills a gap.
            shape_cell=(attempt.plan.shape_cell if attempt.plan is not None else "")
            or claim.shape_cell,
        )

    # ------------------------------------------------------------------ cancel / abandon

    def cancel(self, message: pb.CancelAttempt, *, failure: str = "") -> str:
        """Cooperative, then forceful. Returns what was done, for the activity lane. A
        `failure` stops the attempt the same way and ends it FAILED with that reason."""
        held = self.live.get(message.request_id)
        if held is None or held.attempt != message.attempt_ordinal:
            return "dropped: no such held attempt"
        if held.digest != message.invocation_spec_digest:
            return "dropped: cancel digest does not match the recorded acceptance"
        with held.transition_lock:
            if held.state in ("finalizing", "released", "outcome", "closed"):
                return "ignored: the attempt already left the device"
            if failure:
                held.canceling, held.failure = "failed", failure
            elif not held.canceling:  # the first cause is the terminal's cause
                held.canceling = {
                    pb.CancelReason.CANCEL_REASON_CLIENT: "client",
                    pb.CancelReason.CANCEL_REASON_DRAIN: "drain",
                    pb.CancelReason.CANCEL_REASON_SUPERSEDED: "superseded",
                    pb.CancelReason.CANCEL_REASON_POLICY: "policy",
                    pb.CancelReason.CANCEL_REASON_DEADLINE: "deadline",
                }.get(message.reason, "client")
            held.cancel_started = held.cancel_started or time.monotonic()
            held.poke()
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
            if self.calls is not None:
                self.calls.wake(held.request_id)  # a parent awaiting children reads it now
            return f"cooperative cancel marked ({held.canceling})"

    def escalate(self, attempt: AttemptRecord, stall: Mapping[str, float]) -> dict[str, Any]:
        """The FORCEFUL half, and the exact ordering §5 requires: forced kill -> output
        abort -> epoch invalidation -> record CANCELED. Rebuilding readiness happens
        AFTER, off the cancellation critical path. `stall` is the measurement that
        justified it (`AttemptRecord.cancel_stall`)."""
        executor = attempt.executor
        started = time.perf_counter()
        evidence = None
        if not attempt.begin_reclaim():
            return {
                **stall,
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
            **stall,
            "killed": int(evidence is not None),
            "exit_status": evidence.exit_status if evidence is not None else None,
            "kill_to_gone_ms": round((gone_at - started) * 1000, 2),
            "aborted_outputs": aborted,
            "executor_epoch": attempt.executor_epoch,
            "device_reclaim": evidence.device.state if evidence is not None else "stale",
        }

    def finish_cancel(self, attempt: AttemptRecord, reason: str) -> pb.AttemptOutcome:
        if reason == "failed":
            return self.outcome(
                attempt,
                STATUS.OUTCOME_STATUS_FAILED,
                CAUSE.CAUSE_CODE_LOCAL_SAFETY,
                ORIGIN.CAUSE_ORIGIN_WORKER,
                attempt.failure,
            )
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
        attempt.poke()
        self._release_job_models(attempt)
        return self.wrap(attempt)

    def abandon(self, attempt: AttemptRecord, why: str) -> pb.AttemptOutcome:
        """An accepted-but-incomplete attempt ALWAYS produces a terminal, never silence."""
        if attempt.spool is not None:
            grants.abort_outputs(attempt.spool)
        if attempt.executor is not None and attempt.executions:
            # Only a dispatched attempt can have left the process mid-command.
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
        queued = attempt.state == "staged"
        held = pb.HeldAttempt(
            request_id=attempt.request_id,
            attempt_ordinal=attempt.attempt,
            kind=_KIND[attempt.kind],
            state=pb.AttemptState.ATTEMPT_STATE_OUTCOME_PENDING_ACK
            if pending
            else pb.AttemptState.ATTEMPT_STATE_QUEUED
            if queued
            else pb.AttemptState.ATTEMPT_STATE_RUNNING,
            invocation_spec_digest=attempt.digest,
            placement_id=str(row.get("placement_id", "")) or attempt.placement_id,
            # A queued attempt is bound to no executor (proto-026): epoch 0 until entry.
            executor_epoch=0
            if queued
            else int(row.get("executor_epoch", 0)) or attempt.executor_epoch,
            lane_id="" if attempt.kind == "job" else attempt.lane_id,
        )
        if pending:
            # Set IFF the state is OUTCOME_PENDING_ACK, so the RecordOwner can reconcile the
            # exact outcome it owes an ack for without a second round trip.
            held.outcome_id = attempt.outcome_id
            held.outcome_digest = attempt.outcome_digest
        return held

    #: Every state in which this worker still holds an attempt and owes the owner something.
    #: `staged` is accepted into its lane's queue (wire QUEUED); `entering` is claimed off
    #: the queue for device entry or for a cancel.
    HELD = (
        "staged",
        "entering",
        "accepted",
        "running",
        "reclaiming",
        "finalizing",
        "released",
        "outcome",
    )
    #: The states that hold a LANE SEAT (cr-079 D3, proto-061 G): from the offer's hold to
    #: device release — the lane's queue (`queued` on the reader, `staged`) and its device.
    PRE_RELEASE = (
        "queued",
        "staged",
        "entering",
        "accepted",
        "running",
        "reclaiming",
        "finalizing",
    )

    def snapshot_entries(self) -> list[pb.HeldAttempt]:
        """Every attempt this worker still holds — staged, running, or outcome-pending-ack.

        An offer still crossing to its lane's stage (`queued`, not yet accepted) is absent:
        acceptance is what makes it a claim, and the owner's DISPATCHING-absent rule
        resends it."""
        return [self._active_row(a) for a in self.history.values() if a.state in self.HELD]

    def active(self) -> list[pb.HeldAttempt]:
        """The attempts this worker HOLDS, for the ObservedWorkerState tick.

        Running attempts, released ones in their post phase, AND outcomes pending ack — the
        same set the snapshot carries. Reading only `live` under-reported exactly those: `close()`
        drops the attempt from `live`, but an outcome nobody acked is still owed.

        An offer not yet accepted into its lane's queue is absent: acceptance is what makes
        an attempt a claim.
        """
        return [self._active_row(a) for a in self.history.values() if a.state in self.HELD]

    def in_flight(self) -> int:
        """How many attempts this worker holds — `active()`'s set, counted without building
        its rows. This includes unacknowledged outcomes; it is not an idle signal."""
        return sum(1 for a in self.history.values() if a.state in self.HELD)
