"""The attempt engine: offer -> prepare -> accept -> execute -> terminal, and the warm pass.

Everything the worker may CLAIM about an attempt passes through here, in the one order
worker-protocol/02 fixes:

    VALIDATING  fences, digest recomputation over the resident bytes, local safety
    PREPARING   the PlanChooser fixes ONE plan; the ExecutorPreparer applies it
    ACCEPTED    acceptance journaled (spec + plan + model-construction digests) FIRST
    TERMINAL    the canonical TerminalBody journaled BEFORE it is sent, replayed byte-for-
                byte until a TerminalAck echoes its id AND its digest
    CLOSED      closure recorded

Two things are deliberately NOT here, because they are not the worker's: retryability (a
coordinator projection over status/cause/origin) and attempt-selection order beyond this
worker (dispatch is the coordinator's; below that boundary service class does not exist and
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
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from pathlib import Path
from typing import Any, cast

from cozy_runtime.author._errors import DEVICE_OOM
from cozy_runtime.author._observations import EventRing, Observation
from cozy_runtime.internal import canonical, egress, seam
from cozy_runtime.internal.journal import Journal, JournalError
from cozy_runtime.internal.worker import grants, observe, refusal, triage
from cozy_runtime.internal.worker.child import ExecutorGone, ExecutorSupervision
from cozy_runtime.internal.worker.ledger import Ledger, LedgerRefusal
from cozy_runtime.internal.worker.plan import (
    AttemptPlan,
    DeclaredBinding,
    JobBinding,
    PlanRefusal,
    PreparedModel,
    PreparedRequest,
)
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

#: `safe_message` and `cause.detail` are bounded and SANITIZED (01 §3): printable ASCII
#: only, no secrets, no local paths, no signed URLs. The document profile is integer-only
#: and printable-ASCII, so an unsanitized message is not merely impolite — it cannot be
#: canonicalized at all, and the terminal that carries it could not be journaled.
_PATHY = re.compile(r"(file|https?)://\S+|(?<![\w.])/[\w./-]{4,}")


def safe(text: str, cap: int = 1024) -> str:
    """Bound and sanitize a message for the wire."""
    stripped = _PATHY.sub("<redacted>", text)
    ascii_only = "".join(c if 0x20 <= ord(c) <= 0x7E else "." for c in stripped)
    return ascii_only[:cap]


def _coded(outcome: Mapping[str, Any]) -> str:
    """`<code>: <message>` — the AUTHOR's typed code in front of its own sentence.

    The wire carries a small neutral `CauseCode` on purpose (a coordinator classifies, it
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

#: The fault tail this worker keeps. Every reader takes the last 8.
FAULTS_HELD = 64

CAUSE = pb.CauseCode
ORIGIN = pb.CauseOrigin
STATUS = pb.OutcomeStatus

#: The author-surface outcome vocabulary, mapped onto the wire's NEUTRAL one. The mapping is
#: total and closed: a new author terminal without a wire meaning is a compile-time hole.
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


class UndurableOutcome(Exception):
    """This attempt's outcome could not be journaled, so it will not be sent (#494b).

    Ack-after-journal has no exception for the last resort. An outcome the worker cannot
    replay after a restart, and cannot match against a TerminalAck, is not a terminal — it is
    a claim about bytes nobody wrote down. The attempt stays HELD in `local_fault`, the
    worker takes a LOCAL_SAFETY_REFUSAL fault naming it and goes INTAKE_STATE_ERROR, and the
    owner reads a worker that stopped rather than an outcome it can never close.
    """

    def __init__(self, attempt: AttemptRecord, why: str) -> None:
        super().__init__(
            f"{attempt.request_id}#{attempt.attempt}: the outcome could not be journaled "
            f"({why}); it is held in local_fault and NOTHING was sent"
        )
        self.key = attempt.key()
        self.why = why


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
        origin: pb.CauseOrigin = ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
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
    """The RUN's scratch tree. Keyed on the run and NOT on the attempt, which is the whole
    of resumability: a retried attempt of the same run reads what the killed one left."""
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
    #: what the component-use plane actually did: stages, evictions, leases, resident set
    residency: dict[str, Any] = dataclasses_field(default_factory=dict)
    #: the record the plan was PRICED against, resolved once by the executor's kernel
    prepared: PreparedRequest | None = None
    #: cr-009's PublicationReceipt document, once the output transaction committed one
    publication: dict[str, Any] = dataclasses_field(default_factory=dict)

    def key(self) -> tuple[str, int]:
        return (self.request_id, self.attempt)

    def compact(self) -> None:
        """Keep the REPLAY IDENTITY; drop everything a closed attempt cannot need again.

        A closed attempt is answered by exactly two questions: the (ordinal, digest) fence,
        and a duplicate StartAttempt that must replay its terminal. Everything else — the
        spec, the payload, the grant, the plan, the hydrated inputs, the observation ring,
        the reconciliation — was retained for the life of the process because nothing ever
        released it. The terminal BYTES go too: they can carry a 4 MiB inline result and the
        journal already holds them, so `wrap` re-reads if a replay ever comes (codex audit,
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
        journal: Journal,
        ledger: Ledger,
        chooser: Any,
        supervision: ExecutorSupervision,
        authorizer: grants.Authorizer,
        bindings: Mapping[str, DeclaredBinding],
        prepared_models: Mapping[str, PreparedModel],
        jobs: Mapping[str, JobBinding],
        spool_root: Path,
        scratch_root: Path,
        instance_id: str,
        send: Callable[[pb.WorkerFrame], None],
        progress: Callable[[str, int, dict[str, Any]], None],
        bundles: triage.BundleStore,
        monitor: observe.SilenceMonitor,
        posture: Callable[[], dict[str, Any]],
        device_process: Callable[[], int],
    ) -> None:
        self.journal = journal
        self.ledger = ledger
        self.chooser = chooser
        self.supervision = supervision
        self.authorizer = authorizer
        self.bindings = dict(bindings)
        self.prepared_models = dict(prepared_models)
        self.jobs = dict(jobs)
        self.spool_root = spool_root
        #: Where a RUN's scratch lives. Deliberately OUTSIDE the journal root: §8 lets a
        #: one-shot run delete its journal after a fully acknowledged terminal, and scratch
        #: that lived under it would be deleted by exactly the ABANDONED terminal whose
        #: whole point is that the next attempt resumes from what the killed one left.
        #: Found live: the kill arm resumed from nothing until this moved out.
        self.scratch_root = scratch_root
        self.instance_id = instance_id
        self.send = send
        self.progress = progress
        self.live: dict[str, AttemptRecord] = {}  # request_id -> the attempt held right now
        self.history: dict[tuple[str, int], AttemptRecord] = {}
        self.boot_id = ""
        #: which deployment this worker hosts, read live from the worker ("" in job mode).
        #: Routing only (#446): it rides acceptance and the snapshot, never the digest.
        self.placement_id: Callable[[], str] = lambda: ""
        self.recovered: list[pb.HeldAttempt] = []
        self.executed = 0
        #: the worker's fault tail. BOUNDED: the Report and the triage bundle each read the
        #: last 8, and an unbounded list of the other 65,528 is memory nobody reads.
        self.faults: list[pb.Fault] = []
        self.bundles = bundles
        self.monitor = monitor
        #: the worker's applied posture, read at bundle time — a triage bundle that cannot
        #: say which directive was applied when it ran explains half of what went wrong.
        self.posture = posture
        #: the NVML per-process reader, so the out-of-allocator meter has a real producer
        self.device_process = device_process
        #: binding plan ids whose boot warm pass FAILED. Every attempt served under one
        #: carries a per-request confession: the first-call tax was never paid for it.
        self.degraded: dict[str, str] = {}
        #: the demand falsifier. It ships first, it counts, and it decides nothing — the
        #: `timing-decides-nothing` fence proves the module it lives in cannot reach the
        #: planner, the ledger or the fill plane.
        self.demand = observe.DemandLedger()
        #: cr-009's DURABLE checkpoint ledger: (request_id, operation_key, logical_key) ->
        #: the journaled receipt. Identity replays; the same key with different bytes is a
        #: CONFLICT and never a replacement (§2).
        self.receipts: dict[tuple[str, str, str], dict[str, Any]] = {}
        #: cr-012's egress boundary, rebuilt from the applied binding at each directive.
        #: A default `EgressPolicy()` has an EMPTY allowlist, which means no egress at all —
        #: the fail-closed state is the one you get by not configuring anything.
        self.egress = egress.EgressPolicy()

    def fault(self, row: pb.Fault) -> None:
        """Record ONE fault, bounded. Readers take the last 8; the tail past that is memory
        nobody reads, and an unbounded one is how a long-lived worker grows without a leak
        anywhere (codex audit, adopted 2026-08-25)."""
        self.faults.append(row)
        del self.faults[:-FAULTS_HELD]

    # ------------------------------------------------------------------ offer

    def offer(self, offered: pb.AttemptOffer) -> AttemptRecord:
        """VALIDATING: fences and local safety, before any body field is trusted."""
        if self.journal.stopped:
            raise AttemptRefusal(
                "journal_stopped",
                f"the journal cannot durably commit ({self.journal.stopped}); this worker "
                "acknowledges nothing and executes nothing",
                cause=CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
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

        # The (attempt, digest) fence binds the ORDINAL, not the current state: an attempt
        # is one execution of one spec whether it is running, journaled or already closed.
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
            if held.state in ("queued", "accepted", "running"):
                raise AttemptRefusal(
                    "live_attempt_supersession",
                    f"attempt {offered.attempt_ordinal} arrived while attempt "
                    f"{held.attempt} is live with no journaled outcome: supersession is "
                    "explicit (CancelAttempt(SUPERSEDED) -> journaled outcome -> ordinal+1), never "
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
        with no journaled terminal and refuse supersession.
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
        origin: pb.CauseOrigin = ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
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
        malformed payload must be a refusal the coordinator can read, not an accepted attempt
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
            # The ADDRESS was minted by the coordinator, so a miss here is PLATFORM-FATAL,
            # not a bad request (v1 REF_ORIGIN). This used to fall into the branch below and
            # report a missing file as "not a JSON object" — the wrong system, the wrong
            # person, and a sentence with no remedy in it.
            classified = refusal.classify(
                "payload_unreadable",
                f"the granted payload location could not be read ({type(exc).__name__}); the "
                "coordinator minted this address, so this worker cannot substitute another",
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
        run HANGS — `LocalCoordinator` only deadlines a run that never dispatched. A 42,719 B
        request on the same path succeeds.

        That violates the acceptance-boundary law directly: everything before the journal
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
        endpoint's declared types and this process never imports endpoint code — and it must
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
                self.supervision.invalidate(str(reply.get("detail", "poisoned"))[:256])
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
        spool = self.spool_root / f"{attempt.request_id}-{attempt.attempt}"
        spool.mkdir(parents=True, exist_ok=True)
        attempt.spool = spool
        binding = attempt.declared
        try:
            attempt.inputs = grants.hydrate_inputs(
                attempt.grant,
                self.authorizer,
                spool=spool,
                policy=self.egress,
                max_bytes=binding.max_input_bytes if binding else 64 << 20,
                max_total_bytes=binding.max_inputs_total_bytes if binding else 256 << 20,
            )
        except grants.GrantRefusal as exc:
            # WHOSE fault decides the terminal. A cap, a media-type disagreement or an
            # absent grant entry are facts about the REQUEST; an expired grant or an
            # unreachable coordinator-minted address are facts about the platform, and
            # rendering the two the same way sends the wrong person to the wrong question.
            caller = exc.code in (
                "input_over_cap",
                "inputs_over_total_cap",
                "input_media_type",
                "input_length_mismatch",
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
                origin=ORIGIN.CAUSE_ORIGIN_CLIENT if caller else ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
            ) from exc

    def _accept_job(self, attempt: AttemptRecord) -> pb.AttemptAccepted:
        """BOUNDED ACCEPT-THEN-PREPARE — the ONE mode difference from serving (cr-009).

        Serving never accepts-then-cold-loads: a warm generation is the whole point of a
        serving worker, so an attempt is accepted only against a Ready executor. A job is the
        opposite shape — one immutable build, one bounded attempt, terminal, reclaim — and
        there is nothing to be warm FOR. Preparing before acceptance would only hold a rented
        pod idle while the coordinator waits to hear whether its work was taken.

        What does NOT change: acceptance is journaled BEFORE anything runs, the payload is
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
        self._fetch_payload(attempt)
        try:
            attempt.trees = grants.read_trees(attempt.grant, self.authorizer)
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
        )
        attempt.plan = plan
        # The RUN's scratch, not the attempt's: `Scratch.checkpoint_dir(key=)` must hand a
        # retried attempt the SAME tree, which is the whole of resume. It is created here,
        # before acceptance, so a job never discovers at step 40,000 that it has nowhere
        # durable to write.
        attempt.scratch = self.scratch_root / attempt.request_id
        attempt.scratch.mkdir(parents=True, exist_ok=True)

        attempt.opened = self.ledger.snapshot()
        try:
            self.ledger.grant(
                f"{attempt.request_id}#{attempt.attempt}",
                plan.reserved_host_bytes,
                plan.reserved_pinned_bytes,
                f"job {binding.job} {attempt.request_id}#{attempt.attempt}",
            )
        except LedgerRefusal as exc:
            raise AttemptRefusal(
                exc.code, exc.detail, cause=CAUSE.CAUSE_CODE_CONSTRAINT_INFEASIBLE
            ) from exc

        plan_digest = plan.digest()
        try:
            self.journal.append(
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
                    "executor_generation": self.supervision.incarnation,
                    "input_trees": ",".join(sorted(attempt.trees)),
                },
            )
        except Exception:
            self.ledger.release(f"{attempt.request_id}#{attempt.attempt}")
            raise
        return self._accepted(attempt)

    def prepare_and_accept(self, attempt: AttemptRecord) -> pb.AttemptAccepted:
        """The ATOMIC ACCEPTANCE BOUNDARY, in exactly this order:

            resolve binding -> fetch/verify/decode the payload -> derive REAL features from
            it -> choose the plan -> prepare the executor -> reserve the envelope -> JOURNAL
            -> publish AttemptAccepted.

        Everything before the journal either completes or raises a typed PRE-ACCEPT refusal,
        and the reservation rolls back if the journal cannot commit. The payload is fetched
        HERE and not at execution time on purpose: a missing, expired or malformed payload
        must be a refusal the coordinator can read, not an accepted attempt that later dies
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
        plan_id = serving["entrypoint_binding_plan_id"]
        declared = self.bindings.get(plan_id)
        prepared_model = self.prepared_models.get(plan_id)
        if declared is None or prepared_model is None:
            raise AttemptRefusal(
                "unknown_binding_plan",
                f"no EntrypointBindingPlan {plan_id[:24]}… is bound on this worker",
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
        executor = self.supervision.current
        if executor is None or not executor.alive():
            raise AttemptRefusal(
                "executor_absent",
                "no live executor: an online attempt is accepted only against a Ready one",
                cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
            )
        if plan_id not in executor.ready_plans:
            raise AttemptRefusal(
                "executor_not_ready",
                f"the executor is ready for {sorted(executor.ready_plans) or 'nothing'}, "
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

        prepared = self._prepare_request(attempt, executor, declared)
        attempt.prepared = prepared
        try:
            plan = self.chooser.choose(declared, prepared_model, prepared)
        except PlanRefusal as exc:
            raise AttemptRefusal(
                exc.code, exc.detail, cause=CAUSE.CAUSE_CODE_CONSTRAINT_INFEASIBLE
            ) from exc
        attempt.plan = plan

        # HYDRATE, after the plan and before acceptance (cr-012, §1.3's launch contract:
        # validate -> plan -> hydrate -> prepare -> accept -> invoke). After the plan,
        # because a request that has no plan should download nothing; before acceptance,
        # because an oversized, ungranted, mislabelled or unreachable input must be a
        # refusal the coordinator can read rather than an accepted attempt that dies on a
        # thread. And before the lease, which is the ordering that matters most: a slow URL
        # fetched under a lease would pin staged weights for its whole duration.
        self._hydrate_inputs(attempt)

        attempt.opened = self.ledger.snapshot()
        try:
            self.ledger.grant(
                f"{attempt.request_id}#{attempt.attempt}",
                plan.reserved_host_bytes,
                plan.reserved_pinned_bytes,
                f"attempt {attempt.request_id}#{attempt.attempt}",
            )
        except LedgerRefusal as exc:
            raise AttemptRefusal(
                exc.code, exc.detail, cause=CAUSE.CAUSE_CODE_CONSTRAINT_INFEASIBLE
            ) from exc

        plan_digest = plan.digest()
        try:
            self.journal.append(
                "accepted",
                {
                    "request_id": attempt.request_id,
                    "attempt": attempt.attempt,
                    "worker_boot_id": self.boot_id,
                    "instance_id": self.instance_id,
                    "invocation_spec_digest": documents.spell(attempt.digest),
                    "plan_digest": plan_digest,
                    "model_construction_digest": plan.model_construction_digest,
                    "constructed_model_digest": plan.constructed_model_digest,
                    "request_features_digest": plan.request_features_digest,
                    "preflight_facts_digest": plan.preflight_facts_digest,
                    "grant_epoch": attempt.grant.credential_epoch,
                    "placement_id": self.placement_id(),
                    "executor_generation": self.supervision.incarnation,
                    "entrypoint_binding_plan_id": plan_id,
                },
            )
        except Exception:
            # The envelope was granted a moment ago and nothing will ever run under it:
            # an unjournaled acceptance must not leave the worker's host ledger charged.
            self.ledger.release(f"{attempt.request_id}#{attempt.attempt}")
            raise
        return self._accepted(attempt)

    # ------------------------------------------------------------------ execute

    def execute(self, attempt: AttemptRecord) -> pb.AttemptOutcome:
        """Run ONE accepted attempt. Closing it against the ledger happens inside
        `terminal()`, because the triage bundle the terminal REFERENCES has to carry the
        reconciliation — a bundle that stops one statement short of the numbers an operator
        needs is a bundle they have to leave to go find them."""
        if attempt.canceling:
            # Cancelled between arrival and dispatch — while it queued, or while admission
            # was fetching and hydrating it. Nothing reached the device, so the caller's
            # cancel is complete the moment this terminal is journaled, and spending the
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
        executor = self.supervision.current
        if executor is not None and executor.alive() and not executor.poisoned and attempt.spool:
            try:
                self.ledger.observe_probe(executor.call({"cmd": "probe"}, timeout=30.0))
                self.ledger.device_process = self.device_process()
            except ExecutorGone:
                self.ledger.device_unreadable("the executor died before the ledger was read")
        else:
            self.ledger.device_unreadable(
                "no live usable executor: its CUDA context died with the process"
            )
        record = self.ledger.close_attempt(key, attempt.opened)
        attempt.reconciliation = record
        try:
            self.journal.append("ledger", record)
        except JournalError as exc:
            record["journaled"] = False
            record["journal_error"] = exc.code
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
            self.supervision.invalidate(f"ledger_unreconciled: {record['leaks'][0][:160]}")
        return record

    # ------------------------------------------------------------------ the job lane

    def durable_exchange(self, attempt: AttemptRecord, frame: Mapping[str, Any]) -> dict[str, Any]:
        """Answer ONE durable mid-attempt request from the executor (cr-009's checkpoints).

        Identity is (run, attempt, operation key, logical key, content digest). The write is
        JOURNALED — and fsynced, because that is what `Journal.append` does — before this
        returns, so `Checkpoints.save()` reports success only after the fact is durable.

        Three answers and no fourth:
          * a NEW identity is journaled and gets a fresh receipt;
          * the SAME identity replays its receipt and writes nothing a second time;
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
        try:
            self.journal.append("job_checkpoint", record)
        except JournalError as exc:
            return {
                "ok": False,
                "code": "checkpoint_not_durable",
                "detail": f"the journal cannot commit ({exc.code}); this worker acknowledges "
                "no checkpoint it cannot make durable",
            }
        self.receipts[key] = record
        # The coordinator's copy rides the DURABLE control lane, after the worker's own
        # journal — worker-local durability is what `save()` returned on, and the hub's copy
        # is at-least-once like every other durable message (law 9).
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
            f"checkpoint {logical_key!r} is durable",
            "info",
            receipt=receipt_id,
            bytes=int(frame.get("length", 0)),
        )
        return {"ok": True, "receipt_id": receipt_id, "replayed": False}

    def absorb_receipt(self, receipt: pb.JobCheckpointReceipt) -> None:
        """Record the COORDINATOR's verdict on a checkpoint this worker already made durable.

        It is a second observer, not a second authority: `save()` already returned on the
        worker's own journal. A coordinator CONFLICT is a real fact and is journaled as a
        fault, because it means the run's identity is contested upstream — but it cannot
        retroactively un-write bytes that are durable here.
        """
        if receipt.outcome == pb.CheckpointOutcome.CHECKPOINT_OUTCOME_CONFLICT:
            self.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_LOCAL_SAFETY_REFUSAL,
                    subject=f"{receipt.request_id}#{receipt.attempt_ordinal}",
                    reason="checkpoint_conflict_upstream",
                    detail=safe(
                        f"the coordinator reports {receipt.logical_key!r} under operation key "
                        f"{receipt.operation_key!r} conflicts: {receipt.fault.detail}"
                    ),
                )
            )

    def _run(self, attempt: AttemptRecord) -> pb.AttemptOutcome:
        """ONE attempt: dispatch, freeze point, tail, journaled terminal.

        ONE pipeline for both lanes. They differ in the COMMAND they send and in nothing
        after the executor answers, and keeping two copies of everything after it had
        already let them drift — the job lane never checked the env seal it reports, never
        classified a device OOM, and its oversize refusal named no remedy (codex audit,
        adopted 2026-08-25). What a job actually has that a serving attempt does not is a
        durable mid-attempt exchange, and that is one argument to one call.
        """
        job = attempt.job is not None
        executor = self.supervision.current
        if executor is None:
            return self.abandon(attempt, "the executor vanished before dispatch")
        # The spool already exists — cr-012's hydration made it before acceptance. This
        # stays for the recovered-attempt path, whose acceptance was another process's.
        spool = attempt.spool or self.spool_root / f"{attempt.request_id}-{attempt.attempt}"
        spool.mkdir(parents=True, exist_ok=True)
        attempt.spool = spool
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
            incarnation=self.supervision.incarnation,
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
                incarnation=self.supervision.incarnation,
                canceling=attempt.canceling or "no",
            )
            self.monitor.end(key, "executor_died")
            # WHICHEVER THREAD SEES THE DEATH MARKS IT: `waitpid` answers 0 for a few ms
            # on a dying process, so invalidating at the OBSERVATION removes the race with
            # the watchdog's kill instead of narrowing it (cr-007).
            self.supervision.invalidate("the executor died during an attempt")
            # A job's scratch tree deliberately SURVIVES — it is the run's, not the
            # attempt's — but the attempt SPOOL is aborted on both lanes, so nothing
            # half-written can reach a granted destination.
            grants.abort_outputs(spool)
            # A cancel that ESCALATED killed this executor on purpose: the attempt's
            # terminal is CANCELED, and the executor is already provably dead. Any other
            # death is an invalidation the attempt did not ask for.
            if attempt.canceling:
                return self.finish_cancel(attempt, attempt.canceling)
            return self.abandon(attempt, str(exc))
        return self._finish(attempt, reply, spool, key)

    @staticmethod
    def _subject(attempt: AttemptRecord) -> str:
        if attempt.job is not None:
            return attempt.job.job
        return attempt.declared.entrypoint if attempt.declared else ""

    def _command(
        self, attempt: AttemptRecord, key: str, spool: Path, seconds: float | None
    ) -> dict[str, Any]:
        """The ONE thing the two lanes genuinely differ in: what they ask the executor."""
        if attempt.job is not None:
            return {
                "cmd": "run_job",
                "request_id": key,
                "project": attempt.job.project,
                "job": attempt.job.job,
                "payload": attempt.payload,
                "deadline_s": seconds,
                "spool": str(spool),
                "scratch": str(attempt.scratch) if attempt.scratch else "",
                "trees": {ref: [str(r), d] for ref, (r, d) in attempt.trees.items()},
                "budget": {
                    "gpu_rate_micro_usd_per_hour": attempt.job.gpu_rate_micro_usd_per_hour,
                    "gpu_count": attempt.job.gpu_count,
                    "cap_micro_usd": attempt.job.cap_micro_usd,
                },
            }
        assert attempt.declared is not None and attempt.plan is not None
        return {
            "cmd": "invoke",
            "request_id": key,
            "entrypoint": attempt.declared.entrypoint,
            "deadline_s": seconds,
            "boot_warmup": False,
            "spool": str(spool),
            "placement": attempt.plan.placement,
            "headroom_bytes": attempt.plan.headroom_bytes,
            # cr-012: the VERIFIED inputs, by their stable identity. Paths into this
            # attempt's own spool and nothing else — no URL, no credential and no fetch
            # reaches the executor, which has no outbound network at all.
            "inputs": {
                input_id: {
                    "local": str(granted.local),
                    "media_type": granted.media_type,
                    "digest": granted.digest,
                    "length": granted.length,
                    "order": granted.order,
                }
                for input_id, granted in attempt.inputs.items()
            },
            "max_input_bytes": attempt.declared.max_input_bytes,
        }

    def _finish(
        self, attempt: AttemptRecord, reply: dict[str, Any], spool: Path, key: str
    ) -> pb.AttemptOutcome:
        """Everything after the executor answers. Identical for both lanes, and now once."""
        attempt.resulted_at = time.perf_counter()
        self.monitor.end(key, "returned")
        self.absorb(attempt, reply)
        # What this attempt's kernel allocated ON TOP of the resident weights. Recorded from
        # whatever came back, success or fault: a failed attempt's activation peak is
        # exactly the number an operator needs, and dropping it on the failure path would
        # leave the ledger blind in the one case that matters.
        # THIS attempt's transitions, not the generation's history. Reading the cumulative
        # counters made every attempt after the first staged one claim its activation basis
        # had moved, on attempts that never touched a component.
        moved = bool(
            attempt.residency.get("attempt_stages") or attempt.residency.get("attempt_evictions")
        )
        cell = attempt.plan.shape_cell if attempt.plan else ""
        self.ledger.observe_attempt(reply.get("metrics", {}), resident_moved=moved, cell=cell)
        self.bank_demand(attempt, reply.get("metrics", {}), moved)

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
            # No terminal is journaled while author computation might still mutate GPU or
            # output state. Unproven quiescence is an executor fault, not a success.
            self.supervision.invalidate("unproven quiescence")
            return self.outcome(
                attempt,
                STATUS.OUTCOME_STATUS_FAILED,
                CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
                "the executor could not prove quiescence after the handler returned; a "
                "terminal journaled under unproven quiescence would be a claim about state "
                "that may still be moving",
            )
        if not reply.get("env_intact", True):
            self.supervision.invalidate("env seal broken")
            return self.outcome(
                attempt,
                STATUS.OUTCOME_STATUS_FAILED,
                CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
                "env_seal_broken: the executor's environment changed after CUDA init — the "
                "allocator read it once, so the imposition was a silent no-op",
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
        if reply.get("poisoned"):
            self.supervision.invalidate(str(reply["poisoned"]))

        outcome = reply["outcome"]
        status, cause = _TERMINAL[outcome["terminal"]]
        breach = outcome["code"] == DEVICE_OOM
        if breach:
            # THE ACCEPTED ENVELOPE WAS BREACHED. `_TERMINAL["failed"]` defaults to
            # AUTHOR_EXCEPTION because most failures are the handler's, and a device that ran
            # out is the one class where the endpoint provably had no say — but it is not
            # CONSTRAINT_INFEASIBLE either. That code means "no plan fits", which is a
            # PRE-ENTRY answer this worker gives before it accepts anything; rendering a
            # post-accept OOM as one makes a planner or envelope defect read as a legitimate
            # capacity result, and nobody goes to look at the planner. The runtime priced
            # this attempt and admitted it, so the fault is the runtime's.
            cause = CAUSE.CAUSE_CODE_EXECUTOR_FAULT
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
            manifest, tail_error = self._tail(attempt, reply)
            if tail_error is not None:
                return self.outcome(
                    attempt,
                    STATUS.OUTCOME_STATUS_FAILED,
                    CAUSE.CAUSE_CODE_CAPABILITY_UNAVAILABLE,
                    ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
                    tail_error,
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

    def absorb(self, attempt: AttemptRecord, reply: Mapping[str, Any]) -> None:
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
            self.ledger.observe_residency(attempt.residency)
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
        plan_id = attempt.declared.entrypoint_binding_plan_id if attempt.declared else ""
        why = self.degraded.get(plan_id)
        if why:
            attempt.confess(
                "degraded",
                f"served by a binding whose boot warm pass FAILED ({why}): the first-call "
                "tax was never paid on boot, so this request paid it",
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
                self.ledger.accelerator or "unknown-accelerator",
                f"{plan.delivery}/{plan.materialization or 'verbatim'}/{plan.placement}",
                plan.shape_cell or "-",
            )
        )
        confession = self.demand.bank(lane, plan.headroom_bytes, actual)
        if confession is not None:
            attempt.confess(confession.kind, confession.quantified)

    def _payload(self, attempt: AttemptRecord) -> dict[str, Any]:
        """The payload travels as a grant INPUT, fetched worker-side under the grant."""
        data = grants.read_input(attempt.grant, "payload", self.authorizer)
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError(f"a payload document is an object, not {type(value).__name__}")
        return value

    def _tail(
        self, attempt: AttemptRecord, reply: dict[str, Any]
    ) -> tuple[pb.OutputManifest | None, str | None]:
        """§3.4 steps 4-5, worker-side: the device lease is already released, the bytes
        are already immutable in the attempt spool, and this is where they are written under
        the exact output grant and become a durable manifest.

        The tail runs SYNCHRONOUSLY with the attempt (launch). Overlapping it with the next
        attempt's GPU work is cr-018's door and is not claimed here.

        Outputs bind to granted destinations by STABLE OUTPUT ID — the result's own field
        path (`image`, `pair.thumb`, `frames.0`) — with EXACT SET EQUALITY against the
        grant. Never by list index: two outputs that arrive in a different order must still
        land in their own destinations, and a position cannot express that. A produced
        output the coordinator did not grant, or a granted destination the attempt did not
        fill, refuses the tail WHOLE — an output transaction is not partially successful.
        """
        rows = list(reply.get("outputs", ()))
        spool = attempt.spool
        if not rows and not attempt.grant.outputs:
            return None, None
        if spool is None:
            return None, "the attempt has no spool to read outputs from"
        produced = {str(row["output_id"]): row for row in rows}
        granted = set(attempt.grant.outputs)
        if set(produced) != granted:
            grants.abort_outputs(spool)
            return None, (
                f"destination_absent: the attempt produced {sorted(produced) or 'nothing'} "
                f"and the grant names {sorted(granted) or 'nothing'}; an output the "
                "coordinator did not grant is unwritable however it was produced, and a "
                "granted destination the attempt never filled is not a success"
            )
        entries = []
        try:
            for output_id in sorted(produced):
                row = produced[output_id]
                blob = self._blob(spool, str(row["asset_ref"]))
                if blob is None:
                    return None, (f"output {output_id} has no bytes in the attempt spool")
                entries.append(
                    grants.write_output(
                        attempt.grant,
                        output_id,
                        blob.read_bytes(),
                        self.authorizer,
                        # The AUTHOR's declared media type, carried through instead of
                        # replaced by `application/octet-stream`. cl-010 measured the cost
                        # of dropping it: the product wrote `image` and not `image.png`,
                        # and the client correctly refuses to sniff, so the only place that
                        # could fix it is the place that already knows.
                        media_type=str(row.get("media_type") or ""),
                    )
                )
        except grants.GrantRefusal as exc:
            grants.abort_outputs(spool)
            return None, f"{exc.code}: {exc.detail}"
        receipt = self._publication(attempt, entries)
        try:
            # DURABLE BEFORE IT IS CLAIMED, exactly like the triage bundle: a terminal may
            # only name a receipt that already exists. The bytes are at their granted
            # destinations by now, so a journal that cannot commit makes this a FAILED
            # attempt naming why — never a publication nobody has a record of.
            self.journal.append("publication", receipt.document())
        except JournalError as exc:
            return None, (
                f"publication_not_durable: {exc.code} — this attempt wrote its outputs and "
                "the worker cannot durably record what it published, so it claims nothing"
            )
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
                else (attempt.declared.entrypoint_binding_plan_id if attempt.declared else "")
            ),
            request_id=attempt.request_id,
            attempt=attempt.attempt,
            outputs=tuple(entries),
        )

    @staticmethod
    def _blob(spool: Path, ref: str) -> Path | None:
        tail = ref.rsplit("/", 2)
        if len(tail) < 3:
            return None
        candidate = spool / f"{tail[-2]}-{tail[-1]}"
        return candidate if candidate.is_file() else None

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
        """CLOSE the ledger, commit the triage bundle, build, JOURNAL, then return to send.

        The order is the whole point and none of it is negotiable:

            release the envelope -> reconcile every byte class -> WRITE and fsync the
            bundle -> journal its write receipt -> journal the terminal that names it

        A terminal may only reference a bundle that is already durable, or it would be a
        claim about bytes nobody wrote. And the bundle is committed BEFORE the terminal so
        that a crash between the two leaves an unreferenced bundle (harmless, readable)
        rather than a terminal pointing at nothing.
        """
        # The grant is released FIRST because reconciliation is a claim about the whole
        # envelope: the opening line was taken before the grant, so the closing one must be
        # taken after its release or every attempt reports its own grant as a leak.
        self.ledger.release(f"{attempt.request_id}#{attempt.attempt}")
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
        if manifest is not None:
            body.output_manifest.CopyFrom(manifest)
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
        self.journal.append(
            "outcome",
            {
                "request_id": attempt.request_id,
                "attempt": attempt.attempt,
                "worker_boot_id": self.boot_id,
                "instance_id": self.instance_id,
                "invocation_spec_digest": documents.spell(attempt.digest),
                "outcome_id": attempt.outcome_id,
                "outcome_digest": documents.spell(digest),
                "outcome_canonical_b64": base64.b64encode(canonical_bytes).decode(),
            },
        )
        attempt.state = "outcome"
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
        assembly = triage.Assembly(
            subject_id=attempt.subject_id,
            attempt={
                "request_id": attempt.request_id,
                "attempt": attempt.attempt,
                "worker_boot_id": self.boot_id,
                "instance_id": self.instance_id,
                "executor_generation": self.supervision.incarnation,
                "invocation_spec_digest": documents.spell(attempt.digest),
                "entrypoint": attempt.declared.entrypoint if attempt.declared else "",
                "executions": attempt.executions,
                "replays": attempt.replays,
            },
            plan=plan.document() if plan is not None else {},
            posture=self.posture(),
            terminal={
                "status": pb.OutcomeStatus.Name(status),
                "cause_code": pb.CauseCode.Name(cause),
                "cause_origin": pb.CauseOrigin.Name(origin),
                "safe_message": safe(detail, 4096),
                "outcome_id": attempt.outcome_id,
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
                or {"classes": [c.document() for c in self.ledger.counters()]},
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
        except (triage.TriageError, JournalError, OSError) as exc:
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

        There were three, one per lane plus this one, and they had already drifted: the job
        one omitted `compute_dtype` and the construction digest, which a job plan carries as
        empty anyway. AttemptAccepted always carries the digests that were JOURNALED, so a
        re-send can never move them.
        """
        assert attempt.plan is not None
        plan = attempt.plan
        return pb.AttemptAccepted(
            request_id=attempt.request_id,
            attempt_ordinal=attempt.attempt,
            invocation_spec_digest=attempt.digest,
            plan_digest=documents.raw(plan.digest()),
            # A JOB has no model construction, and an empty digest is the honest answer:
            # spelling one for a construction that never happened would invent an identity.
            model_construction_digest=(
                documents.raw(plan.model_construction_digest)
                if plan.model_construction_digest
                else b""
            ),
            plan=pb.AttemptPlanSummary(
                delivery=plan.delivery,
                materialization=plan.materialization,
                compute_dtype=plan.compute_dtype,
                placement=plan.placement,
                reserved_device_memory_bytes=plan.reserved_vram_bytes,
                reserved_host_bytes=plan.reserved_host_bytes,
                decision_explanation=plan.decision_explanation,
            ),
            # ROUTING plus the generation fence the attempt runs under, fixed at
            # acceptance (#446/#454): the attempt never survives its executor generation.
            placement_id=self.placement_id(),
            executor_generation=self.supervision.incarnation,
        )

    def wrap(self, attempt: AttemptRecord) -> pb.AttemptOutcome:
        """Wrap the SAME journaled body in the CURRENT session envelope. The body is a byte
        fact; only the envelope moves across a reconnect.

        A CLOSED attempt no longer holds those bytes in memory — the journal does, and this
        is the one place that ever needs them again.
        """
        if not attempt.outcome_bytes and attempt.outcome_id:
            attempt.outcome_bytes = self.journal.outcome_bytes(attempt.key())
        return pb.AttemptOutcome(
            request_id=attempt.request_id,
            attempt_ordinal=attempt.attempt,
            invocation_spec_digest=attempt.digest,
            outcome_id=attempt.outcome_id,
            outcome_digest=attempt.outcome_digest,
            outcome_canonical_bytes=attempt.outcome_bytes,
            placement_id=self.placement_id(),
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
            # the endpoint's own schema — never a digest of this particular value, which
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
        # the envelope's whole claim is that the bytes a coordinator holds and the value the
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
            return "dropped: cancel digest does not match the journaled acceptance"
        if held.state == "outcome":
            return "ignored: the attempt already reached a natural terminal"
        held.canceling = {
            pb.CancelReason.CANCEL_REASON_CLIENT: "client",
            pb.CancelReason.CANCEL_REASON_DRAIN: "drain",
            pb.CancelReason.CANCEL_REASON_SUPERSEDED: "superseded",
            pb.CancelReason.CANCEL_REASON_POLICY: "policy",
            pb.CancelReason.CANCEL_REASON_DEADLINE: "deadline",
        }.get(message.reason, "client")
        if held.state != "running":
            # NOTHING OF THIS ATTEMPT IS ON THE DEVICE. The signal is process-wide, so
            # sending it here would cancel whichever attempt IS running — a different
            # request's. The mark is enough: the device lane reads it when it reaches this
            # attempt, and terminates it without dispatching.
            return f"marked before dispatch ({held.canceling}); nothing signalled"
        executor = self.supervision.current
        if executor is None:
            return "executor already gone"
        executor.cooperative_cancel()
        return f"cooperative cancel signalled ({held.canceling})"

    def escalate(self, attempt: AttemptRecord, grace: float) -> dict[str, Any]:
        """The FORCEFUL half, and the exact ordering §5 requires: forced kill -> output
        abort -> epoch invalidation -> journal CANCELED. Rebuilding readiness happens
        AFTER, off the cancellation critical path."""
        executor = self.supervision.current
        started = time.perf_counter()
        killed = executor.kill_group() if executor is not None else 0
        status = executor.reap() if executor is not None else None
        gone_at = time.perf_counter()
        aborted = grants.abort_outputs(attempt.spool) if attempt.spool else 0
        self.supervision.invalidate("forceful cancel")
        return {
            "grace_s": grace,
            "killed": killed,
            "exit_status": status,
            "kill_to_gone_ms": round((gone_at - started) * 1000, 2),
            "aborted_outputs": aborted,
            "executor_generation": self.supervision.incarnation,
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
            ORIGIN.CAUSE_ORIGIN_COORDINATOR
            if reason in ("drain", "policy", "superseded")
            else ORIGIN.CAUSE_ORIGIN_CLIENT,
            f"canceled ({reason}); "
            + (
                "the executor was dead and uncommitted outputs aborted before this terminal "
                "was journaled"
                if attempt.executions
                else "nothing was dispatched, so there is nothing to unwind"
            ),
        )

    def last_resort(self, attempt: AttemptRecord, why: str) -> pb.AttemptOutcome:
        """The SMALLEST terminal this worker can make, for when the ordinary one could not.

        `terminal()` builds a rich body — a reconciliation, a triage bundle, attested
        metrics, an inline result — and every one of those can fail on its own. When one
        does, the attempt used to end in SILENCE: the lane returned, the coordinator held an
        accepted attempt with no terminal, and the liveness detector printed WEDGED and did
        nothing (job-001). An accepted attempt ALWAYS produces a terminal; this is the body
        that has nothing left to fail on.

        It is still journaled before it is sent, and NOT SENT AT ALL when that fails
        (#494b). It used to name the journal failure in the message and ship anyway, which
        gave the owner an outcome with no durable identity behind it: a restart replays
        nothing for it, and an ack naming its terminal_id matches nothing. The attempt is
        held in `local_fault` instead and the worker refuses onward — a worker that stopped
        is readable; an unreplayable terminal is not.
        """
        body = pb.AttemptOutcomeBody(
            request_id=attempt.request_id,
            attempt_ordinal=attempt.attempt,
            invocation_spec_digest=documents.spell(attempt.digest),
            status=STATUS.OUTCOME_STATUS_ABANDONED,
            safe_message=safe(f"outcome_construction_failed: {why}", 4096),
            cause=pb.OutcomeCause(
                code=CAUSE.CAUSE_CODE_EXECUTOR_INVALIDATED,
                origin=ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
                detail=safe(f"outcome_construction_failed: {why}"),
            ),
            execution_started=attempt.executions > 0,
        )
        canonical_bytes, digest = documents.identity(body)
        attempt.outcome_id = f"out-{uuid.uuid4().hex[:24]}"
        attempt.outcome_digest = digest
        attempt.outcome_bytes = canonical_bytes
        try:
            self.journal.append(
                "outcome",
                {
                    "request_id": attempt.request_id,
                    "attempt": attempt.attempt,
                    "worker_boot_id": self.boot_id,
                    "instance_id": self.instance_id,
                    "invocation_spec_digest": documents.spell(attempt.digest),
                    "outcome_id": attempt.outcome_id,
                    "outcome_digest": documents.spell(digest),
                    "outcome_canonical_b64": base64.b64encode(canonical_bytes).decode(),
                },
            )
            attempt.state = "outcome"
        except Exception as exc:
            attempt.state = "local_fault"
            attempt.outcome_id = ""
            attempt.outcome_digest = b""
            attempt.outcome_bytes = b""
            self.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_LOCAL_SAFETY_REFUSAL,
                    subject=f"{attempt.request_id}#{attempt.attempt}",
                    reason="outcome_not_durable",
                    detail=safe(
                        f"the last-resort outcome could not be journaled ({exc}); this "
                        "attempt is HELD and no outcome was sent — ack-after-journal admits "
                        "no last resort"
                    ),
                )
            )
            raise UndurableOutcome(attempt, str(exc)) from exc
        return self.wrap(attempt)

    def abandon(self, attempt: AttemptRecord, why: str) -> pb.AttemptOutcome:
        """An accepted-but-incomplete attempt ALWAYS produces a terminal, never silence."""
        if attempt.spool is not None:
            grants.abort_outputs(attempt.spool)
        self.supervision.invalidate("executor invalidated")
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
            return "dropped: no journaled terminal for that attempt"
        if ack.outcome_id != attempt.outcome_id or ack.outcome_digest != attempt.outcome_digest:
            return (
                f"NOT AN ACK: id {ack.outcome_id!r}/digest {ack.outcome_digest.hex()[:12]} "
                f"does not bind the journaled terminal — replay continues"
            )
        self.journal.append(
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

    # ------------------------------------------------------------------ recovery

    def recover(self) -> list[pb.HeldAttempt]:
        """Replay the journal into `recovered_attempts` for the new session's Register.

        Two states survive a restart and each becomes a DURABLE, REPLAYABLE TERMINAL here —
        never a half-built live attempt:

        * a journaled TERMINAL replays verbatim, if and only if the three-authorization rule
          admits it (persisted assignment, stable `instance_id`, matching execution digest);
        * an ACCEPTED row with no terminal is accepted-but-incomplete. Its execution context
          died with the process, so recovery MINTS `ABANDONED/EXECUTOR_INVALIDATED`,
          journals it, and reports it as terminal-pending-ack. It is never re-executed and
          it never re-enters `live` as an attempt with no plan and no grant — an attempt
          object that cannot answer "what would you run" has no business being dispatchable.

        The identity a recovered ABANDONED terminal needs — request id, ordinal, execution
        digest, instance — is journaled at ACCEPTANCE, before the attempt can start, which
        is what makes this reconstruction possible without guessing.

        A row belonging to ANOTHER instance is not replayable — its terminal was authorized
        under an identity this process is not — but it is not nothing either. It becomes a
        freshly minted, freshly journaled TYPED LOSS (#494d): ABANDONED / EXECUTOR_INVALIDATED
        naming the foreign instance, so the owner learns those attempts existed and died. The
        old behaviour was `continue`, and a coordinator holding a dispatched attempt learned
        nothing at all about it — the worker simply never mentioned it again.
        """
        recovered: list[pb.HeldAttempt] = []
        for (request_id, number), row in self.journal.attempts().items():
            if row["state"] == "closed":
                continue
            digest = documents.raw(row["invocation_spec_digest"])
            if row.get("invocation_spec_digest") != documents.spell(digest):
                continue  # the row cannot even name its own subject
            foreign = str(row.get("instance_id", ""))
            attempt = AttemptRecord(
                request_id=request_id,
                attempt=number,
                digest=digest,
                spec={},
                state="outcome",
                kind=str(row.get("attempt_kind", "serving")),
            )
            if foreign != self.instance_id:
                self.outcome(
                    attempt,
                    STATUS.OUTCOME_STATUS_ABANDONED,
                    CAUSE.CAUSE_CODE_EXECUTOR_INVALIDATED,
                    ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
                    f"foreign_instance_identity: this journal row was authorized under "
                    f"instance {safe(foreign, 64) or '(unnamed)'} and this worker is "
                    f"{safe(self.instance_id, 64) or '(unnamed)'}, so its journaled outcome "
                    "is not replayable here; the attempt existed, it is over, and this is "
                    "the loss said out loud rather than a row dropped in silence",
                )
                self.history[attempt.key()] = attempt
                recovered.append(self._active_row(attempt, row))
                continue
            if row["state"] == "outcome":
                attempt.outcome_id = row["outcome_id"]
                attempt.outcome_digest = documents.raw(row["outcome_digest"])
                attempt.outcome_bytes = base64.b64decode(row["outcome_canonical_b64"])
            elif row["state"] == "accepted":
                self.outcome(
                    attempt,
                    STATUS.OUTCOME_STATUS_ABANDONED,
                    CAUSE.CAUSE_CODE_EXECUTOR_INVALIDATED,
                    ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
                    "accepted-but-incomplete across a control-parent restart: the execution "
                    "context is gone, so this attempt reports ABANDONED and is never "
                    "re-executed",
                )
            else:
                continue
            self.history[attempt.key()] = attempt
            recovered.append(self._active_row(attempt, row))
        self.recovered = recovered
        return recovered

    def _active_row(
        self, attempt: AttemptRecord, journal_row: Mapping[str, Any] | None = None
    ) -> pb.HeldAttempt:
        """One placement-aware `HeldAttempt` (#481: the set holds unacked OUTCOMES too, and
        those are not active — which is what the old `ActiveAttempt` name got wrong).

        A recovered attempt names the placement and executor generation its acceptance
        journaled; a live one names the current. `ATTEMPT_STATE_HELD_UNDURABLE` is the rev-2
        state for #494b's hold (#507d/#510d): the frozen wire could only spell it as RUNNING
        plus a LOCAL_SAFETY fault compound — honest, but compound, and a RecordOwner reading
        RUNNING could not tell a live execution from an outcome nobody could write down.
        """
        row = journal_row or {}
        pending = attempt.state in ("outcome", "closed")
        held = pb.HeldAttempt(
            request_id=attempt.request_id,
            attempt_ordinal=attempt.attempt,
            kind=_KIND[attempt.kind],
            state=pb.AttemptState.ATTEMPT_STATE_OUTCOME_PENDING_ACK
            if pending
            else pb.AttemptState.ATTEMPT_STATE_HELD_UNDURABLE
            if attempt.state == "local_fault"
            else pb.AttemptState.ATTEMPT_STATE_RUNNING,
            invocation_spec_digest=attempt.digest,
            placement_id=str(row.get("placement_id", "")) or self.placement_id(),
            executor_generation=int(row.get("executor_generation", 0))
            or self.supervision.incarnation,
        )
        if pending:
            # Set IFF the state is OUTCOME_PENDING_ACK, so the RecordOwner can reconcile the
            # exact outcome it owes an ack for without a second round trip.
            held.outcome_id = attempt.outcome_id
            held.outcome_digest = attempt.outcome_digest
        return held

    def pending_outcomes(self) -> list[AttemptRecord]:
        return [a for a in self.history.values() if a.state == "outcome"]

    #: Every state in which this worker still HOLDS an attempt and owes the owner something.
    #: `local_fault` is here on purpose (#494b): its outcome was never durable and never
    #: sent, so the attempt is still the worker's — the wire has no state for "held, outcome
    #: undurable", and RUNNING plus the LOCAL_SAFETY_REFUSAL fault naming it is the honest
    #: pair. Dropping it would make the snapshot silent about work the worker is holding.
    HELD = ("accepted", "running", "outcome", "local_fault")

    def snapshot_entries(self) -> list[pb.HeldAttempt]:
        """The recovery snapshot (02 §6): every attempt this worker still HOLDS — accepted,
        running, terminal-pending-ack or locally faulted. A queued one is deliberately absent
        (the wire has no state for held-not-admitted; the owner's DISPATCHING-absent rule
        resends it, and the admission window is one, so there is never a QUEUE of them)."""
        return [self._active_row(a) for a in self.history.values() if a.state in self.HELD]

    def active(self) -> list[pb.HeldAttempt]:
        """The attempts this worker HOLDS, for the ObservedWorkerState tick.

        Running attempts AND outcomes pending ack — the same set the snapshot carries and
        the same set `available_attempt_slots` counts, because a seat held by an unacked
        outcome is a seat (§5). Reading only `live` under-reported exactly those: `close()`
        drops the attempt from `live`, but an outcome nobody acked is still owed.

        A queued one is deliberately absent: acceptance is what makes an attempt a claim,
        and the wire has no state for "held, not yet admitted".
        """
        return [self._active_row(a) for a in self.history.values() if a.state in self.HELD]
