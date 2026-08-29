"""The local slice: one request, through the REAL protocol path, with no network.

This is §8's one-kernel rule made executable. A one-shot local run does not get its own
execution implementation — it gets an EPHEMERAL LOCAL WORKER whose owner is a pair of
in-process queues instead of a dialing gRPC client (#436: the owner dials; here the "dial"
is frame delivery). Everything below that substitution is the production path, unchanged
and un-branched:

    the same Claim -> ClaimAck -> snapshot -> SnapshotAck handshake and envelope fence
    the same `InvocationSpec` canonical bytes and the same digest recomputation
    the same atomic acceptance boundary (payload fetch -> features -> plan -> prepare ->
      reserve -> JOURNAL -> AttemptAccepted)
    the same worker/executor process boundary, exec-first trampoline and env seal
    the same durable journal, the same terminal-before-send rule, the same TerminalAck
    the same output grant, freeze point, tail and manifest
    the same ledger, reconciled at attempt close

The only two things that are LOCAL are stated out loud here rather than hidden:

1. **The owner is this module.** It claims the worker, mints the revision and the
   deployment set, resolves the binding plan id, builds the spec document, grants the
   payload input and one destination per output field path, and acks the terminal. That is
   exactly the local client's job on the network path.
2. **The journal is DELETABLE** after a fully acknowledged terminal (§8). It is deleted at
   the END, never before: a run that is killed mid-attempt leaves its journal behind, which
   is what makes the next run's recovery honest instead of amnesiac.

What this module is NOT: it is not the `run` verb. cr-016 owns the CLI surface, the
descriptor-derived output ids and the exit-code mapping, and it will WRAP this — never the
reverse, and never by growing a second execution path beside it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import shutil
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from cozy_runtime.internal import canonical, descriptor, provisioned
from cozy_runtime.internal.config import RuntimeConfig
from cozy_runtime.internal.pathkey import opaque_key
from cozy_runtime.internal.worker import grants as grants_module
from cozy_runtime.internal.worker import triage
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb

#: The one local deployment's owner-minted routing key (#446). Routing, never identity:
#: the InvocationSpec digest deliberately excludes it.
LOCAL_PLACEMENT = "plc-local"

_M = TypeVar("_M")

#: What an OUTPUT TRANSACTION is allowed AFTER the handler returns: read the attempt spool,
#: write every granted destination, mint the receipt. The grant outlives the attempt by this
#: and by nothing more — it is a tail, never a lifetime.
#:
#: The lifetime itself is DERIVED FROM THE ATTEMPT (`LocalRecordOwner.expiry`). It used to be
#: a 900 s constant derived from nothing, which is not a bound but a ceiling on what a job
#: may be: job-001's dual production ran 20:15.89, landed its 99 GB artifact intact, and died
#: `grant_expired: the output grant expired before the write`.
GRANT_TAIL_SECONDS = 300


class LocalBindingError(Exception):
    """The request names an entrypoint this run staged no binding plan for."""


@dataclass(frozen=True, slots=True)
class StagedWeightlessRelease:
    """The complete local form of one canonical, model-free serving closure."""

    plans: Mapping[str, str]
    plan_refs: Mapping[str, tuple[str, int]]
    descriptor: Path
    artifact_store: Path
    artifact_object_set: Path


@dataclass(frozen=True, slots=True)
class LocalPlacementFacts:
    """Exact qualified release facts for a production-shaped local protocol run.

    This changes only what the local RecordOwner puts in the ordinary PlacementSpec and
    InvocationSpec.  The same worker convergence, executor and attempt path consume it.
    """

    environment_spec_digest: str
    installed_environment_receipt_digest: str
    package_descriptor_digest: str
    environment_root: Path
    wheelhouse_manifest: Path


@dataclass(frozen=True, slots=True)
class WeightlessReleaseDocuments:
    """Exact bytes behind both local staging and pre-spawn subject resolution."""

    descriptor: bytes
    plans: Mapping[str, bytes]
    artifact_object_set: bytes

    def subjects(self) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for entrypoint, raw in sorted(self.plans.items()):
            ref = _exact_ref(raw)
            digest = str(ref["digest"])
            rows.append(
                {
                    "entrypoint": entrypoint,
                    "subject_id": digest,
                    "kind": "plan",
                    "digest": digest,
                    "length": int(ref["length"]),
                }
            )
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class LocalInput:
    """One owner-held file plus its typed position in the request."""

    source: str
    order: int


@dataclass(frozen=True, slots=True)
class LocalRequest:
    """One request, as a RecordOwner would put it on the wire."""

    entrypoint: str
    payload: dict[str, Any]
    #: the result FIELD PATHS the caller grants a destination for (`image`, `detail.thumb`).
    #: cr-016 derives these from the package descriptor; here the driver names them, and a
    #: set that does not match what the handler produced refuses the tail whole (#248).
    outputs: tuple[str, ...] = ("image",)
    request_id: str = "run-1"
    attempt: int = 1
    timeout_ms: int = 0
    """The caller's own bound on this attempt, as a DURATION. 0 means none.

    A RecordOwner mints the absolute deadline when it DISPATCHES, which is what `_dispatch`
    does — a duration written straight into `deadline_unix_ms` is a 1970 timestamp, and
    every attempt under it was silently given the 0.5 s floor.
    """
    #: grant the payload from THIS location instead of a file staged beside the run. A
    #: RecordOwner may grant any location it can authorize; the worker still verifies the
    #: bytes it reads against the grant's own digest and length.
    payload_url: str = ""
    #: `entrypoint` (serving) or `job` (run-to-completion). ONE ExecutionSpec mode is named
    #: either way; a request that named both would be two lifecycles under one identity.
    kind: str = "entrypoint"
    #: JOB mode only: the descriptor id the worker resolves its local job plan against, and
    #: the materialized input trees this run grants (`ref -> local directory`).
    job_descriptor_id: str = ""
    trees: Mapping[str, str] = field(default_factory=dict)
    #: cr-012: the INPUT ASSETS this run grants, as `input_id -> (source, typed order)`.
    #: Order comes from the decoded request walk exactly once; it is never guessed back out
    #: of a field-path string, where a numeric map key is indistinguishable from a list
    #: position. The worker still verifies every byte and sniffs its type. Ordered,
    #: identified subjects — never a digest-sorted anonymous list (worker-protocol/01).
    inputs: Mapping[str, LocalInput] = field(default_factory=dict)
    org: str = "local"
    """The publishing org. A job's outputs are hub-rewritten into the scratch model
    `<org>/_job-<request-id>` — the underscore is deliberately outside the public model
    grammar, so no user ref can squat a job's scratch, and promotion out of it is always an
    explicit later act under the caller's own credentials (jobs.md)."""


@dataclass(slots=True)
class SliceOutcome:
    """Everything one local run OBSERVED. No field is inferred; each has a producer."""

    status: str = ""
    terminal: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    result_schema_digest: str = ""
    invocation: dict[str, Any] = field(default_factory=dict)
    invocation_canonical_bytes: bytes = b""
    accepted: dict[str, Any] = field(default_factory=dict)
    manifest: list[dict[str, Any]] = field(default_factory=list)
    #: cr-009's typed publication receipt — ONE canonical bundle snapshot per publication,
    #: journaled before the terminal named it. `manifest_id` on the terminal IS its digest.
    publication: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    terminal_digest: str = ""
    terminal_canonical_bytes: bytes = b""
    placement_set_digest: str = ""
    placement_set_canonical_bytes: bytes = b""
    placement_spec_digest: str = ""
    worker_resources: dict[str, Any] = field(default_factory=dict)
    accelerator_qualification: dict[str, Any] = field(default_factory=dict)
    ledger: dict[str, Any] = field(default_factory=dict)
    reconciliation: dict[str, Any] = field(default_factory=dict)
    journal: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    warm: list[dict[str, Any]] = field(default_factory=list)
    faults: list[dict[str, str]] = field(default_factory=list)
    progress: int = 0
    executions: int = 0
    #: cr-011: the OPAQUE triage handle from the terminal, and the bundle read back by it.
    #: The subject is the ONLY way in — there is no path anywhere in this outcome.
    triage_subject: str = ""
    triage: dict[str, Any] = field(default_factory=dict)
    triage_explained: list[str] = field(default_factory=list)
    exit_code: int = 0
    journal_deleted: bool = False
    #: JOB mode: the durable checkpoint requests this run received, in arrival order, and
    #: the scratch model its outputs were rewritten into.
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    publication_scope: str = ""
    #: how the run ENDED, which is not always "it produced a terminal": `terminal`,
    #: `worker_error` (the worker reported it could not become ready), `ready_timeout`, or
    #: `record_owner_error` (this side failed). A one-shot run that cannot say this would
    #: have to pretend a hang was a result.
    ended: str = "unstarted"
    error: str = ""
    #: The RecordOwner's PROJECTION over (status, cause, origin) — see `redispatchable()`.
    #: True means the ordinal was consumed and the BUDGET was not, so the next ordinal may
    #: go elsewhere immediately. It is never a worker wire fact and is computed here.
    redispatchable: bool = False


#: THE REFUSED PROJECTION, SPLIT BY (cause, origin) — rev-2 §7/#480b.
#:
#: 02 §2's table used to read "REFUSED => settle refused; never re-dispatch". That was
#: written when every REFUSED was a judgment about the WORK. These four are judgments about
#: the WORKER, made before the author ran: the ordinal is consumed, the budget is untouched,
#: and the RecordOwner may mint `attempt_ordinal + 1` and dispatch it elsewhere immediately —
#: no cooldown, no budget draw. The author/runtime causes (INVALID_REQUEST,
#: UNSUPPORTED_INPUT, LOCAL_SAFETY, CONSTRAINT_INFEASIBLE) still settle refused forever.
#:
#: Retryability is exactly this: a RecordOwner projection, unrepresentable as a worker wire
#: fact, which is why the worker sends the neutral triple and this side reads it.
PRE_EXECUTION_REFUSALS = frozenset(
    {
        pb.CauseCode.CAUSE_CODE_NO_CAPACITY,
        pb.CauseCode.CAUSE_CODE_ADMISSION_GENERATION_STALE,
        pb.CauseCode.CAUSE_CODE_UNKNOWN_PLACEMENT,
        pb.CauseCode.CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE,
    }
)


def redispatchable(body: Mapping[str, Any]) -> bool:
    """Can this attempt's ordinal + 1 go straight to another worker?

    `execution_started` is the BILLING FACT and it is STRUCTURAL (#480c) — the worker states
    it rather than leaving a RecordOwner to infer it from a cause-code allowlist. Both are
    checked: the cause says whose judgment it was, and the bit says whether anything ran.
    """
    if body.get("status") != pb.OutcomeStatus.OUTCOME_STATUS_REFUSED:
        return False
    cause = body.get("cause") or {}
    return (
        cause.get("code") in PRE_EXECUTION_REFUSALS
        and cause.get("origin") == pb.CauseOrigin.CAUSE_ORIGIN_WORKER
        and not body.get("execution_started", False)
    )


class LocalRecordOwner:
    """The in-memory RECORDOWNER — the record-plane half of th-024, in this process.

    It is a real RecordOwner in every way the worker can observe: it claims first, speaks
    only `cozy.worker.v1` frames under the three-field
    envelope, RECOMPUTES the snapshot digest before acknowledging it, dispatches against the
    admission generation it last saw, and learns dispatchability the way a hub does — from a
    durable ObservedWorkerState's per-placement axes, not from a Python attribute.
    """

    def __init__(
        self,
        request: LocalRequest,
        bindings: dict[str, str],
        plan_refs: dict[str, tuple[str, int]],
        workspace: Path,
        *,
        release_id: str,
        placement_facts: LocalPlacementFacts | None = None,
        model_object_set: bytes = b'{"kind":"tensorhub.resolved_object_set/1","roots":[]}',
        auto_ack: bool = True,
        on_progress: Any = None,
        ready_timeout: float = 1800.0,
    ) -> None:
        self.request = request
        self.bindings = bindings  # entrypoint -> entrypoint_binding_plan_id
        self.plan_refs = plan_refs
        #: JOB mode: the durable checkpoint requests this owner answered, in order.
        self.checkpoints: list[dict[str, Any]] = []
        self.workspace = workspace
        self.release_id = release_id
        self.placement_facts = placement_facts
        self.model_object_set = model_object_set
        self.auto_ack = auto_ack
        self.ready_timeout = ready_timeout
        self.ended = "unstarted"
        self.error = ""
        self.host = InMemoryControlHost()
        self.host.on_frame = self._on_frame
        self._on_progress = on_progress
        self.progress_frames: list[pb.AttemptProgress] = []
        #: the accepted envelope this RecordOwner speaks under, learned from ClaimAck
        self.record_owner_epoch = 1
        self.generation = 0
        self.boot_id = ""
        self.held_attempts: list[pb.HeldAttempt] = []
        #: the fence the worker last advertised; every offer ECHOES it (§6)
        self.admission_generation = 0
        self.dispatched = False
        self.accepted: pb.AttemptAccepted | None = None
        self.terminal: pb.AttemptOutcome | None = None
        self.reports: list[pb.ObservedWorkerState] = []
        self.spec_digest = b""
        self.spec_document: dict[str, Any] = {}
        self.spec_canonical_bytes = b""
        self.worker_resources = pb.WorkerResources()
        self.placement_set_digest = b""
        self.placement_set_canonical_bytes = b""
        self.done = threading.Event()
        self.marks: dict[str, float] = {"start": time.perf_counter()}
        self.grant_dir = workspace / "grants"
        self.output_dir = workspace / "outputs"
        # THE CLAIM IS QUEUED BEFORE THE WORKER SERVES: the worker's stream conversation
        # begins by reading it, exactly as a dialing owner's first frame arrives first.
        claim = pb.Claim(
            record_owner_epoch=self.record_owner_epoch,
            record_owner_id="local-run",
            wire_minor=WIRE_MINOR,
        )
        self.host.deliver(pb.RecordOwnerFrame(claim=claim))
        threading.Thread(target=self._deadline, daemon=True, name="ready-deadline").start()

    def destination(self, output_id: str) -> Path:
        """The exact granted destination for one output field path."""
        return self.output_dir / self._filename(output_id)

    def tap_progress(self, frame: pb.AttemptProgress) -> None:
        """The lossy lane's in-process observer (the local door's one watcher)."""
        self.progress_frames.append(frame)
        if self._on_progress is not None:
            self._on_progress(frame)

    def _deadline(self) -> None:
        # A one-shot run gets a readiness DEADLINE: a person is sitting in front of it.
        if self.done.wait(self.ready_timeout):
            return
        if not self.dispatched:
            self.ended = "ready_timeout"
            print(
                f"[run] the worker never reported ready within {self.ready_timeout:g} s; "
                "ending the run rather than waiting on it",
                flush=True,
            )
            self.done.set()
            self.host.close()

    def _env(self, message: _M) -> _M:
        """The owner's half of the three-field envelope (02 §0)."""
        inner: Any = message
        inner.record_owner_epoch = self.record_owner_epoch
        inner.control_stream_generation = self.generation
        inner.worker_boot_id = self.boot_id
        return message

    # ------------------------------------------------------------------ the exchange

    def _on_frame(self, frame: pb.WorkerFrame) -> None:
        """One owner-side frame. A FAILURE HERE ENDS THE RUN, never hangs it."""
        try:
            self._dispatch_frame(frame)
        except LocalBindingError as exc:
            self.ended = "unbound_entrypoint"
            self.error = str(exc)[:400]
            print(f"[run] {self.error}", flush=True)
            self.done.set()
            self.host.close()
        except Exception as exc:
            self.ended = "record_owner_error"
            self.error = f"{type(exc).__name__}: {exc}"[:400]
            print(f"[run] the owner failed: {self.error}", flush=True)
            self.done.set()
            self.host.close()

    def _dispatch_frame(self, frame: pb.WorkerFrame) -> None:
        kind = frame.WhichOneof("msg")
        if kind == "claim_ack":
            self._on_claim_ack(frame.claim_ack)
        elif kind == "snapshot":
            self._on_snapshot(frame.snapshot)
        elif kind == "boot_failure":
            self.ended = "worker_error"
            self.error = f"boot failure: {frame.boot_failure.detail}"[:400]
            self.done.set()
            self.host.close()
        elif kind == "observed_state":
            self._on_observed_state(frame.observed_state)
        elif kind == "attempt_accepted":
            self._on_accepted(frame.attempt_accepted)
        elif kind == "attempt_outcome":
            self._on_outcome(frame.attempt_outcome)
        elif kind == "checkpoint_request":
            self._on_checkpoint(frame.checkpoint_request)

    def _on_accepted(self, accepted: pb.AttemptAccepted) -> None:
        """Retain only a complete acceptance for the exact offered invocation and mode."""
        if self.accepted is not None:
            raise RuntimeError("the worker accepted the same local attempt more than once")
        if accepted.request_id != self.request.request_id:
            raise RuntimeError(
                f"accepted request {accepted.request_id!r}, offered {self.request.request_id!r}"
            )
        if accepted.attempt_ordinal != self.request.attempt:
            raise RuntimeError(
                f"accepted ordinal {accepted.attempt_ordinal}, offered {self.request.attempt}"
            )
        if bytes(accepted.invocation_spec_digest) != self.spec_digest:
            raise RuntimeError("accepted invocation digest differs from the offered spec")
        if len(bytes(accepted.plan_digest)) != hashlib.sha256().digest_size:
            raise RuntimeError("accepted plan digest is not one SHA-256 identity")
        construction = bytes(accepted.model_construction_digest)
        if accepted.executor_generation <= 0:
            raise RuntimeError("accepted executor generation is not positive")
        if self.request.kind == "job":
            if accepted.placement_id:
                raise RuntimeError("a job acceptance carries a serving placement")
            if construction:
                raise RuntimeError("a job acceptance carries a model-construction identity")
            if (
                accepted.plan.delivery != "job"
                or accepted.plan.materialization != "run_to_completion"
                or accepted.plan.placement != "job_attempt"
            ):
                raise RuntimeError("a job acceptance carries a serving attempt plan")
        else:
            if accepted.placement_id != LOCAL_PLACEMENT:
                raise RuntimeError(
                    f"accepted placement {accepted.placement_id!r}, offered {LOCAL_PLACEMENT!r}"
                )
            if construction:
                raise RuntimeError("a local weightless acceptance invents model construction")
            if (
                accepted.plan.delivery != "native"
                or accepted.plan.materialization
                or accepted.plan.placement != "all_resident"
                or accepted.plan.reserved_device_memory_bytes
            ):
                raise RuntimeError("a local weightless acceptance carries a model-bearing plan")
        self.accepted = accepted
        self.marks["accepted"] = time.perf_counter()

    def _on_claim_ack(self, ack: pb.ClaimAck) -> None:
        if not ack.accepted:
            self.ended = "worker_error"
            self.error = f"claim refused: {pb.ClaimRejection.Name(ack.rejection)}"
            self.done.set()
            self.host.close()
            return
        self.generation = ack.control_stream_generation
        self.boot_id = ack.worker_boot_id
        self.worker_resources.CopyFrom(ack.resources)
        self.marks["registered"] = time.perf_counter()

    def _on_snapshot(self, snapshot: pb.WorkerSnapshot) -> None:
        """RECOMPUTE, reconcile, ack the EXACT snapshot, and only then send the desired
        state (02 §6, §5).

        The digest is recomputed over the bytes the snapshot carries, BEFORE the document is
        parsed. There is no entry count to reconcile against any more and there does not
        need to be: a truncated or re-ordered snapshot cannot hash to what it claims, so
        boundedness is a property of the message rather than a promise the sender makes.
        """
        recomputed = documents.digest_of(snapshot.snapshot_canonical_bytes)
        if recomputed != snapshot.snapshot_digest:
            raise RuntimeError(
                f"snapshot digest mismatch: carried {snapshot.snapshot_digest.hex()[:16]} "
                f"and the bytes hash to {recomputed.hex()[:16]}; NOT acknowledged"
            )
        body = pb.WorkerSnapshotBody()
        document = documents.read(snapshot.snapshot_canonical_bytes, pb.WorkerSnapshotBody)
        self.snapshot_document = document
        self.admission_generation = int(document.get("admission_generation", 0))
        del body
        self._send(
            pb.RecordOwnerFrame(
                snapshot_ack=self._env(
                    pb.SnapshotAck(
                        snapshot_id=snapshot.snapshot_id,
                        # ECHOED, never recomputed by the worker: the ack binds the exact
                        # snapshot this owner durably reconciled.
                        snapshot_digest=snapshot.snapshot_digest,
                    )
                )
            )
        )
        self._send(pb.RecordOwnerFrame(desired_state=self._desired_state()))

    def _plan_id(self) -> str:
        """The binding plan this request runs against, or the typed refusal that it has none.

        A request naming an entrypoint no binding plan covers is an ORDINARY RecordOwner
        condition — a stale install, a renamed function, a `--model` for the wrong slot —
        and it has to answer as one. It used to be a bare KeyError inside the message pump,
        which #306 already showed turns a one-line condition into a run that ends with no
        terminal at all.
        """
        plan_id = self.bindings.get(self.request.entrypoint)
        if plan_id is None:
            known = ", ".join(sorted(self.bindings)) or "no entrypoints"
            raise LocalBindingError(
                f"no binding plan covers {self.request.entrypoint!r}: this run staged "
                f"plans for {known}"
            )
        return plan_id

    def _on_observed_state(self, report: pb.ObservedWorkerState) -> None:
        """DISPATCHABILITY is learned from the worker's own durable ObservedWorkerState: the
        placement's SERVING axis plus `dispatchable_plan_ids` in serving mode, job capacity
        in job mode — and, in BOTH, the one worker-level admission fence (§6). The owner
        dispatches against the generation it last saw, and echoes it on the offer."""
        self.reports.append(report)
        self.admission_generation = report.admission_generation
        if report.worker_phase == pb.WorkerPhase.WORKER_PHASE_FAILED:
            # The worker said it cannot serve. That is an answer, and a one-shot run ends on
            # it instead of waiting for a dispatchability that will never come.
            self.marks.setdefault("ready", time.perf_counter())
            if self.terminal is None:
                self.ended = "worker_error"
            self.done.set()
            self.host.close()
            return
        open_window = (
            report.admission_state == pb.AdmissionState.ADMISSION_STATE_OPEN
            and report.available_attempt_slots > 0
        )
        if self.request.kind == "job":
            ready = open_window and report.job_capacity.jobs_available > 0
        else:
            plan_id = self._plan_id()
            status = next((d for d in report.placements if d.placement_id == LOCAL_PLACEMENT), None)
            ready = (
                status is not None
                and open_window
                and status.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
                and plan_id in status.dispatchable_plan_ids
            )
        if ready and not self.dispatched:
            self.marks["ready"] = time.perf_counter()
            self.dispatched = True
            self._dispatch()

    def _on_outcome(self, outcome: pb.AttemptOutcome) -> None:
        self.terminal = outcome
        self.ended = "terminal"
        self.marks["terminal"] = time.perf_counter()
        if not self.auto_ack:
            return
        self._send(
            pb.RecordOwnerFrame(
                outcome_ack=self._env(
                    pb.AttemptOutcomeAck(
                        request_id=outcome.request_id,
                        attempt_ordinal=outcome.attempt_ordinal,
                        invocation_spec_digest=outcome.invocation_spec_digest,
                        outcome_id=outcome.outcome_id,
                        outcome_digest=outcome.outcome_digest,
                    )
                )
            )
        )
        self.done.set()
        self.host.close()

    # ------------------------------------------------------------------ documents

    def expiry(self) -> int:
        """When this run's grant stops being authority — DERIVED, never a constant.

        A deadline is the caller's own bound, so the grant is that plus the tail an output
        transaction needs. A run with NO deadline has no bound to derive one from, and
        `grants.expired()` reads 0 as "does not expire": inventing a ceiling for a
        run-to-completion job is exactly what job-001 hit. The expiry PATH is still walked
        on every deadlined run, so the branch stays proven on the local door.
        """
        if not self.request.timeout_ms:
            return 0
        return int(time.time()) + self.request.timeout_ms // 1000 + GRANT_TAIL_SECONDS

    def _desired_state(self) -> pb.DesiredWorkerState:
        directive = self._env(
            pb.DesiredWorkerState(
                revision=1,
                posture=pb.Posture.POSTURE_ACCEPTING,
                wire_minor=WIRE_MINOR,
            )
        )
        if self.request.kind == "job":
            # A JOB directive names the build and its caps and NOTHING about serving: there
            # is no binding plan to prepare, no warm pool to fill and no serving pointer.
            directive.job.CopyFrom(
                pb.JobDirective(
                    build_id=self.release_id,
                    job_descriptor_id=self.request.job_descriptor_id,
                    resource_caps=pb.ResourceCaps(
                        device_required=False, max_rss_bytes=0, max_disk_bytes=0
                    ),
                    publication_contract=self._publication(),
                    reclaim_on_terminal=True,
                )
            )
            return directive
        # THE SET IS CONTENT-ADDRESSED, NOT ECHO-ADDRESSED (§4). The one-digester rule
        # (#454) is retired with the duplicate structured representation it needed: the
        # RecordOwner authors the canonical `PlacementSet/2` document ONCE and sends its
        # exact bytes, and the worker RECOMPUTES the digest before parsing a field. There is
        # no second place for the set to disagree with itself.
        object_set_ref = _exact_ref(self.model_object_set)
        object_set_digest = str(object_set_ref["digest"])
        placement_set = pb.PlacementSet(
            placements=[
                pb.Placement(
                    placement_id=LOCAL_PLACEMENT,
                    spec=pb.PlacementSpec(
                        # PROVENANCE only — the worker never resolves it (§1).
                        package_release_id=self.release_id,
                        environment_spec_digest=(
                            documents.raw(self.placement_facts.environment_spec_digest)
                            if self.placement_facts is not None
                            else b""
                        ),
                        installed_environment_receipt_digest=(
                            documents.raw(self.placement_facts.installed_environment_receipt_digest)
                            if self.placement_facts is not None
                            else b""
                        ),
                        package_descriptor_digest=(
                            documents.raw(self.placement_facts.package_descriptor_digest)
                            if self.placement_facts is not None
                            else b""
                        ),
                        binding_plans=[
                            pb.ArtifactSubject(
                                subject_id=plan_id,
                                kind="plan",
                                digest=documents.raw(self.plan_refs[plan_id][0]),
                                length=self.plan_refs[plan_id][1],
                            )
                            for plan_id in sorted(set(self.bindings.values()))
                        ],
                        model_object_set=pb.ArtifactSubject(
                            subject_id=object_set_digest,
                            kind="model_object_set",
                            digest=documents.raw(object_set_digest),
                            length=len(self.model_object_set),
                        ),
                    ),
                )
            ]
        )
        canonical_bytes, set_digest = documents.identity(placement_set)
        self.placement_set_digest = set_digest
        self.placement_set_canonical_bytes = canonical_bytes
        directive.placement_set.CopyFrom(
            pb.DesiredPlacementSet(
                placement_set_digest=set_digest,
                placement_set_canonical_bytes=canonical_bytes,
            )
        )
        return directive

    def scratch_model(self) -> str:
        """The opaque scratch model a job's publishes are rewritten into.

        The underscore is deliberately outside the public model grammar (`artifacts.REF`
        admits `[a-zA-Z0-9]` first), so no user ref can ever squat a job's scratch. Promotion
        OUT of scratch is an explicit later act under the caller's own credentials; a verdict
        stamps metadata and never gates persistence — a failed run's writes still land. The
        wire request id is hashed because a model name is authority, not identity.
        """
        return f"{self.request.org}/_job-{opaque_key('job-repository', self.request.request_id)}"

    def _publication(self) -> pb.PublicationContract:
        """What this run authorizes the job to write, by output id. Minted from the job's own
        DECLARATION (`publishes=`/`emits_media=`), never from the fact that it is a job."""
        return pb.PublicationContract(
            grant_id=self.scratch_model(),
            outputs=[
                pb.OutputBinding(output_id=output_id, max_bytes=64 << 20)
                for output_id in self.request.outputs
            ],
        )

    def _input_bindings(self, body: bytes) -> list[pb.InputBinding]:
        """The ORDERED input identities — INSIDE the digest (#439). The payload, the job's
        materialized trees, and cr-012's input assets, each with the digest/length this
        owner holds (a tree's verification is the store's own, so it carries none)."""
        rows = [
            pb.InputBinding(
                input_id="payload",
                digest="sha256:" + hashlib.sha256(body).hexdigest(),
                length=len(body),
                kind_mime="application/json",
                order=0,
            )
        ]
        for ref in sorted(self.request.trees):
            rows.append(
                pb.InputBinding(
                    input_id=f"{grants_module.TREE_PREFIX}{ref}",
                    kind_mime="inode/directory",
                    order=0,
                )
            )
        for input_id, local_input in sorted(self.request.inputs.items()):
            digest, length, media_type = _input_facts(Path(local_input.source))
            rows.append(
                pb.InputBinding(
                    input_id=input_id,
                    digest=digest,
                    length=length,
                    kind_mime=media_type,
                    order=local_input.order,
                )
            )
        return rows

    def _output_bindings(self) -> list[pb.OutputBinding]:
        return [
            pb.OutputBinding(output_id=output_id, max_bytes=64 << 20)
            for output_id in sorted(self.request.outputs)
        ]

    def _dispatch(self) -> None:
        plan_id = "" if self.request.kind == "job" else self._plan_id()
        self.grant_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Invocation payload identity is the exact canonical typed-argument document. A
        # local RecordOwner must not mint a whitespace-dependent dialect beside Tensorhub.
        body = canonical.write(self.request.payload)
        spec = pb.InvocationSpec(
            package_release_id=self.release_id,
            environment_spec_digest=(
                self.placement_facts.environment_spec_digest
                if self.placement_facts is not None
                else ""
            ),
            config_digest="",
            payload_digest="sha256:" + hashlib.sha256(body).hexdigest(),
            inputs=self._input_bindings(body),
            outputs=self._output_bindings(),
            # THE DEADLINE IS MINTED AT DISPATCH, from the caller's duration. It is what the
            # grant's own lifetime is derived from too, so the two cannot disagree.
            deadline_unix_ms=(
                int(time.time() * 1000) + self.request.timeout_ms if self.request.timeout_ms else 0
            ),
        )
        # `config_digest` stays EMPTY. Model configuration is bound and verified separately.
        if self.request.kind == "job":
            spec.job.CopyFrom(
                pb.JobInvocationSpec(
                    build_id=self.release_id,
                    job_descriptor_id=self.request.job_descriptor_id,
                    publication_contract=self._publication(),
                )
            )
        else:
            spec.serving.entrypoint_binding_plan_id = plan_id
            spec.serving.attempt_binding_id = plan_id
        canonical_bytes, digest = documents.identity(spec)
        self.spec_digest = digest
        self.spec_canonical_bytes = canonical_bytes
        self.spec_document = documents.read(canonical_bytes, pb.InvocationSpec)
        self.marks["dispatched"] = time.perf_counter()
        self._send(
            pb.RecordOwnerFrame(
                attempt_offer=self._env(
                    pb.AttemptOffer(
                        request_id=self.request.request_id,
                        attempt_ordinal=self.request.attempt,
                        invocation_spec_digest=digest,
                        invocation_spec_canonical_bytes=canonical_bytes,
                        grant=self._grant(body, digest),
                        placement_id="" if self.request.kind == "job" else LOCAL_PLACEMENT,
                        # The generation this owner OBSERVED when it dispatched (§6). A
                        # stale echo refuses deterministically.
                        admission_generation=self.admission_generation,
                    )
                )
            )
        )

    def _grant(self, body: bytes, invocation_digest: bytes) -> pb.DeliveryGrant:
        """Refreshable ACCESS only (#439): urls, credential, expiry — and the invocation it
        serves. Identity lives in the spec's bindings, which is what makes a refresh
        structurally unable to substitute meaning."""
        if self.request.payload_url:
            payload_url = self.request.payload_url
        else:
            path = self.grant_dir / (
                opaque_key("local-payload-grant", self.request.request_id, self.request.attempt)
                + ".json"
            )
            path.write_bytes(body)
            payload_url = f"file://{path}"
        expires = self.expiry()
        inputs = [pb.InputAccess(input_id="payload", url=payload_url)]
        inputs += [
            pb.InputAccess(
                input_id=f"{grants_module.TREE_PREFIX}{ref}",
                url=f"file://{Path(root).resolve()}",
            )
            for ref, root in sorted(self.request.trees.items())
        ]
        inputs += [
            pb.InputAccess(input_id=input_id, url=f"file://{Path(item.source).resolve()}")
            for input_id, item in sorted(self.request.inputs.items())
        ]
        return pb.DeliveryGrant(
            invocation_spec_digest=invocation_digest,
            credential=pb.DeliveryAccessCredential(
                issuer="cozy-runtime-local",
                key_id="local",
                epoch=1,
                expires_at_unix=expires,
                token=b"local",
            ),
            file_base_url=f"file://{self.grant_dir}",
            expires_at_unix=expires,
            inputs=inputs,
            outputs=[
                pb.OutputAccess(
                    output_id=output_id,
                    url=f"file://{self.output_dir / self._filename(output_id)}",
                )
                for output_id in sorted(self.request.outputs)
            ],
        )

    def _filename(self, output_id: str) -> str:
        return (
            opaque_key(
                "local-output",
                self.request.request_id,
                self.request.attempt,
                output_id,
            )
            + ".bin"
        )

    def _on_checkpoint(self, request: pb.JobCheckpointRequest) -> None:
        """The RecordOwner half of the DURABLE checkpoint exchange (cr-009).

        This is the seam cl-004 and the hub implement for real: the worker already made the
        save durable in its own journal, and this side records the identity and answers with
        its receipt. The RecordOwner's answer is a SECOND observation, not a second
        authority — an at-least-once durable message like every other (law 9).
        """
        digest = documents.spell(request.content_digest) if request.content_digest else ""
        held = next(
            (
                row
                for row in self.checkpoints
                if row["operation_key"] == request.operation_key
                and row["logical_key"] == request.logical_key
            ),
            None,
        )
        conflict = held is not None and held["content_digest"] != digest
        row = {
            "operation_key": request.operation_key,
            "logical_key": request.logical_key,
            "content_digest": digest,
            "seq": request.seq,
            "outcome": "CONFLICT" if conflict else ("REPLAYED" if held else "RECORDED"),
        }
        self.checkpoints.append(row)
        outcome = (
            pb.CheckpointOutcome.CHECKPOINT_OUTCOME_CONFLICT
            if conflict
            else pb.CheckpointOutcome.CHECKPOINT_OUTCOME_RECORDED
        )
        self._send(
            pb.RecordOwnerFrame(
                checkpoint_receipt=self._env(
                    pb.JobCheckpointReceipt(
                        request_id=request.request_id,
                        attempt_ordinal=request.attempt_ordinal,
                        operation_key=request.operation_key,
                        logical_key=request.logical_key,
                        content_digest=request.content_digest,
                        receipt_id=f"crc-{request.operation_key}-{request.seq}",
                        outcome=outcome,
                    )
                )
            )
        )

    # ------------------------------------------------------------------ driving

    def cancel(self, reason: pb.CancelReason = pb.CancelReason.CANCEL_REASON_CLIENT) -> None:
        """Cancel the live attempt — the same frame a hub sends, digest-fenced."""
        self._send(
            pb.RecordOwnerFrame(
                cancel_attempt=self._env(
                    pb.CancelAttempt(
                        request_id=self.request.request_id,
                        attempt_ordinal=self.request.attempt,
                        invocation_spec_digest=self.spec_digest,
                        reason=reason,
                    )
                )
            )
        )

    def close(self) -> None:
        self.host.close()

    def _send(self, frame: pb.RecordOwnerFrame) -> None:
        self.host.deliver(frame)


# --------------------------------------------------------------------------- the runner


def stage_weightless_release(
    cozy_home: Path, package_descriptor: Mapping[str, Any]
) -> StagedWeightlessRelease:
    """Stage the ONE canonical local closure for a wholly weightless release.

    Tensorhub authors these exact Plan and model-root documents for a provisioned package
    release. A local weightless run has no model root and nothing to fetch, but it still
    owes the worker the same stored-byte identities and complete visible-entrypoint set.
    This constructor is shared by the one-shot runner and process-launch acceptance paths;
    neither may revive the retired flat binding record.
    """

    documents = weightless_release_documents(package_descriptor)
    home = cozy_home.resolve()
    documents_root = home / "binding-documents"
    plans_root = home / "binding-plans"
    object_sets_root = home / "artifact-object-sets"
    artifact_store = home / "store"
    artifact_store.mkdir(parents=True, exist_ok=True)

    _, descriptor_path = _stage_exact(documents_root, documents.descriptor)
    plans: dict[str, str] = {}
    plan_refs: dict[str, tuple[str, int]] = {}
    plan_documents: dict[str, provisioned.ExactDocument] = {}
    for entrypoint, plan_bytes in sorted(documents.plans.items()):
        plan_ref, plan_path = _stage_exact(plans_root, plan_bytes)
        plan_id = str(plan_ref["digest"])
        length = int(plan_ref["length"])
        plans[entrypoint] = plan_id
        plan_refs[plan_id] = (plan_id, length)
        plan_documents[plan_id] = provisioned.ExactDocument(
            provisioned.ArtifactRef(plan_id, length), plan_path, plan_bytes
        )

    _, artifact_object_set = _stage_exact(object_sets_root, documents.artifact_object_set)

    # Exercise the same closed reader the worker will use after its process boundary. This
    # proves the local writer produced no private dialect before a caller tries to launch it.
    selected = provisioned.read_provisioned_release(
        plan_documents,
        read_document=provisioned.staged_document_reader(documents_root),
        package_release="local-weightless",
        store_root=artifact_store,
        object_set=provisioned.read_artifact_object_set(documents.artifact_object_set),
        dtype_widths={},
    )
    if set(selected.bindings) != set(plan_refs):
        raise RuntimeError("the staged local binding set did not survive canonical validation")
    return StagedWeightlessRelease(
        plans, plan_refs, descriptor_path, artifact_store, artifact_object_set
    )


def weightless_release_documents(
    package_descriptor: Mapping[str, Any],
) -> WeightlessReleaseDocuments:
    """Construct the exact local weightless documents without choosing storage paths."""

    entries = package_descriptor.get("entrypoints")
    if not isinstance(entries, list):
        raise RuntimeError("a local serving closure requires an PackageDescriptor entrypoint set")
    visible: list[Mapping[str, Any]] = [entry for entry in entries if isinstance(entry, Mapping)]
    if not visible:
        raise RuntimeError("a local serving closure requires at least one visible entrypoint")
    modeled = sorted(str(entry.get("name", "")) for entry in visible if entry.get("models"))
    if modeled:
        raise RuntimeError(
            "a local weightless closure cannot invent model bindings for " + ", ".join(modeled)
        )

    descriptor_bytes = descriptor.canonical_bytes(package_descriptor)
    descriptor_ref = _exact_ref(descriptor_bytes)
    plans: dict[str, bytes] = {}
    for entry in sorted(visible, key=lambda row: str(row.get("name", ""))):
        entrypoint = str(entry.get("name", ""))
        plan_bytes = canonical.write(
            {
                "bindings": [],
                "package_descriptor": descriptor_ref,
                "entrypoint": entrypoint,
                "format": "cozy.package.EntrypointBindingPlan/1",
            }
        )
        plans[entrypoint] = plan_bytes

    object_set_bytes = canonical.write(
        {
            "kind": provisioned.ARTIFACT_OBJECT_SET_KIND,
            "roots": [],
        }
    )
    return WeightlessReleaseDocuments(descriptor_bytes, plans, object_set_bytes)


def _exact_ref(raw: bytes) -> dict[str, Any]:
    return {"digest": "sha256:" + hashlib.sha256(raw).hexdigest(), "length": len(raw)}


def _stage_exact(root: Path, raw: bytes) -> tuple[dict[str, Any], Path]:
    ref = _exact_ref(raw)
    digest = str(ref["digest"])
    root.mkdir(parents=True, exist_ok=True)
    path = (root / f"{digest.removeprefix('sha256:')}.json").resolve()
    try:
        with path.open("xb") as handle:
            handle.write(raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise RuntimeError(f"content-addressed local document changed at {path}") from None
    return ref, path


def _input_facts(path: Path) -> tuple[str, int, str]:
    """Stream one local input once into its exact digest, length, and media claim.

    A local RecordOwner reads the bytes because it holds them; a hub reads its own record
    instead. Either way it is a CLAIM, and the worker sniffs what it actually received and
    refuses a disagreement — which is the point of declaring it at all.
    """
    from cozy_runtime.author._media import SNIFF_BYTES, sniff

    digest = hashlib.sha256()
    length = 0
    head = bytearray()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
            length += len(chunk)
            if len(head) < SNIFF_BYTES:
                head.extend(chunk[: SNIFF_BYTES - len(head)])
    return (
        "sha256:" + digest.hexdigest(),
        length,
        sniff(bytes(head)) or "application/octet-stream",
    )


def stage_job_plans(cozy_home: Path, records: list[dict[str, Any]]) -> dict[str, str]:
    """Write the local JOB plan records the worker resolves job descriptor ids against.

    On the network path th-004 ships the build and the JobDescriptor and tfs-003 the
    materialized input trees. Locally it is a
    plain file under `<cozy_home>/job-plans/`, and the worker cannot tell the difference
    because a RecordOwner never ships a path either way.
    """
    directory = cozy_home / "job-plans"
    directory.mkdir(parents=True, exist_ok=True)
    plans: dict[str, str] = {}
    for record in records:
        descriptor_id = str(record["job_descriptor_id"])
        (directory / f"{descriptor_id.split(':')[-1]}.json").write_text(
            json.dumps(record, indent=2)
        )
        plans[str(record["job"])] = descriptor_id
    return plans


def run_slice(
    config: RuntimeConfig,
    request: LocalRequest,
    binding_records: list[dict[str, Any]] | None = None,
    *,
    workspace: Path,
    release_id: str,
    package_descriptor: Mapping[str, Any] | None = None,
    placement_facts: LocalPlacementFacts | None = None,
    job_records: list[dict[str, Any]] | None = None,
    options: WorkerOptions | None = None,
    delete_journal: bool = True,
    auto_ack: bool = True,
    record_owner_hook: Any = None,
) -> SliceOutcome:
    """One request in, one typed result out, through the whole real path.

    `record_owner_hook` is handed the live `LocalRecordOwner` before the worker starts — that
    is how a driver injects a cancel at a chosen moment WITHOUT a plant switch inside the
    runtime (decisions #249: fault injection belongs to the harness).
    """
    workspace.mkdir(parents=True, exist_ok=True)
    if binding_records:
        raise RuntimeError(
            "flat local binding records were removed; supply the canonical provisioned closure"
        )
    plans: dict[str, str] = {}
    plan_refs: dict[str, tuple[str, int]] = {}
    staged: StagedWeightlessRelease | None = None
    if package_descriptor is not None:
        if request.kind != "entrypoint":
            raise RuntimeError("only a serving run accepts a package binding closure")
        staged = stage_weightless_release(config.cozy_home, package_descriptor)
        plans.update(staged.plans)
        plan_refs.update(staged.plan_refs)
    if placement_facts is not None and staged is None:
        raise RuntimeError("qualified placement facts require a staged package closure")
    stage_job_plans(config.cozy_home, job_records or [])
    record_owner = LocalRecordOwner(
        request,
        plans,
        plan_refs,
        workspace,
        release_id=release_id,
        placement_facts=placement_facts,
        model_object_set=(
            staged.artifact_object_set.read_bytes()
            if staged is not None
            else b'{"kind":"tensorhub.resolved_object_set/1","roots":[]}'
        ),
        auto_ack=auto_ack,
    )
    options = options or WorkerOptions(root=workspace / "worker")
    if staged is not None:
        if options.artifact_store not in (None, staged.artifact_store):
            raise RuntimeError("local closure and worker name different artifact stores")
        if options.artifact_object_set not in (None, staged.artifact_object_set):
            raise RuntimeError("local closure and worker name different artifact object sets")
        options = dataclasses.replace(
            options,
            artifact_store=staged.artifact_store,
            artifact_object_set=staged.artifact_object_set,
            environment_root=(
                placement_facts.environment_root
                if placement_facts is not None
                else options.environment_root
            ),
            base_manifest=(
                placement_facts.wheelhouse_manifest
                if placement_facts is not None
                else options.base_manifest
            ),
        )
    if not options.instance_id:
        options = dataclasses.replace(options, instance_id=f"ins-{uuid.uuid4().hex[:16]}")
    worker = Worker(config, options, record_owner.host)
    worker.watches.tap = record_owner.tap_progress
    if record_owner_hook is not None:
        record_owner_hook(record_owner, worker)
    started = time.perf_counter()
    exit_code = worker.run()
    return _collect(record_owner, worker, started, exit_code, delete_journal)


def _collect(
    record_owner: LocalRecordOwner,
    worker: Worker,
    started: float,
    exit_code: int,
    delete_journal: bool,
) -> SliceOutcome:
    outcome = SliceOutcome(exit_code=exit_code, ended=record_owner.ended, error=record_owner.error)
    outcome.journal = _journal_rows(worker.root / "journal.log")
    outcome.reconciliation = (
        worker.ledger.reconciliations[-1] if worker.ledger.reconciliations else {}
    )
    # The ledger as it stood AT ATTEMPT CLOSE, not after shutdown. Reading it here would
    # report the executor's RSS as unreadable and its device classes as stale, because by
    # now the process that owned them is gone — true, and useless. The reconciliation record
    # is the measurement; this is only where it is published.
    outcome.ledger = (
        {"format": "cozy.runtime.Ledger/0", "classes": outcome.reconciliation["classes"]}
        if outcome.reconciliation
        else worker.ledger.document()
    )
    outcome.warm = list(worker.warm)
    outcome.faults = [
        {"reason": fault.reason, "detail": fault.detail} for fault in worker.engine.faults
    ]
    outcome.progress = len(record_owner.progress_frames)
    outcome.executions = worker.engine.executed
    outcome.checkpoints = list(record_owner.checkpoints)
    outcome.publication_scope = (
        record_owner.scratch_model() if record_owner.request.kind == "job" else ""
    )
    if record_owner.spec_document:
        outcome.invocation = {
            "digest": _spell(record_owner.spec_digest),
            "document": record_owner.spec_document,
        }
        outcome.invocation_canonical_bytes = record_owner.spec_canonical_bytes

    if record_owner.accepted is not None:
        accepted = record_owner.accepted
        outcome.accepted = {
            "request_id": accepted.request_id,
            "attempt_ordinal": accepted.attempt_ordinal,
            "invocation_spec_digest": _spell(accepted.invocation_spec_digest),
            "plan_digest": _spell(accepted.plan_digest),
            # A JOB attempt has NO model construction: no warm pool, no serving pointer, no
            # retained generation. The empty digest is the honest answer and spelling it as
            # one would be inventing an identity for a construction that never happened.
            "model_construction_digest": _spell(accepted.model_construction_digest),
            "delivery": accepted.plan.delivery,
            "materialization": accepted.plan.materialization,
            "placement": accepted.plan.placement,
            "compute_dtype": accepted.plan.compute_dtype,
            "reserved_device_memory_bytes": accepted.plan.reserved_device_memory_bytes,
            "reserved_host_bytes": accepted.plan.reserved_host_bytes,
            "decision_explanation": accepted.plan.decision_explanation,
            "placement_id": accepted.placement_id,
            "executor_generation": accepted.executor_generation,
        }
    if record_owner.terminal is not None:
        outcome.terminal_digest = _spell(record_owner.terminal.outcome_digest)
        outcome.terminal_canonical_bytes = bytes(record_owner.terminal.outcome_canonical_bytes)
        body = documents.read(record_owner.terminal.outcome_canonical_bytes, pb.AttemptOutcomeBody)
        outcome.terminal = body
        # The triage read, exactly as cl-006 will do it: take the opaque subject off the
        # terminal, and read the bundle BY KEY, verified against its journaled receipt.
        # Nothing here holds a path to it.
        outcome.triage_subject = str((body.get("triage_bundle") or {}).get("subject_id", ""))
        if outcome.triage_subject:
            with contextlib.suppress(triage.TriageError, canonical.CanonicalError):
                outcome.triage = triage.BundleStore.read(
                    worker.root / "triage",
                    outcome.triage_subject,
                    worker.root / "journal.log",
                )
                outcome.triage_explained = triage.explain(outcome.triage)
        outcome.status = pb.OutcomeStatus.Name(
            body.get("status", pb.OutcomeStatus.OUTCOME_STATUS_UNSPECIFIED)
        )
        outcome.redispatchable = redispatchable(body)
        envelope = body.get("result") or {}
        inline = envelope.get("inline_result")
        if isinstance(inline, str):
            # The typed result rode the terminal INLINE. Read it with the canonical reader
            # that wrote it, never a looser JSON parser: the whole point of the envelope is
            # that the bytes a RecordOwner holds and the value a caller sees are one thing.
            decoded = canonical.parse(_b64(inline))
            outcome.result = decoded if isinstance(decoded, dict) else {"value": decoded}
        outcome.result_schema_digest = str(envelope.get("result_schema_digest", ""))
        manifest = body.get("output_manifest") or {}
        outcome.manifest = list(manifest.get("outputs", ()))
        outcome.publication = next(
            (row["body"] for row in outcome.journal if row.get("kind") == "publication"), {}
        )
        for row in outcome.manifest:
            path = record_owner.destination(str(row.get("output_id", "")))
            if path.is_file():
                outcome.outputs[str(row["output_id"])] = str(path)

    outcome.placement_set_digest = _spell(record_owner.placement_set_digest)
    outcome.placement_set_canonical_bytes = record_owner.placement_set_canonical_bytes
    outcome.worker_resources = documents.body(record_owner.worker_resources)
    placement = next(
        (
            item
            for report in reversed(record_owner.reports)
            for item in report.placements
            if item.placement_id == LOCAL_PLACEMENT
        ),
        None,
    )
    if placement is not None:
        outcome.placement_spec_digest = _spell(placement.placement_spec_digest)
        outcome.accelerator_qualification = documents.body(placement.accelerator)

    marks = record_owner.marks
    outcome.timings = {
        name: round((marks[key] - marks[base]) * 1000, 2)
        for name, key, base in (
            ("boot_to_register_ms", "registered", "start"),
            ("register_to_ready_ms", "ready", "registered"),
            ("dispatch_to_accept_ms", "accepted", "dispatched"),
            ("accept_to_terminal_ms", "terminal", "accepted"),
            ("request_to_result_ms", "terminal", "dispatched"),
        )
        if key in marks and base in marks
    }
    outcome.timings["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)

    # §8: the journal MAY be deleted after a fully acknowledged terminal — and only then.
    # A run that died mid-attempt leaves it behind on purpose, so the next run recovers.
    closed = any(row.get("kind") == "closed" for row in outcome.journal)
    if delete_journal and closed:
        shutil.rmtree(worker.root, ignore_errors=True)
        outcome.journal_deleted = True
    return outcome


def _spell(raw: bytes) -> str:
    """`sha256:<hex>`, or the empty string for a digest a mode genuinely does not have."""
    return documents.spell(raw) if len(raw) == 32 else ""


def _b64(text: str) -> bytes:
    import base64

    return base64.b64decode(text)


def _journal_rows(path: Path) -> list[dict[str, Any]]:
    """Read the journal back the way it was written: `<sha256 hex> <canonical json>`."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_bytes().split(b"\n"):
        _, _, rest = line.partition(b" ")
        if not rest:
            continue
        try:
            row = canonical.parse_canonical(rest)
        except canonical.CanonicalError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows
