"""The local slice: one request, through the REAL protocol path, with no network.

This is §8's one-kernel rule made executable. A one-shot local run does not get its own
execution implementation — it gets an EPHEMERAL LOCAL WORKER whose owner is a pair of
in-process queues instead of a dialing gRPC client (#436: the owner dials; here the "dial"
is frame delivery). Everything below that substitution is the production path, unchanged
and un-branched:

    the same Claim -> ClaimAck -> snapshot -> SnapshotAck handshake and envelope fence
    the same `InvocationSpec` canonical bytes and the same digest recomputation
    the same atomic acceptance boundary (payload fetch -> features -> plan -> prepare ->
      reserve -> record -> AttemptAccepted)
    the same worker/executor process boundary, exec-first trampoline and env seal
    the same in-memory attempt log, the same terminal-before-send rule, the same TerminalAck
    the same output grant, freeze point, tail and manifest
    the same ledger, reconciled at attempt close

The only two things that are LOCAL are stated out loud here rather than hidden:

1. **The owner is this module.** It claims the worker, mints the revision and the
   deployment set, resolves the binding plan id, builds the spec document, grants the
   payload input and one destination per output field path, and acks the terminal. That is
   exactly the local client's job on the network path.
2. **The workspace is disposable** after a fully acknowledged terminal. Creator owns retry
   and recovery across child processes; Runtime keeps only this process's attempt state.

What this module is NOT: it is not the `run` verb. cr-016 owns the CLI surface, the
descriptor-derived output ids and the exit-code mapping, and it will WRAP this — never the
reverse, and never by growing a second execution path beside it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from cozy_runtime.internal import bindings as binding_table
from cozy_runtime.internal import (
    canonical,
    descriptor,
    fill,
    liveness,
    model_config,
    proctree,
    pull,
)
from cozy_runtime.internal.config import RuntimeConfig
from cozy_runtime.internal.pathkey import opaque_key
from cozy_runtime.internal.worker import grants as grants_module
from cozy_runtime.internal.worker import triage
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.plan import (
    DeclaredBinding,
    ModelBinding,
    model_binding_set_digest,
)
from cozy_runtime.internal.worker.records import WorkerRecords
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb

#: The one local deployment's owner-minted routing key (#446). Routing, never identity:
#: the InvocationSpec digest deliberately excludes it.
LOCAL_PLACEMENT = "plc-local"

_M = TypeVar("_M")


class LocalBindingError(Exception):
    """The request names an entrypoint this run staged no binding for."""


@dataclass(frozen=True, slots=True)
class StagedDevelopmentPlacement:
    """One exact PlacementSet selection for source-checkout development."""

    bindings: Mapping[str, str]
    placement: pb.Placement
    artifact_store: Path
    artifact_cache: Path
    declared_bindings: Mapping[str, DeclaredBinding] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DevelopmentPlacementDocuments:
    """Exact bytes and Placement behind local source-checkout execution."""

    descriptor: bytes
    placement: pb.Placement
    bindings: Mapping[str, DeclaredBinding] = field(default_factory=dict)

    def subjects(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "entrypoint": entrypoint.name,
                "digest": documents.spell(entrypoint.entrypoint_binding_digest),
            }
            for entrypoint in self.placement.entrypoints
        )


@dataclass(frozen=True, slots=True)
class _DevelopmentModel:
    slot: Any
    repo: str
    name: str
    version: str
    lane: str
    manifest_digest: str
    manifest_length: int
    config_digest: str
    config_bytes: bytes
    components: tuple[str, ...]
    assets: Mapping[str, Mapping[str, Any]]
    store: Path
    variant: str
    custody: str
    model_weight_bytes: int


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
    #: JOB mode only: descriptor parameter -> exact Manifest (digest, length). The local
    #: harness stands in for a RecordOwner that already completed model-source preparation.
    models: Mapping[str, tuple[str, int]] = field(default_factory=dict)
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
    #: recorded before the terminal named it. `manifest_id` on the terminal IS its digest.
    publication: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    terminal_digest: str = ""
    terminal_canonical_bytes: bytes = b""
    placement_set_digest: str = ""
    placement_set_canonical_bytes: bytes = b""
    worker_resources: dict[str, Any] = field(default_factory=dict)
    accelerator_qualification: dict[str, Any] = field(default_factory=dict)
    ledger: dict[str, Any] = field(default_factory=dict)
    reconciliation: dict[str, Any] = field(default_factory=dict)
    attempt_records: list[dict[str, Any]] = field(default_factory=list)
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
    workspace_deleted: bool = False
    #: JOB mode: the durable checkpoint requests this run received, in arrival order, and
    #: the scratch model its outputs were rewritten into.
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    publication_scope: str = ""
    #: how the run ENDED, which is not always "it produced a terminal": `terminal`,
    #: `worker_error` (the worker reported it could not become ready), `worker_wedged` (it
    #: provably stopped working before it was ready — `LocalRecordOwner.watch`), or
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
        pb.CauseCode.CAUSE_CODE_ADMISSION_EPOCH_STALE,
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
    admission epoch it last saw, and learns dispatchability the way a hub does — from a
    durable ObservedWorkerState's per-placement axes, not from a Python attribute.
    """

    def __init__(
        self,
        request: LocalRequest,
        bindings: dict[str, str],
        workspace: Path,
        *,
        package_revision_digest: str,
        placement: pb.Placement | None = None,
        auto_ack: bool = True,
        on_progress: Any = None,
    ) -> None:
        self.request = request
        self.bindings = bindings  # entrypoint -> entrypoint_binding_digest
        #: JOB mode: the durable checkpoint requests this owner answered, in order.
        self.checkpoints: list[dict[str, Any]] = []
        self.workspace = workspace
        self.package_revision_digest = package_revision_digest
        self.placement = placement
        self.auto_ack = auto_ack
        self.ended = "unstarted"
        self.error = ""
        self.host = InMemoryControlHost()
        self.host.on_frame = self._on_frame
        self._on_progress = on_progress
        self.progress_frames: list[pb.AttemptProgress] = []
        #: the accepted envelope this RecordOwner speaks under, learned from ClaimAck
        self.record_owner_epoch = 1
        self.control_stream_epoch = 0
        self.boot_id = ""
        self.held_attempts: list[pb.HeldAttempt] = []
        #: the fence the worker last advertised; every offer ECHOES it (§6)
        self.admission_epoch = 0
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

    def destination(self, output_id: str) -> Path:
        """The exact granted destination for one output field path."""
        return self.output_dir / self._filename(output_id)

    def tap_progress(self, frame: pb.AttemptProgress) -> None:
        """The lossy lane's in-process observer (the local door's one watcher)."""
        self.progress_frames.append(frame)
        if self._on_progress is not None:
            self._on_progress(frame)

    def watch(self, worker: Worker) -> None:
        """Start the READINESS WATCH: end the run only when the worker has provably stopped.

        A one-shot run used to get a readiness DEADLINE — 1800 s, no flag to raise it —
        which killed a run whose worker was still downloading or loading a large model,
        and the person in front of it got a hang report for a boot that was working. There
        is no number of seconds a boot may take. What there is, is observation: the
        worker's own boot cursors (`SilenceMonitor`: download, load, warm positions) and
        the measured work of everything the boot runs — this process's working threads,
        its executor, a `uv` install, a native-wheel probe — which `_boot_meter` folds into
        one reading. The run ends `worker_wedged` when that reading has not changed for
        longer than `liveness.Pace` allows against the pauses this boot has itself shown,
        and never before the floor below. Readiness (`dispatched`) or any other end of the
        run stops the watch.

        The floor is DERIVED, not chosen. The deepest silence a healthy boot is permitted
        to show is the pull ledger's own stall floor (`pull.STALL_SAMPLES *
        STALL_SAMPLE_SECONDS`): a stream that has moved nothing for that long is re-asked
        for, and the re-ask is progress this watch can see. An outside observer must be
        more patient than that inner one, by the same factor the inner one applies to its
        own measurements — so the floor is `STILL_FACTOR *` the pull floor, and a boot that
        never paused is called wedged at that and not before.
        """
        threading.Thread(
            target=self._watch, args=(worker,), daemon=True, name="ready-watch"
        ).start()

    def _watch(self, worker: Worker) -> None:
        pace = liveness.Pace()
        while not self.done.wait(liveness.SAMPLE_SECONDS):
            if self.dispatched:
                return
            floor = liveness.STILL_FACTOR * pull.STALL_SAMPLES * pull.STALL_SAMPLE_SECONDS
            pace.observe(self._boot_meter(worker))
            if not pace.wedged(floor):
                continue
            open_steps = sorted(
                f"{cursor.step}:{cursor.subject}"
                for cursor in list(worker.monitor.cursors.values())
                if not cursor.ended
            )
            self.ended = "worker_wedged"
            self.error = (
                f"the worker stopped working before it was ready: "
                f"{pace.verdict(floor)}; open boot steps: {', '.join(open_steps) or 'none'}"
            )[:400]
            print(f"[run] {self.error}; ending the run rather than waiting on it", flush=True)
            self.done.set()
            self.host.close()
            return

    @staticmethod
    def _boot_meter(worker: Worker) -> object:
        """One reading of everything the boot is doing, or None when it cannot be read.

        Work by each of this process's threads (`progress_burn`: CPU plus bytes moved),
        read per thread and summed over the ones that work — the lanes that wake on a clock
        (`Worker.pollers`) and this watch itself are left out, never subtracted, because a
        process total less a thread's own reading is two non-atomic reads and the difference
        jitters by the bytes the watch itself moved in between; PLUS the same for every
        descendant process; PLUS the boot cursors' positions and ends. Any change is
        movement — a thread or child ending lowers the sum, and an ending is progress.
        """
        me = os.getpid()
        quiet = worker.pollers | {threading.get_native_id()}
        try:
            threads = [int(name) for name in os.listdir(f"/proc/{me}/task")]
            working = sum(liveness.burn(me, tid) or 0 for tid in threads if tid not in quiet)
            below = sum(liveness.burn(pid) or 0 for pid in proctree.descendants(me))
            cursors = tuple(
                (cursor.subject, cursor.position, cursor.ended)
                for cursor in list(worker.monitor.cursors.values())
            )
        except (OSError, proctree.ProcessTreeUnsupported, RuntimeError):
            return None
        return (working + below, cursors)

    def _env(self, message: _M) -> _M:
        """The owner's half of the three-field envelope (02 §0)."""
        inner: Any = message
        inner.record_owner_epoch = self.record_owner_epoch
        inner.control_stream_epoch = self.control_stream_epoch
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
        # Acceptance is queue ADMISSION (minor 23, proto-026): it carries routing and the
        # lane, never a plan or an executor epoch — those bind at device entry and ride
        # HeldAttempt from RUNNING on.
        if self.request.kind == "job":
            if accepted.placement_id:
                raise RuntimeError("a job acceptance carries a serving placement")
        elif accepted.placement_id != LOCAL_PLACEMENT:
            raise RuntimeError(
                f"accepted placement {accepted.placement_id!r}, offered {LOCAL_PLACEMENT!r}"
            )
        self.accepted = accepted
        self.marks["accepted"] = time.perf_counter()

    def _on_claim_ack(self, ack: pb.ClaimAck) -> None:
        if not ack.accepted:
            self.ended = "worker_error"
            self.error = f"claim refused: {pb.ClaimRejection.Name(ack.rejection)}"
            self.done.set()
            self.host.close()
            return
        self.control_stream_epoch = ack.control_stream_epoch
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
        self.admission_epoch = int(document.get("admission_epoch", 0))
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

    def _binding_digest(self) -> str:
        """The selected binding this request runs against, or a typed refusal.

        A request naming an entrypoint no binding plan covers is an ORDINARY RecordOwner
        condition — a stale install, a renamed function, a `--model` for the wrong slot —
        and it has to answer as one. It used to be a bare KeyError inside the message pump,
        which #306 already showed turns a one-line condition into a run that ends with no
        terminal at all.
        """
        binding_digest = self.bindings.get(self.request.entrypoint)
        if binding_digest is None:
            known = ", ".join(sorted(self.bindings)) or "no entrypoints"
            raise LocalBindingError(
                f"no binding covers {self.request.entrypoint!r}: this run selected {known}"
            )
        return binding_digest

    def _on_observed_state(self, report: pb.ObservedWorkerState) -> None:
        """DISPATCHABILITY is learned from the worker's own durable ObservedWorkerState: the
        placement's SERVING axis plus `dispatchable_binding_digests` in serving mode, job capacity
        in job mode — and, in BOTH, the one worker-level admission fence (§6). The owner
        dispatches against the epoch it last saw, and echoes it on the offer."""
        self.reports.append(report)
        self.admission_epoch = report.admission_epoch
        if report.worker_phase == pb.WorkerPhase.WORKER_PHASE_FAILED:
            # The worker said it cannot serve. That is an answer, and a one-shot run ends on
            # it instead of waiting for a dispatchability that will never come.
            self.marks.setdefault("ready", time.perf_counter())
            if self.terminal is None:
                self.ended = "worker_error"
            self.done.set()
            self.host.close()
            return
        refusal = self._refusal(report)
        if refusal:
            self.marks.setdefault("ready", time.perf_counter())
            self.ended = "worker_error"
            self.error = f"the worker refused to become ready: {refusal}"[:400]
            print(f"[run] {self.error}", flush=True)
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
            binding_digest = documents.raw(self._binding_digest())
            status = next((d for d in report.placements if d.placement_id == LOCAL_PLACEMENT), None)
            ready = (
                status is not None
                and open_window
                and status.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
                and binding_digest in status.dispatchable_binding_digests
            )
        if ready and not self.dispatched:
            self.marks["ready"] = time.perf_counter()
            self.dispatched = True
            self._dispatch()

    def _refusal(self, report: pb.ObservedWorkerState) -> str:
        """Whatever the worker has published that stops THIS run from ever dispatching.

        A one-shot local run holds one placement, one binding, one attempt and NO retry
        authority: nobody is going to send a second desired revision, so a typed fault raised
        before dispatch is, by construction, the answer. The rule used to be narrower than
        that in two ways, and both were hangs (cr-061): it read three fault kinds in serving
        mode, so every other typed refusal was invisible, and it did not look at all in JOB
        mode. A `run` on a host without cgroup delegation therefore printed a correct,
        complete diagnosis on the worker's own stderr and then sat until SIGKILL.

        Reading EVERY fault is the fix, and it is not a loose one: the worker publishes a
        fault when it has stopped, and a fault a one-shot owner cannot act on has no business
        on the wire in the first place.
        """
        if self.dispatched or self.terminal is not None or self.done.is_set():
            return ""
        faults = list(report.faults)
        for status in report.placements:
            faults.extend(status.faults)
        if not faults:
            return ""
        first = faults[0]
        return f"{first.reason}: {first.detail}" if first.detail else first.reason

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
        """When this run's grant stops being authority: NEVER, which `grants.expired()`
        reads as 0.

        The owner is this process and every destination is a local file, so an expiry on
        this grant protects nothing and can only lose a result. It did: a 900 s constant
        lost job-001's 99 GB artifact (`grant_expired: the output grant expired before the
        write`), and the derived form that replaced it — the caller's deadline plus a 300 s
        tail for the output transaction — was the same ceiling one step removed, on how many
        bytes an attempt may write after its handler returns (xs-007 row 13). The deadline
        is enforced where it belongs, on the attempt (`deadline_s` to the executor and the
        worker's own watchdog); the output transaction that follows runs until its bytes
        have landed. A hub-minted grant carries the presign's real expiry, and the store
        enforces that one.
        """
        return 0

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
                    build_id=self.package_revision_digest,
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
        # RecordOwner authors the canonical `PlacementSet/1` document ONCE and sends its
        # exact bytes, and the worker RECOMPUTES the digest before parsing a field. There is
        # no second place for the set to disagree with itself.
        if self.placement is None:
            raise LocalBindingError("a serving run has no selected PlacementSet/1 placement")
        placement_set = pb.PlacementSet(placements=[self.placement])
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

        The underscore is deliberately outside the public model-name grammar, so no user ref
        can ever squat a job's scratch. Promotion
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
        if self.request.kind == "job":
            for parameter, (manifest, length) in sorted(self.request.models.items()):
                rows.append(
                    pb.InputBinding(
                        input_id=f"{grants_module.MODEL_PREFIX}{parameter}",
                        digest=manifest,
                        length=length,
                        kind_mime=grants_module.MODEL_MIME,
                        order=0,
                    )
                )
        return rows

    def _output_bindings(self) -> list[pb.OutputBinding]:
        return [
            pb.OutputBinding(output_id=output_id, max_bytes=64 << 20)
            for output_id in sorted(self.request.outputs)
        ]

    def _dispatch(self) -> None:
        binding_digest = "" if self.request.kind == "job" else self._binding_digest()
        self.grant_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Invocation payload identity is the exact canonical typed-argument document. A
        # local RecordOwner must not mint a whitespace-dependent dialect beside Tensorhub.
        body = canonical.write(self.request.payload)
        spec = pb.InvocationSpec(
            package_revision_digest=self.package_revision_digest,
            environment_digest=(
                ""
                if self.placement is not None
                and self.placement.WhichOneof("package_mode") == "development"
                else documents.spell(self.placement.environment_digest)
                if self.placement is not None
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
                    build_id=self.package_revision_digest,
                    job_descriptor_id=self.request.job_descriptor_id,
                    publication_contract=self._publication(),
                )
            )
        else:
            spec.serving.entrypoint_binding_digest = binding_digest
            spec.serving.attempt_binding_id = binding_digest
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
                        # The admission epoch this owner OBSERVED when it dispatched (§6). A
                        # stale echo refuses deterministically.
                        admission_epoch=self.admission_epoch,
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
        if self.request.kind == "job":
            inputs += [
                pb.InputAccess(
                    input_id=f"{grants_module.MODEL_PREFIX}{parameter}",
                    url=f"model://{manifest}",
                )
                for parameter, (manifest, _length) in sorted(self.request.models.items())
            ]
        return pb.DeliveryGrant(
            invocation_spec_digest=invocation_digest,
            credential=pb.DeliveryAccessCredential(
                issuer="cozy-runtime-local",
                key_id="local",
                credential_epoch=1,
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
        """The local RecordOwner half of the checkpoint declaration exchange.

        The worker recorded the declaration for this process; this side observes the identity
        and answers. No checkpoint bytes become durable here.
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


def stage_development_placement(
    cozy_home: Path,
    package_descriptor: Mapping[str, Any],
    *,
    package: str,
    release: str,
    source_digest: str,
    artifact_cache: Path,
    artifact_store: Path,
    found: Any = None,
    binding_overrides: Mapping[str, str] | None = None,
) -> StagedDevelopmentPlacement:
    """Stage one exact PlacementSet/1 plus its launch-local descriptor/config bytes.

    The descriptor and config bytes the executor reads land in `artifact_cache` — the
    launcher's install-owned, content-addressed document store (cozy-creator's
    `installs/<id>/artifact-cache/`, a run's workspace), never under `cozy_home`: the home
    root holds what has a reason to be there, and a launch's derived documents are the
    launcher's to place and reclaim. `artifact_store` is named, never created here.
    """

    home = cozy_home.resolve()
    cache = artifact_cache.resolve()
    selected_store = artifact_store.resolve()
    documents_ = development_placement_documents(
        package_descriptor,
        package=package,
        release=release,
        source_digest=source_digest,
        found=found,
        cozy_home=home,
        artifact_store=selected_store,
        artifact_cache=cache,
        binding_overrides=binding_overrides,
    )
    bindings = {
        entrypoint.name: documents.spell(entrypoint.entrypoint_binding_digest)
        for entrypoint in documents_.placement.entrypoints
    }
    return StagedDevelopmentPlacement(
        bindings,
        documents_.placement,
        selected_store,
        cache,
        documents_.bindings,
    )


def development_placement_documents(
    package_descriptor: Mapping[str, Any],
    *,
    package: str,
    release: str,
    source_digest: str,
    found: Any = None,
    cozy_home: Path | None = None,
    artifact_store: Path | None = None,
    artifact_cache: Path | None = None,
    binding_overrides: Mapping[str, str] | None = None,
) -> DevelopmentPlacementDocuments:
    """Construct exact source and local-TensorFS selection facts through existing owners."""

    entries = package_descriptor.get("entrypoints")
    if not isinstance(entries, list):
        raise RuntimeError("a local serving closure requires an PackageDescriptor entrypoint set")
    visible: list[Mapping[str, Any]] = [entry for entry in entries if isinstance(entry, Mapping)]
    if not visible:
        raise RuntimeError("a local serving closure requires at least one visible entrypoint")
    if package.count("/") != 1 or not all(package.split("/")) or not release:
        raise RuntimeError("development package identity or release is invalid")
    try:
        source_raw = documents.raw(source_digest)
    except documents.DocumentError as exc:
        raise RuntimeError("development source digest is invalid") from exc

    descriptor_bytes = descriptor.canonical_bytes(package_descriptor)
    descriptor_ref = _exact_ref(descriptor_bytes)
    descriptor_path = ""
    if artifact_cache is not None:
        _, staged_descriptor = _stage_exact(artifact_cache, descriptor_bytes)
        descriptor_path = str(staged_descriptor)
    modeled_paths = sorted(
        str(model.get("path", ""))
        for entry in visible
        for model in entry.get("models", [])
        if isinstance(model, Mapping)
    )
    resolved_models: Mapping[str, _DevelopmentModel] = {}
    if modeled_paths:
        if found is None or cozy_home is None or artifact_store is None:
            raise RuntimeError(
                "development_model_resolution_absent: modeled development requires the "
                "discovered package, COZY_HOME, and local TensorFS store"
            )
        resolved_models = _resolve_development_models(found, artifact_store, binding_overrides)
        if sorted(resolved_models) != modeled_paths:
            raise RuntimeError("development_model_mismatch: resolved binding paths changed")

    model_ids = {path: f"model-{index:04d}" for index, path in enumerate(modeled_paths)}
    models = [
        pb.Model(
            id=model_ids[path],
            repo=f"{selected.repo}/{selected.name}",
            version=selected.version,
            lane=selected.lane,
            manifest=pb.Ref(
                digest=documents.raw(selected.manifest_digest),
                length=selected.manifest_length,
            ),
        )
        for path, selected in sorted(resolved_models.items())
    ]
    entrypoints: list[pb.Entrypoint] = []
    declared_bindings: dict[str, DeclaredBinding] = {}
    for entry in sorted(visible, key=lambda row: str(row.get("name", ""))):
        name = str(entry["name"])
        selected_paths = sorted(
            str(model.get("path", ""))
            for model in entry.get("models", [])
            if isinstance(model, Mapping)
        )
        slots: list[pb.Slot] = []
        members: list[ModelBinding] = []
        for selected_path in selected_paths:
            selected = resolved_models[selected_path]
            model_id = model_ids[selected.slot.path]
            assets = [
                pb.Asset(
                    name=asset_name,
                    model_id=model_id,
                    ref=pb.Ref(
                        digest=documents.raw(str(asset["digest"])),
                        length=int(asset["length"]),
                    ),
                )
                for asset_name, asset in sorted(selected.assets.items())
            ]
            slots.append(
                pb.Slot(
                    slot=selected.slot.param,
                    reference_model_id=model_id,
                    config=pb.Config(
                        document=pb.Ref(
                            digest=documents.raw(selected.config_digest),
                            length=len(selected.config_bytes),
                        ),
                        assets=assets,
                    ),
                    components=[
                        pb.Component(component=component, model_id=model_id)
                        for component in selected.components
                    ],
                )
            )
            config_path = ""
            if artifact_cache is not None:
                _, staged_config = _stage_exact(artifact_cache, selected.config_bytes)
                config_path = str(staged_config)
            members.append(
                ModelBinding(
                    model_class=selected.slot.class_key,
                    model_binding_path=selected.slot.path,
                    model_parameter_name=selected.slot.param,
                    config_digest=selected.config_digest,
                    store=str(selected.store),
                    config=config_path,
                    variant=selected.variant,
                    reference_snapshot=selected.manifest_digest,
                    components=selected.components,
                    snapshots={
                        component: selected.manifest_digest for component in selected.components
                    },
                    model_weight_bytes=selected.model_weight_bytes,
                    custody=selected.custody,
                    sequence_parallel_degrees=tuple(selected.slot.sequence_parallel),
                )
            )
        slots.sort(key=lambda row: row.slot)
        members.sort(key=lambda row: row.model_binding_path)
        entrypoint = pb.Entrypoint(name=name, slots=slots)
        entrypoint.entrypoint_binding_digest = hashlib.sha256(
            canonical.write({"name": name, "slots": [documents.body(slot) for slot in slots]})
        ).digest()
        entrypoints.append(entrypoint)

        digest = documents.spell(entrypoint.entrypoint_binding_digest)
        model_set = tuple(members)
        first = model_set[0] if model_set else None
        declared_bindings[digest] = DeclaredBinding(
            entrypoint_binding_digest=digest,
            entrypoint=name,
            model_class=first.model_class if first is not None else "",
            model_binding_path=first.model_binding_path if first is not None else "",
            model_parameter_name=first.model_parameter_name if first is not None else "",
            release=source_digest,
            models=model_set,
            application=str(package_descriptor.get("application", "")),
            descriptor_path=descriptor_path,
            development_project=str(found.project) if found is not None else "",
            store=first.store if first is not None else "",
            config=first.config if first is not None else "",
            variant=first.variant if first is not None else "",
            reference_snapshot=first.reference_snapshot if first is not None else "",
            components=first.components if first is not None else (),
            snapshots=first.snapshots if first is not None else {},
            package_revision_digest=source_digest,
            config_digest=model_binding_set_digest(model_set),
            development=True,
            model_weight_bytes=sum(model.model_weight_bytes for model in model_set),
            custody=first.custody if first is not None else "canonical",
        )
    entrypoint_bodies = [documents.body(entrypoint) for entrypoint in entrypoints]
    bindings_digest = hashlib.sha256(
        canonical.write(
            {
                "entrypoints": entrypoint_bodies,
                "models": [documents.body(model) for model in models],
            }
        )
    ).digest()
    placement = pb.Placement(
        placement_id=LOCAL_PLACEMENT,
        development=pb.DevelopmentPackage(
            package=package,
            release=release,
            source_digest=source_raw,
        ),
        package_descriptor=pb.Ref(
            digest=documents.raw(str(descriptor_ref["digest"])), length=len(descriptor_bytes)
        ),
        bindings_digest=bindings_digest,
        models=models,
        entrypoints=entrypoints,
    )
    return DevelopmentPlacementDocuments(descriptor_bytes, placement, declared_bindings)


def _resolve_development_models(
    found: Any,
    artifact_store: Path,
    binding_overrides: Mapping[str, str] | None = None,
) -> Mapping[str, _DevelopmentModel]:
    slots = [
        slot
        for surface in found.surfaces
        if surface.kind == "entrypoint" and not surface.hidden
        for slot in surface.model_bindings
    ]
    try:
        chosen = binding_table.resolve(slots, found.bindings, overrides=binding_overrides)
    except Exception as exc:
        raise RuntimeError(f"development_model_binding_invalid: {exc}") from exc
    try:
        tensorfs = fill.tensorfs_module()
    except fill.FillRefusal as exc:
        raise RuntimeError(f"development_tensorfs_unavailable: {exc}") from exc

    resolved: dict[str, _DevelopmentModel] = {}
    store_root = artifact_store.resolve()
    try:
        if not store_root.is_dir():
            raise FileNotFoundError(store_root)
        store = tensorfs.Store.open(str(store_root))
    except Exception as exc:
        raise RuntimeError(
            f"development_model_store_absent: {store_root}: {type(exc).__name__}: {exc}"
        ) from exc
    if not hasattr(store, "resolve_release"):
        raise RuntimeError(
            "development_tensorfs_resolver_unavailable: install TensorFS with Store.resolve_release"
        )
    for slot in sorted(slots, key=lambda item: item.path):
        binding = chosen[slot.path]
        repo, name, version = _exact_model_ref(binding.ref)
        lane = binding.lane or ""
        if not lane:
            raise RuntimeError(
                f"development_model_lane_missing: {binding.ref}: package.toml must select a lane"
            )
        try:
            release = store.resolve_release(repo, name, version, lane)
        except Exception as exc:
            code = str(getattr(exc, "code", "MODEL_RELEASE_ABSENT")).lower()
            raise RuntimeError(
                f"development_model_missing: {binding.ref} lane={lane}: {code}"
            ) from exc
        manifest_digest = str(release["manifest_digest"])
        manifest_length = int(release["manifest_length"])
        try:
            manifest = store.manifest(manifest_digest)
            manifest_bytes = bytes(manifest["manifest"])
            header = tensorfs.parse_header(bytes(manifest["header"]))
        except Exception as exc:
            raise RuntimeError(
                f"development_model_manifest_invalid: {binding.ref}: {type(exc).__name__}: {exc}"
            ) from exc
        measured_manifest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
        if measured_manifest != manifest_digest or (
            manifest_length and manifest_length != len(manifest_bytes)
        ):
            raise RuntimeError(
                f"development_model_manifest_mismatch: {binding.ref}: expected "
                f"{manifest_digest}/{manifest_length}, measured "
                f"{measured_manifest}/{len(manifest_bytes)}"
            )
        manifest_length = len(manifest_bytes)
        try:
            config_digest, config_bytes = model_config.construction_document(
                store, header.get("configs")
            )
        except model_config.ModelConfigRefusal as exc:
            raise RuntimeError(
                f"development_model_config_invalid: {binding.ref}: {type(exc).__name__}: {exc}"
            ) from exc
        components_value = header.get("components")
        if not isinstance(components_value, Mapping) or not components_value:
            raise RuntimeError(f"development_model_components_absent: {binding.ref}")
        components = tuple(sorted(str(component) for component in components_value))
        selected_assets = _development_assets(binding.ref, header.get("assets", {}))
        try:
            model_weight_bytes = sum(
                int(row["length"]) for row in store.walk_cozytensors(manifest_digest)
            )
        except Exception as exc:
            raise RuntimeError(
                f"development_model_inventory_invalid: {binding.ref}: {exc}"
            ) from exc
        resolved[slot.path] = _DevelopmentModel(
            slot=slot,
            repo=repo,
            name=name,
            version=version,
            lane=lane,
            manifest_digest=manifest_digest,
            manifest_length=manifest_length,
            config_digest=config_digest,
            config_bytes=config_bytes,
            components=components,
            assets=selected_assets,
            store=store_root,
            # Repository lane and accelerator hardware are independent facts. Development
            # carries no caller-authored hardware variant; the executor derives smNN from
            # the measured device after it initializes the accelerator.
            variant="",
            custody="canonical",
            model_weight_bytes=model_weight_bytes,
        )
    return resolved


def _development_assets(binding_ref: str, value: object) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"development_model_assets_invalid: {binding_ref}")
    if value:
        raise RuntimeError(
            "development_model_assets_unsupported: local modeled development does not "
            f"yet stream TensorFS logical assets for {binding_ref}"
        )
    return {}


def _exact_model_ref(value: str) -> tuple[str, str, str]:
    model, separator, version = value.partition("@")
    parts = model.split("/")
    if not separator or len(parts) != 2 or not all((*parts, version)):
        raise RuntimeError(
            f"development_model_ref_inexact: {value!r} must be exact org/name@release"
        )
    return parts[0], parts[1], version


def _exact_ref(raw: bytes) -> dict[str, Any]:
    return {"digest": "sha256:" + hashlib.sha256(raw).hexdigest(), "length": len(raw)}


def development_source_digest(root: Path) -> str:
    """Hash one local source snapshot without Git, caches, environments, or build output."""

    digest = hashlib.sha256()
    ignored = {".git", ".venv", "__pycache__", ".mypy_cache", ".ruff_cache", "dist", "outputs"}
    for current, directories, filenames in os.walk(root):
        directories[:] = sorted(name for name in directories if name not in ignored)
        for filename in sorted(filenames):
            path = Path(current, filename)
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix().encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with path.open("rb") as handle:
                while chunk := handle.read(1 << 20):
                    digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _stage_exact(root: Path, raw: bytes) -> tuple[dict[str, Any], Path]:
    ref = _exact_ref(raw)
    digest = str(ref["digest"])
    root.mkdir(parents=True, exist_ok=True)
    path = (root / digest.removeprefix("sha256:")).resolve()
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
    package_revision_digest: str,
    package_descriptor: Mapping[str, Any] | None = None,
    found: Any = None,
    development_artifact_store: Path | None = None,
    development_binding_overrides: Mapping[str, str] | None = None,
    job_records: list[dict[str, Any]] | None = None,
    options: WorkerOptions | None = None,
    delete_workspace: bool = True,
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
    bindings: dict[str, str] = {}
    staged: StagedDevelopmentPlacement | None = None
    if package_descriptor is not None:
        if request.kind != "entrypoint":
            raise RuntimeError("only a serving run accepts a package binding closure")
        source_digest = development_source_digest(Path.cwd().resolve())
        staged = stage_development_placement(
            config.cozy_home,
            package_descriptor,
            package="local/development-package",
            release="0.0.0",
            source_digest=source_digest,
            artifact_cache=workspace / "artifact-cache",
            artifact_store=development_artifact_store or workspace / "store",
            found=found,
            binding_overrides=development_binding_overrides,
        )
        bindings.update(staged.bindings)
    stage_job_plans(config.cozy_home, job_records or [])
    record_owner = LocalRecordOwner(
        request,
        bindings,
        workspace,
        package_revision_digest=(
            documents.spell(staged.placement.development.source_digest)
            if staged is not None
            else package_revision_digest
        ),
        placement=staged.placement if staged is not None else None,
        auto_ack=auto_ack,
    )
    options = options or WorkerOptions(root=workspace / "worker")
    if staged is not None:
        if options.artifact_store not in (None, staged.artifact_store):
            raise RuntimeError("local closure and worker name different artifact stores")
        options = dataclasses.replace(
            options,
            artifact_store=staged.artifact_store,
            artifact_cache=staged.artifact_cache,
            development_project=Path.cwd().resolve(),
            development_package=staged.placement.development.package,
            development_release=staged.placement.development.release,
            development_source_digest=documents.spell(staged.placement.development.source_digest),
            development_binding_overrides=dict(development_binding_overrides or {}),
        )
    if not options.instance_id:
        options = dataclasses.replace(options, instance_id=f"ins-{uuid.uuid4().hex[:16]}")
    worker = Worker(config, options, record_owner.host)
    worker.watches.tap = record_owner.tap_progress
    record_owner.watch(worker)
    if record_owner_hook is not None:
        record_owner_hook(record_owner, worker)
    started = time.perf_counter()
    exit_code = worker.run()
    return _collect(record_owner, worker, started, exit_code, delete_workspace)


def _collect(
    record_owner: LocalRecordOwner,
    worker: Worker,
    started: float,
    exit_code: int,
    delete_workspace: bool,
) -> SliceOutcome:
    outcome = SliceOutcome(exit_code=exit_code, ended=record_owner.ended, error=record_owner.error)
    outcome.attempt_records = _record_rows(worker.records)
    ledger = worker.primary_ledger()
    outcome.reconciliation = ledger.reconciliations[-1] if ledger.reconciliations else {}
    # The ledger as it stood AT ATTEMPT CLOSE, not after shutdown. Reading it here would
    # report the executor's RSS as unreadable and its device classes as stale, because by
    # now the process that owned them is gone — true, and useless. The reconciliation record
    # is the measurement; this is only where it is published.
    outcome.ledger = (
        {"classes": outcome.reconciliation["classes"]}
        if outcome.reconciliation
        else ledger.measurement()
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
            "placement_id": accepted.placement_id,
            "lane_id": accepted.lane_id,
        }
    if record_owner.terminal is not None:
        outcome.terminal_digest = _spell(record_owner.terminal.outcome_digest)
        outcome.terminal_canonical_bytes = bytes(record_owner.terminal.outcome_canonical_bytes)
        body = documents.read(record_owner.terminal.outcome_canonical_bytes, pb.AttemptOutcomeBody)
        outcome.terminal = body
        # The triage read, exactly as cl-006 will do it: take the opaque subject off the
        # terminal, and read the bundle BY KEY, verified against its in-memory receipt.
        # Nothing here holds a path to it.
        outcome.triage_subject = str((body.get("triage_bundle") or {}).get("subject_id", ""))
        if outcome.triage_subject:
            with contextlib.suppress(triage.TriageError, canonical.CanonicalError):
                outcome.triage = triage.BundleStore.read(
                    worker.root / "triage",
                    outcome.triage_subject,
                    worker.records,
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
            (row["body"] for row in outcome.attempt_records if row.get("kind") == "publication"),
            {},
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

    # A closed one-shot run has no remaining reader for its disposable workspace.
    closed = any(row.get("kind") == "closed" for row in outcome.attempt_records)
    if delete_workspace and closed:
        shutil.rmtree(worker.root, ignore_errors=True)
        outcome.workspace_deleted = True
    return outcome


def _spell(raw: bytes) -> str:
    """`sha256:<hex>`, or the empty string for a digest a mode genuinely does not have."""
    return documents.spell(raw) if len(raw) == 32 else ""


def _b64(text: str) -> bytes:
    import base64

    return base64.b64decode(text)


def _record_rows(source: WorkerRecords) -> list[dict[str, Any]]:
    """Project the current process's ordered worker facts."""
    return [
        {"seq": row.seq, "kind": row.kind, "at_unix_ms": row.at_unix_ms, "body": row.body}
        for row in source.records
    ]
