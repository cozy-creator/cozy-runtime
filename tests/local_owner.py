"""The in-process RecordOwner harness: a worker claimed and driven over queued frames.

It speaks the real protocol (the signed Claim, the snapshot fence, offers and terminal acks)
through an in-memory host, so worker behaviour is tested without a network.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import shutil
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import msgspec

import signed_claims
from cozy_runtime.author._capture import ActivationCapture
from cozy_runtime.author._media import SNIFF_BYTES, sniff
from cozy_runtime.internal import (
    canonical,
    liveness,
    proctree,
)
from cozy_runtime.internal.config import RuntimeConfig
from cozy_runtime.internal.development_launch import (
    LOCAL_PLACEMENT,
)
from cozy_runtime.internal.pathkey import opaque_key
from cozy_runtime.internal.worker import byte_inputs, workspace_byte_outputs
from cozy_runtime.internal.worker import grants as grants_module
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from cozy_runtime.internal.worker.workspace import Workspace


def claimable(config: RuntimeConfig, options: WorkerOptions) -> tuple[RuntimeConfig, WorkerOptions]:
    """A worker config and options that admit this owner's signed Claim."""
    return (
        dataclasses.replace(config, record_owner_public_key=signed_claims.PUBLIC_KEY),
        dataclasses.replace(options, **signed_claims.IDENTITY),
    )


class _Enveloped(Protocol):
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str


class LocalBindingError(Exception):
    """The request names an entrypoint this run staged no binding for."""


@dataclass(frozen=True, slots=True)
class LocalInput:
    """One owner-held file plus its typed position in the request."""

    source: str
    order: int


@dataclass(frozen=True, slots=True)
class LocalRequest:
    """One request, as a RecordOwner would put it on the wire."""

    entrypoint: str
    payload: Mapping[str, object]
    #: the result FIELD PATHS the caller grants a destination for (`image`, `detail.thumb`).
    #: cr-016 derives these from the package interface; here the driver names them, and a
    #: set that does not match what the handler produced refuses the tail whole (#248).
    outputs: tuple[str, ...] = ("image",)
    capture: ActivationCapture | None = None
    #: Optional developer execution-path override; empty leaves runtime auto-selection.
    attention_kernel: str = ""
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


class LocalCheckpoint(msgspec.Struct, frozen=True):
    """One durable checkpoint request this owner answered, and how."""

    operation_key: str
    logical_key: str
    content_digest: str
    seq: int
    outcome: str


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
        package_installation_id: str,
        placement: pb.Placement | None = None,
        auto_ack: bool = True,
        on_progress: Callable[[pb.AttemptProgress], None] | None = None,
    ) -> None:
        self.request = request
        self.bindings = bindings  # entrypoint -> entrypoint_binding_digest
        #: JOB mode: the durable checkpoint requests this owner answered, in order.
        self.checkpoints: list[LocalCheckpoint] = []
        self.workspace = workspace
        self.package_installation_id = package_installation_id
        self.placement = placement
        self.auto_ack = auto_ack
        self.native_workspace: Workspace | None = None
        self._native_exported: bytes = b""
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
        self.spec = pb.InvocationSpec()
        self.spec_digest = b""
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
        claim = signed_claims.claim("local-run", self.record_owner_epoch)
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
        worker's own boot cursors (`SilenceMonitor`: download and load positions) and
        the measured work of everything the boot runs — this process's working threads,
        its executor, a `uv` install, a native-wheel probe — which `_boot_meter` folds into
        one reading. The run ends `worker_wedged` when that reading has not changed for
        longer than `liveness.Pace` allows against the pauses this boot has itself shown,
        and never before the floor below. Readiness (`dispatched`) or any other end of the
        run stops the watch.

        The floor is DERIVED, not chosen. The deepest silence a healthy boot is permitted
        to show is the runtime's one noise floor (`liveness.noise_floor()`) — the same
        floor the transport's own stall ledger acts on (now TensorFS's, cr-090): a stream
        that has moved nothing for that long is re-asked for, and the re-ask is progress
        this watch can see. An outside observer must be more patient than that inner one,
        by the same factor the inner one applies to its own measurements — so the floor is
        `STILL_FACTOR * noise_floor()`, and a boot that never paused is called wedged at
        that and not before.
        """
        threading.Thread(
            target=self._watch, args=(worker,), daemon=True, name="ready-watch"
        ).start()

    def _watch(self, worker: Worker) -> None:
        pace = liveness.Pace()
        while not self.done.wait(liveness.SAMPLE_SECONDS):
            if self.dispatched:
                return
            floor = liveness.STILL_FACTOR * liveness.noise_floor()
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

    def _env[M: _Enveloped](self, message: M) -> M:
        """The owner's half of the three-field envelope (02 §0)."""
        message.record_owner_epoch = self.record_owner_epoch
        message.control_stream_epoch = self.control_stream_epoch
        message.worker_boot_id = self.boot_id
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
        self.admission_epoch = documents.parse(
            snapshot.snapshot_canonical_bytes, pb.WorkerSnapshotBody
        ).admission_epoch
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
        open_window = report.admission_state == pb.AdmissionState.ADMISSION_STATE_OPEN
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
        self._export_native(outcome)
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

    def _export_native(self, outcome: pb.AttemptOutcome) -> None:
        """Preserve local native results before ACK releases their producer roots."""
        if len(outcome.outcome_digest) != 32:
            raise LocalBindingError("native output outcome has no complete digest")
        if self._native_exported == outcome.outcome_digest:
            return
        if hashlib.sha256(outcome.outcome_canonical_bytes).digest() != outcome.outcome_digest:
            raise LocalBindingError("native output outcome digest changed")
        body = documents.parse(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
        for row in body.output_manifest.outputs:
            if not row.HasField("native_tree"):
                continue
            if self.native_workspace is None:
                raise LocalBindingError("native local output has no configured workspace")
            output_id = row.output_id
            hold = pb.NativeByteRetentionRequest(
                source=row.native_tree,
                retention_id=documents.spell(
                    hashlib.sha256(
                        canonical.write(
                            [
                                "local-output",
                                outcome.outcome_digest.hex(),
                                output_id,
                            ]
                        )
                    ).digest()
                ),
            )
            destination = self.destination(output_id)
            staging = self.output_dir / (
                ".native-" + hashlib.sha256(output_id.encode()).hexdigest()
            )
            tree = row.mime_type == workspace_byte_outputs.TREE_MIME
            workspace_byte_outputs.change_hold(
                self.native_workspace, "local-run", hold, release=False
            )
            try:
                byte_inputs.copy_retained(
                    self.native_workspace,
                    "local-run",
                    hold,
                    staging,
                    expected_blob=None if tree else (documents.spell(row.digest), row.length),
                )
                if tree:
                    os.rename(staging, destination)
                else:
                    shutil.copyfile(staging / "payload", destination)
            finally:
                byte_inputs.remove_checkout(staging)
                workspace_byte_outputs.change_hold(
                    self.native_workspace, "local-run", hold, release=True
                )
        self._native_exported = outcome.outcome_digest

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
                    installation_id=self.package_installation_id,
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
            installation_id=self.placement.installation_id
            if self.placement is not None
            else self.package_installation_id,
            payload_digest="sha256:" + hashlib.sha256(body).hexdigest(),
            inputs=self._input_bindings(body),
            outputs=self._output_bindings(),
            attention_kernel=self.request.attention_kernel,
            capture=pb.ActivationCapture(
                components=self.request.capture.components, steps=self.request.capture.steps
            )
            if self.request.capture is not None
            else None,
            # THE DEADLINE IS MINTED AT DISPATCH, from the caller's duration. It is what the
            # grant's own lifetime is derived from too, so the two cannot disagree.
            deadline_unix_ms=(
                int(time.time() * 1000) + self.request.timeout_ms if self.request.timeout_ms else 0
            ),
        )
        if self.request.kind == "job":
            spec.job.CopyFrom(
                pb.JobInvocationSpec(
                    installation_id=self.package_installation_id,
                    job_descriptor_id=self.request.job_descriptor_id,
                    publication_contract=self._publication(),
                )
            )
        else:
            spec.serving.entrypoint_binding_digest = binding_digest
            spec.serving.attempt_binding_id = binding_digest
            if self.placement is None:
                raise RuntimeError("serving dispatch has no exact prepared placement")
            spec.serving.bindings_digest = documents.spell(self.placement.bindings_digest)
        canonical_bytes, digest = documents.identity(spec)
        self.spec = spec
        self.spec_digest = digest
        self.spec_canonical_bytes = canonical_bytes
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
                if row.operation_key == request.operation_key
                and row.logical_key == request.logical_key
            ),
            None,
        )
        conflict = held is not None and held.content_digest != digest
        self.checkpoints.append(
            LocalCheckpoint(
                operation_key=request.operation_key,
                logical_key=request.logical_key,
                content_digest=digest,
                seq=request.seq,
                outcome="CONFLICT" if conflict else ("REPLAYED" if held else "RECORDED"),
            )
        )
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


def _input_facts(path: Path) -> tuple[str, int, str]:
    """Stream one local input once into its exact digest, length, and media claim.

    A local RecordOwner reads the bytes because it holds them; a hub reads its own record
    instead. Either way it is a CLAIM, and the worker sniffs what it actually received and
    refuses a disagreement — which is the point of declaring it at all.
    """
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
