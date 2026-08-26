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

from cozy_runtime.internal import canonical
from cozy_runtime.internal.config import RuntimeConfig
from cozy_runtime.internal.worker import grants as grants_module
from cozy_runtime.internal.worker import triage
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb

#: The one local deployment's owner-minted routing key (#446). Routing, never identity:
#: the InvocationSpec digest deliberately excludes it.
LOCAL_DEPLOYMENT = "dpl-local"

_M = TypeVar("_M")

#: What an OUTPUT TRANSACTION is allowed AFTER the handler returns: read the attempt spool,
#: write every granted destination, mint the receipt. The grant outlives the attempt by this
#: and by nothing more — it is a tail, never a lifetime.
#:
#: The lifetime itself is DERIVED FROM THE ATTEMPT (`LocalCoordinator.expiry`). It used to be
#: a 900 s constant derived from nothing, which is not a bound but a ceiling on what a job
#: may be: job-001's dual production ran 20:15.89, landed its 99 GB artifact intact, and died
#: `grant_expired: the output grant expired before the write`.
GRANT_TAIL_SECONDS = 300


class LocalBindingError(Exception):
    """The request names an entrypoint this run staged no binding plan for."""


@dataclass(frozen=True, slots=True)
class LocalRequest:
    """One request, as a coordinator would put it on the wire."""

    entrypoint: str
    payload: dict[str, Any]
    #: the result FIELD PATHS the caller grants a destination for (`image`, `detail.thumb`).
    #: cr-016 derives these from the endpoint descriptor; here the driver names them, and a
    #: set that does not match what the handler produced refuses the tail whole (#248).
    outputs: tuple[str, ...] = ("image",)
    request_id: str = "run-1"
    attempt: int = 1
    timeout_ms: int = 0
    """The caller's own bound on this attempt, as a DURATION. 0 means none.

    A coordinator mints the absolute deadline when it DISPATCHES, which is what `_dispatch`
    does — a duration written straight into `deadline_unix_ms` is a 1970 timestamp, and
    every attempt under it was silently given the 0.5 s floor.
    """
    #: grant the payload from THIS location instead of a file staged beside the run. A
    #: coordinator may grant any location it can authorize; the worker still verifies the
    #: bytes it reads against the grant's own digest and length.
    payload_url: str = ""
    #: `entrypoint` (serving) or `job` (run-to-completion). ONE ExecutionSpec mode is named
    #: either way; a request that named both would be two lifecycles under one identity.
    kind: str = "entrypoint"
    #: JOB mode only: the descriptor id the worker resolves its local job plan against, and
    #: the materialized input trees this run grants (`ref -> local directory`).
    job_descriptor_id: str = ""
    trees: Mapping[str, str] = field(default_factory=dict)
    #: cr-012: the INPUT ASSETS this run grants, as `input_id -> local file`. The key is
    #: the payload FIELD PATH (`photo`, `refs.0`) and the value is a location the
    #: coordinator can authorize; the worker still verifies every byte it reads against the
    #: grant's own digest and length, and sniffs the type it was told. Ordered, identified
    #: subjects — never a digest-sorted anonymous list (worker-protocol/01).
    inputs: Mapping[str, str] = field(default_factory=dict)
    org: str = "local"
    """The publishing org. A job's outputs are hub-rewritten into the SCRATCH repo
    `<org>/_job-<request-id>` — the underscore is deliberately outside the public repo
    grammar, so no user ref can squat a job's scratch, and promotion out of it is always an
    explicit later act under the caller's own credentials (jobs.md)."""


@dataclass(slots=True)
class SliceOutcome:
    """Everything one local run OBSERVED. No field is inferred; each has a producer."""

    status: str = ""
    terminal: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    result_schema_digest: str = ""
    accepted: dict[str, Any] = field(default_factory=dict)
    manifest: list[dict[str, Any]] = field(default_factory=list)
    #: cr-009's typed publication receipt — ONE canonical bundle snapshot per publication,
    #: journaled before the terminal named it. `manifest_id` on the terminal IS its digest.
    publication: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
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
    #: the scratch repo its outputs were rewritten into.
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    publication_scope: str = ""
    #: how the run ENDED, which is not always "it produced a terminal": `terminal`,
    #: `worker_error` (the worker reported it could not become ready), `ready_timeout`, or
    #: `coordinator_error` (this side failed). A one-shot run that cannot say this would
    #: have to pretend a hang was a result.
    ended: str = "unstarted"
    error: str = ""


class LocalCoordinator:
    """The in-memory OWNER — the record-plane half of th-024, in this process.

    It is a real owner in every way that the worker can observe: it claims first, speaks
    only `cozy.worker.v1` frames under the three-field envelope, reconciles the snapshot
    before dispatching, and learns readiness the way a hub does — from a durable Report's
    per-deployment status, not from a Python attribute.
    """

    def __init__(
        self,
        request: LocalRequest,
        bindings: dict[str, str],
        workspace: Path,
        *,
        release_id: str,
        auto_ack: bool = True,
        on_progress: Any = None,
        ready_timeout: float = 1800.0,
    ) -> None:
        self.request = request
        self.bindings = bindings  # entrypoint -> entrypoint_binding_plan_id
        #: JOB mode: the durable checkpoint requests this owner answered, in order.
        self.checkpoints: list[dict[str, Any]] = []
        self.workspace = workspace
        self.release_id = release_id
        self.auto_ack = auto_ack
        self.ready_timeout = ready_timeout
        self.ended = "unstarted"
        self.error = ""
        self.host = InMemoryControlHost()
        self.host.on_frame = self._on_frame
        self._on_progress = on_progress
        self.progress_frames: list[pb.AttemptProgress] = []
        #: the accepted envelope this owner speaks under, learned from ClaimAck
        self.owner_epoch = 1
        self.generation = 0
        self.boot_id = ""
        self.snapshot_entries: list[pb.ActiveAttempt] = []
        self.dispatched = False
        self.accepted: pb.AttemptAccepted | None = None
        self.terminal: pb.AttemptTerminal | None = None
        self.reports: list[pb.Report] = []
        self.spec_digest = b""
        self.done = threading.Event()
        self.marks: dict[str, float] = {"start": time.perf_counter()}
        self.grant_dir = workspace / "grants"
        self.output_dir = workspace / "outputs"
        # THE CLAIM IS QUEUED BEFORE THE WORKER SERVES: the worker's stream conversation
        # begins by reading it, exactly as a dialing owner's first frame arrives first.
        claim = pb.Claim(
            owner_epoch=self.owner_epoch,
            controller_id="local-run",
            wire_minor=WIRE_MINOR,
        )
        self.host.deliver(pb.OwnerFrame(claim=claim))
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
        inner.owner_epoch = self.owner_epoch
        inner.control_generation = self.generation
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
            self.ended = "coordinator_error"
            self.error = f"{type(exc).__name__}: {exc}"[:400]
            print(f"[run] the owner failed: {self.error}", flush=True)
            self.done.set()
            self.host.close()

    def _dispatch_frame(self, frame: pb.WorkerFrame) -> None:
        kind = frame.WhichOneof("msg")
        if kind == "claim_ack":
            self._on_claim_ack(frame.claim_ack)
        elif kind == "snapshot_begin":
            self.snapshot_entries = []
        elif kind == "snapshot_entry":
            self.snapshot_entries.append(frame.snapshot_entry.attempt)
        elif kind == "snapshot_end":
            self._on_snapshot_end(frame.snapshot_end)
        elif kind == "boot_failure":
            self.ended = "worker_error"
            self.error = f"boot failure: {frame.boot_failure.detail}"[:400]
            self.done.set()
            self.host.close()
        elif kind == "report":
            self._on_report(frame.report)
        elif kind == "attempt_accepted":
            self.accepted = frame.attempt_accepted
            self.marks["accepted"] = time.perf_counter()
        elif kind == "attempt_terminal":
            self._on_terminal(frame.attempt_terminal)
        elif kind == "checkpoint_request":
            self._on_checkpoint(frame.checkpoint_request)

    def _on_claim_ack(self, ack: pb.ClaimAck) -> None:
        if not ack.accepted:
            self.ended = "worker_error"
            self.error = f"claim refused: {pb.ClaimRejection.Name(ack.rejection)}"
            self.done.set()
            self.host.close()
            return
        self.generation = ack.control_generation
        self.boot_id = ack.worker_boot_id
        self.marks["registered"] = time.perf_counter()

    def _on_snapshot_end(self, end: pb.SnapshotEnd) -> None:
        """Reconcile, ack the EXACT snapshot, and only then send the directive (02 §6).
        A one-shot run on a fresh root expects an empty snapshot; entries are recorded so
        a driver can assert on them."""
        if len(self.snapshot_entries) != end.entry_count:
            raise RuntimeError(
                f"snapshot gap: {len(self.snapshot_entries)} entries, end says {end.entry_count}"
            )
        self._send(
            pb.OwnerFrame(snapshot_ack=self._env(pb.SnapshotAck(snapshot_id=end.snapshot_id)))
        )
        self._send(pb.OwnerFrame(directive=self._directive()))

    def _plan_id(self) -> str:
        """The binding plan this request runs against, or the typed refusal that it has none.

        A request naming an entrypoint no binding plan covers is an ORDINARY coordinator
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

    def _on_report(self, report: pb.Report) -> None:
        """READY is learned from the worker's own durable Report — per-deployment status in
        serving mode (#446), job capacity in job mode."""
        self.reports.append(report)
        if report.intake_state == pb.IntakeState.INTAKE_STATE_ERROR:
            # The worker said it cannot become ready. That is an answer, and a one-shot run
            # ends on it instead of waiting for a readiness that will never come.
            self.marks.setdefault("ready", time.perf_counter())
            if self.terminal is None:
                self.ended = "worker_error"
            self.done.set()
            self.host.close()
            return
        if self.request.kind == "job":
            ready = (
                report.intake_state == pb.IntakeState.INTAKE_STATE_READY
                and report.job_capacity.jobs_available > 0
            )
        else:
            plan_id = self._plan_id()
            status = next(
                (d for d in report.deployments if d.deployment_id == LOCAL_DEPLOYMENT), None
            )
            ready = (
                status is not None
                and status.intake_state == pb.IntakeState.INTAKE_STATE_READY
                and status.attempt_credits > 0
                and plan_id in status.ready_entrypoint_binding_plan_ids
            )
        if ready and not self.dispatched:
            self.marks["ready"] = time.perf_counter()
            self.dispatched = True
            self._dispatch()

    def _on_terminal(self, terminal: pb.AttemptTerminal) -> None:
        self.terminal = terminal
        self.ended = "terminal"
        self.marks["terminal"] = time.perf_counter()
        if not self.auto_ack:
            return
        self._send(
            pb.OwnerFrame(
                terminal_ack=self._env(
                    pb.TerminalAck(
                        request_id=terminal.request_id,
                        attempt=terminal.attempt,
                        invocation_digest=terminal.invocation_digest,
                        terminal_id=terminal.terminal_id,
                        terminal_digest=terminal.terminal_digest,
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

    def _directive(self) -> pb.Directive:
        directive = self._env(
            pb.Directive(
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
        # THE ONE-DIGESTER RULE (#454): the owner authors BOTH the structured set and its
        # canonical document digest; the worker echoes the digest as the set's NAME.
        deployment_set = pb.DeploymentSet(
            deployments=[
                pb.Deployment(
                    deployment_id=LOCAL_DEPLOYMENT,
                    endpoint_release_id=self.release_id,
                    entrypoint_binding_plan_ids=sorted(set(self.bindings.values())),
                )
            ]
        )
        _, set_digest = documents.identity(deployment_set)
        directive.deployment_set.CopyFrom(
            pb.DeploymentSetDirective(set_digest=set_digest, set=deployment_set)
        )
        return directive

    def scratch_repo(self) -> str:
        """`<org>/_job-<request-id>` — the SCRATCH repo a job's publishes are rewritten into.

        The underscore is deliberately outside the public repo grammar (`artifacts.REF`
        admits `[a-zA-Z0-9]` first), so no user ref can ever squat a job's scratch. Promotion
        OUT of scratch is an explicit later act under the caller's own credentials; a verdict
        stamps metadata and never gates persistence — a failed run's writes still land.
        """
        return f"{self.request.org}/_job-{self.request.request_id}"

    def _publication(self) -> pb.PublicationContract:
        """What this run authorizes the job to write, by output id. Minted from the job's own
        DECLARATION (`publishes=`/`emits_media=`), never from the fact that it is a job."""
        return pb.PublicationContract(
            grant_id=self.scratch_repo(),
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
        order = 1
        for ref in sorted(self.request.trees):
            rows.append(
                pb.InputBinding(
                    input_id=f"{grants_module.TREE_PREFIX}{ref}",
                    kind_mime="inode/directory",
                    order=order,
                )
            )
            order += 1
        for input_id, source in sorted(self.request.inputs.items()):
            data = Path(source).read_bytes()
            rows.append(
                pb.InputBinding(
                    input_id=input_id,
                    digest="sha256:" + hashlib.sha256(data).hexdigest(),
                    length=len(data),
                    kind_mime=_declared_media_type(Path(source)),
                    order=order,
                )
            )
            order += 1
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
        body = json.dumps(self.request.payload).encode()
        spec = pb.InvocationSpec(
            endpoint_release_id=self.release_id,
            payload_digest="sha256:" + hashlib.sha256(body).hexdigest(),
            inputs=self._input_bindings(body),
            outputs=self._output_bindings(),
            # THE DEADLINE IS MINTED AT DISPATCH, from the caller's duration. It is what the
            # grant's own lifetime is derived from too, so the two cannot disagree.
            deadline_unix_ms=(
                int(time.time() * 1000) + self.request.timeout_ms if self.request.timeout_ms else 0
            ),
        )
        # `image_digest` and `config_digest` stay EMPTY on the local door and that is a
        # statement: a bare-venv run has no container image, and minting a plausible digest
        # for one would be the fabrication class this runtime exists to prevent.
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
        self.marks["dispatched"] = time.perf_counter()
        self._send(
            pb.OwnerFrame(
                start_attempt=self._env(
                    pb.StartAttempt(
                        request_id=self.request.request_id,
                        attempt=self.request.attempt,
                        invocation_digest=digest,
                        invocation_canonical=canonical_bytes,
                        grant=self._grant(body, digest),
                        deployment_id="" if self.request.kind == "job" else LOCAL_DEPLOYMENT,
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
            path = self.grant_dir / f"{self.request.request_id}.json"
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
            pb.InputAccess(input_id=input_id, url=f"file://{Path(source).resolve()}")
            for input_id, source in sorted(self.request.inputs.items())
        ]
        return pb.DeliveryGrant(
            invocation_digest=invocation_digest,
            credential=pb.Credential(
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
        return f"{self.request.request_id}-{output_id.replace('.', '-')}.bin"

    def _on_checkpoint(self, request: pb.JobCheckpointRequest) -> None:
        """The coordinator half of the DURABLE checkpoint exchange (cr-009).

        This is the seam cl-004 and the hub implement for real: the worker already made the
        save durable in its own journal, and this side records the identity and answers with
        its receipt. The coordinator's answer is a SECOND observation, not a second
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
            pb.OwnerFrame(
                checkpoint_receipt=self._env(
                    pb.JobCheckpointReceipt(
                        request_id=request.request_id,
                        attempt=request.attempt,
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
            pb.OwnerFrame(
                cancel_attempt=self._env(
                    pb.CancelAttempt(
                        request_id=self.request.request_id,
                        attempt=self.request.attempt,
                        invocation_digest=self.spec_digest,
                        reason=reason,
                    )
                )
            )
        )

    def close(self) -> None:
        self.host.close()

    def _send(self, frame: pb.OwnerFrame) -> None:
        self.host.deliver(frame)


# --------------------------------------------------------------------------- the runner


def _declared_media_type(path: Path) -> str:
    """What the COORDINATOR declares one input to be (cr-012).

    A local coordinator reads the bytes because it holds them; a hub reads its own record
    instead. Either way it is a CLAIM, and the worker sniffs what it actually received and
    refuses a disagreement — which is the point of declaring it at all.
    """
    from cozy_runtime.author._media import SNIFF_BYTES, sniff

    with open(path, "rb") as handle:
        return sniff(handle.read(SNIFF_BYTES)) or "application/octet-stream"


def stage_binding_plans(cozy_home: Path, records: list[dict[str, Any]]) -> dict[str, str]:
    """Write the local binding-plan records the worker resolves plan ids against.

    THE SEAM: on the network path these records arrive with th-004's EntrypointBindingPlan
    document and tfs-003's artifact fetch. Locally the resolution is the one-shot run's own,
    and it is a plain file under `<cozy_home>/binding-plans/` — the worker cannot tell the
    difference, because a coordinator never ships a path either way.
    """
    directory = cozy_home / "binding-plans"
    directory.mkdir(parents=True, exist_ok=True)
    plans: dict[str, str] = {}
    for record in records:
        plan_id = str(record["entrypoint_binding_plan_id"])
        (directory / f"{plan_id.split(':')[-1]}.json").write_text(json.dumps(record, indent=2))
        plans[str(record["entrypoint"])] = plan_id
    return plans


def stage_job_plans(cozy_home: Path, records: list[dict[str, Any]]) -> dict[str, str]:
    """Write the local JOB plan records the worker resolves job descriptor ids against.

    THE SEAM, one lane over from `stage_binding_plans`: on the network path th-004 ships the
    build and the JobDescriptor and tfs-003 the materialized input trees. Locally it is a
    plain file under `<cozy_home>/job-plans/`, and the worker cannot tell the difference
    because a coordinator never ships a path either way.
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
    binding_records: list[dict[str, Any]],
    *,
    workspace: Path,
    release_id: str,
    job_records: list[dict[str, Any]] | None = None,
    options: WorkerOptions | None = None,
    delete_journal: bool = True,
    auto_ack: bool = True,
    coordinator_hook: Any = None,
) -> SliceOutcome:
    """One request in, one typed result out, through the whole real path.

    `coordinator_hook` is handed the live `LocalCoordinator` before the worker starts — that
    is how a driver injects a cancel at a chosen moment WITHOUT a plant switch inside the
    runtime (decisions #249: fault injection belongs to the harness).
    """
    workspace.mkdir(parents=True, exist_ok=True)
    plans = stage_binding_plans(config.cozy_home, binding_records)
    stage_job_plans(config.cozy_home, job_records or [])
    coordinator = LocalCoordinator(
        request, plans, workspace, release_id=release_id, auto_ack=auto_ack
    )
    options = options or WorkerOptions(root=workspace / "worker")
    if not options.instance_id:
        options = dataclasses.replace(options, instance_id=f"ins-{uuid.uuid4().hex[:16]}")
    worker = Worker(config, options, coordinator.host)
    worker.watches.tap = coordinator.tap_progress
    if coordinator_hook is not None:
        coordinator_hook(coordinator, worker)
    started = time.perf_counter()
    exit_code = worker.run()
    return _collect(coordinator, worker, started, exit_code, delete_journal)


def _collect(
    coordinator: LocalCoordinator,
    worker: Worker,
    started: float,
    exit_code: int,
    delete_journal: bool,
) -> SliceOutcome:
    outcome = SliceOutcome(exit_code=exit_code, ended=coordinator.ended, error=coordinator.error)
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
    outcome.progress = len(coordinator.progress_frames)
    outcome.executions = worker.engine.executed
    outcome.checkpoints = list(coordinator.checkpoints)
    outcome.publication_scope = (
        coordinator.scratch_repo() if coordinator.request.kind == "job" else ""
    )

    if coordinator.accepted is not None:
        accepted = coordinator.accepted
        outcome.accepted = {
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
            "decision_explanation": accepted.plan.decision_explanation,
        }
    if coordinator.terminal is not None:
        body = documents.read(coordinator.terminal.terminal_canonical, pb.TerminalBody)
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
        outcome.status = pb.TerminalStatus.Name(
            body.get("status", pb.TerminalStatus.TERMINAL_STATUS_UNSPECIFIED)
        )
        envelope = body.get("result") or {}
        inline = envelope.get("inline_result")
        if isinstance(inline, str):
            # The typed result rode the terminal INLINE. Read it with the canonical reader
            # that wrote it, never a looser JSON parser: the whole point of the envelope is
            # that the bytes a coordinator holds and the value a caller sees are one thing.
            decoded = canonical.parse(_b64(inline))
            outcome.result = decoded if isinstance(decoded, dict) else {"value": decoded}
        outcome.result_schema_digest = str(envelope.get("result_schema_digest", ""))
        manifest = body.get("output_manifest") or {}
        outcome.manifest = list(manifest.get("outputs", ()))
        outcome.publication = next(
            (row["body"] for row in outcome.journal if row.get("kind") == "publication"), {}
        )
        for row in outcome.manifest:
            path = coordinator.destination(str(row.get("output_id", "")))
            if path.is_file():
                outcome.outputs[str(row["output_id"])] = str(path)

    marks = coordinator.marks
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
