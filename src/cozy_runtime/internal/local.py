"""The local slice: one request, through the REAL protocol path, with no network.

This is §8's one-kernel rule made executable. A one-shot local run does not get its own
execution implementation — it gets an EPHEMERAL LOCAL SUPERVISOR whose coordinator is a pair
of in-process queues instead of a gRPC channel. Everything below that substitution is the
production path, unchanged and un-branched:

    the same `ExecutionSpec` canonical bytes and the same digest recomputation
    the same atomic acceptance boundary (payload fetch -> features -> plan -> prepare ->
      reserve -> JOURNAL -> AttemptAccepted)
    the same supervisor/executor process boundary, exec-first trampoline and env seal
    the same durable journal, the same terminal-before-send rule, the same TerminalAck
    the same output grant, freeze point, tail and manifest
    the same ledger, reconciled at attempt close

The only two things that are LOCAL are stated out loud here rather than hidden:

1. **The coordinator is this module.** It mints the revision, resolves the binding plan id,
   builds the spec document, grants the payload input and one destination per output field
   path, and acks the terminal. That is exactly cl-001's job on the network path.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cozy_runtime.internal import canonical
from cozy_runtime.internal.config import RuntimeConfig
from cozy_runtime.internal.worker import triage
from cozy_runtime.internal.worker.control import InMemoryLink, Link
from cozy_runtime.internal.worker.session import Supervisor, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb

#: How long a granted local file URL stays valid. A local grant still EXPIRES: the expiry
#: path is the same code the network path walks, and a run that never exercises it would
#: leave the worker's most security-relevant branch unproven on the local door.
GRANT_SECONDS = 900


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
    deadline_ms: int = 0
    #: grant the payload from THIS location instead of a file staged beside the run. A
    #: coordinator may grant any location it can authorize; the worker still verifies the
    #: bytes it reads against the grant's own digest and length.
    payload_url: str = ""


@dataclass(slots=True)
class SliceOutcome:
    """Everything one local run OBSERVED. No field is inferred; each has a producer."""

    status: str = ""
    terminal: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    result_schema_digest: str = ""
    accepted: dict[str, Any] = field(default_factory=dict)
    manifest: list[dict[str, Any]] = field(default_factory=list)
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
    #: how the run ENDED, which is not always "it produced a terminal": `terminal`,
    #: `worker_error` (the worker reported it could not become ready), `ready_timeout`, or
    #: `coordinator_error` (this side failed). A one-shot run that cannot say this would
    #: have to pretend a hang was a result.
    ended: str = "unstarted"
    error: str = ""


class LocalCoordinator:
    """The in-memory control adapter — the coordinator half of th-024, in this process.

    It is a real coordinator in every way that the worker can observe: it speaks only
    `cozy.worker.v1` messages, it never reaches inside the supervisor, and it learns
    readiness the way a hub does — from a durable Report, not from a Python attribute.
    """

    reconnects = False

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
        self.workspace = workspace
        self.release_id = release_id
        self.auto_ack = auto_ack
        self.ready_timeout = ready_timeout
        self.ended = "unstarted"
        self.error = ""
        self._link = InMemoryLink()
        self._link.on_message = self._on_message
        self._link.on_progress = on_progress
        self.session_id = ""
        self.dispatched = False
        self.accepted: pb.AttemptAccepted | None = None
        self.terminal: pb.AttemptTerminal | None = None
        self.reports: list[pb.Report] = []
        self.spec_digest = b""
        self.done = threading.Event()
        self.marks: dict[str, float] = {"start": time.perf_counter()}
        self.grant_dir = workspace / "grants"
        self.output_dir = workspace / "outputs"

    @property
    def progress_frames(self) -> list[pb.AttemptProgress]:
        """What arrived on the LOSSY lane. Bounded, and never load-bearing for the result."""
        return self._link.progress_frames

    def destination(self, output_id: str) -> Path:
        """The exact granted destination for one output field path."""
        return self.output_dir / self._filename(output_id)

    # ------------------------------------------------------------------ Coordinator

    def target(self) -> str:
        return "local:in-memory"

    def link(self) -> Link:
        # A one-shot run gets a readiness DEADLINE. With the worker reporting an intake
        # error on a failed preparation this rarely fires, but "rarely" is not "never", and
        # a local run that hangs forever waiting for a worker that will not arrive is worse
        # than one that says so: a person is sitting in front of it.
        threading.Thread(target=self._deadline, daemon=True, name="ready-deadline").start()
        return self._link

    def _deadline(self) -> None:
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
            self._link.close()

    # ------------------------------------------------------------------ the exchange

    def _on_message(self, message: pb.WorkerMessage) -> None:
        """One coordinator-side message. A FAILURE HERE ENDS THE RUN, never hangs it.

        Found live: a `KeyError` in the report handler killed the message-pump thread, and
        the worker then talked to nobody for the full 1,800-second readiness deadline. A
        one-shot run has a person in front of it — a coordinator that cannot do its job says
        so and stops, exactly as it does when the worker reports an intake error.
        """
        try:
            self._dispatch_message(message)
        except LocalBindingError as exc:
            # A CONDITION, not a coordinator fault: the run ends the same way, but it says
            # which entrypoint had no plan instead of naming a KeyError.
            self.ended = "unbound_entrypoint"
            self.error = str(exc)[:400]
            print(f"[run] {self.error}", flush=True)
            self.done.set()
            self._link.close()
        except Exception as exc:
            self.ended = "coordinator_error"
            self.error = f"{type(exc).__name__}: {exc}"[:400]
            print(f"[run] the coordinator failed: {self.error}", flush=True)
            self.done.set()
            self._link.close()

    def _dispatch_message(self, message: pb.WorkerMessage) -> None:
        kind = message.WhichOneof("msg")
        if kind == "register":
            self._on_register(message.register)
        elif kind == "report":
            self._on_report(message.report)
        elif kind == "attempt_accepted":
            self.accepted = message.attempt_accepted
            self.marks["accepted"] = time.perf_counter()
        elif kind == "attempt_terminal":
            self._on_terminal(message.attempt_terminal)

    def _on_register(self, register: pb.Register) -> None:
        self.session_id = register.session_id
        self.marks["registered"] = time.perf_counter()
        self._send(
            pb.CoordinatorMessage(
                register_ack=pb.RegisterAck(
                    session_id=register.session_id,
                    executor_incarnation=register.executor_incarnation,
                    wire_minor=WIRE_MINOR,
                    accepted=True,
                )
            )
        )
        self._send(pb.CoordinatorMessage(directive=self._directive()))

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
        """READY is learned from the worker's own durable Report, like a hub learns it."""
        self.reports.append(report)
        plan_id = self._plan_id()
        ready = (
            report.intake_state == pb.IntakeState.INTAKE_STATE_READY
            and plan_id in report.serving_capacity.ready_entrypoint_binding_plan_ids
        )
        if report.intake_state == pb.IntakeState.INTAKE_STATE_ERROR:
            # The worker said it cannot become ready. That is an answer, and a one-shot run
            # ends on it instead of waiting for a readiness that will never come.
            self.marks.setdefault("ready", time.perf_counter())
            if self.terminal is None:
                self.ended = "worker_error"
            self.done.set()
            self._link.close()
            return
        if ready and not self.dispatched:
            self.marks["ready"] = time.perf_counter()
            self.dispatched = True
            self._dispatch(report.executor_incarnation)

    def _on_terminal(self, terminal: pb.AttemptTerminal) -> None:
        self.terminal = terminal
        self.ended = "terminal"
        self.marks["terminal"] = time.perf_counter()
        if not self.auto_ack:
            return
        self._send(
            pb.CoordinatorMessage(
                terminal_ack=pb.TerminalAck(
                    session_id=terminal.session_id,
                    executor_incarnation=terminal.executor_incarnation,
                    request_id=terminal.request_id,
                    attempt=terminal.attempt,
                    exec_spec_digest=terminal.exec_spec_digest,
                    terminal_id=terminal.terminal_id,
                    terminal_digest=terminal.terminal_digest,
                )
            )
        )
        self.done.set()
        self._link.close()

    # ------------------------------------------------------------------ documents

    def _directive(self) -> pb.Directive:
        directive = pb.Directive(
            session_id=self.session_id,
            revision=1,
            posture=pb.Posture.POSTURE_ACCEPTING,
            wire_minor=WIRE_MINOR,
        )
        directive.credential.CopyFrom(
            pb.Credential(
                issuer="cozy-runtime-local",
                key_id="local",
                epoch=1,
                expires_at_unix=int(time.time()) + GRANT_SECONDS,
                token=b"local",
            )
        )
        directive.serving.entrypoint_binding_plan_ids.extend(sorted(set(self.bindings.values())))
        return directive

    def _dispatch(self, incarnation: int) -> None:
        plan_id = self._plan_id()
        spec = pb.ExecutionSpec(
            endpoint_release_id=self.release_id,
            deadline_unix_ms=self.request.deadline_ms,
        )
        # `image_digest` and `config_digest` stay EMPTY on the local door and that is a
        # statement: a bare-venv run has no container image, and minting a plausible digest
        # for one would be the fabrication class this runtime exists to prevent.
        spec.serving.entrypoint_binding_plan_id = plan_id
        spec.serving.attempt_binding_id = plan_id
        canonical_bytes, digest = documents.identity(spec)
        self.spec_digest = digest
        self.marks["dispatched"] = time.perf_counter()
        self._send(
            pb.CoordinatorMessage(
                start_attempt=pb.StartAttempt(
                    session_id=self.session_id,
                    executor_incarnation=incarnation,
                    request_id=self.request.request_id,
                    attempt=self.request.attempt,
                    exec_spec_digest=digest,
                    exec_spec_canonical=canonical_bytes,
                    grant=self._grant(),
                )
            )
        )

    def _grant(self) -> pb.DeliveryGrant:
        self.grant_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        body = json.dumps(self.request.payload).encode()
        if self.request.payload_url:
            url = self.request.payload_url
        else:
            path = self.grant_dir / f"{self.request.request_id}.json"
            path.write_bytes(body)
            url = f"file://{path}"
        expires = int(time.time()) + GRANT_SECONDS
        return pb.DeliveryGrant(
            credential=pb.Credential(
                issuer="cozy-runtime-local",
                key_id="local",
                epoch=1,
                expires_at_unix=expires,
                token=b"local",
            ),
            file_base_url=f"file://{self.grant_dir}",
            expires_at_unix=expires,
            inputs=[
                pb.InputLocation(
                    digest=hashlib.sha256(body).digest(),
                    url=url,
                    length=len(body),
                    input_id="payload",
                    kind_mime="application/json",
                )
            ],
            outputs=[
                pb.OutputDestination(
                    output_id=output_id,
                    url=f"file://{self.output_dir / self._filename(output_id)}",
                    max_bytes=64 << 20,
                )
                for output_id in self.request.outputs
            ],
        )

    def _filename(self, output_id: str) -> str:
        return f"{self.request.request_id}-{output_id.replace('.', '-')}.bin"

    # ------------------------------------------------------------------ driving

    def cancel(self, reason: pb.CancelReason = pb.CancelReason.CANCEL_REASON_CLIENT) -> None:
        """Cancel the live attempt — the same message a hub sends, on the same lane."""
        self._send(
            pb.CoordinatorMessage(
                cancel_attempt=pb.CancelAttempt(
                    session_id=self.session_id,
                    executor_incarnation=(
                        self.accepted.executor_incarnation if self.accepted else 1
                    ),
                    request_id=self.request.request_id,
                    attempt=self.request.attempt,
                    exec_spec_digest=self.spec_digest,
                    reason=reason,
                )
            )
        )

    def close(self) -> None:
        self._link.close()

    def _send(self, message: pb.CoordinatorMessage) -> None:
        self._link.deliver(message)


# --------------------------------------------------------------------------- the runner


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
        plan_id = str(record["plan_id"])
        (directory / f"{plan_id.split(':')[-1]}.json").write_text(json.dumps(record, indent=2))
        plans[str(record["entrypoint"])] = plan_id
    return plans


def run_slice(
    config: RuntimeConfig,
    request: LocalRequest,
    binding_records: list[dict[str, Any]],
    *,
    workspace: Path,
    release_id: str,
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
    coordinator = LocalCoordinator(
        request, plans, workspace, release_id=release_id, auto_ack=auto_ack
    )
    options = options or WorkerOptions(root=workspace / "worker")
    if not options.instance_id:
        options = dataclasses.replace(options, instance_id=f"ins-{uuid.uuid4().hex[:16]}")
    supervisor = Supervisor(config, options, coordinator)
    if coordinator_hook is not None:
        coordinator_hook(coordinator, supervisor)
    started = time.perf_counter()
    exit_code = supervisor.run()
    return _collect(coordinator, supervisor, started, exit_code, delete_journal)


def _collect(
    coordinator: LocalCoordinator,
    supervisor: Supervisor,
    started: float,
    exit_code: int,
    delete_journal: bool,
) -> SliceOutcome:
    outcome = SliceOutcome(exit_code=exit_code, ended=coordinator.ended, error=coordinator.error)
    outcome.journal = _journal_rows(supervisor.root / "journal.log")
    outcome.reconciliation = (
        supervisor.ledger.reconciliations[-1] if supervisor.ledger.reconciliations else {}
    )
    # The ledger as it stood AT ATTEMPT CLOSE, not after shutdown. Reading it here would
    # report the executor's RSS as unreadable and its device classes as stale, because by
    # now the process that owned them is gone — true, and useless. The reconciliation record
    # is the measurement; this is only where it is published.
    outcome.ledger = (
        {"format": "cozy.runtime.Ledger/0", "classes": outcome.reconciliation["classes"]}
        if outcome.reconciliation
        else supervisor.ledger.document()
    )
    outcome.warm = list(supervisor.warm)
    outcome.faults = [
        {"reason": fault.reason, "detail": fault.detail} for fault in supervisor.engine.faults
    ]
    outcome.progress = len(coordinator.progress_frames)
    outcome.executions = supervisor.engine.executed

    if coordinator.accepted is not None:
        accepted = coordinator.accepted
        outcome.accepted = {
            "plan_digest": documents.spell(accepted.plan_digest),
            "model_construction_digest": documents.spell(accepted.model_construction_digest),
            "delivery": accepted.plan.delivery,
            "materialization": accepted.plan.materialization,
            "placement": accepted.plan.placement,
            "compute_dtype": accepted.plan.compute_dtype,
            "reserved_vram_bytes": accepted.plan.reserved_vram_bytes,
            "quantified_choice": accepted.plan.quantified_choice,
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
                    supervisor.root / "triage",
                    outcome.triage_subject,
                    supervisor.root / "journal.log",
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
        shutil.rmtree(supervisor.root, ignore_errors=True)
        outcome.journal_deleted = True
    return outcome


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
