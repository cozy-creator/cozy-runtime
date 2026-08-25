"""The `cozy.worker.v1` client, and the supervisor entrypoint.

The worker is the gRPC CLIENT; the coordinator is the server (worker-protocol/01). Two RPCs
run at launch and their durability attributes are the whole design:

* **Control** — one durable bidi stream, one causal order, one fence. Durable messages are
  never shed. Terminal authority lives only here.
* **StreamProgress** — bounded and LOSSY, worker->hub, oldest shed first. Saturating it can
  never block Control, because it is a different RPC with a different queue and a different
  thread; there is no shared buffer for backpressure to travel through.

Threads, and why each exists: the READER blocks on the response iterator and must never do
work (a cancel has to arrive while an attempt is running); the ATTEMPT WORKER holds the
single device lease and therefore serializes attempts in arrival order — that is the whole
of worker-local arbitration, and it invents no global precedence; the REPORTER ticks the
durable Report; the WATCHDOG owns the deadline and the grace clock, because the deadline is
supervisor-enforced and never merely handed to author code.

This process never imports torch. `scripts/worker-live.py nocuda` proves that against the
RUNNING process's loaded modules and mapped libraries, not against the import graph.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import itertools
import json
import os
import queue
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cozy_runtime.internal.config import RuntimeConfig, read_config
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.journal import Journal, JournalError
from cozy_runtime.internal.worker import grants
from cozy_runtime.internal.worker.attempts import (
    CAUSE,
    ORIGIN,
    STATUS,
    Attempt,
    AttemptEngine,
    AttemptRefusal,
)
from cozy_runtime.internal.worker.child import GRACE_SECONDS, ExecutorGone, ExecutorSupervision
from cozy_runtime.internal.worker.control import Coordinator, GrpcCoordinator, Link, LinkError
from cozy_runtime.internal.worker.ledger import Ledger, read_vmrss
from cozy_runtime.internal.worker.plan import BindingPlan, PlanChooser
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb

#: The lossy lane's depth. Oldest is shed first; a durable message never enters here.
PROGRESS_QUEUE = 256

#: Report cadence. Report is DURABLE and carries the applied baseline, so the hub can always
#: see a directive it issued but the worker did not apply.
REPORT_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class WorkerOptions:
    """One worker's settings, as a typed value rather than a parser's leftovers.

    The supervisor has two entrances — the long-lived `serve` process and the ephemeral
    local supervisor a one-shot `run` builds in memory (§8) — and only one of them has a
    command line. A frozen dataclass is what makes the second one possible without a fake
    `argparse.Namespace`, and what makes every setting typed at the point it is read.

    The coordinator is NOT here: who drives this worker is a transport (`worker/control.py`),
    not a setting.
    """

    root: Path
    worker_id: str = "wrk-local"
    instance_id: str = ""
    release_id: str = "cozy/sdxl-unet@cr-005"
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
class Fences:
    """The five fence ordinals, and who owns each (worker-protocol/02 §0)."""

    session_id: str = ""  # CP-minted at parent boot; a restart mints a NEW one
    revision: int = 0  # hub-owned
    credential_epoch: int = 0  # hub-owned, echoed by Report
    applied_directive: bytes = b""  # canonical bytes of the applied Directive, for §3's
    #                                 "same revision + different body is a protocol error"
    posture: pb.Posture = pb.Posture.POSTURE_UNSPECIFIED
    intake: pb.IntakeState = pb.IntakeState.INTAKE_STATE_BOOTING
    wire_minor: int = 0
    activity: list[pb.ActivityEvent] = field(default_factory=list)


class Supervisor:
    """The no-CUDA control parent."""

    def __init__(
        self, config: RuntimeConfig, options: WorkerOptions, coordinator: Coordinator
    ) -> None:
        self.config = config
        self.options = options
        self.coordinator = coordinator
        self.root = Path(options.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fences = Fences(session_id=f"ses-{uuid.uuid4().hex[:20]}")
        self.seq = itertools.count(1)  # SUPERVISOR-owned monotone counter
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
        self.bindings: dict[str, BindingPlan] = {}
        self.engine = AttemptEngine(
            journal=self.journal,
            ledger=self.ledger,
            chooser=self.chooser,
            supervision=self.supervision,
            authorizer=self.authorizer,
            bindings={},
            spool_root=self.root / "spool",
            instance_id=options.instance_id,
            send=self.send,
            progress=self.emit_progress,
        )
        self.engine.session_id = self.fences.session_id
        self.outbound: queue.Queue[pb.WorkerMessage | None] = queue.Queue()
        self.progress: collections.deque[pb.AttemptProgress] = collections.deque(
            maxlen=PROGRESS_QUEUE
        )
        self.progress_shed = 0
        self.progress_sent = 0
        self.work: queue.Queue[Attempt | None] = queue.Queue()
        self.stop = threading.Event()
        self.connected = threading.Event()
        self.exit_code = Exit.ok
        self.boot_at = time.perf_counter()
        self.warm: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.link: Link | None = None  # the live control link, so shutdown can end it

    # ------------------------------------------------------------------ plumbing

    def note(self, kind: str, step: str) -> None:
        """The NON-ATTEMPT activity lane: typed, sequence-bearing, on the DURABLE Report —
        never overloaded onto the lossy progress stream."""
        event = pb.ActivityEvent(
            seq=next(self.seq), kind=kind, step=step[:256], at_unix_ms=int(time.time() * 1000)
        )
        self.fences.activity.append(event)
        del self.fences.activity[:-64]
        self.notes.append(f"{event.seq} {kind}: {step}")
        print(f"[supervisor] {kind}: {step}", flush=True)

    def send(self, message: pb.WorkerMessage) -> None:
        """Durable. Queued, never shed, and replayed after a reconnect if unacknowledged."""
        self.outbound.put(message)

    def intake(self, state: pb.IntakeState) -> None:
        """Move the intake state and REPORT IT IMMEDIATELY.

        Readiness is an EDGE, not a poll. The Report ticker exists so a coordinator can
        always see a directive the worker did not apply; it is not a way to learn that
        loading finished. Publishing on the transition costs one durable message and takes
        the cost of learning "ready" from up to a full report period down to the seam
        round-trip — which is the difference between a one-shot local `run` paying a
        two-second tax and paying none.
        """
        if state == self.fences.intake:
            return
        self.fences.intake = state
        if self.connected.is_set():
            self.send(pb.WorkerMessage(report=self.report()))

    def emit_progress(self, request_id: str, attempt: int, frame: dict[str, Any]) -> None:
        """The LOSSY lane. Bounded, oldest-first shed, supervisor-owned sequence numbers."""
        key = (request_id, attempt)
        counter = self.progress_seq.setdefault(key, itertools.count(1))
        if len(self.progress) == PROGRESS_QUEUE:
            self.progress_shed += 1
        self.progress.append(
            pb.AttemptProgress(
                session_id=self.fences.session_id,
                executor_incarnation=self.supervision.incarnation,
                request_id=request_id,
                attempt=attempt,
                seq=next(counter),
                content_type="application/x-cozy-event+json",
                data=json.dumps(
                    {"type": frame.get("kind", "progress"), "payload": frame.get("value")}
                ).encode()[:65536],
            )
        )

    # ------------------------------------------------------------------ directive

    def apply_directive(self, directive: pb.Directive) -> None:
        """FULL REPLACE. A field absent means absent, never unchanged (§3)."""
        body = documents.canonical_bytes(directive)
        if directive.revision < self.fences.revision:
            self.note("directive", f"revision {directive.revision} ignored (older)")
            return
        if directive.revision == self.fences.revision:
            if body != self.fences.applied_directive:
                self.note(
                    "directive",
                    f"PROTOCOL ERROR: revision {directive.revision} arrived with a different "
                    "body; a changed body must carry a new revision",
                )
                self.engine.faults.append(
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
        if directive.HasField("credential"):
            self.fences.credential_epoch = directive.credential.epoch
        if not directive.HasField("serving"):
            self.note("directive", "no serving mode: nothing to prepare")
            return

        self.bindings = {}
        for plan_id in directive.serving.entrypoint_binding_plan_ids:
            record = self.config.cozy_home / "binding-plans" / f"{plan_id.split(':')[-1]}.json"
            if not record.is_file():
                self.engine.faults.append(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=plan_id,
                        reason="unresolved_binding_plan",
                        detail=f"no local record for {plan_id}",
                    )
                )
                continue
            self.bindings[plan_id] = BindingPlan.read(json.loads(record.read_text()))
        self.engine.bindings = self.bindings
        self.note(
            "directive",
            f"revision {directive.revision} applied: {len(self.bindings)} binding plans, "
            f"posture={pb.Posture.Name(directive.posture)}",
        )
        self.prepare_executor()

    # ------------------------------------------------------------------ readiness

    def prepare_executor(self) -> None:
        """The ExecutorPreparer, at directive time: prepare BEFORE advertising ready, and
        before any attempt may be accepted against this executor."""
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
                f"incarnation {executor.incarnation} pid {executor.pid} dialled back in "
                f"{(executor.ready_at - executor.spawned_at) * 1000:.0f} ms "
                f"(no_new_privs={executor.hello.get('no_new_privs')}, "
                f"oom_score_adj={executor.hello.get('oom_score_adj')}, "
                f"torch_loaded={executor.hello.get('torch_loaded')})",
            )
        by_construction: dict[str, list[BindingPlan]] = {}
        for binding in self.bindings.values():
            by_construction.setdefault(binding.construction_key(), []).append(binding)
        for group in by_construction.values():
            first = group[0]
            started = time.perf_counter()
            reply = executor.call(
                {
                    "cmd": "prepare",
                    "binding": {
                        "project": first.project,
                        "model_class": first.model_class,
                        "binding_path": first.binding_path,
                        "param": first.param,
                        "component": first.component,
                        "store": first.store,
                        "plan": first.plan,
                        "config": first.config,
                        "snapshot": first.snapshot,
                        "release": first.release,
                        "variant": first.variant,
                    },
                    "budgets": {
                        "vram": first.vram_bytes,
                        "pinned": first.pinned_bytes,
                    },
                    "expect_construction_digest": first.model_construction_digest,
                },
                timeout=None,
            )
            took = (time.perf_counter() - started) * 1000
            if not reply.get("ok"):
                self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
                self.engine.faults.append(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=first.plan_id,
                        reason=str(reply.get("code")),
                        detail=str(reply.get("detail"))[:1024],
                    )
                )
                self.note(
                    "boot",
                    f"prepare REFUSED for {first.plan_id[:20]}: {reply.get('code')} - "
                    f"peak allocator {reply.get('peak_allocator_bytes', 0)} B / ledger "
                    f"{reply.get('peak_ledger_vram_bytes', 0)} B was LIVE when it failed; "
                    f"after rollback the allocator holds "
                    f"{reply.get('allocator_bytes', -1)} B and the device has "
                    f"{reply.get('device_free_bytes', -1)} B free",
                )
                self.supervision.invalidate(f"prepare refused: {reply.get('code')}")
                return
            digest = reply["model_construction_digest"]
            for binding in group:
                if binding.model_construction_digest and (
                    binding.model_construction_digest != digest
                ):
                    self.note("boot", f"{binding.plan_id[:20]}… construction digest mismatch")
                    continue
                executor.ready_plans.add(binding.plan_id)
                self.bindings[binding.plan_id] = dataclasses.replace(
                    binding, model_construction_digest=digest
                )
            self.engine.bindings = self.bindings
            # The generation's RESIDENT BASELINE enters the ledger here: the destinations the
            # fill actually committed, the allocator's live bytes behind them, and the ring's
            # page-locked PEAK (a closed ring has charged nothing while the budget it held is
            # exactly what the driver had pinned — §3.2's two-number rule). Every later
            # attempt reconciles against this line.
            self.ledger.observe_construction(reply["facts"])
            self.note(
                "boot",
                f"prepared {len(group)} binding plan(s) in {took:.0f} ms — "
                f"{reply['facts']['filled']} destinations, "
                f"{reply['facts']['filled_bytes'] / 2**30:.3f} GiB, fill leg "
                f"{reply['facts']['fill_ms'] / 1000:.3f} s, construction {digest[:23]}…",
            )
        self.run_warm_pass()
        self.intake(pb.IntakeState.INTAKE_STATE_READY)

    def run_warm_pass(self) -> None:
        """One attempted warm case per WARMABLE entrypoint, through ORDINARY dispatch.

        The pass is DRIVEN here and RUN in the executor. That split is a fence, not a
        layering preference: synthesizing a payload needs the entrypoint's schema, and the
        supervisor must never import endpoint code — an endpoint whose module scope touches
        torch would contaminate this long-lived no-CUDA process permanently. The executor
        has the schema because it has the endpoint, and it is disposable.

        The tax lands on boot, never on a paying request. Every attempt lands one durable
        outcome — `warmed`, `skipped` or `failed` — with a typed reason; a failed pass emits
        `serve_degrade` and the worker still serves. Nothing observed here may ever seed,
        fit or arm cr-008b's demand coefficients (se#715), which is why every outcome record
        carries `demand_evidence: false` and the chooser never reads this list.
        """
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
                        "deadline_s": 300.0,
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
            self.journal.append("warm", dict(outcome))
            self.warm.append(outcome)
            self.note(
                "warm",
                f"{binding.entrypoint}: {outcome['outcome']} ({outcome['ms']} ms) "
                f"{outcome['reason']}"[:200],
            )
            if outcome["outcome"] == "failed":
                self.engine.faults.append(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_DEGRADED,
                        subject=plan_id,
                        reason="serve_degrade",
                        detail=f"boot warm pass failed for {binding.entrypoint}: "
                        f"{outcome['reason']} - the worker still serves",
                    )
                )
        # The warm pass IS execution: it allocates the library workspaces the first kernel
        # launch needs, and those belong to the generation rather than to the first paying
        # request. Re-reading the device here is what makes the first paying attempt
        # reconcile to zero — the first-call tax landing on boot, measured rather than
        # asserted. Nothing observed in the warm pass may seed cr-008b's demand
        # coefficients (se#715); a resident BASELINE is not a demand coefficient.
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

    # ------------------------------------------------------------------ the loops

    def reader(self, responses: Any) -> None:
        for message in responses:
            kind = message.WhichOneof("msg")
            if kind == "register_ack":
                if not message.register_ack.accepted:
                    self.note(
                        "session",
                        f"Register REFUSED: "
                        f"{pb.RegisterRejection.Name(message.register_ack.rejection)}",
                    )
                    self.exit_code = Exit.credential
                    self.stop.set()
                    return
                self.fences.wire_minor = min(WIRE_MINOR, message.register_ack.wire_minor)
                self.note(
                    "session",
                    f"registered as {self.fences.session_id} "
                    f"(wire_minor effective {self.fences.wire_minor})",
                )
                self.connected.set()
            elif kind == "directive":
                try:
                    self.apply_directive(message.directive)
                except (ExecutorGone, JournalError) as exc:
                    self.note("directive", f"apply failed: {exc}")
            elif kind == "start_attempt":
                self.on_start(message.start_attempt)
            elif kind == "cancel_attempt":
                self.note("cancel", self.engine.cancel(message.cancel_attempt))
            elif kind == "terminal_ack":
                self.note("terminal", self.engine.close(message.terminal_ack))
        self.connected.clear()

    def on_start(self, start: pb.StartAttempt) -> None:
        """Fence + idempotency in the READER (no I/O), then hand the work to the queue."""
        if start.session_id != self.fences.session_id:
            self.note("fence", f"dropped: wrong session {start.session_id!r}")
            return
        if start.executor_incarnation < self.supervision.incarnation:
            self.note("fence", f"dropped: stale incarnation {start.executor_incarnation}")
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
            self.refuse(start, exc.code, exc.detail, exc.cause)
            return
        if attempt.state in ("terminal", "closed"):
            self.note(
                "attempt",
                f"duplicate StartAttempt for {start.request_id}#{start.attempt}: replaying "
                f"the journaled terminal, executing NOTHING (executions={attempt.executions})",
            )
            self.send(pb.WorkerMessage(attempt_terminal=self.engine.wrap(attempt)))
            return
        if attempt.state in ("accepted", "running"):
            self.note(
                "attempt",
                "duplicate StartAttempt for a held attempt: re-accepting "
                "the SAME acceptance; nothing starts a second time",
            )
            self.send(pb.WorkerMessage(attempt_accepted=self.engine.accepted_message(attempt)))
            return
        try:
            accepted = self.engine.prepare_and_accept(attempt)
        except AttemptRefusal as exc:
            self.note("attempt", f"{start.request_id}#{start.attempt} refused: {exc.code}")
            self.refuse(start, exc.code, exc.detail, exc.cause)
            return
        except JournalError as exc:
            self.note(
                "journal",
                f"{exc.code}: acknowledging nothing and executing nothing for "
                f"{start.request_id}#{start.attempt}",
            )
            self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
            self.engine.faults.append(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_LOCAL_SAFETY_REFUSAL,
                    subject=f"{start.request_id}#{start.attempt}",
                    reason=exc.code,
                    detail=exc.detail[:1024],
                )
            )
            return
        self.send(pb.WorkerMessage(attempt_accepted=accepted))
        self.work.put(attempt)

    def refuse(self, start: pb.StartAttempt, code: str, detail: str, cause: pb.CauseCode) -> None:
        """A typed refusal is a TERMINAL: journaled before send, like any other."""
        attempt = Attempt(
            request_id=start.request_id,
            attempt=start.attempt,
            digest=start.exec_spec_digest,
            spec={},
            grant=start.grant,
        )
        try:
            terminal = self.engine.terminal(
                attempt,
                STATUS.TERMINAL_STATUS_REFUSED,
                cause,
                ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
                f"{code}: {detail}",
            )
        except JournalError as exc:
            self.note("journal", f"cannot journal a refusal ({exc.code}); sending nothing")
            return
        self.engine.history[attempt.key()] = attempt
        self.send(pb.WorkerMessage(attempt_terminal=terminal))

    def worker(self) -> None:
        """The single device lease. Attempts serialize here, in arrival order."""
        while not self.stop.is_set():
            attempt = self.work.get()
            if attempt is None:
                return
            try:
                terminal = self.engine.execute(attempt)
            except JournalError as exc:
                self.note("journal", f"{exc.code}: no terminal can be claimed")
                self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
                continue
            self.send(pb.WorkerMessage(attempt_terminal=terminal))
            current = self.supervision.current
            if self.supervision.needs_rebuild:
                self.rebuild("readiness was invalidated during the attempt")
            elif current is None or not current.alive():
                self.rebuild("the executor is gone")
            elif current.poisoned:
                self.rebuild(f"poisoned: {current.poisoned}")

    def rebuild(self, why: str) -> None:
        """Replace the executor and RELOAD before reporting ready. Never on the
        cancellation critical path — the caller's cancel completed when CANCELED journaled."""
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
        # Cleared LAST, not first: an invalidation raised by another thread WHILE the
        # rebuild ran is about the executor this rebuild just replaced, and clearing at the
        # start would leave the flag set and buy a second, pointless 4-second refill.
        self.supervision.needs_rebuild = False
        self.note(
            "recovery",
            f"ready again in {(time.perf_counter() - started) * 1000:.0f} ms at "
            f"incarnation {self.supervision.incarnation} / readiness epoch "
            f"{self.supervision.readiness_epoch}",
        )

    def watchdog(self) -> None:
        """The deadline and the grace clock — both the SUPERVISOR's, never author code's."""
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
        while not self.stop.wait(REPORT_SECONDS):
            if self.connected.is_set():
                self.send(pb.WorkerMessage(report=self.report()))

    def report(self) -> pb.Report:
        return pb.Report(
            session_id=self.fences.session_id,
            executor_incarnation=self.supervision.incarnation,
            applied_revision=self.fences.revision,
            intake_state=self.fences.intake,
            credential_epoch=self.fences.credential_epoch,
            readiness_epoch=self.supervision.readiness_epoch,
            active_attempts=self.engine.active(),
            faults=self.engine.faults[-8:],
            activity=self.fences.activity[-8:],
            applied_wire_minor=self.fences.wire_minor,
            serving_capacity=pb.ServingCapacity(
                ready_entrypoint_binding_plan_ids=sorted(
                    self.supervision.current.ready_plans if self.supervision.current else ()
                ),
                free_vram_bytes=0,
            ),
        )

    # ------------------------------------------------------------------ transport

    def outbound_iterator(self) -> Any:
        while not self.stop.is_set():
            message = self.outbound.get()
            if message is None:
                return
            yield message

    def pump_progress(self, link: Link) -> None:
        """The lossy lane, on its OWN lane and its OWN thread. Its failure is invisible to
        Control by construction — there is no shared buffer for backpressure to cross."""
        link.progress(self.progress_iterator())

    def progress_iterator(self) -> Any:
        while not self.stop.is_set():
            if self.progress:
                self.progress_sent += 1
                yield self.progress.popleft()
            else:
                time.sleep(0.01)

    def register(self) -> pb.Register:
        recovered = self.engine.recover()
        if recovered:
            self.note(
                "recovery",
                f"the recovered journal is REPORTED FIRST: {len(recovered)} attempt(s) "
                "ride Register, and the hub closes them before minting any next ordinal",
            )
        return pb.Register(
            session_id=self.fences.session_id,
            executor_incarnation=self.supervision.incarnation,
            wire_minor=WIRE_MINOR,
            worker_id=self.options.worker_id,
            instance_id=self.options.instance_id,
            release_id=self.options.release_id,
            image_digest="sha256:" + "00" * 32,
            git_commit=self.options.git_commit,
            resources=self.resources(),
            recovered_attempts=recovered,
        )

    def resources(self) -> pb.WorkerResources:
        """Host facts measured ONCE, by a bounded canary at registration.

        Tri-state honesty: present / absent / UNREADABLE. Anything this supervisor cannot
        read WITHOUT a CUDA context is named in `unreadable` rather than reported as zero —
        a wedged driver is not a CPU pod, and a canary that hangs is worse than one that
        fails, so every probe here is a bounded read of a file or a bounded subprocess.
        """
        import os
        import platform
        import subprocess

        unreadable: list[str] = []
        gpu_name, gpu_sm, vram, driver = "", 0, 0, ""
        try:
            out = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total,driver_version,compute_cap",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                .stdout.strip()
                .splitlines()[0]
            )
            name, total, drv, cap = (part.strip() for part in out.split(","))
            gpu_name, vram, driver = name, int(float(total)) * 1024 * 1024, drv
            gpu_sm = int(float(cap) * 10)
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            unreadable += ["gpu_name", "gpu_sm", "vram_total_bytes", "driver_version"]
        try:
            pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            pages = 0
            unreadable.append("host_ram_total_bytes")
        return pb.WorkerResources(
            gpu_count=1 if gpu_name else 0,
            gpu_name=gpu_name,
            gpu_sm=gpu_sm,
            vram_total_bytes=vram,
            driver_version=driver,
            platform=platform.platform()[:120],
            host_ram_total_bytes=pages,
            vcpu_count=os.cpu_count() or 0,
            pinned_capacity_bytes=self.ledger.pinned_budget,
            unreadable=sorted(unreadable),
        )

    def run(self) -> int:
        """Link, register, serve, relink. The session outlives the connection.

        The loop is the same whichever coordinator drives it — that is the point of the
        transport seam. What differs is `reconnects`: a network peer that drops the stream
        may come back and the worker redials the SAME session; an in-memory coordinator that
        ends the stream has ended the run, and redialling it would be a hang.
        """
        self.journal.append(
            "session",
            {"session_id": self.fences.session_id, "instance_id": self.options.instance_id},
        )
        killed = self.supervision.sweep_orphans()
        if killed:
            self.note(
                "recovery",
                f"killed {len(killed)} orphan executor(s) from before this restart: {killed}",
            )
        threading.Thread(target=self.worker, daemon=True, name="attempt").start()
        threading.Thread(target=self.reporter, daemon=True, name="report").start()
        threading.Thread(target=self.watchdog, daemon=True, name="watchdog").start()

        backoff = 0.2
        while not self.stop.is_set():
            link = self.link = self.coordinator.link()
            try:
                threading.Thread(
                    target=self.pump_progress, args=(link,), daemon=True, name="progress"
                ).start()
                responses = link.control(self.outbound_iterator())
                self.send(pb.WorkerMessage(register=self.register()))
                for attempt in self.engine.pending_terminals():
                    self.note(
                        "terminal",
                        f"replaying the journaled terminal for {attempt.request_id}#"
                        f"{attempt.attempt} in a fresh envelope ({attempt.terminal_id})",
                    )
                    self.send(pb.WorkerMessage(attempt_terminal=self.engine.wrap(attempt)))
                self.reader(responses)
            except LinkError as exc:
                self.note("session", f"stream error: {exc}")
            if self.stop.is_set() or not self.coordinator.reconnects:
                break
            self.note("session", f"reconnecting in {backoff:.1f} s (same session)")
            time.sleep(backoff)
            backoff = min(backoff * 2, 5.0)
        self.shutdown()
        return int(self.exit_code)

    def shutdown(self) -> None:
        seam = (
            self.supervision.current.channel.accounting()
            if self.supervision.current is not None
            else {}
        )
        self.stop.set()
        self.work.put(None)
        self.outbound.put(None)
        if self.link is not None:
            self.link.cancel()
        self.supervision.close()
        self.journal.close()
        print(
            "[supervisor] summary "
            + json.dumps(
                {
                    "session_id": self.fences.session_id,
                    "incarnations": self.supervision.spawns,
                    "readiness_epoch": self.supervision.readiness_epoch,
                    "executed": self.engine.executed,
                    "journal_appends": self.journal.appends,
                    "journal_fsync_ms": round(self.journal.fsync_ns / 1e6, 2),
                    "progress_sent": self.progress_sent,
                    "progress_shed": self.progress_shed,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="cozy-runtime worker supervisor (no CUDA)")
    parser.add_argument("--hub", required=True, help="gRPC target, e.g. unix:/path/hub.sock")
    parser.add_argument("--root", required=True)
    parser.add_argument("--worker-id", default="wrk-local")
    parser.add_argument("--instance-id", default="")
    parser.add_argument("--release-id", default="cozy/sdxl-unet@cr-005")
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--python", default="")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--alloc-conf", default="expandable_segments:True")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--grace", type=float, default=GRACE_SECONDS)
    parser.add_argument("--host-budget", type=int, default=8 << 30)
    parser.add_argument("--pinned-budget", type=int, default=512 << 20)
    parser.add_argument("--no-warm", action="store_true")
    args = parser.parse_args(argv)

    options = WorkerOptions(
        root=Path(args.root),
        worker_id=args.worker_id,
        instance_id=args.instance_id or f"ins-{uuid.uuid4().hex[:16]}",
        release_id=args.release_id,
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
    config = read_config()
    supervisor = Supervisor(config, options, GrpcCoordinator(args.hub))

    def terminate(signum: int, frame: object) -> None:
        supervisor.stop.set()
        supervisor.outbound.put(None)
        if supervisor.link is not None:
            # Half-closing the request stream is not enough: the peer's response generator
            # may be parked waiting to send. Cancelling ends the call from this side.
            supervisor.link.cancel()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        return supervisor.run()
    except JournalError as exc:
        print(f"[supervisor] {exc}", file=sys.stderr, flush=True)
        return int(Exit.internal)


if __name__ == "__main__":
    raise SystemExit(main())
