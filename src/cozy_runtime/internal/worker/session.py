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
import contextlib
import dataclasses
import itertools
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cozy_runtime.internal import egress, hostfacts
from cozy_runtime.internal.config import RuntimeConfig, read_config
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.journal import Journal, JournalError
from cozy_runtime.internal.worker import grants, observe, triage
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
from cozy_runtime.internal.worker.plan import BindingPlan, JobBinding, PlanChooser
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb

#: The lossy lane's depth. Oldest is shed first; a durable message never enters here.
PROGRESS_QUEUE = 256

#: Report cadence. Report is DURABLE and carries the applied baseline, so the hub can always
#: see a directive it issued but the worker did not apply.
REPORT_SECONDS = 2.0

#: The RUN-ONCE RECYCLE exit (cr-009, v1 rc 75). A job worker that completed its one attempt
#: and had its terminal acknowledged asks to be RECLAIMED, and that is not a death: a
#: supervisor exiting 0 on completion is indistinguishable from one that exited 0 because its
#: coordinator went away, which is exactly the ambiguity that makes a pod either leak or get
#: killed while it still had work. It is deliberately NOT in `internal/exits.py`: that matrix
#: is the shared CLI/coordinator vocabulary, and this is a worker PROCESS disposition.
RECYCLE_EXIT = 75

#: How long a StartAttempt that arrives while the executor is being rebuilt WAITS before it
#: gives up and takes the ordinary not-ready refusal. It is generous because the thing it is
#: waiting for is a real refill: about 5 s warm on the launch family and ~48 s from a cold
#: page cache. The alternative is what cl-003 measured — a refusal per dispatch, a requeue
#: budget of three spent in 1.8 s, and a worker that was serving again four seconds later.
REBUILD_WAIT_SECONDS = 120.0


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
    mode: str = ""  # the Directive `mode` oneof this worker APPLIED: "serving", "job", or
    #                 "" before any directive. READ from `WhichOneof`, never inferred from
    #                 what happens to be in `bindings`/`jobs` — the schema already carries
    #                 the discriminator and a second derivation of it is free to disagree.


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
        #: job_descriptor_id -> the local job plan record. A JOB directive fills this and
        #: leaves `bindings` empty: the two modes are alternatives, never a union.
        self.jobs: dict[str, JobBinding] = {}
        # cr-011: the worker's own bounded record, the silence-bounded liveness detector,
        # and the journal-backed bundle store. All three outlive the executor, and the
        # bundle store's receipts outlive THIS process too.
        self.boot = observe.BootRecord()
        self.monitor = observe.SilenceMonitor(self.boot.ring)
        self.bundles = triage.BundleStore(self.root / "triage", self.journal)
        self.engine = AttemptEngine(
            journal=self.journal,
            ledger=self.ledger,
            chooser=self.chooser,
            supervision=self.supervision,
            authorizer=self.authorizer,
            bindings={},
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
        self.engine.session_id = self.fences.session_id
        self.outbound: queue.Queue[pb.WorkerMessage | None] = queue.Queue()
        self.progress: collections.deque[pb.AttemptProgress] = collections.deque(
            maxlen=PROGRESS_QUEUE
        )
        self.progress_shed = 0
        self.progress_sent = 0
        self.work: queue.Queue[Attempt | None] = queue.Queue()
        self.stop = threading.Event()
        #: SET when the intake state is a verdict (READY or ERROR), CLEAR while the worker is
        #: still becoming one. `on_start` waits on it so a request that arrives during a
        #: rebuild queues instead of being refused for a condition that is ending.
        self.settled = threading.Event()
        self.supervision.on_invalidate = self.capacity_dropped
        self.connected = threading.Event()
        self.exit_code: int = int(Exit.ok)
        self.boot_at = time.perf_counter()
        self.warm: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.link: Link | None = None  # the live control link, so shutdown can end it
        self.job_caps = pb.ResourceCaps()
        self.reclaim_on_terminal = False

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
        # SETTLED means the worker has finished deciding what it is: READY (it serves) or
        # ERROR (it does not, and said why). LOADING and BOOTING are transitional — an
        # attempt that arrives in one of them WAITS for the answer instead of being refused
        # for a condition that is about to end. `on_start` is the only reader.
        if state in (pb.IntakeState.INTAKE_STATE_READY, pb.IntakeState.INTAKE_STATE_ERROR):
            self.settled.set()
        else:
            self.settled.clear()
        if self.connected.is_set():
            self.send(pb.WorkerMessage(report=self.report()))

    def capacity_dropped(self, why: str) -> None:
        """The executor's state was invalidated: STOP ADVERTISING READY, right now.

        Installed on `ExecutorSupervision.on_invalidate`, so it runs on whichever thread
        observed the death — the watchdog on a forceful cancel, the attempt thread on a seam
        close — and before that call returns. The ready-plan set is already empty by then, so
        the Report this publishes carries both halves of the same fact: no ready plans, and
        an intake state that is not READY.

        Only a READY worker is moved. An invalidation that arrives while preparation already
        failed must not overwrite the ERROR state with a LOADING that says less.
        """
        if self.fences.intake != pb.IntakeState.INTAKE_STATE_READY:
            return
        self.note("readiness", f"capacity withdrawn: {why[:200]}")
        self.intake(pb.IntakeState.INTAKE_STATE_LOADING)

    def read_device_process(self) -> int:
        """NVML per-process device bytes for the EXECUTOR, read by the supervisor.

        The out-of-allocator meter needs a number the allocator structurally cannot see —
        the CUDA context and the cuDNN/cuBLAS workspaces — and the only place it exists is
        the driver's own per-process accounting. Read by the PARENT, like RSS: the child's
        claim about its own footprint is one the supervisor cannot check, and here it does
        not have to. -1 is UNREADABLE and never a zero.
        """
        pid = self.ledger.executor_pid
        if not pid:
            return -1
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return -1
        for line in out.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
                return int(parts[1]) * 1024 * 1024
        return -1

    def applied_posture(self) -> dict[str, Any]:
        """The deployment state the worker was in when something happened (§3.6).

        A triage bundle that cannot say which directive was applied, which credential epoch
        was live and whether intake was accepting explains half of what went wrong — the
        other half is always "under what configuration".
        """
        return {
            "posture": pb.Posture.Name(self.fences.posture),
            "intake_state": pb.IntakeState.Name(self.fences.intake),
            "applied_revision": self.fences.revision,
            "credential_epoch": self.fences.credential_epoch,
            "applied_wire_minor": self.fences.wire_minor,
            "readiness_epoch": self.supervision.readiness_epoch,
            "executor_incarnation": self.supervision.incarnation,
            "worker_id": self.options.worker_id,
            "release_id": self.options.release_id,
            "boot_steps": list(self.boot.steps),
        }

    def emit_progress(self, request_id: str, attempt: int, frame: dict[str, Any]) -> None:
        """The LOSSY lane. Bounded, oldest-first shed, supervisor-owned sequence numbers."""
        key = (request_id, attempt)
        # Liveness reads the MONOTONE position off this lane and nothing else. The lane is
        # lossy, which is exactly why silence is measured as "the position did not move"
        # rather than "no frame arrived" — a dropped frame is not a wedge.
        position = frame.get("position")
        if isinstance(position, int):
            self.monitor.advance(f"{request_id}#{attempt}", position)
        # A RETAINED observation is held HERE the moment it arrives, so the supervisor owns
        # a copy before the executor can die. The reply's ring fills whatever the lossy lane
        # dropped; the executor's own sequence keeps the two from double-counting.
        if frame.get("kind") != "progress":
            self.engine.forward(request_id, attempt, frame)
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
        # THE MODE, taken from the schema's own discriminator. A Directive is a
        # discriminated COMPLETE replacement, so the mode it carries is the mode this
        # worker is in until the next revision — and the Report's capacity oneof is scoped
        # by exactly this value.
        self.fences.mode = directive.WhichOneof("mode") or ""
        if directive.HasField("job"):
            self.apply_job_directive(directive.job)
            return
        if not directive.HasField("serving"):
            self.note("directive", "no serving mode: nothing to prepare")
            return

        self.bindings = {}
        # SYMMETRIC with `apply_job_directive` clearing `bindings`. A serving directive
        # replaces the job lane too: leaving a previous revision's job plans behind would
        # leave the worker advertising job capacity it will not honour, and "complete
        # replacement" has to mean complete in both directions.
        self.jobs = {}
        self.engine.jobs = {}
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
        # cr-012: the EGRESS boundary is a deployment fact, rebuilt from the applied
        # directive's bindings and from nothing else. A revision that names no host leaves
        # it empty, which means no egress — the fail-closed state is the default state, and
        # nothing at request time can widen it.
        hosts = sorted(
            {
                host.strip()
                for binding in self.bindings.values()
                for host in binding.egress_allowed_hosts.split(",")
                if host.strip()
            }
        )
        self.engine.egress = egress.EgressPolicy(
            allowed_hosts=tuple(hosts),
            max_bytes=max((b.max_input_bytes for b in self.bindings.values()), default=64 << 20),
            allow_private=any(b.tenancy == "local" for b in self.bindings.values()),
        )
        self.note(
            "directive",
            f"revision {directive.revision} applied: {len(self.bindings)} binding plans, "
            f"posture={pb.Posture.Name(directive.posture)}",
        )
        self.prepare_executor()

    def apply_job_directive(self, job: pb.JobDirective) -> None:
        """The JOB lane's directive: resolve the descriptor, spawn a fresh executor, ready.

        No binding plan, no fill, no warm pass and no residency — a job has none of those,
        and inventing them so the two lanes look alike would mean holding weights a job
        never asked for. What the worker DOES prepare is the disposable process the attempt
        will run in; the job's own preparation (its typed inputs, its read-only Model if it
        declares one) happens after acceptance, bounded by the attempt deadline (cr-009).
        """
        self.bindings = {}
        self.engine.bindings = {}
        self.jobs = {}
        plans = self.config.cozy_home / "job-plans"
        record = plans / f"{job.job_descriptor_id.split(':')[-1]}.json"
        if not record.is_file():
            self.engine.faults.append(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                    subject=job.job_descriptor_id,
                    reason="unresolved_job_descriptor",
                    detail=f"no local record for {job.job_descriptor_id}",
                )
            )
            self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
            return
        self.jobs[job.job_descriptor_id] = JobBinding.read(json.loads(record.read_text()))
        self.engine.jobs = self.jobs
        self.job_caps = job.resource_caps
        self.reclaim_on_terminal = job.reclaim_on_terminal
        self.note(
            "directive",
            f"revision {self.fences.revision} applied in JOB mode: "
            f"{self.jobs[job.job_descriptor_id].job!r}, gpu floor {job.gpu_count}, "
            f"reclaim_on_terminal={job.reclaim_on_terminal}",
        )
        self.intake(pb.IntakeState.INTAKE_STATE_LOADING)
        executor = self.supervision.current
        if executor is None or not executor.alive():
            try:
                executor = self.supervision.spawn(imposed=self.imposed())
            except ExecutorGone as exc:
                self.note("boot", f"the job executor never dialled back: {exc}")
                self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
                return
            self.ledger.executor_pid = executor.pid
            self.note(
                "boot",
                f"job executor incarnation {executor.incarnation} pid {executor.pid} dialled "
                f"back in {(executor.ready_at - executor.spawned_at) * 1000:.0f} ms "
                f"(torch_loaded={executor.hello.get('torch_loaded')})",
            )
        self.intake(pb.IntakeState.INTAKE_STATE_READY)

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
            # The LOAD leg gets a liveness subject and `started` at position 0. There is no
            # boot timeout here and there will not be one: a multi-GB load is slow, and the
            # only honest question is whether its integral-MiB position is still moving.
            subject = f"prepare:{first.component}"
            self.monitor.start(subject, "load")
            reply = executor.call(
                {
                    "cmd": "prepare",
                    "binding": {
                        "project": first.project,
                        "model_class": first.model_class,
                        "binding_path": first.binding_path,
                        "param": first.param,
                        "component": first.component,
                        "components": list(first.component_set()),
                        "snapshots": first.snapshot_map(),
                        "store": first.store,
                        "config": first.config,
                        "snapshot": first.snapshot,
                        "release": first.release,
                        "variant": first.variant,
                        "custody": first.custody,
                        "strict_keys": first.strict_keys,
                        # cr-006's persisted probe fact: a DEVICE property, so it is keyed
                        # by the device tuple under cozy_home and shared across every
                        # binding this rig serves.
                        "probe_cache": str(self.config.cozy_home / "probe"),
                        # cr-008c: the OFFERED artifact variants and the measured axis to
                        # order them on. The choice is made in the executor because that is
                        # where the device and the capability table live — this process
                        # never imports torch — and it comes back on the reply as a fact
                        # this supervisor journals and reports.
                        "variants": [dataclasses.asdict(offer) for offer in first.offered()],
                        "objective": first.objective,
                        "steps_basis": first.steps_basis,
                        "delivery_cache": str(self.config.cozy_home / "probe"),
                    },
                    "budgets": {
                        "vram": first.vram_bytes,
                        "pinned": first.pinned_bytes,
                        # The DECLARED construction ceiling. Components past it are PARKED
                        # and staged in on first declared use; 0 means every component
                        # stays resident. Construction is priced against the declaration so
                        # it reproduces; the attempt-time ladder is what reads measurement.
                        "resident": first.resident_budget_bytes,
                    },
                    "expect_construction_digest": first.model_construction_digest,
                },
                timeout=None,
                on_progress=lambda frame: self.monitor.advance(
                    f"prepare:{frame.get('name', '')}", int(frame.get("position", 0) or 0)
                ),
            )
            took = (time.perf_counter() - started) * 1000
            if not reply.get("ok"):
                self.monitor.end(subject, str(reply.get("code")))
                self.boot.step(
                    "load",
                    "refused",
                    took,
                    component=first.component,
                    code=str(reply.get("code")),
                )
                detail = str(reply.get("detail"))[:1024]
                shortfall = reply.get("shortfall") or {}
                if shortfall:
                    # `Fault` has no structured shortfall slot — only `TerminalCause` does
                    # (worker-protocol/01), and that schema is th-024's, not this repo's. So
                    # the two numbers go into `detail` in a fixed shape rather than being
                    # lost: what was needed, what was measured free, on which resource. A
                    # Fault-level ResourceShortfall is a protocol change and is named as a
                    # seam, not smuggled in.
                    detail = (
                        f"{shortfall['resource']} shortfall at {shortfall['scope']}: "
                        f"needed {shortfall['needed_bytes']} B, "
                        f"{shortfall['available_bytes']} B free (measured) - {detail}"
                    )[:1024]
                self.engine.faults.append(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=first.plan_id,
                        reason=str(reply.get("code")),
                        detail=detail,
                    )
                )
                # RECORD THE FAULT, THEN MOVE THE STATE. The transition publishes a Report
                # immediately, so transitioning first tells a coordinator "I failed" in a
                # message that does not yet carry why — and the next Report is a full period
                # away. Found by the harness waiting on a fault report and getting an empty
                # one; it is the ordering hazard every edge-published state has.
                self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
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
            # THE DELIVERY CHOICE, ABSORBED (cr-008c). It is resolved exactly once, here,
            # and every attempt served by this generation prices against it — the
            # PlanChooser reads these two fields and structurally cannot re-choose, which
            # is what keeps a delivery switch from arriving as a placement side effect.
            delivery = reply.get("delivery") or {}
            for binding in group:
                if binding.model_construction_digest and (
                    binding.model_construction_digest != digest
                ):
                    self.note("boot", f"{binding.plan_id[:20]}… construction digest mismatch")
                    continue
                executor.ready_plans.add(binding.plan_id)
                self.bindings[binding.plan_id] = dataclasses.replace(
                    binding,
                    model_construction_digest=digest,
                    constructed_generation_digest=str(reply["constructed_generation_digest"]),
                    delivery_rung=str(delivery.get("rung", "")),
                    delivery_variant=str(delivery.get("variant", "")),
                )
            self.engine.bindings = self.bindings
            # The generation's RESIDENT BASELINE enters the ledger here: the destinations the
            # fill actually committed, the allocator's live bytes behind them, and the ring's
            # page-locked PEAK (a closed ring has charged nothing while the budget it held is
            # exactly what the driver had pinned — §3.2's two-number rule). Every later
            # attempt reconciles against this line.
            self.ledger.observe_construction(reply["facts"])
            self.monitor.end(subject, "filled")
            self.boot.step(
                "load",
                "filled",
                took,
                component=first.component,
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
                f"generation {str(reply['constructed_generation_digest'])[:23]}… "
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
            # A warm invoke is a BENCHMARK OVERRIDE: it runs a synthetic payload at a
            # cheapened schedule, so everything it observes is marked and can never pollute
            # a measurement (§3.6, se#715).
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
                    self.exit_code = int(Exit.credential)
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
                    # A directive whose PREPARATION dies must say so durably. Noting it and
                    # leaving intake at LOADING is the worst of both: the worker will never
                    # become ready and the coordinator has no way to learn that, so it waits
                    # on a worker that is finished. Found live by killing the executor
                    # mid-fill — the run hung instead of converging (cr-008a arm 2).
                    self.note("directive", f"apply failed: {exc}")
                    self.engine.faults.append(
                        pb.Fault(
                            kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                            subject=f"revision {message.directive.revision}",
                            reason="directive_preparation_failed",
                            detail=f"{type(exc).__name__}: {exc}"[:1024],
                        )
                    )
                    self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
            elif kind == "start_attempt":
                self.on_start(message.start_attempt)
            elif kind == "cancel_attempt":
                self.note("cancel", self.engine.cancel(message.cancel_attempt))
            elif kind == "terminal_ack":
                acked = message.terminal_ack
                self.note("terminal", self.engine.close(acked))
                closed = self.engine.history.get((acked.request_id, acked.attempt))
                if self.reclaim_on_terminal and closed is not None and closed.job is not None:
                    # RUN-ONCE: one immutable build, one bounded attempt, terminal, reclaim.
                    # The worker stops HERE, after the terminal is durably acknowledged, with
                    # the recycle disposition rather than a death.
                    self.note(
                        "session",
                        f"job attempt {acked.request_id}#{acked.attempt} is acknowledged and "
                        f"reclaim_on_terminal is set: exiting {RECYCLE_EXIT} (recycle, not a "
                        "death)",
                    )
                    self.exit_code = RECYCLE_EXIT
                    self.stop.set()
                    self.outbound.put(None)
                    if self.link is not None:
                        self.link.cancel()
                    return
            elif kind == "checkpoint_receipt":
                # The COORDINATOR's copy of a checkpoint that is already durable here. It is
                # recorded and acked; it can never un-durable a fact the worker journaled,
                # which is why `save()` did not block on it (law 9: at-least-once).
                receipt = message.checkpoint_receipt
                self.engine.absorb_receipt(receipt)
                self.send(
                    pb.WorkerMessage(
                        checkpoint_ack=pb.JobCheckpointAck(
                            session_id=receipt.session_id,
                            executor_incarnation=receipt.executor_incarnation,
                            request_id=receipt.request_id,
                            attempt=receipt.attempt,
                            operation_key=receipt.operation_key,
                            receipt_id=receipt.receipt_id,
                            content_digest=receipt.content_digest,
                        )
                    )
                )
        self.connected.clear()

    def on_start(self, start: pb.StartAttempt) -> None:
        """READER: fence, record, enqueue. Nothing here waits, fetches or leases.

        Everything slow about an attempt — retrieving the payload, hydrating its inputs,
        pricing a plan, waiting out a rebuild — happens on the DEVICE LANE below. This thread
        is the only one that can hear a cancel, a terminal ack, a directive or a shutdown, so
        work done here is time nobody else's message can be heard in. It used to do all four
        (codex audit, adopted 2026-08-25), and the 120 s rebuild wait made the worst case a
        two-minute deaf reader.
        """
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
            self.refuse(start, exc.code, exc.detail, exc.cause, exc.origin)
            return
        if attempt.state in ("terminal", "closed"):
            self.note(
                "attempt",
                f"duplicate StartAttempt for {start.request_id}#{start.attempt}: replaying "
                f"the journaled terminal, executing NOTHING (executions={attempt.executions})",
            )
            self.send(pb.WorkerMessage(attempt_terminal=self.engine.wrap(attempt)))
            return
        if attempt.state == "queued":
            self.note(
                "attempt",
                "duplicate StartAttempt for a QUEUED attempt: it is already held in arrival "
                "order and its acceptance will be published when the lane admits it",
            )
            return
        if attempt.state in ("accepted", "running"):
            self.note(
                "attempt",
                "duplicate StartAttempt for a held attempt: re-accepting "
                "the SAME acceptance; nothing starts a second time",
            )
            self.send(pb.WorkerMessage(attempt_accepted=self.engine.accepted_message(attempt)))
            return
        self.engine.hold(attempt)
        self.work.put(attempt)

    def await_settled(self, attempt: Attempt) -> None:
        """QUEUE behind a rebuild instead of refusing through one.

        The alternative — the shape cl-003 measured — is a dispatch/refuse/requeue loop: a
        forceful cancel kills the executor, the rebuild takes about five seconds, and every
        attempt that lands in that window is refused `executor_not_ready` with an
        EXECUTOR_FAULT the coordinator can only read as "this worker is broken". One cancel
        spent a requeue budget of three in 1.8 seconds against a worker that was perfectly
        healthy 4 seconds later. Waiting turns a refusal into a queue, which is what a
        coordinator's budget is for.

        The wait is on the DEVICE LANE, which is the thread that would have been idle anyway.
        It used to be on the control reader, where 120 s of waiting was 120 s in which no
        cancel, ack, directive or shutdown could be heard.
        """
        if self.settled.is_set():
            return
        started = time.perf_counter()
        key = f"{attempt.request_id}#{attempt.attempt}"
        self.note(
            "readiness",
            f"{key} reached the lane while the worker is rebuilding: waiting up to "
            f"{REBUILD_WAIT_SECONDS:g} s rather than refusing it",
        )
        settled = self.settled.wait(REBUILD_WAIT_SECONDS)
        waited = (time.perf_counter() - started) * 1000
        self.note(
            "readiness",
            f"{key} waited {waited:.0f} ms; intake is "
            f"{pb.IntakeState.Name(self.fences.intake)}"
            + ("" if settled else " (the wait expired)"),
        )

    def refuse(
        self,
        start: pb.StartAttempt,
        code: str,
        detail: str,
        cause: pb.CauseCode,
        origin: pb.CauseOrigin = ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
    ) -> None:
        """A typed refusal is a TERMINAL: journaled before send, like any other.

        `origin` is carried rather than assumed: the ORIGIN OF THE ADDRESS that missed is
        what separates a caller's typed error from a platform-fatal one, and collapsing both
        onto SUPERVISOR sends the wrong person to look (v1 REF_ORIGIN).
        """
        held = self.engine.history.get((start.request_id, start.attempt))
        attempt = held or Attempt(
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
                origin,
                f"{code}: {detail}",
            )
        except JournalError as exc:
            self.note("journal", f"cannot journal a refusal ({exc.code}); sending nothing")
            return
        self.engine.history[attempt.key()] = attempt
        self.send(pb.WorkerMessage(attempt_terminal=terminal))

    # ------------------------------------------------------------------ the device lane

    def device_lane(self) -> None:
        """ONE SERIALIZED PER-DEVICE STATE MACHINE: admit, execute, reconcile, rebuild.

        Two phases per attempt and they are strictly ordered against each other, which is the
        point rather than an implementation convenience. ADMISSION prices a plan against the
        live ledger; EXECUTION runs it and reconciles the ledger at the terminal. Admitting
        attempt N+1 before N reconciled would price it against a residency N is about to move
        — several attempts accepted against one snapshot, which is what the audit found. So
        the lane admits nothing until the previous attempt's terminal has closed its books.

        Overlapping admission with the previous attempt's GPU work is cr-019's door: it is
        the pipelining this shape makes reachable, and it is not claimed here.
        """
        while not self.stop.is_set():
            attempt = self.work.get()
            if attempt is None:
                return
            if self.admit(attempt):
                self.execute(attempt)
                self.rebuild_if_needed()

    def admit(self, attempt: Attempt) -> bool:
        """FETCH -> HYDRATE -> CHOOSE -> ACCEPT, or a typed terminal. Never a silence."""
        key = f"{attempt.request_id}#{attempt.attempt}"
        self.await_settled(attempt)
        if attempt.canceling:
            # Cancelled while it sat in the queue. Nothing was fetched, planned or reserved,
            # so there is nothing to unwind — but the attempt is HELD and every held attempt
            # ends in a terminal.
            self.note("cancel", f"{key} was cancelled before admission")
            self.terminate(attempt, self.engine.finish_cancel, attempt.canceling)
            return False
        deadline_ms = int(attempt.spec.get("deadline_unix_ms", 0))
        if deadline_ms and time.time() * 1000 > deadline_ms:
            # The queue can hold work now, so a deadline can expire before the lane reaches
            # it. Running it anyway would spend the card on an answer nobody is waiting for.
            self.note("cancel", f"{key} reached the lane past its deadline")
            self.terminate(attempt, self.engine.finish_cancel, "deadline")
            return False
        if self.fences.posture == pb.Posture.POSTURE_DRAINING:
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
            self.engine.faults.append(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_LOCAL_SAFETY_REFUSAL,
                    subject=key,
                    reason=exc.code,
                    detail=exc.detail[:1024],
                )
            )
            self.intake(pb.IntakeState.INTAKE_STATE_ERROR)  # fault first, then the edge
            return False
        self.send(pb.WorkerMessage(attempt_accepted=accepted))
        return True

    def execute(self, attempt: Attempt) -> None:
        """RUN -> TERMINAL -> RECONCILE. The ledger closes inside `terminal()`."""
        try:
            terminal = self.engine.execute(attempt)
        except JournalError as exc:
            self.note("journal", f"{exc.code}: no terminal can be claimed")
            self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
            return
        except Exception as exc:
            # A BUG in the execution path is still an attempt that must terminate. Found
            # live on the job lane: one unhandled ValueError killed this thread, and the
            # coordinator then waited out its whole readiness deadline on an attempt it
            # had already been told was accepted. A worker that cannot run an attempt
            # says so with a terminal; it never converts a defect into a hang.
            self.note("attempt", f"execution faulted: {type(exc).__name__}: {exc}")
            try:
                terminal = self.engine.abandon(
                    attempt, f"the worker faulted while running this attempt: {exc}"
                )
            except Exception as inner:
                self.note("attempt", f"even ABANDONED could not be journaled: {inner}")
                self.intake(pb.IntakeState.INTAKE_STATE_ERROR)
                return
        self.send(pb.WorkerMessage(attempt_terminal=terminal))

    def terminate(
        self, attempt: Attempt, build: Callable[..., pb.AttemptTerminal], *args: Any
    ) -> None:
        """Journal ONE terminal for a held attempt and send it, or say why it could not."""
        try:
            self.send(pb.WorkerMessage(attempt_terminal=build(attempt, *args)))
        except JournalError as exc:
            self.note("journal", f"cannot journal a terminal ({exc.code}); sending nothing")

    def rebuild_if_needed(self) -> None:
        """The queue starts behind the rebuild because BOTH live on this lane."""
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
        """The Report tick, and the liveness SAMPLE that rides it.

        The cadence is a scheduling choice; the VERDICT is not. `sample` compares two
        positions and counts consecutive stills — no elapsed time enters the decision, so a
        transfer that advances one MiB per tick is never called wedged however slow it is,
        and one that stops is called wedged however fast it was.
        """
        while not self.stop.wait(REPORT_SECONDS):
            for subject in list(self.monitor.cursors):
                if self.monitor.sample(subject) == "wedged":
                    self.note("liveness", f"{subject} is WEDGED by silence (no clock involved)")
            if self.connected.is_set():
                self.send(pb.WorkerMessage(report=self.report()))

    def report(self) -> pb.Report:
        """The Report tick. `capacity` is a ONEOF and it is scoped to the applied MODE.

        Setting both members is not "reporting both": in protobuf a oneof holds exactly
        one member, and the second assignment CLEARS the first. cr-009 added
        `job_capacity=` beside `serving_capacity=` in this constructor and every serving
        worker silently began advertising zero ready plans, so no coordinator could
        dispatch to it — output-verified by cl-006's A/B against 54effcc. There is no
        "both" to express and there never was: a Directive is a discriminated complete
        replacement, so a worker is in ONE mode and reports ONE capacity. Before any
        directive it reports NEITHER, which readers already see as empty — an honest
        "no claim yet" rather than a fabricated zero.
        """
        report = pb.Report(
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
        )
        if self.fences.mode == "job":
            report.job_capacity.CopyFrom(self.job_capacity())
        elif self.fences.mode == "serving":
            report.serving_capacity.CopyFrom(self.serving_capacity())
        return report

    def serving_capacity(self) -> pb.ServingCapacity:
        """Which entrypoint binding plans this worker can serve RIGHT NOW.

        The ready set is the executor's, not the directive's: a plan the hub named but the
        executor has not prepared is not ready, which is what makes READY mean
        first-request-servable rather than connected.
        """
        return pb.ServingCapacity(
            ready_entrypoint_binding_plan_ids=sorted(
                self.supervision.current.ready_plans if self.supervision.current else ()
            ),
            free_vram_bytes=0,
        )

    def job_capacity(self) -> pb.JobCapacity:
        """How many JOB attempts this worker can take right now (cr-009).

        Law 8 in the job lane: one active attempt per device group, so a worker with a live
        job attempt advertises zero available. It is only ever called in JOB mode: a
        serving worker does not report a zero here, it reports no job capacity at all,
        because the oneof cannot carry both and a zero would be a claim rather than a
        silence.
        """
        in_flight = sum(1 for a in self.engine.live.values() if a.job is not None)
        ready = bool(self.jobs) and self.fences.intake == pb.IntakeState.INTAKE_STATE_READY
        return pb.JobCapacity(
            jobs_in_flight=in_flight,
            jobs_available=max(0, (1 if ready else 0) - in_flight),
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

        The measurement itself belongs to `internal/hostfacts.py`, which the `doctor` verb
        reads too: a wedged driver has to read the same to a registering worker and to a
        user asking what is wrong, and two probes could disagree.
        """
        facts = hostfacts.measure()
        return pb.WorkerResources(
            gpu_count=facts.gpu_count,
            gpu_name=facts.gpu_name,
            gpu_sm=facts.gpu_sm,
            vram_total_bytes=facts.vram_total_bytes,
            driver_version=facts.driver_version,
            platform=facts.platform,
            host_ram_total_bytes=facts.host_ram_total_bytes,
            vcpu_count=facts.vcpu_count,
            pinned_capacity_bytes=self.ledger.pinned_budget,
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
        """One boot verdict, classified ONCE, with its invoice attached.

        A boot that cannot write its journal is a claim about the POD SHAPE, not a transient
        to retry: requeueing onto an identically-sized disk reproduces it exactly and burns
        the rental twice (th#2246). So the verdict travels with the numbers that justify it
        and the worker exits rather than looping.
        """
        invoice = self.disk_invoice()
        self.boot.step("load", "boot_failure", 0.0, reason=pb.BootFailureReason.Name(reason))
        return pb.BootFailure(
            worker_id=self.options.worker_id,
            instance_id=self.options.instance_id,
            reason=reason,
            detail=f"{detail} | disk invoice: {json.dumps(invoice, sort_keys=True)}"[:1024],
            resources=self.resources(),
            release_id=self.options.release_id,
        )

    def run(self) -> int:
        """Link, register, serve, relink. The session outlives the connection.

        The loop is the same whichever coordinator drives it — that is the point of the
        transport seam. What differs is `reconnects`: a network peer that drops the stream
        may come back and the worker redials the SAME session; an in-memory coordinator that
        ends the stream has ended the run, and redialling it would be a hang.
        """
        try:
            self.journal.append(
                "session",
                {"session_id": self.fences.session_id, "instance_id": self.options.instance_id},
            )
        except JournalError as exc:
            # The worker cannot durably commit anything, so it claims nothing and executes
            # nothing. It still SAYS SO, once, with the disk numbers behind the claim.
            failure = self.boot_failure(
                pb.BootFailureReason.BOOT_FAILURE_REASON_DISK_SHAPE, exc.detail
            )
            self.note("boot", f"BOOT FAILURE (disk shape): {failure.detail}")
            link = self.link = self.coordinator.link()
            with contextlib.suppress(LinkError):
                link.control(self.outbound_iterator())
                self.send(pb.WorkerMessage(boot_failure=failure))
                time.sleep(0.1)
            self.stop.set()
            self.outbound.put(None)
            link.cancel()
            self.journal.close()
            return int(Exit.internal)
        killed = self.supervision.sweep_orphans()
        if killed:
            self.note(
                "recovery",
                f"killed {len(killed)} orphan executor(s) from before this restart: {killed}",
            )
        threading.Thread(target=self.device_lane, daemon=True, name="device-lane").start()
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


def supervise(config: RuntimeConfig, argv: list[str] | None = None) -> int:
    """Parse the launch flags and run ONE supervisor against a real coordinator.

    Not an entrypoint: the config arrives from whoever already read it. `cozy-runtime serve`
    reads the environment for its own verb dispatch and then calls this, and a second read
    inside here refused `config_authority` — one config per process — so every worker the
    CLI launched exited 1 before it could register (cl-004).
    """
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


def main(argv: list[str] | None = None) -> int:
    """The PROCESS entrypoint: read the ONE config here, then supervise."""
    return supervise(read_config(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
