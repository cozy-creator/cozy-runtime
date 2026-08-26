#!/usr/bin/env python
"""The WEIGHTLESS worker lifecycle, live: the platform port's acceptance run (#450).

One run, six sections, no model, no CAS, no GPU: boot -> invoke -> cooperative cancel ->
executor crash -> worker kill + orphan check -> journal recovery. This is the sequence
a platform port must pass before it may call itself a port (cr-020 items 1-5, cr-021's
process half): every section exercises the REAL worker control process, the REAL executor spawn path
(trampoline seal included), and the REAL protocol bytes — only the endpoint is weightless.

    nice -n 19 uv run python scripts/lifecycle-live.py
    nice -n 19 uv run python scripts/lifecycle-live.py --tmp /tmp/cozy-lifecycle

Reuses worker-live.py's coordinator (Hub), spawn and dispatch plumbing, but stages its own
weightless fixture and never touches tensorfs-bench, nvidia-smi or the card — which is what
lets it run on a hosted macOS or Windows runner where worker-live.py structurally cannot.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("worker_live", ROOT / "scripts" / "worker-live.py")
assert _spec is not None and _spec.loader is not None
wl = importlib.util.module_from_spec(_spec)
sys.modules["worker_live"] = wl
_spec.loader.exec_module(wl)

pb = wl.pb

RELEASE = "cozy/lifecycle@live"

#: The fixture endpoint, written into the run's tmp dir. `tick` returns immediately;
#: `hold` runs for payload.seconds checking the cooperative-cancel signal — the payload
#: value is each section's PLANT (how long the attempt would run if nobody stopped it),
#: not a timeout of the harness's.
ENDPOINT_SOURCE = '''\
"""Weightless lifecycle fixture: no model, no torch, no device."""

import os
import time

import msgspec

from cozy_runtime.author import Context
from cozy_runtime.author._app import App

app = App()


class TickInput(msgspec.Struct, forbid_unknown_fields=True):
    note: str = "tick"


class TickOutput(msgspec.Struct):
    note: str
    pid: int


@app.entrypoint
def tick(payload: TickInput) -> TickOutput:
    return TickOutput(note=payload.note, pid=os.getpid())


class HoldInput(msgspec.Struct, forbid_unknown_fields=True):
    seconds: float = 30.0


class HoldOutput(msgspec.Struct):
    held: float


@app.entrypoint
def hold(payload: HoldInput, ctx: Context) -> HoldOutput:
    started = time.monotonic()
    while time.monotonic() - started < payload.seconds:
        ctx.raise_if_cancelled()
        time.sleep(0.05)
    return HoldOutput(held=time.monotonic() - started)
'''

ENDPOINT_TOML = '[application]\nobject = "lifecycle_endpoint:app"\n'


def stage_weightless(home: Path, project: Path, entrypoints: tuple[str, ...]) -> dict[str, str]:
    """A weightless binding-plan record carries the four REQUIRED keys and NO model keys
    (plan.MODEL_KEYS — all or none, cl-013)."""
    project.mkdir(parents=True, exist_ok=True)
    (project / "lifecycle_endpoint.py").write_text(ENDPOINT_SOURCE)
    (project / "endpoint.toml").write_text(ENDPOINT_TOML)
    (home / "binding-plans").mkdir(parents=True, exist_ok=True)
    ids: dict[str, str] = {}
    for entrypoint in entrypoints:
        plan_id = wl.plan_id_for(entrypoint, RELEASE)
        record = {
            "entrypoint_binding_plan_id": plan_id,
            "project": str(project),
            "release": RELEASE,
            "entrypoint": entrypoint,
        }
        (home / "binding-plans" / f"{plan_id.split(':')[1]}.json").write_text(
            json.dumps(record, indent=2)
        )
        ids[entrypoint] = plan_id
    return ids


def start_weightless(
    tmp: Path,
    *,
    instance_id: str,
    fresh: bool,
    hub: Any = None,
    connect: bool = True,
    rlimit_fsize: int = 0,
    root_name: str = "worker",
    posture: Any = None,
) -> Any:
    """worker-live's spawn, minus the quiet-card gate (there is no card in this run) and
    minus the CAS staging. Same control-process argv, same env seal inputs, same log pump.

    `instance_id` EMPTY omits `--instance-id` entirely — that is the bare-restart shape
    #494d is about, and the harness has to be able to express it. `rlimit_fsize` is the same
    HARNESS-SIDE kernel fault injection worker-live uses: a real EFBIG on the one regular
    file the control process appends to, planted in the forked child before exec, so the
    runtime carries no fault switch of its own.
    """
    import resource
    import signal as _signal
    import subprocess
    import threading

    home, root, project = tmp / "home", tmp / root_name, tmp / "project"
    if fresh:
        shutil.rmtree(tmp, ignore_errors=True)
    home.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    ids = stage_weightless(home, project, ("tick", "hold"))
    socket_path = str(tmp / f"control-{len(os.listdir(tmp))}.sock")
    hub = hub or wl.Hub()
    hub.directive = wl.serving_directive(
        RELEASE, sorted(ids.values()), posture or pb.Posture.POSTURE_ACCEPTING
    )
    hub.revision = 1
    log = tmp / f"worker-{len(os.listdir(tmp))}.log"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "COZY_HOME": str(home),
        "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
    }
    (root / "control.addr").unlink(missing_ok=True)
    argv = [
        "nice",
        "-n",
        "19",
        sys.executable,
        "-c",
        "import sys; from cozy_runtime.internal.worker.session import main; "
        "raise SystemExit(main(sys.argv[1:]))",
        "--listen",
        f"unix:{socket_path}",
        "--root",
        str(root),
        "--project",
        str(project),
        *(("--instance-id", instance_id) if instance_id else ()),
        "--release-id",
        RELEASE,
        "--grace",
        "3.0",
        "--no-warm",
    ]

    def plant_disk_full() -> None:
        _signal.signal(_signal.SIGXFSZ, _signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (rlimit_fsize, rlimit_fsize))

    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=0,
        preexec_fn=plant_disk_full if rlimit_fsize else None,
    )
    threading.Thread(target=wl._pump, args=(process, log), daemon=True).start()
    hub.process, hub.log = process, log
    if connect:
        hub.connect(wl.read_addr(root, process))
    worker = wl.Worker(home, root, process, hub, log)
    return worker, ids


def executor_pid_of(worker: Any, budget: float) -> int:
    """The LIVE executor: the worker process's direct child (posix_spawn), by pgrep -P —
    the log's `pid` line names only incarnation 1, and a rebuild mints incarnation 2+.
    Observes until a child exists; `budget` bounds a FAILING run only."""
    import subprocess

    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        out = subprocess.run(
            ["pgrep", "-P", str(worker.process.pid)], capture_output=True, text=True
        )
        pids = [int(x) for x in out.stdout.split()]
        if pids:
            return pids[-1]
        time.sleep(0.05)
    raise SystemExit(f"no executor child under pid {worker.process.pid}")


def pid_gone(pid: int, budget: float, what: str) -> bool:
    """Observe a pid until it is gone. `budget` is this section's observation window —
    kernel signal delivery is near-immediate, so the window only bounds a FAILING run."""
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        time.sleep(0.05)
    print(f"  ... {what}: pid {pid} still alive after {budget:.0f}s window")
    return False


# --------------------------------------------------------------------- the control spine
#
# Sections 7-13 are the CONTROL SPINE arms (decisions #493 item 10, #494, #495d). They are
# here and not in worker-live.py because none of them needs a card, a model or the CAS: every
# one is about the claim fence, the journal, the snapshot and the durable lane, which a
# weightless worker exercises exactly as a loaded one does. Each arm is observed in BOTH
# directions — the plant produces the refusal, removing it produces the ordinary path — and
# every plant is a real kernel or filesystem condition, never a switch in the runtime.


def faults(hub: Any, kind: Any, since: int = 0) -> list[Any]:
    """Every fault of one kind the worker REPORTED after `since` (the durable lane)."""
    out: list[Any] = []
    for message in list(hub.seen)[since:]:
        if message.WhichOneof("msg") != "observed_state":
            continue
        out.extend(row for row in message.observed_state.faults if row.kind == kind)
    return out


def journal_prefix(journal: Path, kind: str) -> int:
    """Bytes of `journal` up to and including its LAST record of `kind`.

    This is the fault plant's calibration (section 9): a second run over the same fixture
    writes the same records at the same lengths, so an RLIMIT_FSIZE of exactly this many
    bytes lets acceptance commit and makes the very next append — the one that would carry
    the OUTCOME — fail with a real EFBIG.
    """
    total = through = 0
    for line in journal.read_bytes().split(b"\n"):
        if not line:
            continue
        total += len(line) + 1
        if json.loads(line.partition(b" ")[2])["kind"] == kind:
            through = total
    return through


def terminal_for(hub: Any, request_id: str, seconds: float, since: int = 0) -> Any:
    return hub.wait(
        lambda m: (
            m.attempt_outcome
            if m.WhichOneof("msg") == "attempt_outcome"
            and m.attempt_outcome.request_id == request_id
            else None
        ),
        seconds,
        f"{request_id} terminal",
        since,
    )


def section_reconnect(args: argparse.Namespace, base: str) -> None:
    """#494a: a reconnect re-presenting the SAME desired revision is not a changed body."""
    wl.head("7. reconnect: the same desired revision is UNCHANGED; a changed one still refuses")
    tmp = Path(base + "-reconnect")
    worker, ids = start_weightless(tmp, instance_id="ins-recon", fresh=True)
    hub = worker.hub
    try:
        hub.ready(ids["tick"], args.observe * 4)
        # GREEN: an ordinary reconnect. The owner re-sends revision 1 verbatim; only the
        # worker-minted control_generation moved, which is not desired state.
        since = hub.mark()
        hub.connect(wl.read_addr(worker.root, worker.process))
        report = hub.wait(
            lambda m: m.observed_state if m.WhichOneof("msg") == "observed_state" else None,
            args.observe,
            "report on the reconnected stream",
            since,
        )
        refused = faults(hub, pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, since)
        wl.check(
            "a reconnect presenting the same revision is accepted as UNCHANGED",
            not refused and report.accepted_desired_state_revision == 1,
            f"stream {hub.streams}, generation {hub.generation}, applied_revision "
            f"{report.accepted_desired_state_revision}, {len(refused)} CONFIG_REFUSED faults",
        )
        # RED: same revision, genuinely different desired state (a posture flip).
        since = hub.mark()
        changed = wl.serving_directive(RELEASE, sorted(ids.values()), pb.Posture.POSTURE_DRAINING)
        hub.send(pb.RecordOwnerFrame(desired_state=hub._env(changed)))
        hub.wait(
            lambda m: (
                m.observed_state
                if m.WhichOneof("msg") == "observed_state"
                and any(
                    row.reason == "same_revision_changed_body"
                    for row in m.observed_state.faults
                    if row.kind == pb.FaultKind.FAULT_KIND_CONFIG_REFUSED
                )
                else None
            ),
            args.observe,
            "the same-revision-changed-body refusal",
            since,
        )
        wl.check(
            "same revision + DIFFERENT desired state is still the permanent refusal",
            True,
            "FAULT_KIND_CONFIG_REFUSED/same_revision_changed_body observed",
        )
    finally:
        worker.stop()
        hub.close()


def section_boot_failure(args: argparse.Namespace, base: str) -> None:
    """#494c: BootFailure INSTEAD of ClaimAck, never an ack followed by a confession."""
    wl.head("8. boot failure: the claiming owner gets BootFailure and NO ClaimAck")
    tmp = Path(base + "-boot")
    # The plant is a real kernel limit: a journal root deep enough that the executor's
    # control socket would not fit `sockaddr_un.sun_path`, which is a boot-fatal
    # CONFIG_INVALID verdict long before any owner dials.
    worker, _ = start_weightless(
        tmp, instance_id="ins-bootfail", fresh=True, connect=False, root_name="d" * 90
    )
    hub = worker.hub
    try:
        hub.connect(wl.read_addr(worker.root, worker.process))
        # `connect` returns on the FIRST of ClaimAck or BootFailure, so what it saw is the
        # arm: an ack here means the worker acknowledged a claim it cannot serve, whatever
        # it confesses afterwards.
        verdicts = [m.boot_failure for m in list(hub.seen) if m.WhichOneof("msg") == "boot_failure"]
        acks = [m for m in list(hub.seen) if m.WhichOneof("msg") == "claim_ack"]
        wl.check(
            "an unbootable worker answers the claim with BootFailure and NOTHING else",
            not acks
            and len(verdicts) == 1
            and verdicts[0].reason == pb.BootFailureReason.BOOT_FAILURE_REASON_CONFIG_INVALID,
            f"{len(hub.seen)} frame(s) on the stream, {len(acks)} ClaimAck, "
            + (
                f"reason {pb.BootFailureReason.Name(verdicts[0].reason)}"
                if verdicts
                else "no BootFailure"
            ),
        )
        wl.check(
            "and it delivered no snapshot to reconcile against a worker that never booted",
            not hub.snapshot,
            f"{len(hub.snapshot)} snapshot entries",
        )
    finally:
        worker.stop()
        hub.close()
    # GREEN: the same fixture on a root that fits acks the claim and serves.
    worker2, ids2 = start_weightless(Path(base + "-boot-ok"), instance_id="ins-bootok", fresh=True)
    try:
        worker2.hub.ready(ids2["tick"], args.observe * 4)
        wl.check(
            "the same fixture on a bindable root acks the claim and becomes DISPATCHABLE",
            not [m for m in list(worker2.hub.seen) if m.WhichOneof("msg") == "boot_failure"],
            f"boot {worker2.hub.boot_id}, no BootFailure on the stream",
        )
    finally:
        worker2.stop()
        worker2.hub.close()


def section_undurable(args: argparse.Namespace, base: str) -> None:
    """#494b: an outcome that could not be journaled is HELD, never sent."""
    wl.head("9. an outcome that cannot be journaled is HELD; nothing unreplayable is sent")
    tmp = Path(base + "-durable")
    # GREEN first, and it doubles as the plant's calibration: the same fixture writes the
    # same journal records at the same lengths, so acceptance's byte offset is the limit
    # that lets acceptance commit and makes the OUTCOME's append fail.
    worker, ids = start_weightless(tmp, instance_id="ins-durable", fresh=True)
    try:
        worker.hub.ready(ids["tick"], args.observe * 4)
        wl.dispatch(
            worker.hub,
            tmp,
            ids["tick"],
            request_id="req-dur",
            payload={"note": "x"},
            release=RELEASE,
            outputs=0,
        )
        body = wl.terminal_body(terminal_for(worker.hub, "req-dur", args.observe))
        wl.check(
            "with a writable journal the same dispatch lands a SUCCEEDED terminal",
            body["status"] == 1,
            f"status={body['status']}",
        )
    finally:
        worker.stop()
        worker.hub.close()
    limit = journal_prefix(tmp / "worker" / "journal.log", "accepted") + 8
    tmp2 = Path(base + "-durable-red")
    worker2, ids2 = start_weightless(
        tmp2, instance_id="ins-durable", fresh=True, rlimit_fsize=limit
    )
    hub = worker2.hub
    try:
        hub.ready(ids2["tick"], args.observe * 4)
        since = hub.mark()
        wl.dispatch(
            hub,
            tmp2,
            ids2["tick"],
            request_id="req-dur",
            payload={"note": "x"},
            release=RELEASE,
            outputs=0,
        )
        hub.wait(
            lambda m: (
                m.attempt_accepted
                if m.WhichOneof("msg") == "attempt_accepted"
                and m.attempt_accepted.request_id == "req-dur"
                else None
            ),
            args.observe,
            "acceptance under the planted limit",
            since,
        )
        # The outcome's append is the one that hits EFBIG. Observe the worker settle into
        # its typed local fault instead of shipping something it cannot replay.
        held = hub.wait(
            lambda m: (
                m.observed_state
                if m.WhichOneof("msg") == "observed_state"
                and m.observed_state.worker_phase == pb.WorkerPhase.WORKER_PHASE_FAILED
                else None
            ),
            args.observe,
            "the worker's FAILED edge",
            since,
        )
        # ORDERING, not a clock: the outcome is enqueued on the durable lane immediately
        # after the FAILED edge, on the same FIFO queue the reporter ticks into. Two ticks
        # landing after that edge with no outcome between them is the observation that no
        # outcome was sent — waiting a wall-clock second would not be.
        for tick in range(2):
            hub.wait(
                lambda m: m.observed_state if m.WhichOneof("msg") == "observed_state" else None,
                args.observe,
                f"observed-state tick {tick + 1} after the FAILED edge",
                hub.mark(),
            )
        wl.check(
            "an unjournalable outcome is NEVER sent",
            not hub.terminals("req-dur"),
            f"{len(hub.terminals('req-dur'))} terminals for req-dur under a journal that "
            f"stopped at {limit} B; worker log: "
            + (worker2.tail("HOLDING", 1)[-1][:110] if worker2.tail("HOLDING", 1) else "-"),
        )
        latest = hub.reports()[-1]
        wl.check(
            "and the attempt is HELD in its OWN wire state, never silently dropped",
            any(
                a.request_id == "req-dur"
                and a.state == pb.AttemptState.ATTEMPT_STATE_HELD_UNDURABLE
                for a in latest.held_attempts
            )
            and held.worker_phase == pb.WorkerPhase.WORKER_PHASE_FAILED,
            "held_attempts="
            + str([(a.request_id, pb.AttemptState.Name(a.state)) for a in latest.held_attempts])
            + f", phase {pb.WorkerPhase.Name(latest.worker_phase)} — rev-2 spells this "
            "HELD_UNDURABLE (#507d/#510d), not RUNNING plus a compound fault",
        )
    finally:
        worker2.stop()
        hub.close()


def section_identity(args: argparse.Namespace, base: str) -> None:
    """#494d: a bare restart recovers its OWN rows; a foreign instance is typed LOSS."""
    wl.head("10. instance identity: a bare restart recovers its own journal rows")
    tmp = Path(base + "-identity")
    worker, ids = start_weightless(tmp, instance_id="", fresh=True)
    hub = worker.hub
    hub.auto_ack = False  # leave the terminal unacked, so recovery has something to find
    try:
        hub.ready(ids["tick"], args.observe * 4)
        wl.dispatch(
            hub,
            tmp,
            ids["tick"],
            request_id="req-ident",
            payload={"note": "x"},
            release=RELEASE,
            outputs=0,
        )
        first = terminal_for(hub, "req-ident", args.observe)
    finally:
        worker.process.send_signal(signal.SIGKILL)
        worker.process.wait(timeout=args.observe)
        hub.close()
    minted = (tmp / "worker" / "instance-id").read_text().strip()
    hub2 = wl.Hub()
    hub2.auto_ack = False
    worker2, _ = start_weightless(tmp, instance_id="", fresh=False, hub=hub2)
    try:
        replay = terminal_for(hub2, "req-ident", args.observe * 2)
        wl.check(
            "a restart with NO --instance-id reuses the root's identity and replays its own row",
            replay.outcome_canonical_bytes == first.outcome_canonical_bytes
            and len(hub2.snapshot) == 1,
            f"identity {minted} persisted; {len(hub2.snapshot)} snapshot entr(y|ies), "
            f"{len(first.outcome_canonical_bytes)} identical canonical bytes",
        )
    finally:
        worker2.stop()
        hub2.close()

    wl.head("11. a GENUINELY foreign instance is a typed LOSS outcome, never silence")
    hub3 = wl.Hub()
    hub3.auto_ack = False
    worker3, _ = start_weightless(tmp, instance_id="ins-someone-else", fresh=False, hub=hub3)
    try:
        loss = terminal_for(hub3, "req-ident", args.observe * 2)
        body = wl.terminal_body(loss)
        wl.check(
            "a row authorized under another instance becomes a durable ABANDONED outcome",
            body["status"] == 5
            and body["cause"]["code"] == 16
            and "foreign_instance_identity" in body["safe_message"]
            and len(hub3.snapshot) == 1,
            f"status={body['status']} cause={body['cause']['code']}; {body['safe_message'][:90]}",
        )
    finally:
        worker3.stop()
        hub3.close()


def section_ownership(args: argparse.Namespace, base: str) -> None:
    """#494e: the accepted (record_owner_epoch, id) survives the process that accepted it."""
    wl.head("12. ownership durability: the accepted epoch outlives the process")
    tmp = Path(base + "-ownership")
    hub = wl.Hub()
    hub.epoch, hub.record_owner_id = 5, "hub-authority"
    worker, ids = start_weightless(tmp, instance_id="ins-owner", fresh=True, hub=hub)
    try:
        hub.ready(ids["tick"], args.observe * 4)
    finally:
        worker.process.send_signal(signal.SIGKILL)
        worker.process.wait(timeout=args.observe)
        hub.close()
    worker2, ids2 = start_weightless(tmp, instance_id="ins-owner", fresh=False, connect=False)
    addr = wl.read_addr(worker2.root, worker2.process)
    try:
        for epoch, controller, expect in (
            (4, "hub-superseded", pb.ClaimRejection.CLAIM_REJECTION_STALE_RECORD_OWNER_EPOCH),
            (5, "hub-imposter", pb.ClaimRejection.CLAIM_REJECTION_EPOCH_HELD),
        ):
            probe = wl.Hub()
            probe.process, probe.log = worker2.process, worker2.log
            probe.epoch, probe.record_owner_id = epoch, controller
            probe.connect(addr, claim_only=True)
            ack = probe.wait(
                lambda m: m.claim_ack if m.WhichOneof("msg") == "claim_ack" else None,
                args.observe,
                "the claim verdict",
            )
            wl.check(
                f"after a restart, epoch {epoch} as {controller!r} is refused "
                f"{pb.ClaimRejection.Name(expect)}",
                not ack.accepted and ack.rejection == expect,
                f"accepted={ack.accepted} rejection={pb.ClaimRejection.Name(ack.rejection)}",
            )
            probe.close()
        keeper = wl.Hub()
        keeper.epoch, keeper.record_owner_id = 5, "hub-authority"
        keeper.process, keeper.log = worker2.process, worker2.log
        keeper.directive = wl.serving_directive(
            RELEASE, sorted(ids2.values()), pb.Posture.POSTURE_ACCEPTING
        )
        keeper.connect(addr)
        ack = keeper.wait(
            lambda m: m.claim_ack if m.WhichOneof("msg") == "claim_ack" else None,
            args.observe,
            "the keeper's claim verdict",
        )
        wl.check(
            "and the owner that HELD the epoch reclaims it",
            ack.accepted,
            f"epoch 5 / hub-authority accepted at generation {ack.control_stream_generation}",
        )
        keeper.close()
    finally:
        worker2.stop()
        worker2.hub.close()


def section_window(args: argparse.Namespace, base: str) -> None:
    """#494f: the admission window is ONE and there is no invisible queue behind it."""
    wl.head("13. the admission window is one, and the snapshot holds everything held")
    tmp = Path(base + "-window")
    worker, ids = start_weightless(tmp, instance_id="ins-window", fresh=True)
    hub = worker.hub
    # No acks: both attempts stay the worker's, so the snapshot has to name both. An acked
    # terminal would leave the snapshot legitimately, which would weaken the count.
    hub.auto_ack = False
    try:
        report = hub.ready(ids["tick"], args.observe * 4)
        wl.check(
            "an idle worker advertises exactly one FREE SEAT in one shared window",
            report.available_attempt_slots == 1
            and report.admission_state == pb.AdmissionState.ADMISSION_STATE_OPEN,
            f"available_attempt_slots={report.available_attempt_slots} in an "
            f"{pb.AdmissionState.Name(report.admission_state)} window at generation "
            f"{report.admission_generation} — one WORKER-level fence, not N per-placement "
            "credits over one serialized device (#472e)",
        )
        since = hub.mark()
        _, digest = wl.dispatch(
            hub,
            tmp,
            ids["hold"],
            request_id="req-holder",
            payload={"seconds": 120.0},
            release=RELEASE,
            outputs=0,
        )
        hub.wait(
            lambda m: (
                m.attempt_accepted
                if m.WhichOneof("msg") == "attempt_accepted"
                and m.attempt_accepted.request_id == "req-holder"
                else None
            ),
            args.observe,
            "the held attempt's acceptance",
            since,
        )
        since = hub.mark()
        wl.dispatch(
            hub,
            tmp,
            ids["tick"],
            request_id="req-overflow",
            payload={"note": "x"},
            release=RELEASE,
            outputs=0,
        )
        body = wl.terminal_body(terminal_for(hub, "req-overflow", args.observe, since))
        wl.check(
            "an offer over the window is a JOURNALED REFUSED outcome, not an invisible hold",
            body["status"] == 2
            and body["cause"]["code"] == 17  # CAUSE_CODE_NO_CAPACITY
            and body["cause"]["origin"] == 4  # CAUSE_ORIGIN_SUPERVISOR
            and not body.get("execution_started", False),
            f"status={body['status']} cause={body['cause']['code']}/"
            f"origin={body['cause']['origin']} execution_started="
            f"{body.get('execution_started', False)} — the ordinal is consumed and the "
            f"BUDGET is not: {body['safe_message'][:70]}",
        )
        # The completeness half: reconnect while one attempt is held and count. Every
        # attempt this owner dispatched and has not closed must be in the snapshot — the
        # invisible queued offer the window used to permit is exactly what would be missing.
        since = hub.mark()
        hub.connect(wl.read_addr(worker.root, worker.process))
        if not hub.connected.wait(args.observe):
            raise SystemExit("the reconnected stream never opened dispatch")
        report = hub.wait(
            lambda m: m.observed_state if m.WhichOneof("msg") == "observed_state" else None,
            args.observe,
            "a report on the reconnected stream",
            since,
        )
        wl.check(
            "the snapshot names every attempt the worker holds, and no others",
            {a["request_id"] for a in hub.snapshot} == {"req-holder", "req-overflow"},
            f"snapshot={sorted(a['request_id'] for a in hub.snapshot)} "
            f"reported={sorted(a.request_id for a in report.held_attempts)} "
            f"slots={report.available_attempt_slots}",
        )
        # Release the holder. NOT this arm's subject, and recorded because it is real:
        # dropping the control stream while an attempt runs already invalidates the executor
        # and ABANDONS that attempt — reproduced identically at 1a93ced and here — so the
        # reconnect above may find req-holder already terminal. This is cleanup, and a
        # cancel the worker answers with "no such held attempt" is a legal answer to it.
        hub.send(
            pb.RecordOwnerFrame(
                cancel_attempt=hub._env(
                    pb.CancelAttempt(
                        request_id="req-holder",
                        attempt_ordinal=1,
                        invocation_spec_digest=digest,
                    )
                )
            )
        )
    finally:
        worker.stop()
        hub.close()


def section_live_delivery(args: argparse.Namespace, base: str) -> None:
    """The live-delivery race: a reconnect must not swallow the next stream's frames."""
    wl.head("14. after a replay-bearing claim, a NEW attempt's terminal arrives LIVE")
    tmp = Path(base + "-delivery")
    worker, ids = start_weightless(tmp, instance_id="ins-delivery", fresh=True)
    hub = worker.hub
    hub.auto_ack = False  # every claim's snapshot replays this, so every round carries one
    try:
        hub.ready(ids["tick"], args.observe * 4)
        wl.dispatch(
            hub,
            tmp,
            ids["tick"],
            request_id="req-unacked",
            payload={"note": "x"},
            release=RELEASE,
            outputs=0,
        )
        terminal_for(hub, "req-unacked", args.observe)
        addr = wl.read_addr(worker.root, worker.process)
        delivered = 0
        for round_no in range(1, 5):
            hub.connect(addr)  # a fresh claim; its snapshot replays the unacked terminal
            # `connect` returns on ClaimAck; the SnapshotAck that OPENS dispatch is sent by
            # the reader thread a moment later, and a StartAttempt that beats it is dropped
            # by the barrier — correctly. Wait for the barrier, then race the sender.
            if not hub.connected.wait(args.observe):
                raise SystemExit(f"round {round_no}: dispatch never opened")
            since = hub.mark()
            request = f"req-live-{round_no}"
            wl.dispatch(
                hub,
                tmp,
                ids["tick"],
                request_id=request,
                payload={"note": "x"},
                release=RELEASE,
                outputs=0,
            )
            try:
                terminal_for(hub, request, args.observe, since)
                delivered += 1
            except (TimeoutError, RuntimeError):
                print(f"  ... round {round_no}: {request} produced no terminal on its stream")
        wl.check(
            "every round's terminal is delivered on the stream that dispatched it",
            delivered == 4,
            f"{delivered}/4 rounds delivered live; the superseded stream's sender drains "
            "its OWN queue and can no longer consume the live stream's frames",
        )
        journaled = [
            line
            for line in worker.log.read_text(errors="replace").splitlines()
            if "outlived its stream" in line
        ]
        wl.check(
            "and no frame is dropped as having outlived its stream",
            not journaled,
            f"{len(journaled)} drop notes in the worker log",
        )
    finally:
        worker.stop()
        hub.close()


def section_claim_convergence_fence(args: argparse.Namespace, base: str) -> None:
    """A claim waits for convergence; an old stream never finishes under the new fence."""
    wl.head("15. claim, snapshot, and convergence are one ownership transition")
    tmp = Path(base + "-claim-convergence")
    worker, ids = start_weightless(tmp, instance_id="ins-claim-convergence", fresh=True)
    old = worker.hub
    successor = wl.Hub()
    successor.process, successor.log = worker.process, worker.log
    successor.directive = wl.serving_directive(
        RELEASE, [ids["tick"]], pb.Posture.POSTURE_ACCEPTING, revision=2
    )
    entered = tmp / "replacement-entered"
    release = tmp / "replacement-release"
    connected = threading.Event()
    connect_error: list[BaseException] = []

    def claim() -> None:
        try:
            successor.connect(wl.read_addr(worker.root, worker.process))
        except BaseException as exc:
            connect_error.append(exc)
        finally:
            connected.set()

    try:
        old.ready(ids["tick"], args.observe * 4)
        old_generation = old.generation

        # The changed binding set forces a fresh executor. Its real endpoint import records
        # that it entered, then waits for the harness to release it; the worker is therefore
        # blocked inside the actual prepare RPC, halfway through revision-2 convergence.
        gate = (
            "import time as _time\n"
            "from pathlib import Path as _Path\n"
            f"_Path({str(entered)!r}).write_text('entered')\n"
            f"while not _Path({str(release)!r}).exists(): _time.sleep(0.01)\n"
        )
        (tmp / "project" / "lifecycle_endpoint.py").write_text(gate + ENDPOINT_SOURCE)
        old.send(pb.RecordOwnerFrame(desired_state=old._env(successor.directive)))
        deadline = time.monotonic() + args.observe
        while not entered.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not entered.exists():
            raise TimeoutError("the replacement executor never entered its endpoint import")

        claimant = threading.Thread(target=claim, daemon=True, name="claim-during-convergence")
        claimant.start()
        time.sleep(0.5)
        early_acks = [m for m in list(successor.seen) if m.WhichOneof("msg") == "claim_ack"]
        wl.check(
            "a successor claim cannot acknowledge while the previous stream is converging",
            not early_acks and not connected.is_set(),
            f"convergence is blocked in a real prepare RPC; successor emitted {len(early_acks)} "
            "ClaimAck frame(s)",
        )

        release.touch()
        claimant.join(args.observe * 4)
        if claimant.is_alive():
            raise TimeoutError("the successor claim did not complete after convergence")
        if connect_error:
            raise connect_error[0]
        if not successor.connected.wait(args.observe):
            raise TimeoutError("the successor never acknowledged its snapshot")

        snapshot = successor.snapshot_document
        wl.check(
            "the successor snapshot is wholly after convergence, never an intermediate state",
            int(snapshot.get("accepted_desired_state_revision", 0)) == 2
            and int(snapshot.get("converged_revision", 0)) == 2,
            f"snapshot accepted={snapshot.get('accepted_desired_state_revision')} "
            f"converged={snapshot.get('converged_revision')}",
        )

        old_generations = {
            getattr(frame, frame.WhichOneof("msg")).control_stream_generation
            for frame in list(old.seen)
            if frame.WhichOneof("msg") is not None
        }
        successor_generations = {
            getattr(frame, frame.WhichOneof("msg")).control_stream_generation
            for frame in list(successor.seen)
            if frame.WhichOneof("msg") is not None
        }
        wl.check(
            "each stream's outgoing frames retain that stream's generation",
            successor.generation > old_generation
            and old_generations <= {old_generation}
            and successor_generations <= {successor.generation},
            f"old stream generations={sorted(old_generations)}, successor generations="
            f"{sorted(successor_generations)}",
        )
    finally:
        release.touch()
        successor.close()
        worker.stop()
        old.close()


def section_generation_replacement(args: argparse.Namespace, base: str) -> None:
    """cr-024: exact set reuse, changed set replacement, then empty-set retirement."""
    wl.head("16. executor generations: reuse exact, replace changed, retire empty")
    tmp = Path(base + "-generation")
    worker, ids = start_weightless(tmp, instance_id="ins-generation", fresh=True)
    hub = worker.hub
    try:
        first_report = hub.ready(ids["tick"], args.observe * 2)
        first_pid = executor_pid_of(worker, args.observe)
        first_generation = hub.deployment(first_report).executor_generation

        since = hub.mark()
        exact = wl.serving_directive(
            RELEASE, sorted(ids.values()), pb.Posture.POSTURE_ACCEPTING, revision=2
        )
        hub.send(pb.RecordOwnerFrame(desired_state=hub._env(exact)))
        exact_report = hub.ready(ids["tick"], args.observe * 2, since=since)
        exact_pid = executor_pid_of(worker, args.observe)
        exact_generation = hub.deployment(exact_report).executor_generation
        wl.check(
            "reapplying the exact prepared construction/binding set reuses its process",
            (exact_pid, exact_generation) == (first_pid, first_generation),
            f"pid {first_pid}->{exact_pid}, generation {first_generation}->{exact_generation}",
        )

        since = hub.mark()
        changed = wl.serving_directive(
            RELEASE, [ids["tick"]], pb.Posture.POSTURE_ACCEPTING, revision=3
        )
        hub.send(pb.RecordOwnerFrame(desired_state=hub._env(changed)))
        changed_report = hub.ready(ids["tick"], args.observe * 2, since=since)
        changed_pid = executor_pid_of(worker, args.observe)
        changed_generation = hub.deployment(changed_report).executor_generation
        wl.check(
            "a changed binding set publishes one successor only after the old PID is gone",
            changed_pid != first_pid
            and changed_generation == first_generation + 1
            and pid_gone(first_pid, args.observe, "replaced executor"),
            f"pid {first_pid}->{changed_pid}, generation {first_generation}->{changed_generation}",
        )

        since = hub.mark()
        empty_set = pb.PlacementSet()
        empty_bytes, empty_digest = wl.documents.identity(empty_set)
        empty = pb.DesiredWorkerState(
            revision=4,
            posture=pb.Posture.POSTURE_ACCEPTING,
            wire_minor=wl.WIRE_MINOR,
            placement_set=pb.DesiredPlacementSet(
                placement_set_digest=empty_digest,
                placement_set_canonical_bytes=empty_bytes,
            ),
        )
        hub.send(pb.RecordOwnerFrame(desired_state=hub._env(empty)))
        hub.wait(
            lambda m: (
                m.observed_state
                if m.WhichOneof("msg") == "observed_state"
                and m.observed_state.accepted_desired_state_revision == 4
                else None
            ),
            args.observe * 2,
            "empty desired state applied",
            since,
        )
        wl.check(
            "an explicit empty placement set retires without spawning an unused successor",
            pid_gone(changed_pid, args.observe, "empty-state executor"),
            f"pid {changed_pid} is absent and the worker owns no executor child",
        )
    finally:
        worker.stop()
        hub.close()


def section_supervision_atomicity(args: argparse.Namespace, base: str) -> None:
    """cr-024: exercise supervision directly, including races the wire serializes."""
    from cozy_runtime.internal import accel, proctree
    from cozy_runtime.internal.worker import child as child_module
    from cozy_runtime.internal.worker.attempts import AttemptEngine, AttemptRecord
    from cozy_runtime.internal.worker.child import (
        ExecutorGone,
        ExecutorSupervision,
        OwnerRecord,
    )

    wl.head("16. executor supervision: refuse stacking, single-flight replacement")
    tmp = Path(base + "-supervision")
    shutil.rmtree(tmp, ignore_errors=True)
    home, root, project = tmp / "home", tmp / "worker", tmp / "project"
    home.mkdir(parents=True)
    root.mkdir(parents=True)
    stage_weightless(home, project, ("tick",))
    detached_file = tmp / "detached.pid"
    endpoint = project / "lifecycle_endpoint.py"
    endpoint.write_text(
        endpoint.read_text()
        + f"""\

# Plant a real executor descendant which defeats process-group containment twice: it
# creates a new session and moves into a nested cgroup. The executor's parent cgroup must
# still kill it, and cleanup must remove its now-empty nested cgroup.
_detached = os.fork()
if _detached == 0:
    os.setsid()
    _relative = next(
        line.split("::", 1)[1]
        for line in open("/proc/self/cgroup").read().splitlines()
        if line.startswith("0::")
    )
    _nested = __import__("pathlib").Path("/sys/fs/cgroup") / _relative.lstrip("/") / "nested"
    _nested.mkdir()
    (_nested / "cgroup.procs").write_text("0\\n")
    __import__("pathlib").Path({str(detached_file)!r}).write_text(str(os.getpid()))
    while True:
        time.sleep(60)
_deadline = time.monotonic() + 10
while not __import__("pathlib").Path({str(detached_file)!r}).exists():
    if time.monotonic() >= _deadline:
        raise RuntimeError("detached descendant did not start")
    time.sleep(0.01)
"""
    )
    base_env = tuple(
        sorted(
            {
                "HOME": os.environ["HOME"],
                "PATH": os.environ["PATH"],
                "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
            }.items()
        )
    )
    imposed = {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTORCH_CUDA_ALLOC_CONF": "",
        "OMP_NUM_THREADS": "1",
    }
    supervision = ExecutorSupervision(
        root=root,
        python=sys.executable,
        base_env=base_env,
        cozy_home=home,
        device_process=lambda _pid: accel.ProcessMemory("absent", 0),
    )
    first = supervision.spawn(imposed=imposed)
    intruder_code = (
        "from pathlib import Path\n"
        "from cozy_runtime.internal.worker.child import ExecutorGone,ExecutorSupervision\n"
        "try:\n"
        f" ExecutorSupervision(root=Path({str(root)!r}),python={sys.executable!r},"
        f"base_env=(),cozy_home=Path({str(home)!r}))\n"
        "except ExecutorGone:\n"
        " raise SystemExit(73)\n"
    )
    intruder = subprocess.run(
        [sys.executable, "-c", intruder_code],
        env=dict(base_env),
        capture_output=True,
        text=True,
        timeout=args.observe,
        check=False,
    )
    wl.check(
        "a concurrent supervision on the same worker root refuses before sweep or spawn",
        intruder.returncode == 73
        and supervision.current is first
        and first.alive()
        and proctree.cgroup_contains(first.cgroup, first.process),
        f"first pid {first.pid} remains the exact live cgroup owner",
    )
    refused = False
    try:
        supervision.spawn(imposed=imposed)
    except ExecutorGone:
        refused = True
    wl.check(
        "spawn refuses while an exact current child is still owned",
        refused and supervision.current is first and first.alive(),
        f"incarnation {first.incarnation} pid {first.pid} remains current and live",
    )

    first_reply = first.call(
        {
            "cmd": "prepare",
            "binding": {"project": str(project), "model_class": ""},
            "budgets": {},
        }
    )
    second_reply = first.call({"cmd": "prepare", "binding": {}, "budgets": {}})
    wl.check(
        "the child refuses a second prepare before another construction can allocate",
        first_reply.get("ok") is True and second_reply.get("code") == "generation_already_prepared",
        f"first={first_reply.get('ok')} second={second_reply.get('code')}",
    )

    detached = proctree.process_identity(int(detached_file.read_text()))

    barrier = threading.Barrier(3)
    successors: list[Any] = []
    errors: list[BaseException] = []

    def replace() -> None:
        barrier.wait()
        try:
            successors.append(supervision.replace(first, imposed=imposed, why="concurrent test"))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=replace) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(args.observe)
    successor = supervision.current
    wl.check(
        "two concurrent replacements of E1 publish exactly one E2",
        len(errors) == 1
        and isinstance(errors[0], ExecutorGone)
        and len(successors) == 1
        and successor is not None
        and successors[0] is successor
        and supervision.spawns == 2
        and pid_gone(first.pid, args.observe, "concurrently replaced executor"),
        f"results={len(successors)} errors={errors} spawns={supervision.spawns} "
        f"successor_pid={successor.pid if successor else 0}",
    )
    wl.check(
        "replacement kills a setsid descendant in a nested cgroup and proves it absent",
        pid_gone(detached.pid, args.observe, "detached executor descendant"),
        f"setsid descendant birth {detached} is absent before E2 is used",
    )
    assert successor is not None
    wl.check(
        "a stale E1 invalidation cannot poison or withdraw E2",
        not supervision.invalidate(first, "stale E1 observation")
        and not supervision.cooperative_cancel(first, "req-stale#1")
        and supervision.current is successor
        and successor.alive()
        and not successor.poisoned,
        f"current pid {successor.pid}, poisoned={successor.poisoned!r}",
    )
    successor_pid = successor.pid
    supervision.close()
    wl.check(
        "shutdown reaps the one executor supervision still owns",
        supervision.current is None and pid_gone(successor_pid, args.observe, "shutdown executor"),
        f"successor pid {successor_pid} absent after close",
    )

    broken_root = tmp / "broken"
    broken_root.mkdir()
    broken = ExecutorSupervision(
        root=broken_root,
        python="/bin/false",
        base_env=base_env,
        cozy_home=home,
    )
    failed_cleanly = False
    try:
        broken.spawn(imposed=imposed)
    except ExecutorGone:
        failed_cleanly = broken.current is None
    finally:
        broken.close()
    wl.check(
        "a child that exits before dial-back is reaped and never published",
        failed_cleanly,
        f"spawn attempts={broken.spawns}, current={broken.current}",
    )

    reclaim_proven = False

    def reclaim_observation(_pid: int) -> accel.ProcessMemory:
        return accel.ProcessMemory("absent" if reclaim_proven else "unreadable")

    unreadable_root = tmp / "unreadable"
    unreadable_root.mkdir()
    guarded = ExecutorSupervision(
        root=unreadable_root,
        python=sys.executable,
        base_env=base_env,
        cozy_home=home,
        device_process=reclaim_observation,
    )
    guarded_executor = guarded.spawn(imposed=imposed)
    failed_prepare = guarded_executor.call(
        {
            "cmd": "prepare",
            "binding": {"project": str(tmp / "missing-project"), "model_class": ""},
            "budgets": {},
        }
    )
    refused_retry = guarded_executor.call({"cmd": "prepare", "binding": {}, "budgets": {}})
    wl.check(
        "a failed first prepare still consumes and poisons the generation",
        not failed_prepare.get("ok") and refused_retry.get("code") == "generation_already_prepared",
        f"first={failed_prepare.get('code')} second={refused_retry.get('code')}",
    )
    previous_reclaim_seconds = child_module.RECLAIM_SECONDS
    child_module.RECLAIM_SECONDS = 0.2
    refused_unproven = False
    blocked_detail = "reclaim unexpectedly returned"
    cancel = AttemptRecord("req-reclaim", 1, b"digest", {}, executor=guarded_executor)
    cancel.state = "running"
    cancel.canceling = "client"
    cancel.executor_generation = guarded_executor.incarnation
    cancel_engine: Any = AttemptEngine.__new__(AttemptEngine)
    cancel_engine.supervision = guarded
    try:
        cancel_engine.escalate(cancel, 0.2)
    except ExecutorGone:
        retained_owner = guarded._read_owner()
        refused_unproven = (
            guarded.current is guarded_executor
            and cancel.state == "reclaiming"
            and not cancel.reclaim_done.wait(0.05)
            and retained_owner is not None
            and bool(retained_owner.members)
        )
        blocked_detail = (
            f"state={cancel.state} done={cancel.reclaim_done.is_set()} "
            f"owner_retained={guarded.current is guarded_executor} "
            f"durable_members={len(retained_owner.members) if retained_owner else 0} "
            f"pid={guarded_executor.pid}"
        )
    finally:
        reclaim_proven = True
        child_module.RECLAIM_SECONDS = previous_reclaim_seconds
    finalization_finished = threading.Event()

    def buffered_reply() -> None:
        cancel.await_reclaim_before_finalizing()
        finalization_finished.set()

    reply_thread = threading.Thread(target=buffered_reply, daemon=True)
    reply_thread.start()
    buffered_reply_blocked = not finalization_finished.wait(0.05)
    completed_after_proof = False
    try:
        cancel_engine.escalate(cancel, 0.2)
        reply_thread.join(1)
        completed_after_proof = (
            cancel.reclaim_done.is_set()
            and finalization_finished.is_set()
            and cancel.state == "finalizing"
            and guarded.current is None
        )
    finally:
        guarded.close()
    wl.check(
        "an unreadable driver observation blocks cancel and a buffered final reply",
        refused_unproven and buffered_reply_blocked,
        blocked_detail,
    )
    wl.check(
        "the same cancel completes only after a retry proves exact driver reclaim",
        completed_after_proof,
        f"done={cancel.reclaim_done.is_set()} current={guarded.current}",
    )

    # A restart can find the durable leader gone while a detached descendant remains. The
    # exact cgroup, unlike a numeric PGID, is still safe authority to reclaim that subtree.
    orphan_root = tmp / "leaderless-owner"
    orphan_root.mkdir()
    leaderless = ExecutorSupervision(
        root=orphan_root,
        python=sys.executable,
        base_env=base_env,
        cozy_home=home,
        device_process=lambda _pid: accel.ProcessMemory("absent", 0),
    )
    owner = OwnerRecord(
        proctree.create_executor_cgroup(proctree.executor_cgroup_namespace(orphan_root))
    )
    leaderless._write_owner(owner)
    helper_file = orphan_root / "helper.pid"
    release_file = orphan_root / "release-leader"
    leader_code = (
        "import os,time,pathlib\n"
        f"scope=pathlib.Path('/sys/fs/cgroup') / {owner.cgroup.relative_path.lstrip('/')!r}\n"
        "(scope/'cgroup.procs').write_text('0\\n')\n"
        "child=os.fork()\n"
        "if child == 0:\n"
        " os.setsid()\n"
        " time.sleep(60)\n"
        " os._exit(0)\n"
        f"open({str(helper_file)!r},'w').write(str(child))\n"
        f"while not os.path.exists({str(release_file)!r}): time.sleep(0.01)\n"
        "os._exit(0)\n"
    )
    leader_pid = os.posix_spawn(
        sys.executable,
        [sys.executable, "-c", leader_code],
        os.environ,
        setpgroup=0,
    )
    leader = proctree.process_identity(leader_pid)
    owner = leaderless._bind_owner(owner, leader)
    deadline = time.monotonic() + args.observe
    while not helper_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    helper = proctree.process_identity(int(helper_file.read_text()))
    release_file.touch()
    os.waitpid(leader_pid, 0)

    try:
        killed = leaderless.sweep_orphans()
    finally:
        leaderless.close()
    wl.check(
        "restart recovery reclaims a detached descendant after its exact leader is gone",
        killed == [] and pid_gone(helper.pid, args.observe, "leaderless descendant"),
        f"leader {leader.pid} absent; detached helper birth {helper} was reclaimed by scope",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmp", default="/tmp/cozy-lifecycle")
    parser.add_argument(
        "--observe",
        type=float,
        default=30.0,
        help="per-section observation window in seconds — bounds a FAILING run only; a "
        "passing section completes on its own progress, never by this clock",
    )
    args = parser.parse_args()
    tmp = Path(args.tmp)
    t0 = time.perf_counter()

    wl.head("1. boot: weightless worker, no card, no CAS")
    worker, ids = start_weightless(tmp, instance_id="ins-lifecycle", fresh=True)
    hub = worker.hub
    try:
        try:
            report = hub.ready(ids["tick"], args.observe * 4)
        except (TimeoutError, RuntimeError) as exc:
            # On a platform whose ProcessTree/AcceleratorOps seams are unfilled, boot
            # refuses TYPED in the worker log — surface it, because that refusal is the
            # port's honest status (#447/#450), not a mystery hang.
            print(f"\nboot never became DISPATCHABLE: {exc}\n--- worker log tail ---")
            for line in worker.tail(lines=40):
                print(f"  {line}")
            raise SystemExit(1) from exc
        boot_ms = (time.perf_counter() - t0) * 1000
        wl.check(
            "worker DISPATCHABLE with both weightless plans",
            set(hub.deployment(report).dispatchable_plan_ids) >= set(ids.values()),
            f"{boot_ms:.0f} ms to DISPATCHABLE; executor pid {worker.executor_pid()}",
        )

        wl.head("2. invoke: tick -> ok terminal")
        t1 = time.perf_counter()
        wl.dispatch(
            hub,
            tmp,
            ids["tick"],
            request_id="req-tick",
            payload={"note": "alive"},
            release=RELEASE,
            outputs=0,
        )
        terminal = hub.wait(
            lambda m: (
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-tick"
                else None
            ),
            args.observe,
            "tick terminal",
        )
        body = wl.terminal_body(terminal)
        wl.check(
            "tick lands a SUCCEEDED terminal",
            body["status"] == 1,
            f"{(time.perf_counter() - t1) * 1000:.0f} ms dispatch->terminal, "
            f"status={body['status']}",
        )

        wl.head("3. cancel: hold(120s plant) stopped cooperatively")
        since = hub.mark()
        _, digest = wl.dispatch(
            hub,
            tmp,
            ids["hold"],
            request_id="req-cancel",
            payload={"seconds": 120.0},
            release=RELEASE,
            outputs=0,
        )
        hub.wait(
            lambda m: (
                m.attempt_accepted
                if m.WhichOneof("msg") == "attempt_accepted"
                and m.attempt_accepted.request_id == "req-cancel"
                else None
            ),
            args.observe,
            "hold accepted",
            since,
        )
        t2 = time.perf_counter()
        hub.send(
            pb.RecordOwnerFrame(
                cancel_attempt=hub._env(
                    pb.CancelAttempt(
                        request_id="req-cancel",
                        attempt_ordinal=1,
                        invocation_spec_digest=digest,
                    )
                )
            )
        )
        terminal = hub.wait(
            lambda m: (
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-cancel"
                else None
            ),
            args.observe,
            "cancel terminal",
        )
        body = wl.terminal_body(terminal)
        wl.check(
            "hold lands a CANCELED terminal with a client-cancel cause",
            body["status"] == 4 and body["cause"]["code"] == 11,
            f"{(time.perf_counter() - t2) * 1000:.0f} ms cancel->terminal, "
            f"status={body['status']} cause={body.get('cause', {}).get('code', '')}",
        )
        # A cancel can poison the serving generation; the next dispatch is only owed
        # acceptance after the worker reports DISPATCHABLE again (the rebuild is off the
        # cancellation critical path, so this waits for a FRESH report).
        hub.ready(ids["hold"], args.observe * 2, since=hub.mark())

        wl.head("4. executor crash: SIGKILL the executor mid-hold")
        # A refused attempt is the coordinator's to retry: re-dispatch a HIGHER attempt
        # after a fresh DISPATCHABLE (a post-cancel dispatch can race the poison-triggered
        # rebuild, and the refusal is typed, not a failure of anything).
        attempt_no = 0
        while True:
            attempt_no += 1
            if attempt_no > 3:
                raise SystemExit("req-crash refused three times — the worker never converged")
            since = hub.mark()
            wl.dispatch(
                hub,
                tmp,
                ids["hold"],
                request_id="req-crash",
                attempt=attempt_no,
                payload={"seconds": 120.0},
                release=RELEASE,
                outputs=0,
            )
            outcome = hub.wait(
                lambda m: (
                    ("accepted", m.attempt_accepted)
                    if m.WhichOneof("msg") == "attempt_accepted"
                    and m.attempt_accepted.request_id == "req-crash"
                    else ("refused", m.attempt_outcome)
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-crash"
                    and wl.terminal_body(m.attempt_outcome)["status"] == 2
                    else None
                ),
                args.observe,
                "crash-hold accepted or refused",
                since,
            )
            if outcome[0] == "accepted":
                break
            hub.ready(ids["hold"], args.observe * 2, since=hub.mark())
        executor = executor_pid_of(worker, args.observe)
        os.kill(executor, signal.SIGKILL)
        terminal = hub.wait(
            lambda m: (
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-crash"
                and m.attempt_outcome.attempt_ordinal == attempt_no
                and wl.terminal_body(m.attempt_outcome)["status"] != 2
                else None
            ),
            args.observe,
            "crash terminal",
        )
        body = wl.terminal_body(terminal)
        alive = worker.process.poll() is None
        wl.check(
            "executor SIGKILL -> ABANDONED/EXECUTOR_INVALIDATED, worker process survives",
            body["status"] == 5 and body["cause"]["code"] == 16 and alive,
            f"killed pid {executor}; status={body['status']} "
            f"cause={body.get('cause', {}).get('code', '')} worker_alive={alive}",
        )
        hub.ready(ids["hold"], args.observe * 2, since=hub.mark())
        wl.check("dispatchability rebuilt after the crash", True, "fresh DISPATCHABLE observed")

        wl.head("5. orphan: SIGKILL the WORKER control process mid-hold; the executor must die")
        since = hub.mark()
        wl.dispatch(
            hub,
            tmp,
            ids["hold"],
            request_id="req-orphan",
            payload={"seconds": 120.0},
            release=RELEASE,
            outputs=0,
        )
        hub.wait(
            lambda m: (
                m.attempt_accepted
                if m.WhichOneof("msg") == "attempt_accepted"
                and m.attempt_accepted.request_id == "req-orphan"
                else None
            ),
            args.observe,
            "orphan-hold accepted",
            since,
        )
        executor = executor_pid_of(worker, args.observe)
        control_pid = worker.process.pid
        worker.process.send_signal(signal.SIGKILL)
        worker.process.wait(timeout=args.observe)
        wl.check(
            "parent-death containment reaps the executor",
            pid_gone(executor, args.observe, "orphaned executor"),
            f"worker control process {control_pid} SIGKILLed; executor {executor} gone "
            "(pdeathsig on linux; the seam-EOF exit is the portable backstop)",
        )
        hub.close()

        wl.head("6. recovery: restart on the same root, journal converges")
        t3 = time.perf_counter()
        worker2, ids2 = start_weightless(tmp, instance_id="ins-lifecycle", fresh=False)
        hub2 = worker2.hub
        try:
            terminal = hub2.wait(
                lambda m: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-orphan"
                    else None
                ),
                args.observe * 2,
                "recovered terminal for the interrupted attempt",
            )
            body = wl.terminal_body(terminal)
            wl.check(
                "the interrupted attempt converges to a typed terminal on restart",
                body["status"] in (3, 5),
                f"{(time.perf_counter() - t3) * 1000:.0f} ms restart->replayed terminal, "
                f"status={body['status']} cause={body.get('cause', {}).get('code', '')}",
            )
            hub2.ready(ids2["tick"], args.observe * 2)
            wl.check("worker DISPATCHABLE again after recovery", True, "second boot serves")
        finally:
            worker2.stop()
            hub2.close()
    finally:
        if worker.process.poll() is None:
            worker.stop()

    for section in (
        section_reconnect,
        section_boot_failure,
        section_undurable,
        section_identity,
        section_ownership,
        section_window,
        section_live_delivery,
        section_claim_convergence_fence,
        section_generation_replacement,
        section_supervision_atomicity,
    ):
        section(args, args.tmp)

    total = time.perf_counter() - t0
    print(f"\nlifecycle: {wl.PASS} ok, {wl.FAIL} failed, {total:.1f} s total")
    return 1 if wl.FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
