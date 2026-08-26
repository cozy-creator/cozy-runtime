#!/usr/bin/env python
"""The WEIGHTLESS worker lifecycle, live: the platform port's acceptance run (#450).

One run, six sections, no model, no CAS, no GPU: boot -> invoke -> cooperative cancel ->
executor crash -> supervisor kill + orphan check -> journal recovery. This is the sequence
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
import sys
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
        if message.WhichOneof("msg") != "report":
            continue
        out.extend(row for row in message.report.faults if row.kind == kind)
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
            m.attempt_terminal
            if m.WhichOneof("msg") == "attempt_terminal"
            and m.attempt_terminal.request_id == request_id
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
            lambda m: m.report if m.WhichOneof("msg") == "report" else None,
            args.observe,
            "report on the reconnected stream",
            since,
        )
        refused = faults(hub, pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, since)
        wl.check(
            "a reconnect presenting the same revision is accepted as UNCHANGED",
            not refused and report.applied_revision == 1,
            f"stream {hub.streams}, generation {hub.generation}, applied_revision "
            f"{report.applied_revision}, {len(refused)} CONFIG_REFUSED faults",
        )
        # RED: same revision, genuinely different desired state (a posture flip).
        since = hub.mark()
        changed = wl.serving_directive(RELEASE, sorted(ids.values()), pb.Posture.POSTURE_DRAINING)
        hub.send(pb.OwnerFrame(directive=hub._env(changed)))
        hub.wait(
            lambda m: (
                m.report
                if m.WhichOneof("msg") == "report"
                and any(
                    row.reason == "same_revision_changed_body"
                    for row in m.report.faults
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
            "the same fixture on a bindable root acks the claim and reaches READY",
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
                m.report
                if m.WhichOneof("msg") == "report"
                and m.report.intake_state == pb.IntakeState.INTAKE_STATE_ERROR
                else None
            ),
            args.observe,
            "the worker's ERROR edge",
            since,
        )
        # ORDERING, not a clock: the outcome is enqueued on the durable lane immediately
        # after the ERROR edge, on the same FIFO queue the reporter ticks into. Two report
        # ticks landing after that edge with no terminal between them is the observation
        # that no terminal was sent — waiting a wall-clock second would not be.
        for tick in range(2):
            hub.wait(
                lambda m: m.report if m.WhichOneof("msg") == "report" else None,
                args.observe,
                f"report tick {tick + 1} after the ERROR edge",
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
            "and the attempt is still HELD and still reported, never silently dropped",
            any(a.request_id == "req-dur" for a in latest.active_attempts)
            and held.intake_state == pb.IntakeState.INTAKE_STATE_ERROR,
            f"active_attempts={[a.request_id for a in latest.active_attempts]}, intake "
            f"{pb.IntakeState.Name(latest.intake_state)}",
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
            replay.terminal_canonical == first.terminal_canonical and len(hub2.snapshot) == 1,
            f"identity {minted} persisted; {len(hub2.snapshot)} snapshot entr(y|ies), "
            f"{len(first.terminal_canonical)} identical canonical bytes",
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
    """#494e: the accepted (owner_epoch, controller) survives the process that accepted it."""
    wl.head("12. ownership durability: the accepted epoch outlives the process")
    tmp = Path(base + "-ownership")
    hub = wl.Hub()
    hub.epoch, hub.controller_id = 5, "hub-authority"
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
            (4, "hub-superseded", pb.ClaimRejection.CLAIM_REJECTION_STALE_OWNER_EPOCH),
            (5, "hub-imposter", pb.ClaimRejection.CLAIM_REJECTION_EPOCH_HELD),
        ):
            probe = wl.Hub()
            probe.process, probe.log = worker2.process, worker2.log
            probe.epoch, probe.controller_id = epoch, controller
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
        keeper.epoch, keeper.controller_id = 5, "hub-authority"
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
            f"epoch 5 / hub-authority accepted at generation {ack.control_generation}",
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
            "an idle worker advertises exactly one admission credit",
            hub.deployment(report).attempt_credits == 1,
            f"attempt_credits={hub.deployment(report).attempt_credits}",
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
            "a dispatch over the window is a typed REFUSED terminal, not an invisible hold",
            body["status"] == 2 and "no_admission_credit" in body["safe_message"],
            f"status={body['status']} cause={body['cause']['code']}: {body['safe_message'][:90]}",
        )
        # The completeness half: reconnect while one attempt is held and count. Every
        # attempt this owner dispatched and has not closed must be in the snapshot — the
        # invisible queued offer the window used to permit is exactly what would be missing.
        since = hub.mark()
        hub.connect(wl.read_addr(worker.root, worker.process))
        if not hub.connected.wait(args.observe):
            raise SystemExit("the reconnected stream never opened dispatch")
        report = hub.wait(
            lambda m: m.report if m.WhichOneof("msg") == "report" else None,
            args.observe,
            "a report on the reconnected stream",
            since,
        )
        wl.check(
            "the snapshot names every attempt the worker holds, and no others",
            {a.request_id for a in hub.snapshot} == {"req-holder", "req-overflow"},
            f"snapshot={sorted(a.request_id for a in hub.snapshot)} "
            f"reported={sorted(a.request_id for a in report.active_attempts)} "
            f"credits={hub.deployment(report).attempt_credits}",
        )
        # Release the holder. NOT this arm's subject, and recorded because it is real:
        # dropping the control stream while an attempt runs already invalidates the executor
        # and ABANDONS that attempt — reproduced identically at 1a93ced and here — so the
        # reconnect above may find req-holder already terminal. This is cleanup, and a
        # cancel the worker answers with "no such held attempt" is a legal answer to it.
        hub.send(
            pb.OwnerFrame(
                cancel_attempt=hub._env(
                    pb.CancelAttempt(request_id="req-holder", attempt=1, invocation_digest=digest)
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
            print(f"\nboot never reached READY: {exc}\n--- worker log tail ---")
            for line in worker.tail(lines=40):
                print(f"  {line}")
            raise SystemExit(1) from exc
        boot_ms = (time.perf_counter() - t0) * 1000
        wl.check(
            "worker READY with both weightless plans",
            set(hub.deployment(report).ready_entrypoint_binding_plan_ids) >= set(ids.values()),
            f"{boot_ms:.0f} ms to READY; executor pid {worker.executor_pid()}",
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
                m.attempt_terminal
                if m.WhichOneof("msg") == "attempt_terminal"
                and m.attempt_terminal.request_id == "req-tick"
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
            pb.OwnerFrame(
                cancel_attempt=hub._env(
                    pb.CancelAttempt(
                        request_id="req-cancel",
                        attempt=1,
                        invocation_digest=digest,
                    )
                )
            )
        )
        terminal = hub.wait(
            lambda m: (
                m.attempt_terminal
                if m.WhichOneof("msg") == "attempt_terminal"
                and m.attempt_terminal.request_id == "req-cancel"
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
        # acceptance after the worker reports READY again (the rebuild is off the
        # cancellation critical path, so this waits for a FRESH report).
        hub.ready(ids["hold"], args.observe * 2, since=hub.mark())

        wl.head("4. executor crash: SIGKILL the executor mid-hold")
        # A refused attempt is the coordinator's to retry: re-dispatch a HIGHER attempt
        # after a fresh READY (a post-cancel dispatch can race the poison-triggered
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
                    else ("refused", m.attempt_terminal)
                    if m.WhichOneof("msg") == "attempt_terminal"
                    and m.attempt_terminal.request_id == "req-crash"
                    and wl.terminal_body(m.attempt_terminal)["status"] == 2
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
                m.attempt_terminal
                if m.WhichOneof("msg") == "attempt_terminal"
                and m.attempt_terminal.request_id == "req-crash"
                and m.attempt_terminal.attempt == attempt_no
                and wl.terminal_body(m.attempt_terminal)["status"] != 2
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
            f"cause={body.get('cause', {}).get('code', '')} supervisor_alive={alive}",
        )
        hub.ready(ids["hold"], args.observe * 2, since=hub.mark())
        wl.check("readiness rebuilt after the crash", True, "fresh READY observed")

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
        supervisor = worker.process.pid
        worker.process.send_signal(signal.SIGKILL)
        worker.process.wait(timeout=args.observe)
        wl.check(
            "parent-death containment reaps the executor",
            pid_gone(executor, args.observe, "orphaned executor"),
            f"worker control process {supervisor} SIGKILLed; executor {executor} gone "
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
                    m.attempt_terminal
                    if m.WhichOneof("msg") == "attempt_terminal"
                    and m.attempt_terminal.request_id == "req-orphan"
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
            wl.check("worker READY again after recovery", True, "second boot serves")
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
    ):
        section(args, args.tmp)

    total = time.perf_counter() - t0
    print(f"\nlifecycle: {wl.PASS} ok, {wl.FAIL} failed, {total:.1f} s total")
    return 1 if wl.FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
