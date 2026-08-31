#!/usr/bin/env python
"""The WEIGHTLESS worker lifecycle, live: the platform port's acceptance run (#450).

One run, six core sections, no model, no CAS, no GPU: boot -> invoke -> cooperative cancel ->
executor crash -> worker kill + orphan check -> fresh process, then the control-spine and
executor-liveness and egress arms (`--only <name>` runs one of those alone). This is the sequence
a platform port must pass before it may call itself a port (cr-020 items 1-5, cr-021's
process half): every section exercises the REAL worker control process, the REAL executor spawn path
(trampoline seal included), and the REAL protocol bytes — only the package is weightless.

    nice -n 19 uv run python scripts/lifecycle-live.py
    nice -n 19 uv run python scripts/lifecycle-live.py --tmp /tmp/cozy-lifecycle
    nice -n 19 uv run python scripts/lifecycle-live.py --only egress

Reuses worker-live.py's RecordOwner (Hub), spawn and dispatch plumbing, but stages its own
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

#: The fixture package, written into the run's tmp dir. `tick` returns immediately;
#: `hold` runs for payload.seconds checking the cooperative-cancel signal — the payload
#: value is each section's PLANT (how long the attempt would run if nobody stopped it),
#: not a timeout of the harness's. `dial`/`dial_native` are section 17's egress plants.
PACKAGE_SOURCE = '''\
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


class DialInput(msgspec.Struct, forbid_unknown_fields=True):
    port: int


class DialOutput(msgspec.Struct):
    rc: int
    how: str


@app.entrypoint
def dial(payload: DialInput) -> DialOutput:
    """cr-042 section 17's PLANT: package code dialling a reachable address, in Python."""
    import socket

    sock = socket.create_connection(("127.0.0.1", payload.port), timeout=5)
    sock.close()
    return DialOutput(rc=0, how="socket.create_connection")


@app.entrypoint
def dial_native(payload: DialInput) -> DialOutput:
    """The SAME dial, below CPython. `ctypes` reaches `connect(2)` without raising any audit
    event, so this measures exactly what the executor's Python fence does NOT bound."""
    import ctypes
    import socket
    import struct

    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    if fd < 0:
        return DialOutput(rc=-1, how="libc.socket failed")
    address = (
        struct.pack("H", socket.AF_INET)
        + struct.pack("!H", payload.port)
        + socket.inet_aton("127.0.0.1")
        + bytes(8)
    )
    rc = libc.connect(fd, ctypes.create_string_buffer(address, 16), 16)
    libc.close(fd)
    return DialOutput(rc=int(rc), how="ctypes -> libc connect(2)")
'''

PACKAGE_TOML = '[application]\nobject = "lifecycle_package:app"\n'


def write_weightless_project(project: Path) -> None:
    project.mkdir(parents=True, exist_ok=True)
    (project / "lifecycle_package.py").write_text(PACKAGE_SOURCE)
    (project / "package.toml").write_text(PACKAGE_TOML)


def stage_development(home: Path, project: Path) -> Any:
    """Stage this fixture through the production local development constructor."""
    from cozy_runtime.internal import descriptor, local
    from cozy_runtime.internal.discovery import discover

    write_weightless_project(project)
    found = discover(project)
    staged = local.stage_development_placement(
        home,
        descriptor.build(found),
        package="local/development-package",
        release="0.0.0",
        source_digest=local.development_source_digest(project),
    )
    staged.placement.placement_id = wl.PLACEMENT
    return staged


def start_weightless(
    tmp: Path,
    *,
    fresh: bool,
    hub: Any = None,
    connect: bool = True,
    root_name: str = "worker",
    posture: Any = None,
) -> Any:
    """worker-live's spawn, minus the quiet-card gate (there is no card in this run) and
    minus the CAS staging. Same control-process argv, same env seal inputs, same log pump.
    """
    import subprocess
    import threading

    home, root, project = tmp / "home", tmp / root_name, tmp / "project"
    if fresh:
        shutil.rmtree(tmp, ignore_errors=True)
    home.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    write_weightless_project(project)
    from cozy_runtime.internal.local import development_source_digest

    source_digest = development_source_digest(project)
    socket_path = str(tmp / f"control-{len(os.listdir(tmp))}.sock")
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "COZY_HOME": str(home),
        "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
    }
    resolved = subprocess.run(
        [
            sys.executable,
            "-m",
            "cozy_runtime.cli.main",
            "--dir",
            str(project),
            "--json",
            "bindings",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=project,
    )
    if resolved.returncode:
        raise SystemExit(
            f"weightless binding resolution exited {resolved.returncode}: {resolved.stderr.strip()}"
        )
    binding_result = json.loads(resolved.stdout)
    subjects = binding_result.get("weightless_plans", [])
    if binding_result.get("bindings") != [] or not subjects:
        raise SystemExit(f"weightless binding resolution returned {binding_result!r}")
    ids = {str(row["entrypoint"]): str(row["digest"]) for row in subjects}
    staged = stage_development(home, project)
    if ids != dict(staged.bindings):
        raise SystemExit(f"bindings output differs from development Placement: {ids!r}")
    hub = hub or wl.Hub()
    hub.package_revision_digest = source_digest
    hub.development = True
    hub.revision = 1
    log = tmp / f"worker-{len(os.listdir(tmp))}.log"
    (root / "control.addr").unlink(missing_ok=True)
    argv = [
        "nice",
        "-n",
        "19",
        sys.executable,
        "-m",
        "cozy_runtime.cli.main",
        "serve",
        "--socket",
        socket_path,
        "--out",
        str(root),
        "--development-project",
        str(project),
        "--development-package",
        "local/development-package",
        "--development-release",
        "0.0.0",
        "--development-source-digest",
        source_digest,
        "--release-id",
        RELEASE,
        "--grace",
        "3.0",
        "--no-warm",
    ]

    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=project,
        bufsize=0,
    )
    threading.Thread(target=wl._pump, args=(process, log), daemon=True).start()
    hub.process, hub.log = process, log
    addr = wl.read_addr(root, process)
    hub.directive = wl.serving_directive(
        staged.placement,
        posture or pb.Posture.POSTURE_ACCEPTING,
    )
    if connect:
        hub.connect(addr)
    worker = wl.Worker(home, root, process, hub, log, ids)
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
# The remaining sections are same-process control and executor-lifecycle arms. They are
# here and not in worker-live.py because none of them needs a card, a model or the CAS: every
# one is about the claim fence, snapshot and control lane, which a
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
    worker, ids = start_weightless(tmp, fresh=True)
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
        changed = wl.serving_directive(
            stage_development(worker.home, Path(base + "-reconnect") / "project").placement,
            pb.Posture.POSTURE_DRAINING,
        )
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
    # The plant is a real kernel limit: a worker root deep enough that the executor's
    # control socket would not fit `sockaddr_un.sun_path`, which is a boot-fatal
    # CONFIG_INVALID verdict long before any owner dials.
    worker, _ = start_weightless(tmp, fresh=True, connect=False, root_name="d" * 90)
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
    worker2, ids2 = start_weightless(Path(base + "-boot-ok"), fresh=True)
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


def section_window(args: argparse.Namespace, base: str) -> None:
    """#494f: the admission window is ONE and there is no invisible queue behind it."""
    wl.head("13. the admission window is one, and the snapshot holds everything held")
    tmp = Path(base + "-window")
    worker, ids = start_weightless(tmp, fresh=True)
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
            and body["cause"]["origin"] == 4  # CAUSE_ORIGIN_WORKER
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
    worker, ids = start_weightless(tmp, fresh=True)
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
            except TimeoutError, RuntimeError:
                print(f"  ... round {round_no}: {request} produced no terminal on its stream")
        wl.check(
            "every round's terminal is delivered on the stream that dispatched it",
            delivered == 4,
            f"{delivered}/4 rounds delivered live; the superseded stream's sender drains "
            "its OWN queue and can no longer consume the live stream's frames",
        )
        dropped = [
            line
            for line in worker.log.read_text(errors="replace").splitlines()
            if "outlived its stream" in line
        ]
        wl.check(
            "and no frame is dropped as having outlived its stream",
            not dropped,
            f"{len(dropped)} drop notes in the worker log",
        )
    finally:
        worker.stop()
        hub.close()


def section_claim_convergence_fence(args: argparse.Namespace, base: str) -> None:
    """A claim stays responsive during convergence and snapshots both honest clocks."""
    wl.head("15. claim and snapshot stay live while convergence works off-stream")
    tmp = Path(base + "-claim-convergence")
    worker, ids = start_weightless(tmp, fresh=True)
    old = worker.hub
    successor = wl.Hub()
    successor.process, successor.log = worker.process, worker.log
    successor.directive = wl.serving_directive(
        RELEASE + "-replacement",
        sorted(ids.values()),
        pb.Posture.POSTURE_ACCEPTING,
        worker.home / "binding-plans",
        revision=2,
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

        # The changed release identity forces a fresh executor while retaining the complete
        # visible-entrypoint plan set the canonical reader requires. Its real import records
        # that it entered, then waits for the harness to release it; the worker is therefore
        # blocked inside the actual prepare RPC, halfway through revision-2 convergence.
        gate = (
            "import time as _time\n"
            "from pathlib import Path as _Path\n"
            f"_Path({str(entered)!r}).write_text('entered')\n"
            f"while not _Path({str(release)!r}).exists(): _time.sleep(0.01)\n"
        )
        (tmp / "project" / "lifecycle_package.py").write_text(gate + PACKAGE_SOURCE)
        old.send(pb.RecordOwnerFrame(desired_state=old._env(successor.directive)))
        deadline = time.monotonic() + args.observe
        while not entered.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not entered.exists():
            raise TimeoutError("the replacement executor never entered its package import")

        claimant = threading.Thread(target=claim, daemon=True, name="claim-during-convergence")
        claimant.start()
        claimant.join(args.observe)
        early_acks = [m for m in list(successor.seen) if m.WhichOneof("msg") == "claim_ack"]
        wl.check(
            "a successor claim ACKS while slow convergence remains blocked",
            len(early_acks) == 1 and connected.is_set(),
            f"convergence is blocked in a real prepare RPC; successor emitted "
            f"{len(early_acks)} ClaimAck frame(s)",
        )
        if connect_error:
            raise connect_error[0]
        if not successor.connected.wait(args.observe):
            raise TimeoutError("the successor never acknowledged its snapshot")

        snapshot = successor.snapshot_document
        wl.check(
            "the successor snapshot reports accepted intent ahead of observed convergence",
            int(snapshot.get("accepted_desired_state_revision", 0)) == 2
            and int(snapshot.get("converged_revision", 0)) == 1,
            f"snapshot accepted={snapshot.get('accepted_desired_state_revision')} "
            f"converged={snapshot.get('converged_revision')}",
        )

        release.touch()
        converged = successor.ready(ids["tick"], args.observe * 4)
        wl.check(
            "the off-stream lane converges after the blocked import is released",
            converged.converged_revision == 2,
            f"converged_revision={converged.converged_revision}",
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
    """cr-024: exact release reuse, changed release replacement, then retirement."""
    wl.head("16. executor generations: reuse exact, replace changed release, retire empty")
    tmp = Path(base + "-generation")
    worker, ids = start_weightless(tmp, fresh=True)
    hub = worker.hub
    try:
        first_report = hub.ready(ids["tick"], args.observe * 2)
        first_pid = executor_pid_of(worker, args.observe)
        first_generation = hub.deployment(first_report).executor_generation

        since = hub.mark()
        exact = wl.serving_directive(
            RELEASE,
            sorted(ids.values()),
            pb.Posture.POSTURE_ACCEPTING,
            worker.home / "binding-plans",
            revision=2,
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
            RELEASE + "-replacement",
            sorted(ids.values()),
            pb.Posture.POSTURE_ACCEPTING,
            worker.home / "binding-plans",
            revision=3,
        )
        hub.send(pb.RecordOwnerFrame(desired_state=hub._env(changed)))
        changed_report = hub.ready(ids["tick"], args.observe * 2, since=since)
        changed_pid = executor_pid_of(worker, args.observe)
        changed_generation = hub.deployment(changed_report).executor_generation
        wl.check(
            "a changed release publishes one successor only after the old PID is gone",
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


def section_idle_executor_liveness(args: argparse.Namespace, base: str) -> None:
    """cr-024: idle death withdraws capacity and rebuilds without an offer."""

    wl.head("16. idle executor liveness: withdraw immediately, rebuild on the device lane")
    tmp = Path(base + "-idle-liveness")
    worker, ids = start_weightless(tmp, fresh=True)
    hub = worker.hub
    try:
        initial = hub.ready(ids["hold"], args.observe * 2)
        first = hub.deployment(initial)
        first_pid = executor_pid_of(worker, args.observe)
        first_generation = first.executor_generation
        first_admission = initial.admission_generation
        since = hub.mark()

        os.kill(first_pid, signal.SIGKILL)
        withdrawn = hub.wait(
            lambda m: (
                m.observed_state
                if m.WhichOneof("msg") == "observed_state"
                and m.observed_state.admission_state == pb.AdmissionState.ADMISSION_STATE_CLOSED
                and hub.deployment(m.observed_state).serving
                == pb.ServingState.SERVING_STATE_ACTIVATING
                and not hub.deployment(m.observed_state).dispatchable_binding_digests
                else None
            ),
            args.observe,
            "idle executor capacity withdrawal",
            since,
        )
        wl.check(
            "idle child exit withdraws admission before any offer observes stale capacity",
            withdrawn.admission_generation > first_admission,
            f"pid {first_pid}, admission {first_admission}->{withdrawn.admission_generation}, "
            f"serving={pb.ServingState.Name(hub.deployment(withdrawn).serving)}",
        )

        recovered = hub.ready(ids["hold"], args.observe * 2, since=since)
        second_pid = executor_pid_of(worker, args.observe)
        second_generation = hub.deployment(recovered).executor_generation
        wl.check(
            "the exact idle death rebuilds automatically with no offer as a trigger",
            second_pid != first_pid
            and second_generation == first_generation + 1
            and pid_gone(first_pid, args.observe, "idle dead executor")
            and not hub.terminals(),
            f"pid {first_pid}->{second_pid}, generation "
            f"{first_generation}->{second_generation}, outcomes={len(hub.terminals())}",
        )

        since = hub.mark()
        wl.dispatch(
            hub,
            tmp,
            ids["hold"],
            request_id="req-exit-race",
            payload={"seconds": 120.0},
            release=RELEASE,
            outputs=0,
        )
        hub.wait(
            lambda m: (
                m.attempt_accepted
                if m.WhichOneof("msg") == "attempt_accepted"
                and m.attempt_accepted.request_id == "req-exit-race"
                else None
            ),
            args.observe,
            "exit-race attempt accepted",
            since,
        )
        os.kill(second_pid, signal.SIGKILL)
        terminal = hub.wait(
            lambda m: (
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-exit-race"
                else None
            ),
            args.observe,
            "exit-race outcome",
            since,
        )
        body = wl.terminal_body(terminal)
        raced_recovery = hub.ready(ids["hold"], args.observe * 2, since=since)
        third_pid = executor_pid_of(worker, args.observe)
        third_generation = hub.deployment(raced_recovery).executor_generation
        outcomes = hub.terminals("req-exit-race", 1)
        wl.check(
            "watcher and attempt-side death observations produce one outcome and one rebuild",
            body["status"] == 5
            and body["cause"]["code"] == 16
            and len(outcomes) == 1
            and third_pid != second_pid
            and third_generation == second_generation + 1,
            f"status={body['status']} cause={body.get('cause', {}).get('code')} "
            f"outcomes={len(outcomes)} pid {second_pid}->{third_pid}, generation "
            f"{second_generation}->{third_generation}",
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
    staged = stage_development(home, project)
    detached_file = tmp / "detached.pid"
    package = project / "lifecycle_package.py"
    package.write_text(
        package.read_text()
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
    previous_cwd = Path.cwd()
    os.chdir(project)
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
        and isinstance(first.scope, proctree.CgroupScope)
        and proctree.cgroup_contains(first.scope, first.process),
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
            "binding": {
                "application": "lifecycle_package:app",
                "package_descriptor": str(staged.descriptor),
                "model_class": "",
            },
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
            "binding": {
                "application": "missing_package:app",
                "package_descriptor": str(staged.descriptor),
                "model_class": "",
            },
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
        proctree.create_executor_cgroup(proctree.executor_scope_namespace(orphan_root))
    )
    assert isinstance(owner.scope, proctree.CgroupScope)
    leaderless._write_owner(owner)
    helper_file = orphan_root / "helper.pid"
    release_file = orphan_root / "release-leader"
    leader_code = (
        "import os,time,pathlib\n"
        f"scope=pathlib.Path('/sys/fs/cgroup') / {owner.scope.relative_path.lstrip('/')!r}\n"
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
    os.chdir(previous_cwd)


def section_executor_egress(args: argparse.Namespace, base: str) -> None:
    """cr-042: what the executor's "no outbound network" claim IS, both halves, live.

    The executor runs in PID 1's network namespace with no seccomp filter, so a reachable
    address really is reachable from it — this section establishes that FIRST, by dialling
    the harness's own listener from inside the real serving executor through `ctypes`, and
    watching the listener accept. Only then is the Python dial run: same address, same
    process, refused by `sandbox.refuse_network` before a packet leaves.

    Both halves are asserted because either alone is a lie. The refusal alone would read as
    "the executor cannot reach the network"; the connection alone would read as "the fence
    does nothing". Together they are the exact claim the comments in `internal/executor.py`
    and `internal/worker/grants.py` make: a bound on a BUGGY package at the Python layer,
    and no bound at all on a hostile one (law 15's third rung, unarmed and named).
    """
    import socket as _socket
    import threading

    wl.head("17. egress: a Python dial REFUSES; the same dial below CPython does not")
    tmp = Path(base + "-egress")
    listener = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    listener.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    port = listener.getsockname()[1]
    accepted: list[str] = []

    def accept_forever() -> None:
        while True:
            try:
                conn, peer = listener.accept()
            except OSError:
                return
            accepted.append(f"{peer[0]}:{peer[1]}")
            conn.close()

    threading.Thread(target=accept_forever, daemon=True).start()

    worker, ids = start_weightless(tmp, fresh=True)
    hub = worker.hub
    try:
        hub.ready(ids["dial_native"], args.observe * 4)
        wl.dispatch(
            hub,
            tmp,
            ids["dial_native"],
            request_id="req-dial-native",
            payload={"port": port},
            release=RELEASE,
            outputs=0,
        )
        body = wl.terminal_body(terminal_for(hub, "req-dial-native", args.observe))
        deadline = time.monotonic() + 5.0
        while not accepted and time.monotonic() < deadline:
            time.sleep(0.02)
        wl.check(
            "THE PREMISE: the executor CAN reach the network — a ctypes connect(2) lands",
            body["status"] == 1 and len(accepted) == 1,
            f"status={body['status']}, the harness listener on 127.0.0.1:{port} accepted "
            f"{accepted or 'nothing'} from inside the executor: no netns, no seccomp",
        )

        since = hub.mark()
        hub.ready(ids["dial"], args.observe * 4, since=since)
        wl.dispatch(
            hub,
            tmp,
            ids["dial"],
            request_id="req-dial",
            payload={"port": port},
            release=RELEASE,
            outputs=0,
        )
        body = wl.terminal_body(terminal_for(hub, "req-dial", args.observe, since))
        said = json.dumps(body)
        wl.check(
            "THE FENCE: the same address dialled through Python is refused, typed",
            body["status"] == 3 and "no_network" in said,
            f"status={body['status']}, {body.get('cause', {}).get('safe_message', said)[:160]}",
        )
        wl.check(
            "and no packet left: the listener still shows only the ctypes connection",
            len(accepted) == 1,
            f"accepted={accepted} — the refusal fires on the first NETWORK_EVENTS spelling "
            "the dial reaches (`socket.getaddrinfo` here), before any connect(2)",
        )
    finally:
        worker.stop()
        hub.close()
        listener.close()


#: The post-core sections, by name. `--only <name>` runs one without the boot/invoke/cancel
#: spine, which is what makes a single arm cheap to rerun on a shared machine.
SECTIONS = {
    "reconnect": section_reconnect,
    "boot-failure": section_boot_failure,
    "window": section_window,
    "live-delivery": section_live_delivery,
    "claim-convergence": section_claim_convergence_fence,
    "generation-replacement": section_generation_replacement,
    "idle-liveness": section_idle_executor_liveness,
    "supervision-atomicity": section_supervision_atomicity,
    "egress": section_executor_egress,
}


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
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="stop after boot/invoke/cancel/executor-crash/worker-crash/fresh-process",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(SECTIONS),
        default=[],
        help="run only these post-core sections, skipping the core spine",
    )
    parser.add_argument(
        "--expect-containment",
        choices=("cgroup_v2", "process_tree"),
        help="require the production selector to publish this durable owner backend",
    )
    args = parser.parse_args()
    tmp = Path(args.tmp)
    t0 = time.perf_counter()

    if args.only:
        for name in args.only:
            SECTIONS[name](args, args.tmp)
        total = time.perf_counter() - t0
        print(
            f"\nlifecycle {'+'.join(args.only)}: {wl.PASS} ok, {wl.FAIL} failed, "
            f"{total:.1f} s total"
        )
        return 1 if wl.FAIL else 0

    wl.head("1. boot: weightless worker, no card, no CAS")
    worker, ids = start_weightless(tmp, fresh=True)
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
            set(hub.deployment(report).dispatchable_binding_digests)
            >= {wl.documents.raw(value) for value in ids.values()},
            f"{boot_ms:.0f} ms to DISPATCHABLE; executor pid {worker.executor_pid()}",
        )
        owner = json.loads((worker.root / "executor.owner").read_text())
        backend = owner.get("scope", {}).get("backend")
        if args.expect_containment:
            wl.check(
                f"the production selector chose {args.expect_containment}",
                backend == args.expect_containment,
                f"durable owner backend={backend}",
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
        # A refused attempt is the RecordOwner's to retry: re-dispatch a HIGHER attempt
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

        wl.head("6. fresh process: Runtime owns no restart state")
        worker2, ids2 = start_weightless(tmp, fresh=False)
        hub2 = worker2.hub
        try:
            wl.check(
                "the fresh Runtime snapshot contains no previous-process attempt",
                not hub2.snapshot and not hub2.terminals("req-orphan"),
                f"snapshot={hub2.snapshot}, terminals={len(hub2.terminals('req-orphan'))}; "
                "the outer RecordOwner owns settlement",
            )
            hub2.ready(ids2["tick"], args.observe * 2)
            wl.check("the fresh process becomes DISPATCHABLE", True, "second boot serves")
        finally:
            worker2.stop()
            hub2.close()
    finally:
        if worker.process.poll() is None:
            worker.stop()

    if args.core_only:
        total = time.perf_counter() - t0
        print(f"\nlifecycle core: {wl.PASS} ok, {wl.FAIL} failed, {total:.1f} s total")
        return 1 if wl.FAIL else 0

    for section in SECTIONS.values():
        section(args, args.tmp)

    total = time.perf_counter() - t0
    print(f"\nlifecycle: {wl.PASS} ok, {wl.FAIL} failed, {total:.1f} s total")
    return 1 if wl.FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
