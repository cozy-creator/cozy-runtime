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


def start_weightless(tmp: Path, *, instance_id: str, fresh: bool) -> Any:
    """worker-live's spawn, minus the quiet-card gate (there is no card in this run) and
    minus the CAS staging. Same control-process argv, same env seal inputs, same log pump."""
    home, root, project = tmp / "home", tmp / "worker", tmp / "project"
    if fresh:
        shutil.rmtree(tmp, ignore_errors=True)
    home.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    ids = stage_weightless(home, project, ("tick", "hold"))
    socket_path = str(tmp / "hub.sock")
    server, hub = wl.serve_hub(socket_path)
    hub.directive = pb.Directive(
        revision=1,
        posture=pb.Posture.POSTURE_ACCEPTING,
        wire_minor=wl.WIRE_MINOR,
        credential=pb.Credential(
            issuer="cozy-live",
            key_id="k1",
            epoch=1,
            expires_at_unix=int(time.time()) + 3600,
            token=b"never-crosses-the-seam",
        ),
        serving=pb.ServingDirective(
            endpoint_release_id=RELEASE,
            entrypoint_binding_plan_ids=sorted(ids.values()),
        ),
    )
    hub.revision = 1
    log = tmp / f"supervisor-{len(os.listdir(tmp))}.log"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "COZY_HOME": str(home),
        "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
    }
    argv = [
        "nice",
        "-n",
        "19",
        sys.executable,
        "-c",
        "import sys; from cozy_runtime.internal.worker.session import main; "
        "raise SystemExit(main(sys.argv[1:]))",
        "--hub",
        f"unix:{socket_path}",
        "--root",
        str(root),
        "--instance-id",
        instance_id,
        "--release-id",
        RELEASE,
        "--grace",
        "3.0",
        "--no-warm",
    ]
    import subprocess
    import threading

    process = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, bufsize=0
    )
    threading.Thread(target=wl._pump, args=(process, log), daemon=True).start()
    hub.process, hub.log = process, log
    worker = wl.Worker(home, root, process, hub, server, log)
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
            set(report.serving_capacity.ready_entrypoint_binding_plan_ids) >= set(ids.values()),
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
            pb.CoordinatorMessage(
                cancel_attempt=pb.CancelAttempt(
                    session_id=hub.session(),
                    executor_incarnation=hub.incarnation(),
                    request_id="req-cancel",
                    attempt=1,
                    exec_spec_digest=digest,
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
        hub.send(None)
        worker.server.stop(0)

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
            hub2.send(None)
            worker2.server.stop(0)
    finally:
        if worker.process.poll() is None:
            worker.stop()

    total = time.perf_counter() - t0
    print(f"\nlifecycle: {wl.PASS} ok, {wl.FAIL} failed, {total:.1f} s total")
    return 1 if wl.FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
