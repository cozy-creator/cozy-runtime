#!/usr/bin/env python3
"""Focused first-party process-tree containment and restart-recovery proof."""

from __future__ import annotations

import argparse
import atexit
import contextlib
import os
import secrets
import shutil
import signal
import sys
import time
from pathlib import Path

from cozy_runtime.internal import accel, proctree, spawn
from cozy_runtime.internal.child_env import EXECUTOR_SCOPE_ENV
from cozy_runtime.internal.worker.child import ExecutorSupervision, OwnerRecord


def wait_file(path: Path, seconds: float) -> int:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            return int(path.read_text())
        except (FileNotFoundError, ValueError):
            time.sleep(0.01)
    raise TimeoutError(path)


def wait_absent(process: proctree.ProcessIdentity, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while proctree.process_state(process) is not None:
        proctree.reap_child(process)
        if time.monotonic() >= deadline:
            raise TimeoutError(process)
        time.sleep(0.01)


def cleanup_scope(scope: proctree.ProcessTreeScope) -> None:
    """Best-effort harness cleanup; the production path never weakens its proof this way."""
    with contextlib.suppress(BaseException):
        processes = proctree.freeze_process_tree(
            scope, None, set(), worker_pid=os.getpid(), seconds=1.0
        )
        proctree.kill_process_tree(processes)
        for process in processes:
            with contextlib.suppress(BaseException):
                wait_absent(process, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmp", default="/tmp/cozy-process-containment")
    parser.add_argument("--observe", type=float, default=10.0)
    args = parser.parse_args()
    root = Path(args.tmp)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    home = root / "home"
    home.mkdir()
    helper_file = root / "detached.pid"
    release_file = root / "release-leader"

    proctree.arm_subreaper()
    scope = proctree.ProcessTreeScope(secrets.token_hex(32))
    atexit.register(cleanup_scope, scope)
    base_env = dict(os.environ)
    base_env[EXECUTOR_SCOPE_ENV] = scope.token
    leader_code = (
        "import os,pathlib,signal,time\n"
        "child=os.fork()\n"
        "if child == 0:\n"
        " os.setsid()\n"
        " grand=os.fork()\n"
        " if grand: os._exit(0)\n"
        f" pathlib.Path({str(helper_file)!r}).write_text(str(os.getpid()))\n"
        " while True: time.sleep(60)\n"
        f"while not pathlib.Path({str(release_file)!r}).exists(): time.sleep(0.01)\n"
    )

    supervision = ExecutorSupervision(
        root=root / "worker",
        python=sys.executable,
        base_env=(),
        cozy_home=home,
        device_process=lambda _pid: accel.ProcessMemory("absent", 0),
    )
    provisional = OwnerRecord(scope)
    supervision._write_owner(provisional)
    spawned = spawn.spawn_executor(
        python=sys.executable,
        module_argv=["-c", leader_code],
        base_env=tuple(sorted(os.environ.items())),
        imposed={EXECUTOR_SCOPE_ENV: scope.token},
        cwd=str(root),
        scope=scope,
    )
    leader_pid = spawned.pid
    leader = proctree.process_identity(spawned.pid)
    proctree.wait_process_stopped(leader, seconds=args.observe)
    if not proctree.process_scope_contains(scope, leader):
        raise RuntimeError("stopped leader is outside its process-tree scope")
    supervision._bind_owner(provisional, leader)
    proctree.signal_process(leader, signal.SIGCONT)
    detached = proctree.process_identity(wait_file(helper_file, args.observe))

    wrong_scope = proctree.ProcessTreeScope(secrets.token_hex(32))
    atexit.register(cleanup_scope, wrong_scope)
    wrong_env = dict(os.environ)
    wrong_env[EXECUTOR_SCOPE_ENV] = wrong_scope.token
    wrong_pid = os.posix_spawn(
        sys.executable,
        [sys.executable, "-c", "import time; time.sleep(60)"],
        wrong_env,
        setpgroup=0,
    )
    wrong = proctree.process_identity(wrong_pid)

    release_file.touch()
    os.waitpid(leader_pid, 0)
    if proctree.same_process(leader):
        raise RuntimeError("leader did not exit before recovery")

    # Simulate the old worker birth ending while its durable owner and detached helper
    # survive. A new supervision must follow the RECORDED backend, not reselect this host's
    # writable cgroup-v2 path.
    proctree.release_worker_root_lease(supervision._root_lease)
    supervision._closed = True
    successor = ExecutorSupervision(
        root=root / "worker",
        python=sys.executable,
        base_env=(),
        cozy_home=home,
        device_process=lambda _pid: accel.ProcessMemory("absent", 0),
    )
    try:
        killed = successor.sweep_orphans()
        wait_absent(detached, args.observe)
        if killed or successor._read_owner() is not None:
            raise RuntimeError("leaderless owner did not clear after exact reclaim")
        if not proctree.same_process(wrong):
            raise RuntimeError("a wrong-token sibling was killed as collateral")
    finally:
        successor.close()
        if proctree.same_process(wrong):
            proctree.signal_process(wrong, signal.SIGKILL)
        wait_absent(wrong, args.observe)

    print("process_backend=process_tree")
    print("spawn_bound_before_continue=true")
    print(f"leader_gone=true detached_reclaimed={detached.pid}")
    print("wrong_token_survived=true")
    print("durable_owner_cleared=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
