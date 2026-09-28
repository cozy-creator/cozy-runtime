"""Operator restart and durable root admission share one authoritative worker lock."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import signed_claims
from cozy_runtime.cli.runtime_worker import _restart_handler, _watch_supervisor
from cozy_runtime.internal import proctree
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_execution import offer


@pytest.fixture
def prepared(monkeypatch: pytest.MonkeyPatch) -> Iterator[threading.Event]:
    """An accepted execution's unit holds at its preparation (a download, say) until the
    returned event is set: its work is owed, and the worker says so."""
    done = threading.Event()
    dispatch = Worker.dispatch_machine

    def preparing(self: Worker, *args: Any) -> Any:
        done.wait()
        return dispatch(self, *args)

    monkeypatch.setattr(Worker, "dispatch_machine", preparing)
    try:
        yield done
    finally:
        done.set()


def settled(ready: Any) -> None:
    bound = time.monotonic() + 60
    while not ready():
        assert time.monotonic() < bound
        time.sleep(0.01)


def _worker(root: Path) -> tuple[Worker, pb.Claim]:
    worker = Worker(
        RuntimeConfig(
            cozy_home=root / "home",
            credentials=Credentials(),
            record_owner_public_key=signed_claims.PUBLIC_KEY,
        ),
        WorkerOptions(**signed_claims.IDENTITY, root=root / "worker", tensorfs_root=root / "store"),
        InMemoryControlHost(),
    )
    claim = signed_claims.claim()
    assert worker.accept_claim(claim, lambda frame: None)[0] == 1
    return worker, claim


def test_runtime_cli_reports_guarded_restart_capability() -> None:
    command = [str(Path(sys.executable).with_name("cozy-runtime")), "version", "--json"]
    value = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
    assert value["supports_guarded_restart"] is True
    selected = json.loads(
        subprocess.run(
            [*command, "--fields", "distribution"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    assert set(selected) == {"distribution", "next"}


def test_running_guard_capability_names_process_birth_and_worker_boot(tmp_path: Path) -> None:
    worker, _claim = _worker(tmp_path)
    try:
        worker.mark_claim_ready()
        path = worker.root / "restart-capability.json"
        document = json.loads(path.read_bytes())
        process = proctree.process_identity(os.getpid())
        assert document == {
            "supports_guarded_restart": True,
            "pid": process.pid,
            "started_ticks": process.started_ticks,
            "worker_boot_id": worker.fence.worker_boot_id,
        }
        assert path.stat().st_mode & 0o777 == 0o400
        assert worker.request_idle_restart()
        verdict = json.loads((worker.root / "restart-status.json").read_bytes())
        assert all(
            verdict[key] == document[key] for key in ("pid", "started_ticks", "worker_boot_id")
        )
    finally:
        worker.shutdown()


def test_real_hup_cannot_reenter_main_thread_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared: threading.Event,
) -> None:
    worker, claim = _worker(tmp_path)
    read, write = os.pipe()
    entered = threading.Event()
    guarded_restart = worker.request_idle_restart

    def observed_guard() -> bool:
        entered.set()
        return guarded_restart()

    monkeypatch.setattr(worker, "request_idle_restart", observed_guard)
    previous = signal.signal(signal.SIGHUP, _restart_handler(worker))
    watcher = threading.Thread(
        target=_watch_supervisor,
        args=(worker, worker.request_stop, read),
        daemon=True,
    )
    watcher.start()
    try:
        # A real signal interrupts the main thread while it owns the reentrant lock.
        # The watcher must wait for its admission to commit before deciding idle/busy.
        with worker.control_lock:
            os.kill(os.getpid(), signal.SIGHUP)
            assert entered.wait(5)
            assert not worker.stop.is_set()
            worker.submit_execution(
                claim,
                "root",
                b"c" * 32,
                offer(job=False),
                expected_execution_workspace_id=worker._execution_service(claim)[0].workspace_id,
            )
        with worker.control_lock:
            pass
        worker.request_stop()
        watcher.join(5)
        assert not watcher.is_alive()
        verdict = json.loads((worker.root / "restart-status.json").read_bytes())
        assert verdict["status"] == "busy" and verdict["active_executions"] == 1
        assert not worker.restart_requested
    finally:
        prepared.set()
        signal.signal(signal.SIGHUP, previous)
        worker.request_stop()
        watcher.join(5)
        os.close(read)
        os.close(write)
        worker.shutdown()


def test_queued_detached_root_refuses_update_then_can_finish(
    tmp_path: Path, prepared: threading.Event
) -> None:
    worker, claim = _worker(tmp_path)
    try:
        receipt = worker.submit_execution(
            claim,
            "root",
            b"c" * 32,
            offer(job=False),
            expected_execution_workspace_id=worker._execution_service(claim)[0].workspace_id,
        )
        # No Control stream/snapshot acknowledgement. Queued work is absent from held_attempts.
        assert not worker.snapshot_body().held_attempts
        assert not worker.request_idle_restart()
        verdict = json.loads((worker.root / "restart-status.json").read_bytes())
        assert verdict["status"] == "busy" and verdict["active_executions"] == 1
        assert not worker.stop.is_set()
        prepared.set()
        settled(lambda: not worker.supervisor.busy())
        assert worker.execution_status(claim, receipt.request_id).state == "failed"
        outcome = worker.collect_execution(claim, receipt.request_id)
        assert worker.request_idle_restart()
        assert worker.restart_requested and worker.stop.is_set()
        verdict2 = json.loads((worker.root / "restart-status.json").read_bytes())
        assert verdict2["status"] == "accepted" and verdict2["sequence"] == verdict["sequence"] + 1
        assert worker.collect_execution(claim, receipt.request_id) == outcome
        with pytest.raises(WorkspaceRefusal, match="stopping"):
            worker.submit_execution(
                claim,
                "later",
                b"d" * 32,
                offer("later", job=False),
                expected_execution_workspace_id=worker._execution_service(claim)[0].workspace_id,
            )
        assert worker.stop.is_set()  # Cannot reopen admission after a successful guard.
    finally:
        worker.shutdown()


def test_a_preparation_is_busy(tmp_path: Path) -> None:
    worker, _claim = _worker(tmp_path)
    try:
        with worker.preparation_lock:
            assert not worker.request_idle_restart()
            assert json.loads((worker.root / "restart-status.json").read_bytes())["preparing"]
    finally:
        worker.shutdown()


@pytest.mark.parametrize("iteration", range(8))
def test_restart_and_root_admission_have_exactly_one_winner(
    tmp_path: Path, iteration: int, prepared: threading.Event
) -> None:
    worker, claim = _worker(tmp_path / str(iteration))
    barrier = threading.Barrier(2)

    def submit() -> bool:
        barrier.wait()
        try:
            worker.submit_execution(
                claim,
                "root",
                b"c" * 32,
                offer(job=False),
                expected_execution_workspace_id=worker._execution_service(claim)[0].workspace_id,
            )
        except WorkspaceRefusal as exc:
            assert "stopping" in str(exc)
            return False
        return True

    def restart() -> bool:
        barrier.wait()
        return worker.request_idle_restart()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            admitted = pool.submit(submit)
            stopped = pool.submit(restart)
            assert admitted.result(timeout=10) != stopped.result(timeout=10)
        assert worker.executions is not None
        assert worker.executions.owns("owner", "root") != worker.stop.is_set()
    finally:
        prepared.set()
        worker.shutdown()
