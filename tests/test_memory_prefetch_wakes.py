"""Real queued acceptance/native bytes, event-only wakes, no executor process or CUDA."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author._executor_requests import Answer, StoreRead
from cozy_runtime.internal import memory
from cozy_runtime.internal.worker.attempts import AttemptEngine
from cozy_runtime.internal.worker.gpu_scheduler import Demand, GpuScheduler, Want
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.internal.worker.supervisor import Supervisor
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_executions import Executions
from test_memory_first_use_pages import queued
from test_weights_resume_native import _committed


def accepted_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Worker, list[str], GpuScheduler, dict[str, Any]]:
    worker, selected, notes = queued(tmp_path, monkeypatch)
    executions = Executions(Workspace(Path(str(worker.options.tensorfs_root))))
    offer = selected.row_value.attempt_offer()
    offer.invocation_spec_digest = hashlib.sha256(offer.invocation_spec_canonical_bytes).digest()
    offer.grant.invocation_spec_digest = offer.invocation_spec_digest
    executions.submit(
        "owner",
        "next-submission",
        b"c" * 32,
        offer,
        expected_execution_workspace_id=executions.workspace_id,
        worker_boot="boot",
        preparation=canonical_json.encode({"state": selected.prepared_value.state}),
        worker_id="worker",
    )
    worker.executions = executions
    scheduler = GpuScheduler(2)
    roots = {"older": 1, "next": 2}
    scheduler.sync(roots, [Demand("older#1", "older", 2), Demand("next#1", "next", 1)])
    before = scheduler.view()
    template = worker._memory_waiting[0]
    worker.gpu = cast(
        Any,
        SimpleNamespace(wants={"older#1": Want("older", 2), "next#1": Want("next", 1, template)}),
    )
    worker.prespawns = cast(Any, SimpleNamespace(reconcile=lambda *_: None))
    worker._gpu_observed([], roots, before, {})
    worker.memory.store_ready = worker._wake_memory_prefetch
    return worker, notes, scheduler, before


def supervise(
    worker: Worker, notes: list[str], monkeypatch: pytest.MonkeyPatch
) -> tuple[threading.Event, threading.Event]:
    blocked, read = threading.Event(), threading.Event()

    def note(_kind: str, text: str) -> None:
        notes.append(text)
        if "foreground" in text:
            blocked.set()
        if "2048 B" in text:
            read.set()

    monkeypatch.setattr(worker, "note", note)
    worker.supervisor = Supervisor(
        lambda text: notes.append("integrity:" + text), lambda text: notes.append("crash:" + text)
    )
    return blocked, read


def finish(worker: Worker) -> None:
    worker.stop.set()
    worker.supervisor.stop()
    worker.supervisor.join()


def test_store_read_release_rearms_same_durable_queue_without_new_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, notes, scheduler, before = accepted_queue(tmp_path, monkeypatch)
    blocked, read = supervise(worker, notes, monkeypatch)
    origin = SimpleNamespace(pid=101, epoch=7)
    lane = cast(Any, SimpleNamespace(row=lambda _: SimpleNamespace(executor=lambda: origin)))
    answer = worker._room(lane, "older")
    try:
        begun = answer(StoreRead(reservation="read"))
        assert isinstance(begun, Answer) and begun.ok
        worker._refresh_memory_queue()
        assert blocked.wait(5), notes
        worker.supervisor.join()
        assert not read.is_set() and worker._memory_prefetch_deferred
        released = answer(StoreRead(reservation="read", release=True))
        assert isinstance(released, Answer) and released.ok
        assert read.wait(5), notes
        worker.supervisor.join()
        assert scheduler.view() == before
        row = worker.executions.row("owner", "next") if worker.executions is not None else None
        assert row is not None and not row.dispatched and row.desired == "run"
        assert len([line for line in notes if "foreground store demand" in line]) == 1
        assert not worker.memory.host_reserved
    finally:
        finish(worker)


def execute_to_transport(worker: Worker, tmp_path: Path, running: Any) -> None:
    class TransportReached(Exception):
        pass

    def reached(*_: Any) -> None:
        raise TransportReached

    executor = SimpleNamespace(
        worst_pause=0, epoch=1, pid=101, process=SimpleNamespace(started_ticks=1)
    )
    attempt = SimpleNamespace(
        job=None,
        executor=executor,
        spool=tmp_path / "spool",
        deadline_ms=0,
        request_id="older",
        attempt=1,
        transition_lock=threading.Lock(),
        canceling="",
        state="accepted",
        executions=0,
        poke=lambda: None,
    )
    engine = SimpleNamespace(
        history={},
        live={"older": attempt},
        supervision_for=lambda _: SimpleNamespace(owns=lambda _: True),
        _command=lambda *_: {},
        dispatched=lambda _: None,
        running=running,
        executed=0,
        progress=lambda *_: None,
        monitor=SimpleNamespace(start=reached),
    )
    engine._run = lambda current: AttemptEngine._run(cast(Any, engine), current)
    worker.engine = cast(Any, engine)
    # An accepted/preparing execution still blocks optional reads, even if the gate opened.
    worker._prefetch_pages()
    assert bool(worker._memory_prefetch_deferred)
    with pytest.raises(TransportReached):
        AttemptEngine.execute(cast(Any, engine), cast(Any, attempt))
    assert attempt.state == "running" and engine.executed == 1
    assert not attempt.transition_lock.locked()


def test_running_transition_rearms_after_gate_opened_while_still_preparing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, notes, scheduler, before = accepted_queue(tmp_path, monkeypatch)
    _, read = supervise(worker, notes, monkeypatch)

    def running() -> None:
        attempt = worker.engine.live["older"]
        assert attempt.state == "running" and not attempt.transition_lock.locked()
        worker._wake_memory_prefetch()

    try:
        execute_to_transport(worker, tmp_path, running)
        assert read.wait(5), notes
        worker.supervisor.join()
        assert scheduler.view() == before
    finally:
        finish(worker)


def test_release_before_deferred_interest_is_armed_is_not_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, notes, _, _ = accepted_queue(tmp_path, monkeypatch)
    _, read = supervise(worker, notes, monkeypatch)
    worker.memory.begin_store_read("older")
    original = worker._prefetch_blocked
    once = False

    def raced() -> str:
        nonlocal once
        reason = original()
        if reason and not once:
            once = True
            worker.memory.end_store_read("older")  # callback sees deferred=False
        return reason

    monkeypatch.setattr(worker, "_prefetch_blocked", raced)
    try:
        worker._refresh_memory_queue()
        assert read.wait(5), notes
        worker.supervisor.join()
        assert once and not worker._memory_prefetch_deferred
    finally:
        finish(worker)


def test_optional_callback_failure_cannot_fail_foreground_release_or_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail() -> None:
        raise RuntimeError("optional unit cannot start")

    machine = memory.Machine({}, lambda _: None, lambda _: None, store_ready=fail)
    assert machine.begin_store_read("read")
    machine.end_store_read("read")
    with machine.store_demand("phase"):
        pass
    assert machine.optional_store_wanted()
    worker, _, _, _ = accepted_queue(tmp_path, monkeypatch)
    execute_to_transport(worker, tmp_path, fail)


def test_verified_only_proof_is_readable_by_another_cpu_process(tmp_path: Path) -> None:
    store, _, _, receipt = _committed(tmp_path)
    code = """import json,sys,tensorfs
store=tensorfs.Store.open(sys.argv[1])
with store.acquire_cozytensors_if_verified(sys.argv[2]) as lease:
    print(json.dumps({'header_bytes':len(lease.header),'torch_imported':'torch' in sys.modules}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(store.root), receipt.artifact.manifest.digest],
        capture_output=True,
        text=True,
        check=True,
    )
    observed = json.loads(result.stdout)
    assert observed["header_bytes"] > 0 and not observed["torch_imported"]


def test_gate_ready_edges_are_outside_lock_and_only_when_last_demand_leaves() -> None:
    calls = []
    machine = memory.Machine({}, lambda _: None, lambda _: None)

    def ready() -> None:
        assert not cast(Any, machine._store_condition)._is_owned()
        assert machine.optional_store_wanted()
        calls.append("ready")

    machine.store_ready = ready
    assert machine.begin_store_read("executor:1:old")
    assert machine.begin_store_read("executor:2:new")
    machine.end_store_read("executor:1:old")
    assert not calls
    machine.release_store_prefix("executor:2:")
    assert calls == ["ready"]
    machine.end_store_read("executor:2:new")
    assert calls == ["ready"]  # idempotent release is not another eligibility edge
    with machine.store_demand("phase"):
        assert not machine.optional_store_wanted()
    assert calls == ["ready", "ready"]


def test_shutdown_or_removed_queue_does_not_restart_optional_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, notes, _, _ = accepted_queue(tmp_path, monkeypatch)
    _, read = supervise(worker, notes, monkeypatch)
    try:
        worker.memory.begin_store_read("older")
        worker._prefetch_pages()
        assert bool(worker._memory_prefetch_deferred)
        worker._memory_waiting = ()
        worker._first_use_waiting = ()
        worker._refresh_memory_queue()
        worker.memory.end_store_read("older")
        worker.supervisor.join()
        assert not read.is_set() and not worker._memory_prefetch_deferred
        worker.stop.set()
        worker.supervisor.stop()
        assert worker.memory.begin_store_read("shutdown")
        worker.memory.end_store_read("shutdown")
        assert not worker.supervisor.keys("memory-prefetch")
    finally:
        finish(worker)


def test_first_use_projection_failure_cannot_undo_durable_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, _notes, _scheduler, _view = accepted_queue(tmp_path, monkeypatch)

    def unavailable() -> None:
        raise RuntimeError("optional scheduling unavailable")

    monkeypatch.setattr(worker, "_refresh_memory_queue", unavailable)
    attempt = SimpleNamespace(request_id="next", attempt=1)
    worker._dispatched(cast(Any, attempt))
    assert worker.executions is not None
    row = worker.executions.row("owner", "next")
    assert row is not None and row.dispatched
    assert not worker._first_use_waiting
