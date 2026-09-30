"""Child admission and deferred progress must not acquire each other's locks."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker.call_timing import Timing
from cozy_runtime.internal.worker.machine_calls import Calls
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_calls import Calls as JournalCalls
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_calls import request, running
from test_machine_execution import offer


@pytest.mark.parametrize("separate_writer", [False, True])
def test_call_admission_drains_child_progress_without_a_lock_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, separate_writer: bool
) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "parent",
        b"p" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    parent = running(executions)
    journal = JournalCalls(workspace)
    previous = request(parent)
    child = journal.accept("owner", previous)
    executions.submit(
        "owner",
        child.child_request,
        b"c" * 32,
        offer(child.child_request),
        expected_execution_workspace_id=executions.workspace_id,
    )
    running(executions, child.child_request)

    calls = Calls.__new__(Calls)
    calls.workspace, calls.executions, calls.journal = workspace, executions, journal
    calls.lock, calls.requests = threading.Lock(), {child.child_request: previous}
    calls.timing = Timing(executions, journal, lambda *_: "Next segment")
    nudges: list[str] = []
    calls.worker = cast(
        Any,
        SimpleNamespace(
            fence=SimpleNamespace(record_owner_id="owner"),
            machine_calls=calls,
            calls=SimpleNamespace(wake=nudges.append),
            execution=lambda *_: None,
        ),
    )
    incoming = request(parent)
    incoming.call_index = 1
    attempt = cast(Any, SimpleNamespace(request_id="parent", attempt=1))
    callback_entered, release_callback = threading.Event(), threading.Event()

    def changed(request_id: str, kind: str) -> None:
        if separate_writer and kind == "progress":
            callback_entered.set()
            assert release_callback.wait(5), "admission did not reach its journal lookup"
        Worker._journal_changed(calls.worker, request_id, kind)

    executions.changed = changed
    # Retain a real progress write for the next journal access. Only its optional
    # background flush is disabled; the production drain and callback both run.
    monkeypatch.setattr(workspace, "_flush", lambda: None)
    executions.progress(
        "owner",
        child.child_request,
        1,
        {
            "type": "progress",
            "payload": {"stage": "decode_video"},
        },
    )
    errors: list[BaseException] = []
    responses: list[pb.ChildCallResult] = []

    def admit() -> None:
        try:
            responses.append(calls.handle(attempt, incoming, "call"))
        except BaseException as error:
            errors.append(error)

    def drain() -> None:
        try:
            with workspace.locked():
                pass
        except BaseException as error:
            errors.append(error)

    writer = None
    if separate_writer:
        original_get = journal.get

        def get(owner: str, parent_id: str, index: int) -> Any:
            # The other thread already owns the workspace mutex. Let its real
            # callback acquire the requests lock while this lookup waits for it.
            release_callback.set()
            return original_get(owner, parent_id, index)

        monkeypatch.setattr(journal, "get", get)
        writer = threading.Thread(target=drain, daemon=True)
        writer.start()
        assert callback_entered.wait(5), "deferred progress did not reach its callback"
    admission = threading.Thread(target=admit, daemon=True)
    admission.start()
    # These are test hang bounds; events above determine the ordering, not sleeps.
    admission.join(5)
    assert not admission.is_alive(), "child admission deadlocked with deferred progress"
    if writer is not None:
        writer.join(5)
        assert not writer.is_alive(), "progress writer deadlocked with child admission"
    assert not errors
    assert nudges == ["parent"]
    accepted = journal.get("owner", "parent", 1)
    assert responses[0].child_request_id == accepted.child_request
    assert responses[0].state == pb.CHILD_CALL_STATE_PENDING

    # Replay retains one accepted child and its original queued observation.
    assert calls.handle(attempt, incoming, "call") == responses[0]
    with workspace.locked() as db:
        phases = [
            canonical_json.decode(row[0])
            for row in db.execute(
                "SELECT body FROM execution_events WHERE kind='call.phase'",
            )
        ]
    assert len(phases) == 1
    assert phases[0]["request"] == accepted.child_request
    assert phases[0]["phase"] == "queued"
    journal.freeze("owner", accepted, b"{}")
    journal.observe_result("owner", accepted, b'{"value":3}')
    settled = calls.handle(attempt, incoming, "poll")
    assert settled.state == pb.CHILD_CALL_STATE_SUCCEEDED
    assert settled.result_canonical_bytes == b'{"value":3}'
    controlled = executions.control("owner", "parent", "cancel-parent", 1, "cancel")
    assert controlled.generation == 2
    with workspace.locked() as db:
        assert (
            db.execute("SELECT desired FROM executions WHERE request='parent'").fetchone()[0]
            == "cancel"
        )


def test_real_worker_nested_progress_keeps_completion_and_cancel_responsive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_end_to_end import NO_EXECUTOR
    from test_machine_partial_work import machine

    if NO_EXECUTOR:
        pytest.skip(NO_EXECUTOR)
    entered, gate = tmp_path / "entered", tmp_path / "gate"
    gate.touch()
    source = f"""import asyncio, msgspec
from pathlib import Path
from cozy_runtime.author import App, Context, Telemetry, invocable
app = App()
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    value: int
@invocable
async def leaf(ctx: Context, tel: Telemetry, *, n: int) -> Result:
    tel.progress(0.25, stage="Child ready")
    Path({str(entered)!r}).touch()
    while not Path({str(gate)!r}).exists():
        await asyncio.sleep(0.01)
    tel.progress(1, stage="Child complete")
    return Result(n)
app.job(leaf)
@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    total = 0
    for n in range(3):
        total += (await leaf(n=n)).value
    return Result(total)
"""
    with machine(monkeypatch, source) as pod:
        pod.start(("leaf",), entrypoint="nested")
        coordinator = pod.worker.machine_calls
        workspace, executions = pod.worker.workspace, pod.worker.executions
        assert coordinator is not None and workspace is not None and executions is not None
        original_get = coordinator.journal.get
        injected = threading.Event()

        def get(owner: str, parent_id: str, index: int) -> Any:
            if parent_id == "responsive" and index == 1 and not injected.is_set():
                injected.set()
                executions.progress(
                    owner,
                    parent_id,
                    1,
                    {
                        "type": "progress",
                        "payload": {"stage": "Admitting next child"},
                    },
                )
            return original_get(owner, parent_id, index)

        monkeypatch.setattr(workspace, "_flush", lambda: None)
        monkeypatch.setattr(coordinator.journal, "get", get)
        pod.submit("responsive")
        result = pod.outcome("responsive")
        assert injected.is_set()
        assert result["status"] == pb.OUTCOME_STATUS_SUCCEEDED, result
        assert (
            len(
                pod.rows(
                    "SELECT result FROM execution_calls "
                    "WHERE parent_request='responsive' AND result<>x''"
                )
            )
            == 3
        )

        entered.unlink()
        gate.unlink()
        pod.submit("cancel-responsive")
        try:
            pod.wait("nested child to await its gate", entered.exists, 120)
            # Uses the real authenticated Get/ControlMachineExecution API while a
            # child is active, after the earlier root drained progress during admission.
            pod.cancel("cancel-responsive")
            canceled = pod.outcome("cancel-responsive")
            assert canceled["status"] == pb.OUTCOME_STATUS_CANCELED, canceled
        finally:
            gate.touch()
        pod.wait(
            "canceled child attempts to close",
            lambda: all(
                attempt.state == "closed" for attempt in pod.worker.engine.history.values()
            ),
            120,
        )
        executors = [
            supervision.current
            for supervision in (
                pod.worker.supervision,
                *(slot.supervision for slot in pod.worker.job_slots.values()),
                *(hosted.supervision for hosted in pod.worker.hosted.values()),
            )
            if supervision.current is not None
        ]
    assert not pod.thread.is_alive()
    assert all(not executor.alive() for executor in executors)
