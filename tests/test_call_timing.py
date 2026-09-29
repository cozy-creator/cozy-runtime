"""Call timing is journaled observation, with no inferred compute across a restart."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import msgspec
import pytest

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker.call_timing import Timing
from cozy_runtime.internal.worker.machine_calls import CallRecord
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_calls import Calls
from cozy_runtime.internal.worker.workspace_executions import Executions
from test_machine_calls import request, running
from test_machine_execution import offer


def test_call_phases_use_monotonic_intervals_and_freeze_during_finalization(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "parent",
        b"c" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    calls = Calls(workspace)
    call = calls.accept("owner", request(running(executions)))
    clock = [0.0]
    wall = [1000]
    timing = Timing(
        executions,
        calls,
        lambda *_: "Same label",
        monotonic=lambda: clock[0],
        unix_ms=lambda: wall[0],
    )
    timing.phase("owner", call.child_request, 1, "queued", fresh=True)
    clock[0] = 90
    timing.phase("owner", call.child_request, 1, "preparing")
    clock[0] = 120
    timing.phase("owner", call.child_request, 1, "queued")
    clock[0] = 125
    timing.phase("owner", call.child_request, 1, "preparing")
    clock[0] = 128
    timing.phase("owner", call.child_request, 1, "running")
    clock[0] = 134
    timing.phase("owner", call.child_request, 1, "finalizing")
    clock[0] = 999  # output custody is not author execution
    measured = timing.phase("owner", call.child_request, 1, "terminal", status="succeeded")
    assert measured == {
        "queued_ms": 95000,
        "preparation_ms": 33000,
        "execution_ms": 6000,
        "execution_started_unix_ms": 1004,
    }
    timing.phase("owner", call.child_request, 1, "running")  # terminal does not reopen
    with workspace.locked() as db:
        events = [
            canonical_json.decode(row[0])
            for row in db.execute(
                "SELECT body FROM execution_events WHERE kind='call.phase' ORDER BY sequence"
            )
        ]
    assert [event["phase"] for event in events] == [
        "queued",
        "preparing",
        "queued",
        "preparing",
        "running",
        "finalizing",
        "terminal",
    ]
    assert [event["at_unix_ms"] for event in events] == list(range(1000, 1007))
    assert all(
        event["request"] == call.child_request
        and event["label"] == "Same label"
        and event["called_unix_ms"] == call.created_ms
        for event in events
    )
    assert events[-2]["execution_ms"] == events[-1]["execution_ms"] == 6000
    # A new attempt is separate, and old callbacks cannot overwrite its clock.
    clock[0] = 1100
    timing.phase("owner", call.child_request, 2, "queued", fresh=True)
    assert timing.phase("owner", call.child_request, 1, "terminal") == {}
    clock[0] = 1102
    assert timing.phase("owner", call.child_request, 2, "terminal", status="paused") == {
        "queued_ms": 2000,
        "preparation_ms": 0,
        "execution_ms": 0,
    }


def test_reopened_attempt_reports_unknown_duration_instead_of_offline_compute(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "parent",
        b"c" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    calls = Calls(workspace)
    call = calls.accept("owner", request(running(executions)))
    first = Timing(executions, calls, lambda *_: "label")
    first.phase("owner", call.child_request, 1, "queued", fresh=True)
    first.phase("owner", call.child_request, 1, "running")
    reopened = Timing(
        Executions(Workspace(workspace.store_root)),
        Calls(Workspace(workspace.store_root)),
        lambda *_: "label",
    )
    assert reopened.phase("owner", call.child_request, 1, "terminal", status="failed") == {}
    with workspace.locked() as db:
        raw = db.execute(
            "SELECT body FROM execution_events WHERE kind='call.phase' "
            "ORDER BY sequence DESC LIMIT 1"
        ).fetchone()[0]
    terminal = canonical_json.decode(raw)
    assert terminal["phase"] == "terminal"
    assert not any(
        key in terminal
        for key in ("queued_ms", "preparation_ms", "execution_ms", "execution_started_unix_ms")
    )
    # Older settled records remain readable; additive measurements are absent on write.
    record = CallRecord(
        request=call.child_request,
        parent="parent",
        index=0,
        attempt=1,
        module="library",
        export="publish",
        label="label",
        status="failed",
        error="lost",
        called_unix_ms=call.created_ms,
        stages={},
        steps={},
    )
    assert "execution_ms" not in msgspec.to_builtins(record)
    assert msgspec.json.decode(msgspec.json.encode(record), type=CallRecord).execution_ms is None


@pytest.mark.parametrize("fails", [False, True])
def test_worker_execution_boundary_excludes_preparation_and_postprocessing(
    tmp_path: Path, fails: bool
) -> None:
    from cozy_runtime.internal.worker.session import Worker

    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "parent",
        b"c" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    calls = Calls(workspace)
    call = calls.accept("owner", request(running(executions)))
    clock = [0.0]
    timing = Timing(executions, calls, lambda *_: "Render", monotonic=lambda: clock[0])
    timing.phase("owner", call.child_request, 1, "preparing", fresh=True)
    clock[0] = 90
    settled = []
    outcome = object()

    def execute(attempt: Any) -> object:
        with workspace.locked() as db:
            latest = canonical_json.decode(
                db.execute(
                    "SELECT body FROM execution_events WHERE kind='call.phase' "
                    "ORDER BY sequence DESC LIMIT 1"
                ).fetchone()[0]
            )
        assert latest["phase"] == "running"
        assert latest["preparation_ms"] == 90000 and latest["execution_ms"] == 0
        clock[0] = 96
        if fails:
            raise ValueError("executor failed")
        return outcome

    worker = SimpleNamespace(
        machine_calls=SimpleNamespace(timing=timing),
        fence=SimpleNamespace(record_owner_id="owner"),
        memory=SimpleNamespace(watch=lambda _: nullcontext()),
        _record_executor=lambda _: None,
        engine=SimpleNamespace(execute=execute, abandon=lambda *_: outcome),
        settle=settled.append,
        note=lambda *_: None,
    )
    attempt = SimpleNamespace(request_id=call.child_request, attempt=1)
    Worker.execute(cast(Any, worker), cast(Any, attempt), cast(Any, object()))
    assert settled == [outcome]
    clock[0] = 150
    measured = timing.phase(
        "owner", call.child_request, 1, "terminal", status="failed" if fails else "succeeded"
    )
    assert measured["execution_ms"] == 6000 and measured["preparation_ms"] == 90000


def test_unrecordable_timing_does_not_refuse_an_accepted_call(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "parent",
        b"c" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    calls = Calls(workspace)
    call = calls.accept("owner", request(running(executions)))
    timing = Timing(executions, calls, lambda *_: "x" * (128 << 10))
    assert timing.phase("owner", call.child_request, 1, "queued", fresh=True) == {}
    assert executions.status("owner", "parent").state == "running"
    timing.label = lambda *_: "label"
    assert timing.phase("owner", call.child_request, 1, "terminal", status="failed") == {}
