from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import msgspec
import pytest

from cozy_runtime.author import CapabilityError
from cozy_runtime.author._calls import MAX_ACTIVE_CALLS, _Broker, _CallType
from cozy_runtime.author._executor_requests import MAX_CALL_INDEX, CallProgress, CallState
from cozy_runtime.author._services import Attempt, ProgressFrame, Telemetry
from cozy_runtime.author.fakes import fake_context
from durable_seam import wire


class Arguments(msgspec.Struct):
    value: int


class Result(msgspec.Struct):
    value: int


def test_pending_bound_orders_activation_and_releases_completed_history() -> None:
    key = ("sha256:" + "11" * 32, "fixture", "measure")
    activated: list[int] = []
    forgotten: list[int] = []

    def exchange(kind: str, frame: dict[str, Any]) -> dict[str, Any]:
        index = frame["call_index"]
        if kind == "child_call":
            activated.append(index)
            return {"ok": True}
        if kind == "child_forget":
            forgotten.append(index)
            return {"ok": True}
        return {"ok": True, "state": "succeeded", "result": f'{{"value":{index}}}'}

    broker = _Broker("parent", {key[1:]: _CallType(*key, Arguments, Result)}, wire(exchange))
    broker.bind(fake_context(request_id="parent"))

    async def run() -> None:
        for batch in range(3):
            calls = [broker.reserve(*key[1:], {"value": i}) for i in range(MAX_ACTIVE_CALLS)]
            with pytest.raises(CapabilityError, match="child_fanout"):
                broker.reserve(*key[1:], {"value": 99})
            results = await asyncio.gather(*calls)
            assert [r.value for r in results] == list(
                range(batch * MAX_ACTIVE_CALLS, (batch + 1) * MAX_ACTIVE_CALLS)
            )
            assert not broker.calls and not broker.activated

    asyncio.run(run())
    broker.finish()
    assert activated == list(range(3 * MAX_ACTIVE_CALLS))
    assert sorted(forgotten) == activated


def test_call_index_refuses_overflow_without_reusing_an_accepted_index() -> None:
    key = ("sha256:" + "11" * 32, "fixture", "measure")

    def exchange(kind: str, frame: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "state": "succeeded", "result": '{"value":1}'}

    broker = _Broker("parent", {key[1:]: _CallType(*key, Arguments, Result)}, wire(exchange))
    broker.bind(fake_context(request_id="parent"))
    broker.next_index = MAX_CALL_INDEX

    async def run() -> None:
        last = broker.reserve(*key[1:], {"value": 1})
        assert last.index == MAX_CALL_INDEX
        await last
        with pytest.raises(CapabilityError, match="child_index_exhausted"):
            broker.reserve(*key[1:], {"value": 2})

    asyncio.run(run())
    assert not broker.calls


def test_child_progress_uses_its_reserved_scope_and_deduplicates_polls(tmp_path: Path) -> None:
    key = ("sha256:" + "11" * 32, "fixture", "measure")
    frames: list[Any] = []
    ctx = fake_context(request_id="parent")
    telemetry = Telemetry(Attempt("parent", tmp_path, sink=frames.append), ctx)
    broker = _Broker("parent", {key[1:]: _CallType(*key, Arguments, Result)}, wire(lambda *a: {}))
    broker.bind(ctx, telemetry)
    with telemetry.scope("Shot 2 of 7", overall_range=(0.2, 0.4)):
        first = broker.reserve(*key[1:], {"value": 1})
        update = CallState(
            ok=True,
            child_request_id="child-1",
            progress=CallProgress(
                sequence=10,
                payload={
                    "stage": "denoise",
                    "stage_fraction": 0.375,
                    "overall_fraction": 0.5,
                    "position": 3,
                    "total": 8,
                    "step_ms": 100,
                    "call_request": "child-1",
                    "call_attempt": 2,
                },
            ),
        )
        broker._observe(first, update)
        measured = frames[-1]
        assert isinstance(measured, ProgressFrame)
        assert (measured.stage, measured.overall_fraction, measured.position, measured.total) == (
            "Shot 2 of 7 / denoise",
            0.3,
            3,
            8,
        )
        assert (measured.call_request, measured.call_attempt) == ("child-1", 2)
        count = len(frames)
        broker._observe(first, update)
        assert len(frames) == count
    with telemetry.scope("Shot 3 of 7", overall_range=(0.4, 0.6)):
        second = broker.reserve(*key[1:], {"value": 2})
        broker._observe(second, msgspec.structs.replace(update, child_request_id="child-2"))
        assert frames[-1].stage == "Shot 3 of 7 / denoise"
        assert frames[-1].overall_fraction == 0.5
