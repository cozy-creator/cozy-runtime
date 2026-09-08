from __future__ import annotations

import asyncio
from typing import Any

import msgspec
import pytest

from cozy_runtime.author import CapabilityError
from cozy_runtime.author._calls import MAX_ACTIVE_CALLS, MAX_CALL_INDEX, _Broker, _CallType
from cozy_runtime.author.fakes import fake_context


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

    broker = _Broker("parent", {key: _CallType(*key, Arguments, Result)}, exchange)
    broker.bind(fake_context(request_id="parent"))

    async def run() -> None:
        for batch in range(3):
            calls = [broker.reserve(*key, {"value": i}) for i in range(MAX_ACTIVE_CALLS)]
            with pytest.raises(CapabilityError, match="child_fanout"):
                broker.reserve(*key, {"value": 99})
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

    broker = _Broker("parent", {key: _CallType(*key, Arguments, Result)}, exchange)
    broker.bind(fake_context(request_id="parent"))
    broker.next_index = MAX_CALL_INDEX

    async def run() -> None:
        last = broker.reserve(*key, {"value": 1})
        assert last.index == MAX_CALL_INDEX
        await last
        with pytest.raises(CapabilityError, match="child_index_exhausted"):
            broker.reserve(*key, {"value": 2})

    asyncio.run(run())
    assert not broker.calls
