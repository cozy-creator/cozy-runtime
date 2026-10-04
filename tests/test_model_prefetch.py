"""Known-next preparation does not require a future inference payload or GPU admission."""

from __future__ import annotations

import time
from typing import Any, cast

import msgspec
import pytest

from cozy_runtime import author
from cozy_runtime.author._calls import _Broker, _CallType, _current
from durable_seam import wire


class Payload(msgspec.Struct):
    image: str


class Result(msgspec.Struct):
    value: str


def next_model(*, image: str) -> None:
    raise AssertionError("a hint invoked inference")


def test_hint_uses_locked_callable_without_reserving_a_call_or_future_payload() -> None:
    received: list[tuple[str, dict[str, Any]]] = []

    def exchange(kind: str, body: dict[str, Any]) -> dict[str, Any]:
        received.append((kind, body))
        return {"ok": True}

    binding = _CallType("sha256:" + "a" * 64, __name__, "next_model", Payload, Result)
    broker = _Broker("parent", {(__name__, "next_model"): binding}, wire(exchange))
    broker.bind(author.Context("parent", time.monotonic() + 60))
    token = _current.set(broker)
    try:
        author.prefetch(next_model)
        assert received == [
            (
                "model_prefetch",
                {
                    "module": __name__,
                    "export": "next_model",
                    "payload": "{}",
                },
            )
        ]
        assert broker.calls == {} and broker.next_index == 0
        with pytest.raises(author.CapabilityError, match="locked callable"):
            author.prefetch(lambda: None)
        with pytest.raises(author.ConformanceError, match="ModelArtifact"):
            author.prefetch(next_model, models=cast(Any, {"model": "unresolved/alias"}))
        broker.finish()
    finally:
        _current.reset(token)


@pytest.mark.parametrize("code", ["unknown_durable_request", "model_prefetch_unsupported"])
def test_old_worker_can_decline_optional_hint(code: str) -> None:
    binding = _CallType("sha256:" + "a" * 64, __name__, "next_model", Payload, Result)
    broker = _Broker(
        "parent", {(__name__, "next_model"): binding}, wire(lambda *_: {"ok": False, "code": code})
    )
    broker.bind(author.Context("parent", time.monotonic() + 60))
    token = _current.set(broker)
    try:
        author.prefetch(next_model)
        broker.finish()
    finally:
        _current.reset(token)
