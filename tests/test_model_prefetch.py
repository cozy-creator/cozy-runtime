"""Known-next hints carry explicit model artifacts over the real control framing."""

from __future__ import annotations

import time
from typing import Any, cast

import msgspec
import pytest

from cozy_runtime import author
from cozy_runtime.author._calls import _Broker, _CallType, _current
from cozy_runtime.author._executor_requests import Answer, ModelPrefetch, Request
from durable_seam import seam


class Payload(msgspec.Struct):
    image: str


class Result(msgspec.Struct):
    value: str


def next_model(*, image: str) -> None:
    raise AssertionError("a hint invoked inference")


def test_hint_uses_locked_callable_without_reserving_a_call_or_future_payload() -> None:
    received: list[tuple[str, dict[str, Any]]] = []

    def exchange(request: Request) -> Answer:
        assert isinstance(request, ModelPrefetch)
        received.append(("model_prefetch", {
            "module": request.module, "export": request.export, "payload": request.payload,
        }))
        return Answer(ok=True)

    binding = _CallType("sha256:" + "a" * 64, __name__, "next_model", Payload, Result)
    broker = _Broker("parent", {(__name__, "next_model"): binding}, seam(exchange))
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
        selected = author.ModelArtifact(
            "earlier-native-output", "model", author.ObjectRef("sha256:" + "b" * 64, 123),
            "sha256:" + "c" * 64,
        )
        author.prefetch(next_model, models={"model": selected, "vae": None})
        assert msgspec.json.decode(received[-1][1]["payload"]) == {
            "model": msgspec.to_builtins(selected), "vae": None,
        }
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
    def decline(request: Request) -> Answer:
        assert isinstance(request, ModelPrefetch)
        return Answer(ok=False, code=code)

    broker = _Broker("parent", {(__name__, "next_model"): binding}, seam(decline))
    broker.bind(author.Context("parent", time.monotonic() + 60))
    token = _current.set(broker)
    try:
        author.prefetch(next_model)
        broker.finish()
    finally:
        _current.reset(token)
