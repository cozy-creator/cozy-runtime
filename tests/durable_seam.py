"""The executor's durable request lane against a worker handler or a wire-level fake."""

from __future__ import annotations

import socket
from collections.abc import Callable, Mapping
from typing import Any

import msgspec

from cozy_runtime.author._executor_requests import (
    Answer,
    Exchange,
    Handler,
    Request,
    encode,
    respond,
)
from cozy_runtime.internal.seam import Channel


def seam(handler: Handler) -> Exchange:
    """Each request crosses a real seam socket pair, its capability included."""

    def exchange[A: Answer](request: Request, into: type[A], /) -> A:
        left, right = socket.socketpair()
        with left, right:
            executor, worker = Channel(left), Channel(right)
            executor.send({"event": "request", "seq": 1, **encode(request)})
            frame = worker.recv()
            assert frame is not None
            answer, handoff = respond(frame, handler)
            try:
                worker.send(answer)
                if handoff is not None:
                    worker.send_descriptor(handoff)
            finally:
                if handoff is not None:
                    handoff.close()
            reply = executor.recv()
            assert reply is not None and reply["seq"] == 1
            if reply.get("descriptor") is True:
                reply["descriptor"] = executor.recv_descriptor()
            return msgspec.convert(reply, into)

    return exchange


def wire(fake: Callable[[str, dict[str, Any]], Mapping[str, object]]) -> Exchange:
    """A fake worker that reads each request's `kind` and body and answers a wire document."""

    def exchange[A: Answer](request: Request, into: type[A], /) -> A:
        body = encode(request)
        kind = body.pop("kind")
        assert isinstance(kind, str)
        return msgspec.convert(fake(kind, body), into)

    return exchange
