"""Native broker fixtures use the same descriptor framing as the executor."""

from __future__ import annotations

import socket
from collections.abc import Callable
from typing import Any

from cozy_runtime.internal.seam import Channel, DescriptorReply


def exchange(
    handle: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    def call(kind: str, frame: dict[str, Any], *, descriptor: bool = False) -> dict[str, Any]:
        assert kind == "weights_writer"
        answer = handle(frame)
        if not isinstance(answer, DescriptorReply):
            return answer
        assert descriptor
        left, right = socket.socketpair()
        try:
            with left, right:
                sender, receiver = Channel(left), Channel(right)
                sender.send(dict(answer))
                sender.send_descriptor(answer.descriptor)
                result = receiver.recv()
                assert result is not None
                result["descriptor"] = receiver.recv_descriptor()
                return result
        finally:
            answer.descriptor.close()

    return call
