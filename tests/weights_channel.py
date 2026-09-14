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


def open_output(
    broker: Any,
    attempt: Any,
    transaction: str,
    definition: Any,
    *,
    output: str = "model",
    checkpoint: tuple[str, int] | None = None,
    frames: list[dict[str, Any]] | None = None,
    maximum: int | None = None,
) -> Any:
    """Supply the fixture's execution binding through the real executor output door."""
    import hashlib

    from cozy_runtime.internal.weights_writer import ExecutionStorage

    def bind(current: Any, slot: str, declaration: bytes) -> tuple[str, int]:
        broker.authorize(
            current,
            transaction,
            current.attempt,
            hashlib.sha256(declaration).digest(),
            slot,
            checkpoint=checkpoint,
        )
        return transaction, current.attempt

    def handle(frame: dict[str, Any]) -> dict[str, Any]:
        if frames is not None:
            frames.append(dict(frame))
        answer = broker.handle(attempt, frame)
        assert isinstance(answer, dict)
        return answer

    broker.bind_output = bind
    limit = next(
        row.get("max_bytes", 0) for row in attempt.spec["outputs"] if row["output_id"] == output
    )
    client = ExecutionStorage(
        attempt.spool, exchange(handle), {output: limit if maximum is None else maximum}
    )
    return client.open_output(output, definition)
