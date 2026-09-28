"""Native broker fixtures use the same descriptor framing as the executor."""

from __future__ import annotations

import hashlib
from typing import Any

from cozy_runtime.author._executor_requests import (
    Handler,
    Reply,
    Request,
    WriterAdopt,
    WriterOutput,
    WriterSource,
)
from cozy_runtime.internal.weights_writer import ExecutionStorage, WriterAttempt, WriterBroker
from durable_seam import seam


def broker_lane(broker: WriterBroker, attempt: WriterAttempt) -> Handler:
    """One attempt's native writer broker as a durable request handler."""

    def handle(request: Request) -> Reply:
        assert isinstance(request, WriterSource | WriterOutput | WriterAdopt)
        return broker.handle(attempt, request)

    return handle


def open_output(
    broker: Any,
    attempt: Any,
    transaction: str,
    definition: Any,
    *,
    output: str = "model",
    checkpoint: tuple[str, int] | None = None,
    frames: list[Request] | None = None,
    maximum: int | None = None,
) -> Any:
    """Supply the fixture's execution binding through the real executor output door."""

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

    handle = broker_lane(broker, attempt)

    def recorded(request: Request) -> Reply:
        if frames is not None:
            frames.append(request)
        return handle(request)

    broker.bind_output = bind
    limit = next(
        row.get("max_bytes", 0) for row in attempt.spec["outputs"] if row["output_id"] == output
    )
    client = ExecutionStorage(
        attempt.spool, seam(recorded), {output: limit if maximum is None else maximum}
    )
    return client.open_output(output, definition)
