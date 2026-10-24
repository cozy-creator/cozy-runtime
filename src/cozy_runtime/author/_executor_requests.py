"""The executor's durable requests to its worker, and the worker's answers.

The executor sends `{"event": "request", "seq": N, **request}` and blocks until the worker answers
`{"event": "answer", "seq": N, **answer}`; a `child_events` answer is then followed by one
connected socket. Both are `tensorfs.executor`'s classes, the Rust structs TensorD reads and writes
them as (`protocol::executor`): built by name and written by serde here, decoded by serde there.
"""

from __future__ import annotations

import json
from typing import Protocol

from tensorfs.executor import (
    Answer,
    CheckpointItem,
    ChildCall,
    ChildCancel,
    ChildEvents,
    ChildForget,
    ChildPoll,
    GpuRelease,
    ModelPrefetch,
    Publish,
    PublishPart,
    StageMemoLookup,
    StageMemoStore,
    decode_answer,
)

__all__ = [
    "MAX_CALL_INDEX",
    "Answer",
    "CheckpointItem",
    "ChildCall",
    "ChildCancel",
    "ChildEvents",
    "ChildForget",
    "ChildPoll",
    "Exchange",
    "GpuRelease",
    "ModelPrefetch",
    "Publish",
    "PublishPart",
    "Request",
    "StageMemoLookup",
    "StageMemoStore",
    "decode_answer",
    "encode",
]

MAX_CALL_INDEX = (1 << 32) - 1

type Request = (
    ChildEvents
    | ChildCall
    | ChildPoll
    | ChildCancel
    | ChildForget
    | GpuRelease
    | ModelPrefetch
    | Publish
    | CheckpointItem
    | StageMemoLookup
    | StageMemoStore
)


class Exchange(Protocol):
    """One durable request; blocks until the worker answers it."""

    def __call__(self, request: Request, /) -> Answer: ...


def encode(request: Request) -> dict[str, object]:
    """A request as the JSON object the seam carries, `kind` included."""
    document = json.loads(request.json())
    assert isinstance(document, dict)
    return document
