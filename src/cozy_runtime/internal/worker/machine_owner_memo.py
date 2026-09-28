"""Memoized results shared with a machine execution's record owner (workspace memo_lookup).

The owner keeps them on its own machine; nothing goes to Tensorhub. `memo.record` announces
a completed upload's checkpoint. Under `owner_memo`, `memo.lookup` asks before computing and
parks only that call, and only while the owner observes the root's events: from its wait read
until one ends with the owner gone. Between its reads an observer stays attached, so a lookup
recorded then is read next. The owner's answer decides a lookup, or its read past the lookup
does, or its leaving does: a miss. No clock decides.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author.publication import CheckpointRef
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from .workspace_executions import Executions, ExecutionWorkspaceRefusal

_LOG = logging.getLogger(__name__)
RESULT_MAX_BYTES = 48 << 10
DECIDED_KEPT = 256  # answered lookups kept so a repeated answer replays its reply


@dataclass
class _Lookup:
    computation: bytes
    destination: str
    decided: bool = False
    answer: bytes = b""
    refusal: ExecutionWorkspaceRefusal | None = None
    result: bytes | None = None


def _checkpoint(raw: bytes, destination: str) -> bytes:
    """The owner's recorded result as the upload's own completion would record it."""
    try:
        if len(raw) > RESULT_MAX_BYTES:
            raise ValueError("result exceeds its bound")
        ref = msgspec.convert(canonical_json.decode(raw), type=CheckpointRef, strict=True)
    except (ValueError, TypeError, msgspec.MsgspecError) as exc:
        raise ExecutionWorkspaceRefusal(
            "memo_result_invalid", "memo answer is not an upload's checkpoint"
        ) from exc
    if ref.destination.lower() != destination.lower() or not ref.checkpoint:
        raise ExecutionWorkspaceRefusal(
            "memo_result_invalid", "memo answer names no checkpoint in the call's destination"
        )
    return canonical_json.encode(msgspec.to_builtins(ref))


class OwnerMemo:
    def __init__(self, executions: Executions) -> None:
        self.executions = executions
        self.condition = threading.Condition()
        self.reading: Counter[tuple[str, str]] = Counter()  # open owner wait reads
        self.observed: set[tuple[str, str]] = set()
        self.lookups: dict[tuple[str, str, int], _Lookup] = {}

    @contextmanager
    def wait_read(self, owner: str, request: str, gone: Callable[[], bool]) -> Iterator[None]:
        key = owner, request
        with self.condition:
            self.reading[key] += 1
            self.observed.add(key)
        try:
            yield
        finally:
            with self.condition:
                self.reading[key] -= 1
                if not self.reading[key]:
                    del self.reading[key]
                    if gone():
                        self.observed.discard(key)
                        for (o, r, _), lookup in self.lookups.items():
                            lookup.decided |= (o, r) == key
                        self.condition.notify_all()

    def read(self, owner: str, request: str, after: int) -> None:
        """An owner read from `after` has passed every lookup through it: unanswered, a miss."""
        with self.condition:
            for (o, r, sequence), lookup in self.lookups.items():
                if (o, r) == (owner, request) and sequence <= after:
                    lookup.decided = True
            self.condition.notify_all()

    def wake(self) -> None:
        with self.condition:
            self.condition.notify_all()

    def _root(self, owner: str, call: pb.ChildCallRequest) -> str:
        if not self.executions.owns(owner, call.parent_request_id):
            return ""
        return self.executions.scheduling_root(owner, call.parent_request_id)[0]

    def record(
        self,
        owner: str,
        call: pb.ChildCallRequest,
        operation: str,
        computation: bytes,
        result: bytes,
    ) -> None:
        try:
            if root := self._root(owner, call):
                self.executions.notice(
                    owner,
                    root,
                    "memo.record",
                    {
                        "call_index": call.call_index,
                        "operation": operation,
                        "computation_digest": documents.spell(computation),
                        "result": canonical_json.decode(result),
                    },
                )
        except Exception:
            _LOG.exception("memo.record was not journaled")  # an optional cache

    def lookup(
        self,
        owner: str,
        call: pb.ChildCallRequest,
        operation: str,
        computation: bytes,
        destination: str,
        stopped: Callable[[], bool],
    ) -> bytes | None:
        """The owner's validated result for this computation, or None to compute."""
        try:
            root = self._root(owner, call)
            if not root or not self.executions.owner_memo(owner, root):
                return None
            with self.condition:
                if (owner, root) not in self.observed:
                    return None
                sequence = self.executions.notice(
                    owner,
                    root,
                    "memo.lookup",
                    {
                        "call_index": call.call_index,
                        "operation": operation,
                        "computation_digest": documents.spell(computation),
                    },
                )
                lookup = self.lookups[owner, root, sequence] = _Lookup(computation, destination)
                decided = [key for key, row in self.lookups.items() if row.decided]
                for key in decided[: max(0, len(self.lookups) - DECIDED_KEPT)]:
                    del self.lookups[key]
                while not lookup.decided and not stopped():
                    self.condition.wait()
                lookup.decided = True
                return lookup.result
        except Exception:
            _LOG.exception("memo.lookup failed; the call computes")
            return None

    def answer(self, owner: str, request: str, memo: pb.MachineMemoAnswer) -> None:
        raw = memo.SerializeToString(deterministic=True)
        with self.condition:
            lookup = self.lookups.get((owner, request, memo.lookup_sequence))
            if lookup is not None and lookup.answer == raw:
                if lookup.refusal is not None:
                    raise lookup.refusal
                return
            if lookup is None or lookup.decided:
                raise ExecutionWorkspaceRefusal(
                    "memo_lookup_not_outstanding", "no memo.lookup at this sequence awaits one"
                )
            lookup.decided, lookup.answer = True, raw
            self.condition.notify_all()
            try:
                if memo.computation_digest != lookup.computation:
                    raise ExecutionWorkspaceRefusal(
                        "memo_digest_mismatch", "memo answer names another computation"
                    )
                if memo.result_canonical_bytes:
                    lookup.result = _checkpoint(memo.result_canonical_bytes, lookup.destination)
            except ExecutionWorkspaceRefusal as exc:
                lookup.refusal = exc
                raise
