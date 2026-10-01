"""What the stage scheduler (tracker #297, W4) asks the memo tier (#302, #304).

Not wired yet: the scheduler that calls this is W4's draft (cozy-runtime#1036). This file is
the agreed seam, so neither side re-derives the other's facts. Its notes are in
outputs/comfy-memory-design-20260930/memo-scheduler-interface.md.

- A memo hit opens no scope, so it asks for no turn. The executor still reports it, as a
  `memo.call` observation with outcome `hit:*`, so a stage graph keeps the stage.
- Admission (#302): a request's preflight names its memoized calls; the Worker turns them
  into keys with the executor's published `KeyContext` and probes them. A component every
  declared call of the request hits is left out of its restore, prefetch and prefill.
- Coalescing (#304, on by default): while one request holds a memoized method's components,
  the same turn may compute queued requests' missing calls of that method, one at a time at
  batch 1, when `worth_coalescing` says the reload it saves exceeds the encodes it adds.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import msgspec


class StageCall(msgspec.Struct, frozen=True):
    """One memoized call a request's preflight declares: the method and the canonical digest
    of its bound arguments, computed by the executor with the same code as the call itself."""

    model: str
    method: str
    arguments: str


class KeyContext(msgspec.Struct, frozen=True):
    """What the executor publishes per generation (and again when numerics change) so the
    Worker builds the same key without importing package code."""

    stages: dict[str, str]
    """method -> stage identity (`describe`)"""
    components: dict[str, tuple[str, ...]]
    """method -> declared components"""
    environment: str
    weights: dict[str, str]
    """component -> weights identity"""
    assets: str
    numerics: dict[str, str]
    """method -> numerics digest at the last call"""


class Promise(msgspec.Struct, frozen=True):
    """Admission's answer for one request: the keys it may count on, retained until the
    request ends, and the components none of its declared calls will stage."""

    keys: tuple[str, ...]
    skip: tuple[str, ...]


class MemoAdmission(Protocol):
    def probe(self, request: str, calls: Sequence[StageCall]) -> Promise:
        """Keys present or in flight count as hits; a promised entry is not evicted until
        `release(request)`. Admission can only be wrong in the slower direction: a
        promised hit that misses stages its components lazily."""
        ...

    def release(self, request: str) -> None: ...

    def coalescible(self, holder: str, method: str, queued: Sequence[str]) -> tuple[StageCall, ...]:
        """The missing calls of `method` among `queued` requests that `holder`'s turn may
        compute: same generation and numerics, not cancelled, worth it by the cost model."""
        ...


def worth_coalescing(reload_s: float, encode_s: float, resident: bool) -> bool:
    """Compute one queued request's missing call now when the staging it would pay later
    (measured reload of the method's components) exceeds the warm encode added to this turn.
    An encoder that stays resident anyway saves nothing."""
    return not resident and reload_s > encode_s
