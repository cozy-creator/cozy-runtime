"""`retained`: state a model keeps across steps, held by the Runtime's memory.

A step that reads again what an earlier step wrote (a prefix's K/V, computed in step 0 and read
by every later step) puts it here instead of keeping the tensors. Each `put` becomes memory the
Runtime manages like weights: resident while the card has room, else evicted and refilled from a
pinned host copy taken behind compute, and filled ahead of its next use. Its bytes count in the
room every other decision sees.
"""

from __future__ import annotations

from typing import Any, Protocol


class State(Protocol):
    """What one `put` holds."""

    def get(self) -> tuple[Any, ...]:
        """The tensors `put` was given, as read-only views valid while they live."""
        ...


class Store(Protocol):
    def put(self, *tensors: Any) -> State:
        """Hold `tensors` from now on: the caller drops its own and keeps the returned `State`."""
        ...


class Memory(Protocol):
    def retained(self, nbytes: int) -> Store | None: ...


_RUNTIME: Memory | None = None


def install(runtime: Memory | None) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def retained(nbytes: int) -> Store | None:
    """Where a call keeps `nbytes` of cross-step state, decided once, before its loop: None when
    neither the card nor host RAM has room for them, and the call recomputes that state instead
    of keeping it (the general fallback, never a model's special case)."""
    if _RUNTIME is None:
        raise RuntimeError("retained state is held inside a Runtime executor")
    return _RUNTIME.retained(nbytes)
