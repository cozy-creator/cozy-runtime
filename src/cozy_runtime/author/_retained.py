"""`retained`: state a model keeps across steps, held by the Runtime's memory.

A step that reads again what an earlier step wrote (a prefix's K/V, computed in step 0 and read
by every later step) puts it here instead of keeping the tensors. Each `put` becomes memory the
Runtime manages like weights: resident while the card has room, else evicted and refilled from a
pinned host copy taken behind compute, and filled ahead of its next use. Its bytes count in the
room every other decision sees. The state dies with the loop that made it: the store is closed
(or left as a `with`) when that loop returns, and its memory is free for what comes after.
Outside an executor (plain Python) the store holds the tensors as they are.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor


class State(Protocol):
    """What one `put` holds."""

    def get(self) -> tuple[Tensor, ...]:
        """The tensors `put` was given, as read-only views valid while they live. They hold the
        state on the card while they live: read each state where it is used, so a step holds
        only those (qwen-image-2 read 32 layers' prefix K/V at once, 24 GB, run 47471)."""
        ...


class Store(Protocol):
    def put(self, *tensors: Tensor) -> State:
        """Hold `tensors` from now on: the caller drops its own and keeps the returned `State`."""
        ...

    def close(self) -> None:
        """The loop is done: every state goes now, whoever still holds one."""
        ...

    def __enter__(self) -> Store: ...

    def __exit__(self, *_: object) -> None: ...


class Memory(Protocol):
    def retained(self, nbytes: int) -> Store: ...


class _Held:
    def __init__(self, tensors: tuple[Tensor, ...]) -> None:
        self.tensors: tuple[Tensor, ...] | None = tensors

    def get(self) -> tuple[Tensor, ...]:
        if self.tensors is None:
            raise RuntimeError("retained state read after its loop ended")
        return self.tensors


class Plain:
    """A store of the tensors as they are: where no Runtime memory runs."""

    def __init__(self) -> None:
        self.states: list[_Held] = []

    def put(self, *tensors: Tensor) -> State:
        self.states.append(_Held(tensors))
        return self.states[-1]

    def close(self) -> None:
        for state in self.states:
            state.tensors = None
        self.states.clear()

    def __enter__(self) -> Plain:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


_RUNTIME: Memory | None = None


def install(runtime: Memory | None) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def retained(nbytes: int) -> Store:
    """Where a loop keeps `nbytes` of cross-step state, decided once, before it. Always a store:
    the Runtime's memory keeps trying to hold it (on the card, else spilled to host). Close the
    store when the loop returns."""
    return Plain() if _RUNTIME is None else _RUNTIME.retained(nbytes)
