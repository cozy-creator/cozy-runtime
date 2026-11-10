"""The group an attempt runs on, as the Runtime's own helpers see it (`spread`, caches, saves).

Every GPU of a group runs the same handler (`internal/parallel/spmd`). The executor installs
the attempt's group here for its duration; None outside a group attempt."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Protocol


class Group(Protocol):
    rank: int
    world: int

    def all(self, flag: bool, what: str) -> bool:
        """True on every GPU when `flag` is True on every GPU."""
        ...

    def spread[*Ts, R](
        self, method: Callable[[*Ts], R], arguments: Iterable[tuple[*Ts]]
    ) -> Iterator[R]:
        """`method(*args)` per entry, each on one GPU, every result on every GPU, in order."""
        ...


#: The running attempt's group, set by the executor (one group attempt runs at a time).
CURRENT: list[Group | None] = [None]
