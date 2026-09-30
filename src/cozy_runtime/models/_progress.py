"""Pipeline iteration without console output or background display workers.

Runtime reports model progress through Telemetry. A disabled tqdm display still
starts a monitor thread, so supported pipelines use this adapter instead.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from types import TracebackType
from typing import Literal, Self


class PipelineProgress[T]:
    def __init__(self, iterable: Iterable[T] | None = None, total: int | None = None) -> None:
        if iterable is None and total is None:
            raise ValueError("Either total or iterable must be defined")
        self.iterable = iterable

    def __iter__(self) -> Iterator[T]:
        if self.iterable is None:
            raise TypeError("progress has no iterable")
        return iter(self.iterable)

    def update(self, n: int = 1) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        return False
