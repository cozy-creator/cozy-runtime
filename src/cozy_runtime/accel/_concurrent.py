"""Independent Model calls, run in the order given."""

from __future__ import annotations

from collections.abc import Callable
from typing import overload


@overload
def concurrently[A](a: Callable[[], A], /) -> tuple[A]: ...
@overload
def concurrently[A, B](a: Callable[[], A], b: Callable[[], B], /) -> tuple[A, B]: ...
@overload
def concurrently[A, B, C](
    a: Callable[[], A], b: Callable[[], B], c: Callable[[], C], /
) -> tuple[A, B, C]: ...
@overload
def concurrently[A, B, C, D](
    a: Callable[[], A], b: Callable[[], B], c: Callable[[], C], d: Callable[[], D], /
) -> tuple[A, B, C, D]: ...
@overload
def concurrently[T](*calls: Callable[[], T]) -> tuple[T, ...]: ...
def concurrently(*calls: Callable[[], object]) -> tuple[object, ...]:
    """Run independent calls; the results, in call order.

    The author's statement: no call reads what another writes before this returns, and each
    returns its result rather than publishing it through shared state. The Runtime runs them
    in the order given: every GPU of a group runs this same program, and calls overlapped on
    threads would meet the group's collectives in a different order on each GPU. The first
    failure is raised and the calls after it do not run.
    """
    return tuple(call() for call in calls)
