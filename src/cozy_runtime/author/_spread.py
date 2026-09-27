"""Spread one component's pure method calls over the ranks of its group."""

from collections.abc import Iterable, Iterator
from typing import Any


def spread(component: Any, method: str, arguments: Iterable[tuple[Any, ...]]) -> Iterator[Any]:
    """`getattr(component, method)(*args)` for each entry of `arguments`, results in order.

    The author's statement: the method is a pure function of its tensor and JSON arguments
    and returns one tensor. Inside a sequence-parallel group the runtime runs the calls on
    the ranks with room, rank 0 included, a round at a time; each result is what rank 0
    alone computes on an identical card. Elsewhere they run here, one after another.
    """
    runner = getattr(component, "_cozy_spread", None)
    calls = iter(arguments)
    if runner is None:
        local = getattr(component, method)
        return (local(*args) for args in calls)
    result: Iterator[Any] = runner(method, calls)
    return result
