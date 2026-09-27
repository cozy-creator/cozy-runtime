"""Spread one component's pure method calls over the ranks of its group."""

from collections.abc import Iterable, Iterator
from typing import Any


def spread(
    component: Any,
    method: str,
    arguments: Iterable[tuple[Any, ...]],
    *,
    spare: bool = False,
) -> Iterator[Any]:
    """`getattr(component, method)(*args)` for each entry of `arguments`, results in order.

    The author's statement: the method is a pure function of its tensor and JSON arguments
    and returns one tensor. Inside a sequence-parallel group the runtime offers the calls to
    the idle ranks, rank 0 included, a round at a time; each result is what rank 0 alone
    computes on an identical card. A follower without room for the call declines it and the
    call runs on another rank. Elsewhere they run here, one after another.

    `spare=True`: no follower may free anything to take a call, because the attempt still
    needs what the followers hold (a spread before the sharded denoise). A follower homing a
    `placeable` component always spares.
    """
    runner = getattr(component, "_cozy_spread", None)
    calls = iter(arguments)
    if runner is None:
        local = getattr(component, method)
        return (local(*args) for args in calls)
    result: Iterator[Any] = runner(method, calls, spare)
    return result
