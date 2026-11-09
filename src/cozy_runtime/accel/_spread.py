"""Spread one component's pure method calls over the GPUs of its group."""

from collections.abc import Callable, Iterable, Iterator
from typing import cast


def spread[*Ts, R](
    method: Callable[[*Ts], R],
    arguments: Iterable[tuple[*Ts]],
    *,
    spare: bool = False,
) -> Iterator[R]:
    """`method(*args)` for each entry of `arguments`, results in order. `method` is a bound
    method of a component (`vae.encode`).

    The author's statement: the method is a pure function of its tensor and JSON arguments
    and returns one tensor. Inside a sequence-parallel group the runtime offers the calls to
    the idle GPUs, this one included, a round at a time; each result is what one GPU alone
    computes on an identical card. A GPU whose memory refuses the call up front declines it
    and the call runs on another; one that runs out partway fails the call. Elsewhere they run
    here, one after another.

    `spare=True`: no other GPU may free anything to take a call, because the attempt still
    needs what they hold (a spread before the sharded denoise). A GPU homing a `placeable`
    component always spares.
    """
    component = getattr(method, "__self__", None)
    name = getattr(method, "__name__", None)
    if component is None or not isinstance(name, str):
        raise TypeError("spread takes a bound method of a component: spread(vae.encode, args)")
    runner = getattr(component, "_cozy_spread", None)
    calls = iter(arguments)
    if runner is None:
        return (method(*args) for args in calls)
    return cast("Iterator[R]", runner(name, calls, spare))
