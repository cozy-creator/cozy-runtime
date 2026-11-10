"""Spread one component's pure method calls over the GPUs of its group."""

from collections.abc import Callable, Iterable, Iterator

from cozy_runtime.author import _group


def spread[*Ts, R](
    method: Callable[[*Ts], R],
    arguments: Iterable[tuple[*Ts]],
    *,
    spare: bool = False,
) -> Iterator[R]:
    """`method(*args)` for each entry of `arguments`, results in order. `method` is a bound
    method of a component (`vae.encode`).

    The author's statement: the method is a pure function of its tensor and JSON arguments
    and returns one tensor. Inside a sequence-parallel group call n runs on GPU n mod K and
    its result crosses to every GPU; each result is what one GPU alone computes on an
    identical card. Elsewhere they run here, one after another. `spare` has no effect.
    """
    component = getattr(method, "__self__", None)
    name = getattr(method, "__name__", None)
    if component is None or not isinstance(name, str):
        raise TypeError("spread takes a bound method of a component: spread(vae.encode, args)")
    group = _group.CURRENT[0]
    if group is None or group.world < 2:
        return (method(*args) for args in arguments)
    return group.spread(method, arguments)
