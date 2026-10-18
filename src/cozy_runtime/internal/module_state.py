"""What a module tree's forward can change that no journal sees: its plain attributes (a step
counter), the tensors it holds by identity (a cache swapped), and its public lists and dicts.
Read before a run that may happen again (a replay, a measurement pass) and put back after it,
so a call runs the same however often it runs. Imports no torch."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn import Module

type Pair = tuple[Module, str]
type Taken = list[tuple[Module, str, object]]

_PLAIN = (int, float, bool, str, bytes, type(None))
_MISSING = object()


def _kept(name: str, value: object) -> bool:
    if isinstance(value, _PLAIN) or hasattr(value, "untyped_storage"):
        return True
    return not name.startswith("_") and isinstance(value, list | dict)


def _capture(value: object) -> object:
    """A value as it is now: a container with a copy of its items."""
    if isinstance(value, list):
        return (value, list(value))
    if isinstance(value, dict):
        return (value, dict(value))
    return value


def _same(a: object, b: object) -> bool:
    if a is b:
        return True
    if isinstance(a, tuple) and isinstance(b, tuple):  # captured containers
        x, y = a[1], b[1]
        if a[0] is not b[0] or type(x) is not type(y) or len(x) != len(y):
            return False
        if isinstance(x, dict) and isinstance(y, dict):
            return x.keys() == y.keys() and all(x[k] is y[k] for k in x)
        return all(p is q for p, q in zip(x, y, strict=True))
    return isinstance(a, _PLAIN) and type(a) is type(b) and a == b


def snapshot(module: Module) -> dict[Pair, object]:
    return {
        (m, name): _capture(value)
        for m in module.modules()
        for name, value in vars(m).items()
        if _kept(name, value)
    }


def changed(before: dict[Pair, object], after: dict[Pair, object]) -> tuple[Pair, ...]:
    return tuple(
        k for k in {**before, **after} if not _same(before.get(k, _MISSING), after.get(k, _MISSING))
    )


def take(pairs: Iterable[Pair]) -> Taken:
    return [(m, n, _capture(vars(m).get(n, _MISSING))) for m, n in pairs]


def put_back(taken: Taken) -> None:
    for module, name, value in taken:
        if value is _MISSING:
            vars(module).pop(name, None)
            continue
        if isinstance(value, tuple) and isinstance(value[0], list | dict):
            container, items = value
            if isinstance(container, list):
                container[:] = items
            else:
                container.clear()
                container.update(items)
            value = container
        setattr(module, name, value)


def restore(module: Module, before: dict[Pair, object]) -> None:
    """Put the whole tree back as `before` found it."""
    put_back([(m, n, before.get((m, n), _MISSING)) for m, n in changed(before, snapshot(module))])
