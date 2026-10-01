"""Memoized Model methods: the author's mark and the wrapper's seam to the Runtime (#298).

`@uses_components(..., memoize=True)` marks ONE method as eligible: its result is a pure
function of its code, declared helpers, component weights, arguments and numerics. Whether a
call is served from memory, from the machine's store or computed is the Runtime's decision,
made before the component scope opens, so a hit stages nothing. Nothing here names a package,
model or component; the Runtime body lives in `internal/stage_memo.py`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol

from cozy_runtime.author._errors import ConformanceError


class StageMemo(Protocol):
    """Installed by the Runtime on a constructed generation (`Model._cozy_memo`)."""

    def call(
        self,
        model: object,
        fn: Callable[..., object],
        components: tuple[str, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        run: Callable[[], object],
    ) -> object: ...


_EXACT: ContextVar[list[bool] | None] = ContextVar("cozy_stage_memo_exact", default=None)


def inexact() -> None:
    """Runtime code that changed this call's arithmetic (a tiled retry after OOM) marks its
    result unstorable: it is returned, never memoized."""
    if (flags := _EXACT.get()) is not None:
        flags[0] = False


@contextmanager
def exactness() -> Iterator[list[bool]]:
    flags = [True]
    token = _EXACT.set(flags)
    try:
        yield flags
    finally:
        _EXACT.reset(token)


def memoized(
    fn: Callable[..., Any],
    scoped: Callable[..., Any],
    components: tuple[str, ...],
    *,
    is_async: bool,
) -> Callable[..., Any]:
    """Wrap `scoped` (the scope-opening wrapper) so the lookup runs BEFORE admission."""
    if is_async:
        raise ConformanceError(
            f"{fn.__qualname__}: memoize=True needs a synchronous method; an encoder returns "
            "its result, it does not stream it",
            code="memo_async",
        )

    @functools.wraps(fn)
    def wrapper(self: Any, /, *args: object, **kwargs: object) -> object:
        memo: StageMemo | None = self._cozy_memo
        if memo is None:
            return scoped(self, *args, **kwargs)
        return memo.call(self, fn, components, args, kwargs, lambda: scoped(self, *args, **kwargs))

    wrapper.__dict__["__memoize__"] = True
    return wrapper


def is_memoized(member: object) -> bool:
    return getattr(member, "__memoize__", False) is True
