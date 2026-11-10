"""Memoized Model methods: the author's mark and the wrapper's seam to the Runtime (#298).

`@memoize` marks ONE method as eligible: its result is a pure
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
from typing import Concatenate, Protocol, cast

from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._handler import Handler


class StageMemo(Protocol):
    """Installed by the Runtime on a constructed generation (`Model._cozy_memo`)."""

    def call(
        self,
        model: object,
        fn: Handler,
        components: tuple[str, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        run: Callable[[], object],
    ) -> object: ...


class Memoizing(Protocol):
    """A model whose methods `memoized` wraps: the Runtime installs its `StageMemo`."""

    @property
    def _cozy_memo(self) -> StageMemo | None: ...


_EXACT: ContextVar[list[bool] | None] = ContextVar("cozy_stage_memo_exact", default=None)


def inexact() -> None:
    """Runtime code that changed this call's arithmetic (a lossy retry after OOM) marks its
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


def memoized[M: Memoizing, **P, R](
    fn: Callable[Concatenate[M, P], R],
    scoped: Callable[Concatenate[M, P], R],
    components: tuple[str, ...],
    *,
    is_async: bool,
) -> Callable[Concatenate[M, P], R]:
    """Wrap `scoped` (the scope-opening wrapper) so the lookup runs BEFORE admission."""
    if is_async:
        raise ConformanceError(
            f"{fn.__qualname__}: @memoize needs a synchronous method; an encoder returns "
            "its result, it does not stream it",
            code="memo_async",
        )

    @functools.wraps(fn)
    def wrapper(self: M, /, *args: P.args, **kwargs: P.kwargs) -> R:
        memo = self._cozy_memo
        if memo is None:
            return scoped(self, *args, **kwargs)
        found = memo.call(self, fn, components, args, kwargs, lambda: scoped(self, *args, **kwargs))
        return cast("R", found)  # `run()`'s result, or the stored copy of one

    wrapper.__dict__["__memoize__"] = True
    return wrapper


def is_memoized(member: object) -> bool:
    return getattr(member, "__memoize__", False) is True
