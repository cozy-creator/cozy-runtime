"""Kernel work that blocks a request: a compile, tune, self-check, build or CUDA graph capture on
the executor's request thread while it serves a request (`serving`). Each is a row the attempt
records as `request_path.blocking` with a warning; none fails the run. Work on another thread runs
beside the request and is none: a run starts on a kernel that is ready while better ones build in
the background (owner, 2026-10-09). A request waiting on another's turn says whom it waits on
(`wait`)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

#: The request path's known violations, by the function holding each; nothing else is allowed.
KNOWN = {
    # cr-210: a graphed component records a signature at its second call, on the request thread.
    "cozy_runtime.graphs._record": "cr-210",
    # cr-210: LM decode records each step it replays on the thread of the request driving it.
    "cozy_runtime.lm._Engine._replay": "cr-210",
}
_LOCAL = threading.local()
#: This process's threads serving a request now.
_SERVING = [0]
_COUNTED = threading.Lock()
#: Tells the machine the stage of the call this thread serves (the executor's, when it serves).
told: Callable[[str], None] | None = None


@contextmanager
def serving() -> Iterator[dict[str, tuple[float, int]]]:
    """This thread serves a request: what blocked it, as work -> (ms, times). Re-entered, the
    request's own rows."""
    held = getattr(_LOCAL, "rows", None)
    rows = _LOCAL.rows = {} if held is None else held
    with _COUNTED:
        _SERVING[0] += 1
    try:
        yield rows
    finally:
        _LOCAL.rows = held
        with _COUNTED:
            _SERVING[0] -= 1


def in_flight() -> bool:
    """Whether any thread of this process serves a request now."""
    return _SERVING[0] > 0


def note(work: str, ms: float) -> None:
    """`work` took `ms`: a row on a request thread, nothing elsewhere."""
    rows = getattr(_LOCAL, "rows", None)
    if rows is not None and ms >= 1:  # a wait that was none is no row
        total, times = rows.get(work, (0.0, 0))
        rows[work] = (total + ms, times + 1)


@contextmanager
def _quiet() -> Iterator[None]:
    rows = getattr(_LOCAL, "rows", None)
    _LOCAL.rows = None
    try:
        yield
    finally:
        _LOCAL.rows = rows


@contextmanager
def forbid(work: str) -> Iterator[None]:
    """Work that must never block a request, timed: one row, which covers guarded work inside
    it. Also a decorator."""
    started = time.perf_counter()
    try:
        with _quiet():
            yield
    finally:
        note(work, (time.perf_counter() - started) * 1000)


def known(site: str) -> AbstractContextManager[None]:
    """`site`'s violation, which `KNOWN` names: the guarded work inside is no row."""
    return _quiet()


def wait(
    changed: threading.Condition, ready: Callable[[], bool], holder: Callable[[], str]
) -> None:
    """Wait on `changed`, held, until `ready()`: woken by the holders' own notifications, never
    a clock. Each holder `holder()` names meanwhile is told as this call's stage ("waiting for
    run 17"), outside the lock."""
    named = ""
    while not ready():
        now = holder()
        if told is None or not now or now == named:
            changed.wait()
            continue
        named = now
        changed.release()
        try:
            told(f"waiting for {named}")
        finally:
            changed.acquire()
