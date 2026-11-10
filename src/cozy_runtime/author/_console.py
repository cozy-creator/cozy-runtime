"""What a package writes the ordinary way reaches its run: each line it prints and each `logging`
record a logger lets through is one of the run's log lines, a `warnings.warn` is a warning line,
and a tqdm bar is the run's progress (`ctx.progress`). The executor installs this once per process;
outside an attempt, and in the executor's own log files, nothing changes."""

from __future__ import annotations

import contextlib
import importlib
import io
import logging
import os
import sys
import threading
import warnings
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Protocol, TextIO

from cozy_runtime.author._context import CURRENT

if TYPE_CHECKING:
    from cozy_runtime.author._services import Attempt

#: Log lines one attempt sends to its run.
LINES = 2000
#: A tqdm loop of up to this many steps reports each step, timed as `ctx.progress` times it; a
#: longer one reports as often as tqdm redraws.
COUNTED = 10_000
#: The Runtime's own loggers: their records stay in the executor's log.
_OWN = ("cozy_runtime", "tensorfs")

_busy = threading.local()


def attempt_log(attempt: Attempt) -> Callable[[str, str], None]:
    """`attempt`'s log lines (`Context._log`): at most `LINES`, none once it has ended."""
    sent = [0]
    lock = threading.Lock()

    def log(line: str, level: str) -> None:
        with lock:
            sent[0] += 1
            count = sent[0]
        if attempt.closed or count > LINES + 1:
            return
        if count > LINES:
            line, level = f"more than {LINES} log lines: the rest are not shown", "warning"
        attempt.emit("log", line, level)

    return log


def _send(text: str, level: str) -> None:
    context = CURRENT.get()
    log = None if context is None else context._log
    if log is None or getattr(_busy, "on", False):
        return
    _busy.on = True
    try:
        with contextlib.suppress(Exception):  # a line never fails the code that wrote it
            for line in text.splitlines():
                if line.strip():
                    log(line.rstrip(), level)
    finally:
        _busy.on = False


class _Lines(io.TextIOWrapper):
    """`sys.stdout`: written as before, and each whole line an attempt prints is a log line."""

    def __init__(self, stream: io.TextIOWrapper) -> None:
        stream.flush()
        super().__init__(
            stream.buffer, encoding=stream.encoding, errors=stream.errors, line_buffering=True
        )
        self._held = threading.local()

    def write(self, s: str) -> int:
        written = super().write(s)
        if CURRENT.get() is not None:
            whole, _, rest = (getattr(self._held, "text", "") + s).rpartition("\n")
            self._held.text = rest
            if whole:
                _send(whole, "info")
        return written


def _level(number: int) -> str:
    if number >= logging.ERROR:
        return "error"
    if number >= logging.WARNING:
        return "warning"
    return "info" if number >= logging.INFO else "debug"


_call_handlers = logging.Logger.callHandlers


def _logged(self: logging.Logger, record: logging.LogRecord) -> None:
    """Every record a logger lets through (its level, filters), as Python passes it on."""
    _call_handlers(self, record)
    name = record.name
    if CURRENT.get() is None or any(name == own or name.startswith(own + ".") for own in _OWN):
        return
    with contextlib.suppress(Exception):
        text = record.getMessage()
        if record.exc_info:
            text += "\n" + logging.Formatter().formatException(record.exc_info)
        _send(text, _level(record.levelno))


_showwarning = warnings.showwarning


def _warned(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    _showwarning(message, category, filename, lineno, file, line)
    if file is None:
        name = os.path.basename(filename)
        _send(f"{name}:{lineno}: {category.__name__}: {message}", "warning")


class _Bar(Protocol):
    """The part of `tqdm.std.tqdm` read here."""

    n: float
    total: float | None
    disable: bool
    miniters: float
    mininterval: float
    dynamic_miniters: bool


def _step(bar: _Bar) -> None:
    """The bar's count as the attempt's step count; also a cancellation point."""
    context = CURRENT.get()
    if context is None or bar.disable:
        return
    total = bar.total
    if total is None or total < 1:
        context.raise_if_cancelled()
        return
    state = vars(bar)
    done, before = min(int(bar.n), int(total)), state.get("_cozy_done", 0)
    state["_cozy_done"] = done
    if done >= 1 and done != before:
        context.progress(done - 1, int(total))


def _patch_tqdm() -> None:
    try:
        std = importlib.import_module("tqdm.std")
    except ImportError:
        return
    bar = std.tqdm
    update: Callable[[_Bar, float | None], bool | None] = bar.update
    iterate: Callable[[_Bar], Iterator[object]] = bar.__iter__

    def counted(self: _Bar, n: float | None = 1) -> bool | None:
        shown = update(self, n)
        _step(self)
        return shown

    def each(self: _Bar) -> Iterator[object]:
        total = self.total
        if CURRENT.get() is not None and not self.disable and total and total <= COUNTED:
            self.miniters, self.mininterval, self.dynamic_miniters = 1, 0.0, False
        return iterate(self)

    for name, patched in (("update", counted), ("__iter__", each)):
        setattr(bar, name, patched)


_INSTALLED: list[bool] = []


def install() -> None:
    """Once per executor process."""
    if _INSTALLED:
        return
    _INSTALLED.append(True)
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout = _Lines(sys.stdout)
    logging.Logger.callHandlers = _logged  # type: ignore[method-assign]  # every record, once
    warnings.showwarning = _warned
    _patch_tqdm()
