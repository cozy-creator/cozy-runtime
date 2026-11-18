"""A call's own place (cl-984). TensorD names a folder inside the call's project with its Invoke
(`scratch`; the machine's for a call of no project). An executor runs one Invoke at a time, so
while it runs, the process's temporary directory, working directory, HOME (and the XDG homes)
and fds 1 and 2 are the call's: `tmp/`, `work/`, `home/`, `stdout.log` and `stderr.log` in that
folder. After the call they are put back and its `work/`, `tmp/` and `home/` removed; the logs
stay for TensorD.

The one thing beside an Invoke is TensorD's lookahead `PrepareRequest` for the next request,
maybe another project's. What it prints is kept with its request and written to that request's
own log when its Invoke comes (`preparing`). Its `preflight` only resolves the payload: it gets
no file bytes and writes no files.
"""

from __future__ import annotations

import collections
import contextlib
import contextvars
import os
import shutil
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import IO, cast

#: What a preparation printed, by request, until its Invoke (as many as the executor holds).
_HELD: collections.OrderedDict[str, list[tuple[str, str]]] = collections.OrderedDict()
_HELD_REQUESTS, _HELD_BYTES = 32, 64 << 10
_LOCK = threading.Lock()
#: Held by the placed call: a second one waits rather than move the process under the first.
_ONE = threading.Lock()
#: The request whose preparation this command serves.
_PREPARING: contextvars.ContextVar[str] = contextvars.ContextVar("cozy_preparing", default="")


class _Placed:
    """The call that holds the process now: its names, and the process's own temporary directory,
    values of those names (None: unset) and working directory."""

    names: dict[str, str] = {}
    tempdir = ""
    saved: dict[str, str | None] = {}
    cwd = ""


def names() -> dict[str, str]:
    """The environment names the running call holds as its own (none outside a placed call)."""
    return _Placed.names


def process_tempdir() -> str:
    """The process's temporary directory, during a placed call too: for the Runtime's own files."""
    return _Placed.tempdir or tempfile.gettempdir()


def process_environment() -> dict[str, str]:
    """The process's own environment, during a placed call too: for a child that may outlive the
    call (a kernel builder), so it never lives in the call's place."""
    environ, saved = dict(os.environ), _Placed.saved
    for name, value in saved.items():
        if value is None:
            environ.pop(name, None)
        else:
            environ[name] = value
    return environ


def process_cwd() -> str | None:
    """The process's own working directory during a placed call; None outside one (it is the
    current one)."""
    return _Placed.cwd or None


def _own(scratch: Path) -> dict[str, str]:
    home, tmp = scratch / "home", str(scratch / "tmp")
    own = {"HOME": str(home), "TMPDIR": tmp, "TEMP": tmp, "TMP": tmp}
    own["XDG_CONFIG_HOME"] = str(home / ".config")
    own["XDG_CACHE_HOME"] = str(home / ".cache")
    own["XDG_DATA_HOME"] = str(home / ".local" / "share")
    own["XDG_STATE_HOME"] = str(home / ".local" / "state")
    own["MPLCONFIGDIR"] = str(home / ".config" / "matplotlib")
    if sys.platform == "win32":
        own["USERPROFILE"] = str(home)
    return own


@contextlib.contextmanager
def entered(scratch: Path | None, request: str) -> Iterator[bool]:
    """One Invoke in its own place (`scratch`); none: the executor's, as before. Yields whether
    it is placed."""
    if scratch is None:
        _replay(request)
        yield False
        return
    if not scratch.is_absolute():
        raise ValueError(f"a call's scratch is an absolute path, not {scratch}")
    with _ONE:
        with _placed(scratch, request):
            yield True


@contextlib.contextmanager
def _placed(scratch: Path, request: str) -> Iterator[None]:
    work, tmp, home = scratch / "work", scratch / "tmp", scratch / "home"
    for directory in (scratch, work, tmp, home):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    own = _own(scratch)
    environ, saved = os.environ, {name: os.environ.get(name) for name in own}
    process_tmp, cwd = tempfile.gettempdir(), os.getcwd()
    logs = [_log(scratch / "stdout.log"), _log(scratch / "stderr.log")]
    _flush()  # what the process wrote stays the process's
    kept = [os.dup(1), os.dup(2)]
    try:
        for fd, log in zip((1, 2), logs, strict=True):
            os.dup2(log, fd)
        _Placed.saved, _Placed.cwd = saved, cwd  # before the call's stand in for them
        environ.update(own)
        _Placed.names, _Placed.tempdir = own, process_tmp
        tempfile.tempdir = str(tmp)
        os.chdir(work)
        _replay(request)
        yield
    finally:
        _flush()
        for fd, back in zip((1, 2), kept, strict=True):
            os.dup2(back, fd)
        for descriptor in (*kept, *logs):
            os.close(descriptor)
        with contextlib.suppress(OSError):
            os.chdir(cwd)
        tempfile.tempdir = process_tmp
        _Placed.names, _Placed.tempdir = {}, ""
        for name, value in saved.items():
            if value is None:
                environ.pop(name, None)
            else:
                environ[name] = value
        _Placed.saved, _Placed.cwd = {}, ""  # after the process's are back
        for directory in (work, tmp, home):
            shutil.rmtree(directory, ignore_errors=True)


@contextlib.contextmanager
def preparing(request: str) -> Iterator[None]:
    """A request's preparation: what it prints is held for its own call's log (`entered`)."""
    token = _PREPARING.set(request)
    try:
        yield
    finally:
        _PREPARING.reset(token)


def install() -> None:
    """Route Python's stdout and stderr (`_Writer`), once, before any package is imported."""
    if not isinstance(sys.stdout, _Writer):
        sys.stdout = cast("IO[str]", _Writer(sys.stdout, "stdout"))
        sys.stderr = cast("IO[str]", _Writer(sys.stderr, "stderr"))


class _Writer:
    """`sys.stdout` (`sys.stderr`): a preparation writes to its request's held output; anything
    else to the process's stream, which is the running call's log while one runs."""

    def __init__(self, stream: IO[str], name: str) -> None:
        self.stream, self._name = stream, name

    def write(self, text: str) -> int:
        request = _PREPARING.get()
        if not request:
            return self.stream.write(text)
        with _LOCK:
            held = _HELD.setdefault(request, [])
            _HELD.move_to_end(request)
            held.append((self._name, text))
            while len(held) > 1 and sum(len(t) for _, t in held) > _HELD_BYTES:
                held.pop(0)
            while len(_HELD) > _HELD_REQUESTS:
                _HELD.popitem(last=False)
        return len(text)

    def flush(self) -> None:
        if not _PREPARING.get():
            self.stream.flush()

    def __getattr__(self, name: str) -> object:  # the stream's own
        return getattr(self.stream, name)


def _replay(request: str) -> None:
    """What `request`'s preparation printed, into its own call's streams."""
    with _LOCK:
        held = _HELD.pop(request, [])
    for name, text in held:
        stream = getattr(sys, name)
        (stream.stream if isinstance(stream, _Writer) else stream).write(text)


def _flush() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(OSError, ValueError):
            stream.flush()


def _log(path: Path) -> int:
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC, 0o600)
