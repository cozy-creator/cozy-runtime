"""Watcher threads: they tick while other threads work (a sampler, a reporter, a timeout), so their
ticks show nothing of the work. TensorD's hang watch (cl-977) reads no life in a thread whose name
begins with `~`; each is named so in the kernel too (Python before 3.14 names no thread there)."""

from __future__ import annotations

import ctypes
import sys
import threading
from collections.abc import Callable

WATCHER = "~"
_PR_SET_NAME = 15


def native_name(name: str) -> None:
    """The calling thread's name in the kernel too (the kernel keeps 15 bytes)."""
    if sys.platform == "linux":
        ctypes.CDLL(None).prctl(_PR_SET_NAME, name.encode()[:15], 0, 0, 0)


def thread(target: Callable[[], object], name: str) -> threading.Thread:
    """An unstarted daemon thread running `target`, named `~name`."""
    named = f"{WATCHER}{name}"

    def run() -> None:
        native_name(named)
        target()

    return threading.Thread(target=run, name=named, daemon=True)
