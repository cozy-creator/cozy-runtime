"""A kernel serves only once it is ready here: built, loaded and self-checked, none of it on a
request thread (owner: no compile wait; compiles, self-checks and builds never block a request).

Dispatch asks `ready(name, opener)`, which never blocks a request. While one is in flight
here, the opener runs on a thread of its own and dispatch serves the best kernel already ready,
by its family's own rules. While none is (a preparation, beside the weight download), an opener
whose build is done runs at once, and one whose build is not goes to its own thread. A failure
stays in this process: its family serves the fallback and says why (`failed`). A request that
names its kernel waits for it (`wait`).

A kernel a background opener produced serves from the next boundary (`boundary`): a step or
stage of a request, the start of one, a follower's next call; with no request in flight, the
next ask. A call never sees its kernels change, and the run records each kernel that began
serving and where. `epoch` counts the changes (`LANDED`), for what keys on the kernels serving.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

from cozy_runtime.author import _request_path as request_path
from cozy_runtime.author._activity import own_thread
from cozy_runtime.author._errors import is_capacity
from cozy_runtime.author._model import LANDED
from cozy_runtime.internal import kernel_compile

#: `opener(wait)`: the kernel ready to dispatch. With `wait` it may wait for its build; without,
#: None while the build runs. It raises when the kernel cannot serve here.
type Opener[T] = Callable[[bool], T | None]


@dataclass(slots=True)
class _Slot:
    state: str  # opening | landed | ready | failed
    value: object = None
    error: Exception | None = None


_SLOTS: dict[str, _Slot] = {}
_LOCK = threading.Lock()
#: A background opener's thread name, before its kernel's.
THREAD = "kernel-ready: "


def ready[T](name: str, opener: Opener[T] | None = None) -> T | None:
    """`name`'s kernel once it serves here, else None: opening (started now by `opener` if it
    was not), or failed. Never blocks a request."""
    quiet = not request_path.in_flight()
    with _LOCK:
        slot = _SLOTS.get(name)
        if slot is not None or opener is None:
            if slot is not None and slot.state == "landed" and quiet:
                slot.state = "ready"
                LANDED[0] += 1
            return cast("T", slot.value) if slot is not None and slot.state == "ready" else None
        slot = _SLOTS[name] = _Slot("opening")
    if quiet:
        try:
            value = opener(False)
        except Exception as exc:
            _ended(name, slot, exc)
            if is_capacity(exc):
                raise  # the memory manager's to answer
            return None
        if value is not None:
            _landed(slot, value, "ready")
            return value
    opening = threading.Thread(target=_background, args=(name, slot, opener), daemon=True)
    opening.name = f"{THREAD}{name}"
    opening.start()
    return None


def wait[T](name: str, opener: Opener[T]) -> T:
    """`name`'s kernel, its build waited for on this thread, under this thread's cancel: a
    request that names its kernel holds it for every step (owner, 2026-10-10). Raises why it
    cannot serve."""
    with _LOCK:
        slot = _SLOTS.get(name)
        mine = slot is None
        if slot is None:
            slot = _SLOTS[name] = _Slot("opening")
        elif slot.state == "landed":
            slot.state = "ready"
            LANDED[0] += 1
    if slot.error is not None:
        raise slot.error
    if slot.state == "ready":
        return cast("T", slot.value)
    try:
        value = opener(True)
        if value is None:
            raise RuntimeError("its opener waited and returned nothing")
    except Exception:
        if mine:  # asked again, it opens again: a failed build is on the machine's record
            with _LOCK:
                _SLOTS.pop(name, None)
        raise
    _landed(slot, value, "ready")
    return value


def _background(name: str, slot: _Slot, opener: Opener[object]) -> None:
    own_thread()
    kernel_compile.told.set(False)  # no one waits for it
    try:
        value = opener(True)
        if value is None:
            raise RuntimeError("its opener waited and returned nothing")
    except Exception as exc:
        _ended(name, slot, exc)
        return
    _landed(slot, value, "landed")


def _landed(slot: _Slot, value: object, state: str) -> None:
    with _LOCK:
        slot.value, slot.state = value, state
        LANDED[0] += state == "ready"


def _ended(name: str, slot: _Slot, exc: Exception) -> None:
    """A failed opener's kernel stays failed here; one the device ran out under asks again."""
    with _LOCK:
        if is_capacity(exc):
            _SLOTS.pop(name, None)
        slot.state, slot.error = "failed", exc


def failed(name: str) -> Exception | None:
    """Why `name`'s kernel does not serve in this process, or None."""
    slot = _SLOTS.get(name)
    return slot.error if slot is not None and slot.state == "failed" else None


def opening(name: str) -> bool:
    """Whether `name`'s kernel is on its way: opening, or landed for the next boundary."""
    slot = _SLOTS.get(name)
    return slot is not None and slot.state in ("opening", "landed")


def served(prefix: str) -> list[object]:
    """Every kernel serving here whose name starts with `prefix`."""
    with _LOCK:
        return [s.value for n, s in _SLOTS.items() if n.startswith(prefix) and s.state == "ready"]


def failures(prefix: str) -> dict[str, Exception]:
    """Every kernel whose name starts with `prefix` that failed here, and why."""
    with _LOCK:
        return {n: s.error for n, s in _SLOTS.items() if n.startswith(prefix) and s.error}


def boundary() -> list[str]:
    """A boundary of a request: every kernel that landed since serves from here on. Their
    names."""
    with _LOCK:
        landed = [name for name, slot in _SLOTS.items() if slot.state == "landed"]
        for name in landed:
            _SLOTS[name].state = "ready"
        LANDED[0] += bool(landed)
    return landed


def epoch() -> int:
    """How many times the kernels serving in this process changed."""
    return LANDED[0]
