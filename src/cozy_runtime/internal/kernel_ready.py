"""A kernel serves only once it is ready here: built, loaded and self-checked, none of it on a
request thread (owner: no compile wait; compiles, self-checks and builds never block a request).

Dispatch asks `ready(name, opener)`, which never blocks a request. While one is in flight
here, the opener runs on a thread of its own and dispatch serves the best kernel already ready,
by its family's own rules. While none is (a preparation, beside the weight download), an opener
whose build is done runs at once, and one whose build is not goes to its own thread. A failure
stays in this process: its family serves the fallback and says why (`failed`).

A kernel a background opener produced serves from the next boundary (`boundary`): a step or
stage of a request, the start of one, a follower's next call; with no request in flight, the
next ask. A call never sees its kernels change, and the run records each kernel that began
serving and where. `epoch` counts the changes (`LANDED`), for what keys on the kernels serving.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cozy_runtime.author import _request_path as request_path
from cozy_runtime.author._activity import own_thread
from cozy_runtime.author._errors import is_capacity
from cozy_runtime.author._model import LANDED

#: `opener(wait)`: the kernel ready to dispatch. With `wait` it may wait for its build; without,
#: None while the build runs. It raises when the kernel cannot serve here.
type Opener[T] = Callable[[bool], T | None]


@dataclass(slots=True)
class _Slot:
    state: str  # opening | landed | ready | failed
    value: Any = None
    detail: str = ""


_SLOTS: dict[str, _Slot] = {}
_LOCK = threading.Lock()


def ready[T](name: str, opener: Opener[T]) -> T | None:
    """`name`'s kernel once it serves here, else None (opening it, or failed); never blocks a
    request."""
    quiet = not request_path.in_flight()
    with _LOCK:
        slot = _SLOTS.get(name)
        if slot is not None:
            if slot.state == "landed" and quiet:
                slot.state = "ready"
                LANDED[0] += 1
            return slot.value if slot.state == "ready" else None
        slot = _SLOTS[name] = _Slot("opening")
    if quiet:
        try:
            value = opener(False)
        except Exception as exc:
            _ended(name, slot, exc)
            return None
        if value is not None:
            with _LOCK:
                slot.value, slot.state = value, "ready"
                LANDED[0] += 1
            return value
    opening = threading.Thread(target=_open, args=(name, slot, opener), daemon=True)
    opening.start()
    return None


def _open(name: str, slot: _Slot, opener: Opener[Any]) -> None:
    own_thread()
    try:
        value = opener(True)
        if value is None:
            raise RuntimeError("its opener waited and returned nothing")
    except Exception as exc:
        _ended(name, slot, exc)
        return
    with _LOCK:
        slot.value, slot.state = value, "landed"


def _ended(name: str, slot: _Slot, exc: Exception) -> None:
    """A failed opener's kernel stays failed here; one the device ran out under asks again."""
    with _LOCK:
        if is_capacity(exc):
            _SLOTS.pop(name, None)
        else:
            slot.state, slot.detail = "failed", f"{type(exc).__name__}: {exc}"[:300]


def failed(name: str) -> str:
    """Why `name`'s kernel does not serve in this process, or ""."""
    slot = _SLOTS.get(name)
    return slot.detail if slot is not None and slot.state == "failed" else ""


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
