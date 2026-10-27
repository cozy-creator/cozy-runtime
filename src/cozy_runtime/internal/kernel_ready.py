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

Every opener ends in `verified`: the kernel serves once its probe agrees with the exact law on
this card (`verify`). A call asks through `serve`, which keeps a CUDA graph's recording whole.

In a group every GPU switches at the same boundary, or none does. Each follower states its
kernels in its load's and every call's reply (`pending`, `report`); at its boundary rank 0 serves
what landed on some GPU and opens or failed on none, and sends that with its next call to each
follower, which serves it from that call on. A GPU that never asked for an agreed kernel opens it
at its first ask: it is built on this machine, since another GPU serves it.
"""

from __future__ import annotations

import re
import sys
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

from cozy_runtime.author import _request_path as request_path
from cozy_runtime.author._errors import is_capacity
from cozy_runtime.author._model import LANDED, Building
from cozy_runtime.internal import kernel_compile, verify

if TYPE_CHECKING:
    from cozy_runtime.author.kernels import Launch

#: `opener(wait)`: the kernel ready to dispatch. With `wait` it may wait for its build; without,
#: None while the build runs. It raises when the kernel cannot serve here.
type Opener[T] = Callable[[bool], T | None]


@dataclass(slots=True)
class _Slot:
    state: str  # opening | landed | ready | failed
    value: object = None
    error: Exception | None = None


_SLOTS: dict[str, _Slot] = {}
_LOCK = threading.RLock()
# Explicit wait may overlap a background opener for the same slot. Track actual opener
# lifetimes as well as slot state; a second publisher must not hide a still-running probe.
_OPENERS = 0
_RECORDING = threading.local()
#: In a group: this GPU's index as its slots name it (`cuda:1`), shared as `cuda:*`, and the
#: group's size. Alone: None.
_GROUP: list[tuple[re.Pattern[str], int] | None] = [None]
#: In a group, the kernels every GPU serves: rank 0's decisions so far, as the group names them.
_AGREED: set[str] = set()
#: Rank 0 of a group: each follower's kernels not serving yet, as its latest reply stated them.
_REPORTS: dict[int, dict[str, str]] = {}
#: A background opener's thread name, before its kernel's.
THREAD = "kernel-ready: "
#: `triton_kernels.opened(kernels, probe, wait, *, device)`: a Runtime Triton set's opener,
#: bound by `triton_kernels` when the executor imports it past its device boundary (the
#: families that open one import no torch).
TRITON: list[Callable[..., Launch | None]] = []


class Refused(RuntimeError):
    """A kernel whose probe disagreed with the exact law on this card: it never serves here."""


@contextmanager
def recording() -> Iterator[None]:
    """Refuse cold GPU probes before capture and exclude new openers until capture ends.

    Do not wait for compilation/probing on a request. The existing slot lock is reentrant
    so capture may query already-ready kernels through serve(); it cannot open a new one.
    """
    with _LOCK:
        opening = next((name for name, slot in _SLOTS.items() if slot.state == "opening"), None)
        if opening is not None or _OPENERS:
            raise Building(opening or "another kernel probe is still opening")
        depth = getattr(_RECORDING, "depth", 0)
        _RECORDING.depth = depth + 1
        try:
            yield
        finally:
            _RECORDING.depth = depth


@contextmanager
def _opening(name: str) -> Iterator[None]:
    global _OPENERS
    with _LOCK:
        if getattr(_RECORDING, "depth", 0):
            raise Building(name)
        _OPENERS += 1
    try:
        yield
    finally:
        with _LOCK:
            _OPENERS -= 1


def ready[T](name: str, opener: Opener[T] | None = None) -> T | None:
    """`name`'s kernel once it serves here, else None: opening (started now by `opener` if it
    was not), or failed. Never blocks a request, except to open a kernel its group serves."""
    quiet = not request_path.in_flight()
    grouped = _GROUP[0] is not None
    if grouped and opener is not None and _AGREED and _shared(name) in _AGREED:
        held = _SLOTS.get(name)
        if held is None or held.state != "ready":
            return wait(name, opener)
    with _LOCK:
        slot = _SLOTS.get(name)
        if slot is not None or opener is None:
            if slot is not None and slot.state == "landed" and quiet and not grouped:
                slot.state = "ready"
                LANDED[0] += 1
            return cast("T", slot.value) if slot is not None and slot.state == "ready" else None
        if getattr(_RECORDING, "depth", 0):
            raise Building(name)
        slot = _SLOTS[name] = _Slot("opening")
    if quiet:
        try:
            with _opening(name):
                value = opener(False)
        except Exception as exc:
            _ended(name, slot, exc)
            if is_capacity(exc):
                raise  # the memory manager's to answer
            return None
        if value is not None:
            selected = _landed(name, slot, value, "landed" if grouped else "ready")
            return None if grouped else cast("T | None", selected)
    opening = threading.Thread(target=_background, args=(name, slot, opener), daemon=True)
    opening.name = f"{THREAD}{name}"
    opening.start()
    return None


def serve[T](name: str, opener: Opener[T], capturing: bool) -> T | None:
    """`ready`, for a call. While its CUDA graph records (`capturing`) an open would end the
    recording: a kernel not ready yet raises `Building`, so the graph records once it lands, and
    one failed for good is None, so the graph records the fallback."""
    if not capturing:
        return ready(name, opener)
    value = cast("T | None", ready(name))
    if value is None and failed(name) is None:
        raise Building(name)
    return value


def verified[T](
    value: T, kernel: str, artifact: str, sm: int, probe: Callable[[T], verify.Verdict]
) -> T:
    """An opener's last step: `value` once `kernel` (`artifact`'s bytes) passes its probe on this
    card (`verify.verified`), said once; raises `Refused` with why it did not."""
    verdict = verify.verified(kernel, artifact, sm, partial(probe, value))
    said = "verified" if verdict.ok else "refused"
    print(f"[kernels] {kernel} {said} on sm{sm}: {verdict.detail}"[:400], file=sys.stderr)
    if not verdict.ok:
        raise Refused(verdict.detail)
    return value


def wait[T](name: str, opener: Opener[T]) -> T:
    """`name`'s kernel, its build waited for on this thread, under this thread's cancel: a
    request that names its kernel holds it for every step (owner, 2026-10-10). Raises why it
    cannot serve."""
    with _LOCK:
        slot = _SLOTS.get(name)
        mine = slot is None
        if getattr(_RECORDING, "depth", 0) and (slot is None or slot.state != "ready"):
            raise Building(name)
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
        with _opening(name):
            value = opener(True)
        if value is None:
            raise RuntimeError("its opener waited and returned nothing")
    except Exception:
        if mine:  # asked again, it opens again: a failed build is on the machine's record
            with _LOCK:
                if _SLOTS.get(name) is slot and slot.state == "opening":
                    _SLOTS.pop(name)
        raise
    with _LOCK:  # its slot may be gone (another opener failed): this verified kernel serves then
        current = _SLOTS.setdefault(name, _Slot("opening"))
        if current.error is not None:
            raise current.error
        return cast("T", _landed(name, current, value, "ready"))


def _background(name: str, slot: _Slot, opener: Opener[object]) -> None:
    kernel_compile.told.set(False)  # no one waits for it
    try:
        with _opening(name):
            value = opener(True)
        if value is None:
            raise RuntimeError("its opener waited and returned nothing")
    except Exception as exc:
        _ended(name, slot, exc)
        return
    _landed(name, slot, value, "landed")


def _landed(name: str, slot: _Slot, value: object, state: str) -> object | None:
    """The kernel `name` serves once `slot`'s opener verified `value`: the first verified one,
    never replaced by a later one. None when `slot` is no longer `name`'s, or failed."""
    with _LOCK:
        if _SLOTS.get(name) is not slot or slot.state == "failed":
            return None
        if slot.state == "opening":
            slot.value, slot.state = value, state
            LANDED[0] += state == "ready"
        elif slot.state == "landed" and state == "ready":
            slot.state = "ready"
            LANDED[0] += 1
        return slot.value


def _ended(name: str, slot: _Slot, exc: Exception) -> None:
    """A failed opener's kernel stays failed here, even one another opener made serve (the epoch
    counts that change); one the device ran out under asks again, unless another's serves."""
    with _LOCK:
        if _SLOTS.get(name) is not slot:
            return
        if is_capacity(exc):
            if slot.state == "opening":
                _SLOTS.pop(name)
            return
        LANDED[0] += slot.state == "ready"
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


def boundary(agreed: Iterable[str] | None = None) -> list[str]:
    """A boundary of a request: every kernel that landed since serves from here on. Their
    names. In a group, only the kernels the group serves: rank 0 decides (`agreed` None), a
    follower takes rank 0's decision."""
    with _LOCK:
        if _GROUP[0] is not None:
            _AGREED.update(_decided() if agreed is None else agreed)
        landed = [
            name
            for name, slot in _SLOTS.items()
            if slot.state == "landed" and (_GROUP[0] is None or _shared(name) in _AGREED)
        ]
        for name in landed:
            _SLOTS[name].state = "ready"
        LANDED[0] += bool(landed)
    return landed


def group(index: int, world: int) -> None:
    """This process is GPU `index` of a group of `world`: its kernels change only as the group's
    do (`boundary`)."""
    _GROUP[0] = (re.compile(rf"\bcuda:{index}\b"), world)


def agreed() -> tuple[str, ...]:
    """What rank 0 decided the group serves, for its next call to each follower."""
    with _LOCK:
        return tuple(sorted(_AGREED))


def pending() -> dict[str, str]:
    """This GPU's kernels not serving yet, as the group names them: opening, landed or failed."""
    with _LOCK:
        return _pending()


def _pending() -> dict[str, str]:
    return {_shared(name): slot.state for name, slot in _SLOTS.items() if slot.state != "ready"}


def report(rank: int, kernels: Mapping[str, str]) -> None:
    """Rank 0: follower `rank`'s `pending`, from its latest reply."""
    with _LOCK:
        _REPORTS[rank] = dict(kernels)


def _decided() -> set[str]:
    """Rank 0's decision: a kernel landed on some GPU that none still opens or failed. Nothing
    until every follower has stated its kernels once."""
    assert _GROUP[0] is not None
    if len(_REPORTS) < _GROUP[0][1] - 1:
        return set()
    states = [_pending(), *_REPORTS.values()]
    landed = {name for state in states for name, at in state.items() if at == "landed"}
    return landed - {name for state in states for name, at in state.items() if at != "landed"}


def _shared(name: str) -> str:
    """`name` as every GPU of the group names it: its device index is `cuda:*`."""
    return name if _GROUP[0] is None else _GROUP[0][0].sub("cuda:*", name)


def epoch() -> int:
    """How many times the kernels serving in this process changed."""
    return LANDED[0]
