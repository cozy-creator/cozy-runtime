"""The Runtime's own work a call waits on, named while it lasts (cl-921): a weight load in bytes,
a compile, a graph capture, an import, the GPU's start.

A phase that has held its call `SHOWN_S` shows as `<stage> / <name>` with its numbers, a few times
a second, and when it ends the stage shows its own name again. Its time is then off the stage and
step clocks (`_activity.off_clock`) and its own row of the record, so a first step that read its
weights does not read as the steps' pace. A load or a burst of compiles is a run of spells: it ends
once its call has spent `IDLE_S` on other work, or a load once it has read all it had to.

A phase is told where its call runs: the attempt's live lane under the stage open on its thread
(`attempt`, `_services`), else the executor command it serves (`command`, the executor); anywhere
else (a background build, a forkable executor before its start) it is not a phase. Every frame
goes out from one watching thread, never the call's own: a frame from the call's thread is a step
or stage boundary to the executor (`Executor.invoke`), and a phase is neither.
"""
from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Protocol

from cozy_runtime.author import _activity

#: A phase shows once it has held its call this long.
SHOWN_S = 1.0
#: A shown phase says its numbers at most this often, and that it goes on at least this often.
EVERY_S = 0.5
STILL_S = 5.0
#: A run of spells whose call has done other work this long has ended.
IDLE_S = 0.5
_TICK_S = 0.25
#: The longest stage path a frame carries (`Telemetry._stage_name`).
_PATH = 120


class Teller(Protocol):
    """Where a call's phases are told, and under what."""

    @property
    def owner(self) -> object:
        """Its call: the attempt, or the command's request."""
        ...

    @property
    def base(self) -> str:
        """The stage path it shows under ("" none)."""
        ...

    def tell(self, stage: str, position: int | None, total: int | None, unit: str) -> bool:
        """`stage` with its numbers; False once its call moved on to another stage."""
        ...

    def back(self) -> None:
        """Its stage shows its own name again."""
        ...

    def row(self, name: str, ms: float) -> None:
        """`ms` of its call's time went to `name`: a row of its record."""
        ...


def _none() -> Teller | None:
    return None


#: The attempt running on this thread, as a teller (`_services` installs it).
attempt: Callable[[], Teller | None] = _none
#: The executor command served on this thread, as a teller (the executor installs it).
command: Callable[[], Teller | None] = _none


@dataclass(eq=False)
class Phase:
    """One phase: what it says, its numbers, and how long it has held its call."""

    name: str
    teller: Teller
    unit: str = ""
    total: int | None = None
    #: The phase it runs inside on its thread: it shows under that one.
    parent: Phase | None = None
    #: A run of spells, ended by idleness or its total; else one spell, ended with its scope.
    run: bool = False
    #: Its record row; "" for none (one the record has from elsewhere).
    row: str = ""
    counted: bool = False
    position: int = 0
    #: Seconds its finished spells held the call, and of those the ones off the clocks.
    held: float = 0.0
    off: float = 0.0
    #: Seconds its first spells held the call before it showed: off the clocks once it does.
    unclocked: float = 0.0
    since: float | None = None
    #: When its last spell ended (it opened): a run idle `IDLE_S` since has ended.
    last: float = field(default_factory=time.perf_counter)
    shown: bool = False
    told_at: float = 0.0
    told: int = -1
    #: Its call moved on: it is no longer told.
    quiet: bool = False
    ended: bool = False

    @property
    def stage(self) -> str:
        return _join(self.parent.stage if self.parent is not None else self.teller.base, self.name)

    def add(self, count: int) -> None:
        """`count` more of what it counts done."""
        self.counted = True
        self.position += count
        if self.total is not None and self.position > self.total:
            self.total = self.position


#: An open spell of a run (`begin`): its run, when it began, whether it is the outermost on its
#: thread, whether its time stays the step's, and whether it nests others.
type Spell = tuple[Phase, float, bool, bool, bool]


def _join(base: str, name: str) -> str:
    room = _PATH - len(name) - 3
    return f"{base[:room]} / {name}" if base and room > 0 else name[:_PATH]


_LOCK = threading.Condition()
#: Open phases, and the runs among them by (call, stage, parent, key).
_LIVE: list[Phase] = []
_RUNS: dict[tuple[int, str, int, str], Phase] = {}
#: Ended shown phases whose stage says its own name again at the next look.
_BACKS: list[Phase] = []
_WATCHING: list[threading.Thread] = []
_LOCAL = threading.local()
#: The scoped phase open on this thread: one opened inside it shows under it.
_OPEN: ContextVar[Phase | None] = ContextVar("cozy_phase", default=None)


def _teller() -> Teller | None:
    return attempt() or command()


def _register(phase: Phase) -> None:
    with _LOCK:
        _LIVE.append(phase)
        if not _WATCHING:
            watcher = threading.Thread(target=_watch, name="phases", daemon=True)
            _WATCHING.append(watcher)
            watcher.start()
        _LOCK.notify()


def _end(phase: Phase) -> None:
    """It ends: its row, and its stage's own name again where it showed."""
    with _LOCK:
        if phase.ended:
            return
        phase.ended = True
        with contextlib.suppress(ValueError):
            _LIVE.remove(phase)
        for key, run in list(_RUNS.items()):
            if run is phase:
                del _RUNS[key]
        if phase.shown and not phase.quiet:
            _BACKS.append(phase)
            _LOCK.notify()
    if phase.row and phase.off > 0:
        with contextlib.suppress(Exception):
            phase.teller.row(phase.row, phase.off * 1000)


def finish(owner: object) -> None:
    """Its call is over: every phase it left open ends now, before its record is read."""
    with _LOCK:
        mine = [phase for phase in _LIVE if phase.teller.owner is owner]
    for phase in mine:
        phase.quiet = True
        _end(phase)


def _begin(phase: Phase, nests: bool = True) -> tuple[float, bool]:
    """A spell of `phase` begins on this thread: when, and whether it is the outermost here (an
    inner one's time is already its outer one's). One that `nests` makes those after it inner."""
    depth: int = getattr(_LOCAL, "depth", 0)
    if nests:
        _LOCAL.depth = depth + 1
    began = time.perf_counter()
    if phase.since is None:
        phase.since = began
    return began, depth == 0


def _close(
    phase: Phase, began: float, outermost: bool, clocked: bool, nests: bool = True
) -> None:
    """Its spell ends. Once the phase has held its call `SHOWN_S`, an unclocked spell's time
    (a first read, a compile) leaves this thread's clocks; a clocked one (a read again) stays its
    step's."""
    if nests:
        _LOCAL.depth = getattr(_LOCAL, "depth", 1) - 1
    now = time.perf_counter()
    spent = now - began
    phase.since, phase.last = None, now
    phase.held += spent
    if outermost and not clocked:
        phase.unclocked += spent
    if phase.held >= SHOWN_S and phase.unclocked > 0:
        _activity.off_clock(phase.unclocked)
        phase.off += phase.unclocked
        phase.unclocked = 0.0


@contextlib.contextmanager
def phase(
    name: str, *, unit: str = "", total: int | None = None, row: bool = True, told: bool = True
) -> Iterator[Phase | None]:
    """`name` holds the call on this thread for as long as it runs: `with phase("compiling sol")`.
    None where no call is told (`Phase.add` counts it otherwise). `row`: its time is its own row of
    the record; False where the record has it from elsewhere. `told`: False where its call is told
    of it otherwise (a compile wait, `kernel_compile.wait`); its time still leaves the clocks."""
    teller = _teller()
    if teller is None:
        yield None
        return
    parent = _OPEN.get()
    opened = Phase(
        name,
        teller,
        unit,
        total,
        parent if parent is not None and parent.teller.owner is teller.owner else None,
        row=_join(teller.base, name) if row else "",
        counted=total is not None,
        quiet=not told,
    )
    _register(opened)
    token = _OPEN.set(opened)
    began, outermost = _begin(opened)
    try:
        yield opened
    finally:
        _OPEN.reset(token)
        _close(opened, began, outermost, clocked=False)
        _end(opened)


def begin(
    key: str,
    name: str,
    *,
    unit: str = "",
    total: Callable[[], int | None] | None = None,
    row: str = "",
    clocked: bool = False,
    told: bool = True,
    nests: bool = True,
) -> Spell | None:
    """A spell of the run `key` of the call on this thread (a read of a component's weights, one
    compile of a burst), the run named `name` from now on; `end` it. The run opens at its first
    spell, its total `total()` then. `clocked`: the spell stays its step's (a read again).
    `told`: as `phase`'s. `nests`: False for one whose end may never come (a hook's, `drop`):
    it makes no spell after it an inner one."""
    teller = _teller()
    if teller is None:
        return None
    parent = _OPEN.get()
    if parent is not None and parent.teller.owner is not teller.owner:
        parent = None
    index = (id(teller.owner), teller.base, id(parent) if parent is not None else 0, key)
    run = _RUNS.get(index)
    whole: int | None = None
    if total is not None and (run is None or run.ended):
        # Read before the lock (it may walk a model's blocks); a phase never fails its call.
        with contextlib.suppress(Exception):
            whole = total()
    with _LOCK:
        run = _RUNS.get(index)
        if run is None or run.ended:
            named = _join(teller.base, row) if row else ""
            run = Phase(name, teller, unit, whole, parent, run=True, row=named, quiet=not told)
            run.counted = whole is not None
            _RUNS[index] = run
            _register(run)
        else:
            run.name = name
    began, outermost = _begin(run, nests)
    return run, began, outermost, clocked, nests


def end(spell: Spell | None, done: int = 0) -> None:
    """`begin`'s spell ends, `done` more of what its run counts done; a run that has done its
    total ends with it."""
    if spell is None:
        return
    run, began, outermost, clocked, nests = spell
    _close(run, began, outermost, clocked, nests)
    if done:
        run.add(done)
    if run.total is not None and run.position >= run.total:
        _end(run)


def drop(spell: Spell | None) -> None:
    """`begin`'s spell never ended (what it timed raised): it ends now, its time no one's."""
    if spell is None:
        return
    run, _, _, _, nests = spell
    if nests:
        _LOCAL.depth = getattr(_LOCAL, "depth", 1) - 1
    run.since, run.last = None, time.perf_counter()


@contextlib.contextmanager
def spell(
    key: str,
    name: str,
    *,
    unit: str = "",
    total: Callable[[], int | None] | None = None,
    row: str = "",
    clocked: bool = False,
    told: bool = True,
    done: int = 0,
) -> Iterator[None]:
    """`begin` ... `end` around the spell: `done` counts only when it returns."""
    opened = begin(key, name, unit=unit, total=total, row=row, clocked=clocked, told=told)
    finished = 0
    try:
        yield
        finished = done
    finally:
        end(opened, finished)


def loading(
    component: str, nbytes: int, unread: Callable[[], int | None], *, again: bool = False
) -> contextlib.AbstractContextManager[None]:
    """One read of `component`'s weights (`nbytes`) for the call on this thread: a spell of its
    load, `loading <component>` in bytes of what `unread()` says it had to read when it began.
    `again`: these bytes were read before in this call (paging), so the read is its step's."""
    name = f"loading {component}"
    return spell(name, name, unit="bytes", total=unread, row=name, clocked=again, done=nbytes)


def _watch() -> None:
    """The one thread that tells: each phase that has held its call `SHOWN_S`, as it moves; a run
    whose call went on to other work ends here."""
    while True:
        with _LOCK:
            while not _LIVE and not _BACKS:
                _LOCK.wait()
            live, backs = list(_LIVE), list(_BACKS)
            _BACKS.clear()
        now = time.perf_counter()
        for ended in backs:
            if ended.parent is not None and ended.parent.shown and not ended.parent.ended:
                ended.parent.told_at = 0.0  # its outer phase says itself again now
                continue
            with contextlib.suppress(Exception):
                ended.teller.back()
        for phase in live:
            with contextlib.suppress(Exception):
                _look(phase, now)
        time.sleep(_TICK_S)


def _look(phase: Phase, now: float) -> None:
    if phase.ended:
        return
    if phase.run and phase.since is None and now - phase.last >= IDLE_S:
        _end(phase)
        return
    held = phase.held + (now - phase.since if phase.since is not None else 0.0)
    if phase.quiet or held < SHOWN_S:
        return
    if phase.shown:
        since = now - phase.told_at
        if since < EVERY_S or (phase.position == phase.told and since < STILL_S):
            return
    position = phase.position if phase.counted else None
    if not phase.teller.tell(phase.stage, position, phase.total, phase.unit):
        phase.quiet = True
        return
    phase.shown, phase.told, phase.told_at = True, phase.position, now
