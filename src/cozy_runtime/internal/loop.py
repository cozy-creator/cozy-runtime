"""`author.denoise_loop`: eager, or resident reuse.

A call runs its site's kept program natively from start to end, else eagerly, for real, on the
caller's stream (only recorded regions run on the recording's streams, which end with the program).
A keyed call whose blocks all stay resident gets its whole call (`before`, the steps and `after`)
recorded and kept for its call site, until a new key replaces it or residency retires it under
pressure. The first call of a key and layout runs `before` and its steps through each label's
first eagerly (one step, unlabeled), warming their kernels, naming their blocks and measuring each
part's working peak, then records its whole call on copies of its tensors, one graph per label
(every scope model held and loaded), and runs the rest natively from them (xs-134 F2): its eager
steps cost more than the recording on a host-bound CPU (SDXL int8 on an RTX 4090: 173 ms eager
against 79 native, a 0.9 s recording), and the next call runs natively whenever it comes. An
`after` that has never run is not recorded: that call records its rest, runs its `after` eagerly,
which chooses its split levels and is measured, and queues the whole call for the idle period after
it (`_IdleRecording`, xs-134 F), in the call's component scope, as a refused recording does; a
command that arrives preempts it between its parts (never inside package code). Its copies are a
cache residency may drop. A call that finds it unrecorded (no idle period yet, preempted, or no
room at idle) records the whole call before running any of it and runs it natively from its start:
one plan and no eager step. No idle period opens between a call's prepare and its invoke
(`Executor.awaiting_invoke`), so a call queued behind another never waits for idle work.

A recording's mode is decided before anything is captured (`_admit`): its pool is `vt.pool_bytes`
of the most any part it records held as varena measured it eagerly, and varena makes room for it
beside the program's weights with `after`, else without it (`after` then runs eagerly after each
replay), else not at all: the call runs eagerly and the run says what it needed. A part never
measured is never recorded. A recording refused or unrecordable is not tried again for that layout;
one whose code cannot run outside its request records only in a call; the calls say why. Nothing
catches an allocation that fails after that: it fails the call. Other component loops stay eager:
recording a streamed loop did not repay its recording and extra reads (Qwen-Image int8 1 MP on an
RTX 3070: eager 45.6-46.8 s, native 49.3-50.7 s; MiniMax-H3 FL2VA 15 s on an RTX 5090: 716 vs
737 s). So does one whose scope holds an `@uncaptured` component, and nothing queues for it. An
unkeyed component loop whose inputs hold `retained` States runs eagerly, reading them again each
step; any other reads them once and holds them pinned for the call.

A loop outside any component scope (tensor work only) records resident every call and is never kept:
`before` and the same warm steps run eagerly, then the rest natively. A refusal before anything is
captured drops that attempt and the rest runs eagerly; nothing runs twice. After submission
nothing falls back.
"""

from __future__ import annotations

import bisect
import hashlib
import itertools
import marshal
import threading
import time
import weakref
from collections.abc import Callable, Hashable, Iterable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import cached_property, partial
from typing import Any

import tensorfs.varena.torch as vt
import torch
from tensorfs import varena
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from cozy_runtime import graphs, lora, stepcache
from cozy_runtime.author import _loop
from cozy_runtime.author._context import CURRENT, Context
from cozy_runtime.author._errors import (
    CAPTURE_ERRORS,
    CapabilityError,
    Fault,
    OutputError,
    fault,
    unwind,
)
from cozy_runtime.author._model import _LOCAL_SCOPES, LANDED, PLACED, Building
from cozy_runtime.graphs import domain
from cozy_runtime.internal import attention, cache_owners, kernel_ready, pacing
from cozy_runtime.internal.accel import ResidencyError
from cozy_runtime.internal.execution import phase as working_phase
from cozy_runtime.internal.prepared import (
    PreparedRepeat,
    PreparedSequence,
    _native_error,
    _SequenceCapture,
    _zero_dropout_attention,
)
from cozy_runtime.internal.residency import (
    _REQUEST,
    _UNPROVEN,
    ENTERED,
    Binding,
    Model,
)

_execution: Any
try:
    import cozy_runtime._kernels._execution as _execution
except ModuleNotFoundError as missing:  # CPU and macOS wheels carry no native execution module
    if missing.name != "cozy_runtime._kernels._execution":
        raise
    _execution = None

#: Codes a preparation refusal carries: the loop continues in the next mode.
_REFUSED = (
    "prepared_graph_ineligible",
    "prepared_load_unavailable",
    "prepared_load_events_unavailable",
)
#: a refusal by one of them (`_refusal`)
_UNRECORDABLE = "the step cannot be recorded"
#: varena's refusal of a recording's weights or pool, before anything was captured (`_NoRoom`)
_NO_ROOM = "no room to record it"
#: a refusal that says nothing against the graph (`_refusal`)
_WAITING = "capture waits for kernel readiness"
#: request id -> its loops, for the run record (`execution.loops`)
ROWS: dict[str, list[dict[str, Any]]] = {}


class _Fixed(str):
    """A refusal that is a property of the code (`_REFUSED`): in the record, and no warning."""


class _NoRoom(Exception):
    """varena had no room for a recording's weights or pool (`varena.Exhausted` at `_admit` or a
    hold), before anything was captured: the call runs eagerly."""


def _call(step: Callable[..., object], index: Any, tensors: tuple[Any, ...], label: Any) -> None:
    if label is _NONE:
        step(index, *tensors)
    else:
        step(index, *tensors, label=label)


_NONE = object()
#: The recording call's parts, for the run record (`parts_ms`): [{part: ms}, last lap's clock].
_LAPS: ContextVar[list[Any] | None] = ContextVar("cozy_loop_laps", default=None)
#: Set around eager parts whose working peaks size a recording's pool (`_menu`): each part's
#: ("before", "steps", "after") as varena charged its pass (its phase's `peak_bytes`).
_PEAKS: ContextVar[dict[str, int] | None] = ContextVar("cozy_loop_peaks", default=None)
#: The call's row, for its GPU timeline (`_mark`).
_ROW: ContextVar[dict[str, Any] | None] = ContextVar("cozy_loop_row", default=None)


def _mark(name: str) -> None:
    """A timed CUDA event on the current stream, for the call's GPU timeline (`_timeline`)."""
    row = _ROW.get()
    if row is None or not torch.cuda.is_initialized() or torch.cuda.is_current_stream_capturing():
        return
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    row.setdefault("_marks", []).append((name, event))


def _timeline(row: dict[str, Any]) -> None:
    """The GPU's time between consecutive marks (`gpu_ms`: [mark, ms since the one before]): each
    eager step's span, and what lies between the steps (a recording's, a native run's)."""
    marks = row.get("_marks") or []
    if len(marks) > 1:
        marks[-1][1].synchronize()
        row["gpu_ms"] = [
            [name, round(earlier.elapsed_time(event), 2)]
            for (_, earlier), (name, event) in itertools.pairwise(marks)
        ]


@contextmanager
def _laps(row: dict[str, Any]) -> Iterator[None]:
    """Time a recording call's parts into its row (`parts_ms`)."""
    token = _LAPS.set([row.setdefault("parts_ms", {}), time.perf_counter()])
    try:
        yield
    finally:
        _LAPS.reset(token)


def _lap(name: str) -> None:
    """Close the current part of a recording call as `name` (milliseconds since the last lap)."""
    laps = _LAPS.get()
    if laps is not None:
        now = time.perf_counter()
        laps[0][name] = round(laps[0].get(name, 0.0) + (now - laps[1]) * 1000, 1)
        laps[1] = now


def _now() -> float:
    return round(time.time() * 1000, 1)


def _refusal(error: BaseException) -> str | None:
    """The reason a preparation error lets the loop continue in another mode, else None. An
    allocation that fails once its recording was admitted is none: it fails the call."""
    if isinstance(error, _NoRoom):
        return f"{_NO_ROOM}: {error}"
    if isinstance(error, CapabilityError):
        return _Fixed(error) if error.code in _REFUSED else None
    if isinstance(error, Building):
        return f"{_WAITING}: {error}"
    if isinstance(error, ResidencyError):
        return f"residency refused preparation: {error}"
    code = getattr(error, "error_code", None)
    if isinstance(code, int) and code in CAPTURE_ERRORS:
        return f"{_UNRECORDABLE}: {error}"
    if isinstance(error, Exception) and fault(error) is Fault.OTHER:
        return f"recording failed: {type(error).__name__}: {error}"
    return None


def _non_finite(done: int, total: int) -> OutputError:
    return OutputError(
        f"the latents went non-finite by step {done} of {total}", code="non_finite_latents"
    )


def _warn(why: str) -> None:
    """A refusal the run hears of. A kernel still compiling says nothing against the graph, and
    a property of the code is no run going wrong: every call would warn of it."""
    if not why.startswith(_WAITING) and not isinstance(why, _Fixed):
        graphs.safeguard("capture_refused", why)


def _refused(row: dict[str, Any], where: str, why: str) -> None:
    row["refusals"].append(f"{where}: {why}")
    _warn(why)


@dataclass
class _Program:
    """A recorded native loop and the tensors its graphs read.

    `first` runs the steps after the eager ones (and `after` when it is recorded); `again`,
    when recorded, runs `before`, every step and `after` for a later call, with no eager step.
    Each is one sequence: a stage per run of equal labels, its label's graph repeated.
    """

    first: tuple[tuple[PreparedSequence, int, int], ...]
    again: PreparedSequence | None
    mode: str
    tensors: tuple[Any, ...]
    counter: Any
    ident: Hashable
    stages: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: a kept program's cache handle: residency releases it under pressure while it is idle
    region: Any = None
    #: its idle recording's parts and total (ms)
    recorded: dict[str, float] | None = None
    #: a step cache's device counters and flags its graphs write (`stepcache._Run.adopt`)
    verdicts: tuple[Any, ...] | None = None
    #: `again` holds `before` and the steps only: its `after` runs eagerly once it has run
    eager_after: bool = False
    #: what its recording's private pool holds reserved for as long as it is kept
    pool_bytes: int = 0
    closed: bool = False

    def sequences(self) -> tuple[PreparedSequence, ...]:
        rest = () if self.again is None else (self.again,)
        return (*(sequence for sequence, _, _ in self.first), *rest)

    def close(self) -> None:
        if self.closed:
            return
        for sequence in self.sequences():
            sequence.close()
        self.closed = True
        # Closed definitions must not keep their Python static tensors/outputs alive merely
        # because the cache dictionary has not yet removed this stale program.
        self.first, self.again, self.tensors, self.counter = (), None, (), None

    def bindings(self) -> tuple[Binding, ...]:
        """The weights its graphs read."""
        stages = tuple(stage for sequence in self.sequences() for stage in sequence.stages)
        found = {id(binding): binding for stage in stages for binding in stage._owned_bindings}
        for stage in stages:
            for reference, _ in stage._bindings:
                if (binding := reference()) is not None:
                    found[id(binding)] = binding
        return tuple(found.values())

    def try_close(self) -> bool:
        sequences = self.sequences()
        stages = tuple(stage for sequence in sequences for stage in sequence.stages)
        bindings = self.bindings()
        domains = {
            id(shared.domain): shared.domain for sequence in sequences
            if (shared := sequence._capture_domain) is not None and shared.domain is not None
        }
        locks = [
            self.lock, *(sequence._submission for sequence in sequences),
            *(stage._submission for stage in stages), *(owner.lock for owner in domains.values()),
            *(binding.lock for binding in bindings),
            *(binding.model.memory.lock for binding in bindings),
        ]
        with cache_owners.try_locks(locks) as claimed:
            if not claimed:
                return False
            for definition in (*sequences, *stages):
                active = definition._active
                if active is not None and active._finished is None and not active.poll().done:
                    return False
            for owner in domains.values():
                if not owner.fenced or (owner.done is not None and not owner.done.query()):
                    return False
            for binding in bindings:
                if binding.completion_lost or (
                    binding.last is not None and not binding.last.query()
                ):
                    return False
            self.close()
            return True

    def release(self) -> None:
        if self.region is None:
            self.close()
        else:
            self.region.close()  # calls close unless a reclaim already did


@dataclass
class _Site:
    """What a call site's keyed calls learned, per layout: the blocks an eager `after` ran, each
    eager part's measured working peak, the whole call queued to record at idle, and the whole
    calls ruled out (and why)."""

    after: dict[Hashable, dict[Model, tuple[str, ...]]] = field(default_factory=dict)
    #: each part's most working memory as varena measured it eagerly (`_PEAKS`): its pools' size
    measured: dict[Hashable, dict[str, int]] = field(default_factory=dict)
    unfit: set[Hashable] = field(default_factory=set)
    #: layouts whose idle recording failed: code that runs only inside its request records there
    inside: set[Hashable] = field(default_factory=set)
    pending: dict[Hashable, _IdleRecording] = field(default_factory=dict)  # by key and layout
    why: dict[Hashable, str] = field(default_factory=dict)


class Loops:
    """The installed `author._loop.Loops`: one per executor process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._kept: dict[Hashable, _Program] = {}
        self._sites: dict[Hashable, _Site] = {}

    def run(
        self,
        step: Callable[..., object],
        steps: range,
        state: tuple[Any, ...],
        carried: tuple[Any, ...],
        inputs: tuple[Any, ...],
        labels: tuple[Hashable, ...] | None,
        key: Hashable | None,
        on_step: Callable[[int], object] | None,
        before: Callable[..., object] | None,
        after: Callable[..., object] | None,
        cache: stepcache.Policy | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "steps": len(steps),
            "mode": "eager",
            "native_steps": 0,
            "reused": False,
            "refusals": [],
            "at_ms": {"entered": _now()},  # unix ms: entered, submitted, finished, returned
        }
        request = _REQUEST.get()
        if request is not None:
            ROWS.setdefault(request.id, []).append(row)
        # The loop records its own programs: component region graphs neither replay nor learn
        # its calls (a replayed step enters no blocks, and its graph holds their weights).
        if cache is not None and _ranks() > 1:  # each rank's verdict would differ
            row["refusals"].append("step cache: off across ranks (its sums are not all-reduced)")
            cache = None
        bypass, marked = graphs.BYPASS.set(True), _ROW.set(row)
        overlays = lora.LOOP.set(_overlays(_phase_models()))
        try:
            _mark("entered")
            # `retained` States among its inputs: one unkeyed in a component scope never records,
            # so each step reads them again (`_Loop.refills`) and their regions may leave between
            # steps. Any other reads them once, held for the call: a program recorded on them
            # reads them only while they stay pinned, or reads its own copies.
            refills = False
            if any(not isinstance(value, torch.Tensor) for value in inputs):
                refills = key is None and bool(_LOCAL_SCOPES.get())
                if not refills:
                    inputs = _filled(inputs)
            loop = _Loop(
                step, steps, labels, on_step, before, after, key, len(state), cache, refills
            )
            with (
                nullcontext() if cache is None else stepcache.running(cache, steps, state[0].device)
            ) as run:
                self._run(loop, state, carried, inputs, key, row)
            if run is not None:
                row["cache"] = run.document()
            _mark("returned")
            _timeline(row)
        finally:
            lora.LOOP.reset(overlays)
            graphs.BYPASS.reset(bypass)
            _ROW.reset(marked)
            row.pop("_marks", None)
            row["at_ms"]["returned"] = _now()

    def _run(
        self,
        loop: _Loop,
        state: tuple[Any, ...],
        carried: tuple[Any, ...],
        inputs: tuple[Any, ...],
        key: Hashable | None,
        row: dict[str, Any],
    ) -> None:
        ctx = CURRENT.get()
        step, steps = loop.step, loop.steps
        mutable = state + carried
        with torch.inference_mode(False), torch.no_grad(), _owned(mutable) as work:
            tensors = work + inputs
            scope, why = (
                ((), "its retained inputs are read again each step")
                if loop.refills
                else _eligible(ctx, steps, tensors)
            )
            if why:
                row["refusals"].append(why)
                loop.eager_all(tensors, ctx)
                return
            assert ctx is not None
            site = (step.__code__, tuple(id(model) for model in scope))
            # What a call site learns (`_Site`) depends on a keyed call's layout, not its key.
            shape = None if key is None else (
                _layouts(tensors), steps, loop.labels, loop.cache, lora.LOOP.get()
            )
            ident = None if key is None else (key, shape)
            with self._lock:
                known = self._sites.setdefault(site, _Site())
                kept = self._kept.get(site)
                if kept is not None and (why := _stale(kept, ident)):
                    row["dropped"] = why
                    del self._kept[site]
                    _close_idle(kept)
                    kept = None
            if kept is not None:
                with _taken(kept.lock, ctx), _pinned(kept) as why:
                    why = why or _stale(kept, ident, checked=True)
                    if not why:
                        self._reuse(kept, loop, tensors, work, ctx, row)
                if not why:
                    _eager_after(kept, loop, tensors, row)
                    return
                row["dropped"] = why
                with self._lock:
                    if self._kept.get(site) is kept:
                        del self._kept[site]
                _close_idle(kept)
            job = None if ident is None else known.pending.get(ident)
            if job is not None and self._take_over(
                site, known, job, loop, tensors, work, scope, ctx, row
            ):
                return
            if scope:
                self._first(site, known, loop, tensors, work, scope, ident, shape, ctx, row)
                return
            program = None
            try:
                with _laps(row):
                    program, warmed = _record(loop, tensors, work, key, ctx, row)
                if program is None:
                    return
                for sequence, first, count in program.first:
                    _submit(program, sequence, range(first, first + count), loop, ctx, row)
                warmed()
            except BaseException:
                if program is not None:
                    program.close()
                raise
            # No call keeps a componentless program; it closes before the work after it (an
            # eager decode) needs its pool, and its `close` lets go of its tensors.
            finished = program.tensors
            program.close()
            _after(loop, finished, scope, known, shape)
            _back(work, finished)

    def _first(
        self,
        site: Hashable,
        known: _Site,
        loop: _Loop,
        tensors: tuple[Any, ...],
        work: tuple[Any, ...],
        scope: tuple[Model, ...],
        ident: Hashable,
        shape: Hashable,
        ctx: Context,
        row: dict[str, Any],
    ) -> None:
        """A component call its site has no program for: `before` and its steps through each
        label's first eagerly, warming their kernels and naming their blocks. A keyed whole call
        with steps left then records on its program's own copies and runs them natively; the
        program is kept for the site. Otherwise, or refused, the rest runs eagerly and the
        recording is queued for the idle period after it."""
        with _laps(row):
            opened = None
            if ident is not None and shape not in known.unfit:
                with suppress(CapabilityError, domain.Unavailable):  # its own recording says why
                    opened = _opened(work)
            pre, body, rest, warmed, peaks = _warm(loop, tensors, scope, ctx)
            if shape is not None:
                with self._lock:
                    peaks = _measured(known, shape, peaks)
            program, refused, began = None, "", time.perf_counter()
            if rest and not _unkept(known, ident, shape, body):
                whole = loop.after is None or shape in known.after
                menu = _menu(
                    peaks,
                    before=whole and loop.before is not None,
                    after=whole and loop.after is not None,
                )
                program, refused = _record_first(
                    loop, rest, tensors, len(work), scope, pre, body, ident, menu, opened, whole
                )
            elif opened is not None:
                _abandon(opened)
        if program is not None and program.again is None:
            row.update(
                mode=program.mode,
                stages=program.stages,
                pool_bytes=program.pool_bytes,
                whole_call="its rest recorded; the whole call records once its after has run",
                capture_ms=round((time.perf_counter() - began) * 1000, 3),
            )
            with _pinned(program) as why:
                if not why:
                    ((sequence, first, count),) = program.first
                    try:
                        _submit(program, sequence, range(first, first + count), loop, ctx, row)
                    except BaseException:
                        program.close()
                        raise
                    warmed()
            finished = program.tensors
            program.close()  # before the eager `after` needs its pool
            if not why:
                kept = _unkept(known, ident, shape, body)  # what `after` evicts, a replay refills
                _after(loop, finished, scope, known, shape)
                _back(work, finished)
                row["eager"] = self._queue(
                    site, known, loop, tensors, len(work), scope, pre, body, ident, shape, kept
                )
                return
            row["dropped"] = why
            program = None
        if program is not None:
            row.update(
                mode=program.mode,
                stages=program.stages,
                pool_bytes=program.pool_bytes,
                whole_call="recorded on its first call, after its warm steps",
                capture_ms=round((time.perf_counter() - began) * 1000, 3),
            )
            self._install(site, program, scope[0].memory)
            with _taken(program.lock, ctx), _pinned(program) as why:
                if not why:
                    ((sequence, first, count),) = program.first
                    try:
                        _submit(program, sequence, range(first, first + count), loop, ctx, row)
                    except BaseException:
                        program.close()
                        raise
                    sequence.close()  # later calls run `again`
                    program.first = ()
                    warmed()
                    _back(work, program.tensors)
            if not why:
                _eager_after(program, loop, tensors, row)
                return
            row["dropped"] = why
        if refused:
            _refused(row, "whole call on its first call", refused)
            if refused.startswith((_UNRECORDABLE, _NO_ROOM)):
                # Its idle recording would meet the same refusal and keep the next call waiting
                # on it (an RTX 3060's SDXL w1 559 ms, then its own doomed recording).
                with self._lock:
                    known.unfit.add(shape)
                    known.why[shape] = refused[:400]
        loop.eager(rest, tensors, ctx)
        warmed()
        _after(loop, tensors, scope, known, shape)
        row["eager"] = self._queue(
            site, known, loop, tensors, len(work), scope, pre, body, ident, shape
        )

    def _take_over(
        self,
        site: Hashable,
        known: _Site,
        job: _IdleRecording,
        loop: _Loop,
        tensors: tuple[Any, ...],
        work: tuple[Any, ...],
        scope: tuple[Model, ...],
        ctx: Context,
        row: dict[str, Any],
    ) -> bool:
        """Its idle recording has not run (no idle period yet, preempted, or no room at idle):
        this call records the whole call before running any of it, from what its first call
        measured, and runs it natively from its start (one plan, no eager step). False when
        reclaim dropped the job first: the call is its site's first again."""
        shape = job.ident[1]  # type: ignore[index]
        pin = ExitStack()
        try:
            pin.enter_context(job.region.pin())
        except ResidencyError:
            return False
        program, began, taken, waiting = None, time.perf_counter(), False, False
        try:
            with pin:
                with self._lock:
                    if known.pending.get(job.ident) is not job:
                        return False
                    del known.pending[job.ident]
                    measured, taken = dict(known.measured.get(shape, {})), True
                try:
                    with _laps(row):
                        program = _record_whole(
                            loop, job.tensors, job.nwork, job.pre, job.body, job.post, job.ident,
                            measured,
                        )
                except Exception as error:
                    if isinstance(error, Building) or isinstance(error.__cause__, Building):
                        with self._lock:
                            known.pending[job.ident] = job  # a kernel still opens: the idle period
                        graphs.IDLE.add(job)
                        waiting = True
                        row["refusals"].append("whole call: a kernel still opens")
                    elif (why := _refusal(error)) is not None:
                        _refused(row, "whole call", why)
                        with self._lock:
                            known.unfit.add(shape)
                            known.why[shape] = why[:400]
                    else:
                        raise
        finally:
            if taken and not waiting:
                job.region.close()  # its copies are the program's now, or go
        if program is None:
            loop.eager_all(tensors, ctx)
            return True
        row.update(
            pool_bytes=program.pool_bytes,
            whole_call="recorded on this call, its idle recording not run",
            capture_ms=round((time.perf_counter() - began) * 1000, 3),
        )
        self._install(site, program, scope[0].memory)
        with _taken(program.lock, ctx), _pinned(program) as why:
            why = why or _stale(program, job.ident, checked=True)
            if not why:
                self._reuse(program, loop, tensors, work, ctx, row)
        if not why:
            _eager_after(program, loop, tensors, row)
            return True
        row["dropped"] = why
        loop.eager_all(tensors, ctx)
        return True

    def _queue(
        self,
        site: Hashable,
        known: _Site,
        loop: _Loop,
        tensors: tuple[Any, ...],
        nwork: int,
        scope: tuple[Model, ...],
        pre: dict[Model, tuple[str, ...]],
        body: dict[Model, tuple[str, ...]],
        ident: Hashable,
        shape: Hashable,
        unkept: str | None = None,
    ) -> str:
        """Why the call ran eagerly: its whole call queued to record at idle, or why none does;
        `unkept`, `_unkept` read before an eager `after` that evicts what a replay refills."""
        if why := _unkept(known, ident, shape, body) if unkept is None else unkept:
            return why
        if lora.LOOP.get():
            return "resident, and its overlays are armed only in its call"
        if shape in known.inside:
            return f"resident, recorded only in a call: {known.why.get(shape, '')}"[:400]
        with self._lock:
            if ident in known.pending:
                return "resident: the whole call records at idle"
        job = _IdleRecording(
            self,
            site,
            known,
            replace(loop, on_step=None),  # the call's progress hook stays with the call
            tuple(value.clone() for value in tensors),  # the program's own copies
            nwork,
            scope,
            pre,
            body,
            known.after.get(shape, {}),
            ident,
            _LOCAL_SCOPES.get(),
            tuple(model.wanted_identity() for model in scope),
        )
        # Its copies wait for an idle period beside every call before it: a cache reclaim drops.
        job.region = scope[0].memory.region(
            "denoise_loop.pending", release=job.drop, shrink=job.shrink
        )
        with self._lock:
            if ident in known.pending:
                queued = None
            else:
                known.pending[ident] = queued = job
        if queued is None:
            job.region.close()
        else:
            graphs.IDLE.add(job)
        return "resident: the whole call records at idle"

    def _install(self, site: Hashable, program: _Program, memory: Any) -> None:
        program.region = memory.region("denoise_loop", release=program.close, shrink=program.try_close)
        with self._lock:
            previous = self._kept.pop(site, None)
            self._kept[site] = program
        if previous is not None:
            _close_idle(previous)

    def _reuse(
        self,
        kept: _Program,
        loop: _Loop,
        tensors: tuple[Any, ...],
        work: tuple[Any, ...],
        ctx: Context,
        row: dict[str, Any],
    ) -> None:
        """A kept program: this call's values into its tensors, then natively `before`, every
        step and `after`, and the state back out."""
        assert kept.again is not None  # only whole calls are kept
        row.update(
            mode=kept.mode, reused=True, stages=kept.stages, pool_bytes=kept.pool_bytes,
        )
        if kept.recorded is not None:
            row["recorded_idle_ms"] = kept.recorded
        try:
            for slot, value in zip(kept.tensors, tensors, strict=True):
                slot.copy_(value)
            if kept.verdicts is not None and (run := stepcache._RUN.get()) is not None:
                run.adopt(kept.verdicts)
            _submit(kept, kept.again, loop.steps, loop, ctx, row)
            _back(work, kept.tensors)
        except BaseException:
            kept.close()
            raise


@dataclass(frozen=True)
class _Loop:
    """One call's declaration."""

    step: Callable[..., object]
    steps: range
    labels: tuple[Hashable, ...] | None
    on_step: Callable[[int], object] | None
    before: Callable[..., object] | None
    after: Callable[..., object] | None
    key: Hashable | None = None
    #: how many of a call's first tensors are its latents (`state`)
    latents: int = 1
    cache: stepcache.Policy | None = None
    #: its inputs hold `retained` States, read again (`get`, which fills and pins them) for each
    #: step and for `before` and `after`: it only runs eagerly (`Loops.run`)
    refills: bool = False

    def label(self, i: int) -> Any:
        return _NONE if self.labels is None else self.labels[i - self.steps.start]

    def warm(self) -> range:
        """The steps a first call runs eagerly: through each label's first (one, unlabeled), so
        every label's kernels and blocks have run before its body records."""
        if self.labels is None:
            return self.steps[:1]
        return self.steps[: max(self.labels.index(label) for label in set(self.labels)) + 1]

    def eager(
        self, indices: range, tensors: tuple[Any, ...], ctx: Context | None, settle: bool = True
    ) -> graphs.Finite | None:
        """The steps in order, each reported (`on_step(i)`) once queued; the call returns once the
        last has run on the device. Returning with it still queued put its time into the caller's
        next stage (Qwen-Image's last 0.9 s step into its decode). Unsettled, it returns with them
        queued, so a recording overlaps them, and gives back their finite check (`Finite.last`)
        for the caller to read once it waits anyway."""
        if not len(indices):
            return None
        request = _REQUEST.get()
        measured = request is not None and tensors[0].is_cuda
        models = _phase_models() if measured else ()
        mode: working_phase.Mode = (
            "preparation" if request is not None and request.idle else "eager"
        )
        configuration = _phase_configuration(self, indices, tensors, models) if measured else None
        epoch = LANDED[0]
        with working_phase.observing(
            tensors[0].device.index if measured else None, mode,
            request.working_phases if measured and request is not None else None,
            key=working_phase.execution_key(configuration, "runtime.eager/1"),
        ) as observation:
            index = torch.zeros(1, dtype=torch.int64, device=tensors[0].device)
            finite = graphs.Finite(torch, tensors[: self.latents])
            _mark("eager")
            for i in indices:
                if ctx is not None:
                    ctx.raise_if_cancelled()
                index.fill_(i)
                if (cached := stepcache._RUN.get()) is not None:
                    cached.begin(index, i - self.steps.start)
                read = _filled(tensors) if self.refills else tensors
                _call(self.step, index, read, self.label(i))
                _mark(f"step {i}")
                if index.is_cuda:
                    pacing.tick()
                if not finite.step():
                    raise _non_finite(i - self.steps.start, len(self.steps))
                if self.on_step is not None:
                    if observation is None:
                        self.on_step(i)
                    else:
                        with vt.external_callback(index.device.index):
                            self.on_step(i)
            if settle and index.is_cuda:
                stream = torch.cuda.current_stream(index.device)
                stream.synchronize()
                if not finite.last():
                    raise _non_finite(indices[-1] - self.steps.start + 1, len(self.steps))
                if (
                    observation is not None and mode == "eager"
                    and (
                        observation.completion_stream is None
                        or int(observation.completion_stream.cuda_stream) != int(stream.cuda_stream)
                    )
                ):
                    observation.invalidate("completion_stream_changed")
            if settle and observation is not None and mode == "eager":
                if ctx is not None and ctx.cancelled:
                    observation.invalidate("cancelled")
                else:
                    actual = _phase_configuration(self, indices, tensors, models)
                    if epoch != LANDED[0] or actual != configuration:
                        observation.invalidate("dispatch_changed")
                    observation.complete()
        if observation is not None and observation.result is not None:
            _peak("steps", observation)
            key = observation.result.get("key")
            if settle and key is not None and request is not None and models:
                request.memory.remember_working_phase(str(key), models)
        return None if settle else finite

    def eager_all(self, tensors: tuple[Any, ...], ctx: Context | None) -> None:
        _once(self.before, _filled(tensors) if self.refills else tensors)
        self.eager(self.steps, tensors, ctx)
        _once(self.after, _filled(tensors) if self.refills else tensors)


def _ranks() -> int:
    dist = torch.distributed
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _once(fn: Callable[..., object] | None, tensors: tuple[Any, ...]) -> None:
    if fn is not None:
        fn(*tensors)


def _filled(values: tuple[Any, ...]) -> tuple[Any, ...]:
    """`values` with each `retained` State in it replaced by the tensors its `get` returns now,
    filled and pinned while they live."""
    return tuple(
        tensor for value in values
        for tensor in ((value,) if isinstance(value, torch.Tensor) else value.get())
    )


@contextmanager
def _measuring() -> Iterator[dict[str, int]]:
    """The working peak of each eager part run inside (`_PEAKS`)."""
    peaks: dict[str, int] = {}
    token = _PEAKS.set(peaks)
    try:
        yield peaks
    finally:
        _PEAKS.reset(token)


def _peak(part: str, observation: working_phase.Observation | None) -> None:
    """Under `_measuring`, the working peak varena charged `part`'s observed pass. It counts the
    pass's frees still running, which an `after`'s host issues far ahead of its device (SDXL's
    uncapped decode charged 3.5 GiB and held 2.4 live): its pool is that much larger. The steps'
    capture needs their charged peak: live alone fell 87 MB short (run 37755)."""
    peaks = _PEAKS.get()
    result = None if observation is None else observation.result
    if peaks is not None and result is not None and result.get("peak_bytes") is not None:
        peaks[part] = max(int(result["peak_bytes"]), peaks.get(part, 0))


def _part(fn: Callable[..., object] | None, tensors: tuple[Any, ...], part: str) -> None:
    """`before` or `after` eagerly; under `_measuring`, one observed pass, measured as `part` (its
    components' passes are its own: an enclosing observation owns the device)."""
    request = _REQUEST.get()
    if fn is None or request is None or _PEAKS.get() is None or not tensors[0].is_cuda:
        _once(fn, tensors)
        return
    with working_phase.observing(
        tensors[0].device.index, "eager", request.working_phases
    ) as observation:
        fn(*tensors)
    _peak(part, observation)


def _measured(known: _Site, shape: Hashable, peaks: dict[str, int]) -> dict[str, int]:
    """`peaks` joined to what the calls of `shape` before measured, each part's most: under the
    site's lock."""
    seen = known.measured.setdefault(shape, {})
    for part, peak in peaks.items():
        seen[part] = max(peak, seen.get(part, 0))
    return dict(seen)


def _menu(measured: dict[str, int], *, before: bool, after: bool) -> list[tuple[int, bool]]:
    """What a recording may hold, most first, each as (its pool, with `after`): every part, then
    without `after`, which runs eagerly after each replay. A pool is `vt.pool_bytes` of the most
    any part it records held (one runs at a time), and a part never measured is never recorded.
    The core will answer this (`record(parts)`); `_admit` asks it room for each in turn."""
    parts = ("steps", "before") if before else ("steps",)
    modes = [(True, (*parts, "after"))] if after else []
    modes.append((False, parts))
    return [
        (vt.pool_bytes(max(measured[part] for part in recorded)), with_after)
        for with_after, recorded in modes
        if all(part in measured for part in recorded)
    ]


@contextmanager
def _entering() -> Iterator[set[Any]]:
    """The models whose forwards run inside: residency adds each one it opens."""
    seen: set[Any] = set()
    token = ENTERED.set(seen)
    try:
        yield seen
    finally:
        ENTERED.reset(token)


def _warm(
    loop: _Loop, tensors: tuple[Any, ...], scope: tuple[Model, ...], ctx: Context
) -> tuple[
    dict[Model, tuple[str, ...]],
    dict[Model, tuple[str, ...]],
    range,
    Callable[[], None],
    dict[str, int],
]:
    """A component call's `before` and warm steps (`_Loop.warm`) eagerly, on the caller's
    tensors, left queued for a recording to overlap: the blocks each ran, the steps left, their
    finite check, which raises once their values are in, and their working peaks (`_PEAKS`)."""
    with _measuring() as peaks:
        with _entering() as seen:
            _part(loop.before, tensors, "before")
        _lap("before")
        pre = _used(seen, scope)
        warm = loop.warm()
        with _entering() as seen:
            finite = loop.eager(warm, tensors, ctx, settle=False)
    _lap("first_step")
    body = _used(seen, scope) or {model: ("",) for model in scope}
    rest = loop.steps[len(warm) :]
    return pre, body, rest, partial(_finished, finite, len(warm), loop), peaks


def _finished(finite: graphs.Finite | None, done: int, loop: _Loop) -> None:
    """Unsettled eager steps' finite check (`_Loop.eager`), read where the call waits anyway."""
    if finite is not None and not finite.last():
        raise _non_finite(done, len(loop.steps))


def _unkept(
    known: _Site, ident: Hashable, shape: Hashable, body: dict[Model, tuple[str, ...]]
) -> str:
    """Why no call keeps this call's whole program ("" when one does)."""
    resident = all(not _unresident(model, paths) for model, paths in body.items())
    if ident is None or not resident:
        return f"{'resident' if resident else 'streamed'}, and no call keeps the program"
    if shape in known.unfit:
        return f"no whole call: {known.why.get(shape, 'it did not fit')}"
    return ""


def _record_first(
    loop: _Loop,
    rest: range,
    tensors: tuple[Any, ...],
    nwork: int,
    scope: tuple[Model, ...],
    pre: dict[Model, tuple[str, ...]],
    body: dict[Model, tuple[str, ...]],
    ident: Hashable,
    menu: list[tuple[int, bool]],
    recording: _SequenceCapture | None,
    whole: bool,
) -> tuple[_Program | None, str]:
    """A first call's program, recorded after its warm steps on copies of its tensors, into
    `recording` when one was opened (`_opened`): `first` runs its `rest` (`after` included when
    it has room), `again` a later call, every scope model held, and loaded first. Not `whole` (its
    `after` has never run), only its `rest`, kept by no call: an `after` records at the levels its
    own eager run chose, into a pool its measure sized (SDXL's first decode recorded whole at 6 GiB
    needed a 2.7 GiB pool beside its weights, and every call ran eagerly). (None, why) when
    refused, its attempt gone."""
    own = tuple(value.clone() for value in tensors)
    mutable = own[:nwork]
    counter = torch.zeros(1, dtype=torch.int64, device=own[0].device)
    every: dict[Model, tuple[str, ...]] = {model: ("",) for model in scope}
    if recording is None:
        recording = PreparedSequence.record()
        recording.__enter__()
    failure: BaseException | None = None
    try:
        with ExitStack() as loading:
            _hold(scope if whole else (), loading)  # until the recording's bindings hold them
            _lap("load")
            recording._prepare((*mutable, counter))
            _lap("prepare")
            with recording.capturing():
                parts, verdicts, with_after = _record_resident(
                    recording, loop, rest, own, mutable, counter, body,
                    pre if whole else None, every if whole else None, whole, menu,
                )
            first, *again = recording.finish_parts(
                outputs=mutable, parts=[(b, e, 1) for b, e in parts]
            )
        _lap("finish")
    except BaseException as error:
        failure = error
    if failure is not None:
        _abandon(recording, failure)
        if (why := _refusal(failure)) is None:
            raise failure
        return None, why
    stages = len((again or [first])[0].stages)
    return _Program(
        ((first, rest.start, len(rest)),), again[0] if again else None, "resident", own, counter,
        ident, stages, verdicts=verdicts,
        eager_after=whole and loop.after is not None and not with_after,
        pool_bytes=recording._working_bytes or 0,
    ), ""


def _opened(tensors: tuple[Any, ...]) -> _SequenceCapture:
    """A recording with its domain made (its library streams and handles, which wait for the
    device): before a first call's warm steps, so the recording after them overlaps them."""
    recording = PreparedSequence.record()
    recording.__enter__()
    try:
        recording._prepare(tensors)
    except BaseException:
        recording.close()
        raise
    return recording


def _record(
    loop: _Loop,
    tensors: tuple[Any, ...],
    work: tuple[Any, ...],
    key: Hashable | None,
    ctx: Context,
    row: dict[str, Any],
) -> tuple[_Program | None, Callable[[], None]]:
    """A componentless loop: `before` and its warm steps (`_Loop.warm`) eagerly, then the rest
    recorded, overlapping them; None when nothing was left or the recording was refused, after
    the loop ran eagerly to its end. With it, the warm steps' finite check (`_warm`)."""
    if key is not None:  # the program owns copies; this call's values go in and out
        tensors = tuple(value.clone() for value in tensors)
    mutable = tensors[: len(work)]
    counter = torch.zeros(1, dtype=torch.int64, device=tensors[0].device)
    with _measuring() as peaks:
        _part(loop.before, tensors, "before")
        _lap("before")
        warm = loop.warm()
        finite = loop.eager(warm, tensors, ctx, settle=False)
    warmed = partial(_finished, finite, len(warm), loop)
    _lap("first_step")
    rest = loop.steps[len(warm) :]
    if not rest:
        warmed()
        _back(work, tensors)
        _once(loop.after, work + tensors[len(work) :])
        return None, warmed
    recording = PreparedSequence.record()
    recording.__enter__()
    failure: BaseException | None = None
    try:
        recording._prepare((*mutable, counter))
        _lap("prepare")
        began = time.perf_counter()
        ctx.raise_if_cancelled()
        with recording.capturing():
            parts, verdicts, _ = _record_resident(
                recording, loop, rest, tensors, mutable, counter, {}, None, None, False,
                _menu(peaks, before=False, after=False),
            )
        (made,) = recording.finish_parts(outputs=mutable, parts=[(b, e, 1) for b, e in parts])
        _lap("finish")
    except BaseException as error:
        failure = error
    if failure is not None:
        # Its frames go first: they no longer hold the attempt's graphs, so its pool returns
        # before the eager rest.
        _abandon(recording, failure)
        if (why := _refusal(failure)) is None:
            raise failure
        _refused(row, "resident", why)
        _back(work, tensors)  # the eager rest runs on the caller's tensors
        loop.eager(rest, work + tensors[len(work) :], ctx)
        warmed()
        _once(loop.after, work + tensors[len(work) :])
        return None, warmed
    stages = len(made.stages)
    row.update(
        mode="resident",
        stages=stages,
        pool_bytes=recording._working_bytes or 0,
        capture_ms=round((time.perf_counter() - began) * 1000, 3),
        after_native=False,
    )
    return _Program(
        ((made, rest.start, len(rest)),), None, "resident", tensors, counter, None, stages,
        verdicts=verdicts, pool_bytes=recording._working_bytes or 0,
    ), warmed


def _yield_point() -> None:
    """At idle, a command that arrived ends the recording here (`graphs.Yielded`): at a boundary
    between its parts, never inside package or library code, whose broad handlers and native
    frames are not made to unwind it (a yield inside the int8 op served its slower path, then
    the executor crashed)."""
    if (idle := graphs._IDLE.get(None)) is not None:
        idle.check()


def _record_whole(
    loop: _Loop,
    tensors: tuple[Any, ...],
    nwork: int,
    pre: dict[Model, tuple[str, ...]],
    body: dict[Model, tuple[str, ...]],
    post: dict[Model, tuple[str, ...]],
    ident: Hashable,
    measured: dict[str, int],
) -> _Program:
    """The whole call (`before`, every step and `after`) recorded on `tensors` as one program a
    later call runs natively, from its parts' `measured` peaks; without `after` when its pool has
    no room beside the program's weights (SDXL's 1024px decode at 6 GiB holds 1.3 GiB of full-size
    activations at any split level, and beside its 4 GiB of weights no pool for it fits the card).
    It yields to a command between its parts (`_yield_point`). A failed or preempted attempt goes
    as a refused one does: its frames first, then its stages and the tensors of its pool, then its
    domain (a preemption that closed the domain under its live frames crashed the executor)."""
    mutable = tensors[:nwork]
    counter = torch.zeros(1, dtype=torch.int64, device=tensors[0].device)
    recording = PreparedSequence.record()
    recording.__enter__()
    failure: BaseException | None = None
    try:
        recording._prepare((*mutable, counter))
        _lap("prepare")
        _yield_point()
        with recording.capturing():
            parts, verdicts, with_after = _record_resident(
                recording, loop, loop.steps[:0], tensors, mutable, counter, body, pre, post, True,
                _menu(measured, before=loop.before is not None, after=loop.after is not None),
            )
        (again,) = recording.finish_parts(outputs=mutable, parts=[(b, e, 1) for b, e in parts])
        _lap("finish")
    except BaseException as error:
        failure = error
    if failure is not None:
        _abandon(recording, failure)
        raise failure
    return _Program(
        (), again, "resident", tensors, counter, ident, len(again.stages), verdicts=verdicts,
        eager_after=loop.after is not None and not with_after,
        pool_bytes=recording._working_bytes or 0,
    )


def _eager_after(
    program: _Program, loop: _Loop, tensors: tuple[Any, ...], row: dict[str, Any]
) -> None:
    """A kept program's `after` that had no room in it, eagerly on the call's tensors."""
    if program.eager_after:
        row["after"] = "eager: no room for it in the program beside its weights"
        _once(loop.after, tensors)


@dataclass(eq=False)
class _IdleRecording:
    """A keyed call's whole program, recorded at an idle period (`graphs.IDLE`) on copies of its
    tensors, in the call's component scope, and kept for its site. A command preempts it at its
    next torch op inside the recorded code; it records again at the next idle. Its copies are a
    cache (`region`): residency's reclaim drops the job with them. One whose models changed fill
    or left the device goes (the next call queues again); one varena has no room for at idle,
    which takes only free room, waits for the next call to record it (`Loops._take_over`); one
    that cannot be recorded (its code cannot run outside its request) is ruled out for its layout,
    and the next calls say why."""

    loops: Loops
    site: Hashable
    known: _Site
    loop: _Loop
    tensors: tuple[Any, ...]
    nwork: int
    scope: tuple[Model, ...]
    pre: dict[Model, tuple[str, ...]]
    body: dict[Model, tuple[str, ...]]
    post: dict[Model, tuple[str, ...]]
    ident: Hashable
    #: the call's component scopes, outermost first: each one's owner, method and components
    entry: tuple[tuple[Any, str, tuple[str, ...]], ...]
    identities: tuple[Any, ...]
    #: it yields between its parts (`_yield_point`), not per torch op (`graphs.IDLE`): under the
    #: per-op mode an RTX 4090's SDXL plan took 3.4 s instead of 0.46 s
    yields_per_op = False
    #: its copies' cache handle (`Loops._queue`)
    region: Any = None

    def __call__(self) -> bool:
        """True once kept, gone, or left for the next call; False for a later idle period."""
        shape = self.ident[1]  # type: ignore[index]
        pin, end = ExitStack(), False
        try:
            pin.enter_context(self.region.pin())
        except ResidencyError:
            return True  # reclaimed: its copies went, and the next call queues again
        try:
            with pin:
                with self.loops._lock:
                    if self.known.pending.get(self.ident) is not self:
                        end = True
                        return True
                try:
                    end = self._record()
                except graphs.Yielded:
                    raise
                except Exception as error:
                    if isinstance(error, Building) or isinstance(error.__cause__, Building):
                        return False  # a kernel still opens: a later idle period
                    if isinstance(error, _NoRoom):
                        return True  # the next call records it, before running any of it
                    # A step that cannot be recorded (an RTX 3060's SDXL step frees memory
                    # inside the capture: 900 from cuMemFree) is refused here as on a call's
                    # path, not a device fault: one at idle poisoned the executor and failed
                    # every next request. Anything else fails the job, and the period says so.
                    end = True
                    if (why := _refusal(error)) is None:
                        raise
                    with self.loops._lock:
                        self.known.inside.add(shape)
                        self.known.why[shape] = f"its idle recording failed: {why}"[:400]
                    _warn(why)
                    return True
                return end
        finally:
            if end:
                self.region.close()

    def drop(self) -> None:
        """Its end: the job and its copies go (a program recorded on them holds its own)."""
        with self.loops._lock:
            if self.known.pending.get(self.ident) is self:
                del self.known.pending[self.ident]
            self.tensors = ()

    def shrink(self) -> bool:
        """A cache reclaim on varena's worker: a queued job goes with its copies. It declines
        while its site's lock is held, whose holder may be the caller allocating."""
        if not self.loops._lock.acquire(blocking=False):
            return False
        try:
            if self.known.pending.get(self.ident) is not self:
                return False
            del self.known.pending[self.ident]
            self.tensors = ()
            return True
        finally:
            self.loops._lock.release()

    def _record(self) -> bool:
        """True once kept or gone; False to try at a later idle period."""
        if any(model.closed or model.retiring for model in self.scope):
            return True
        if tuple(model.wanted_identity() for model in self.scope) != self.identities:
            return True  # another fill than the call's (a LoRA'd call's): the next call queues
        with self.loops._lock:
            kept = self.loops._kept.get(self.site)
        if kept is not None and not _stale(kept, self.ident):
            return True
        paths: dict[Model, set[str]] = {}
        for used in (self.body, self.pre, self.post):
            for model, entered in used.items():
                paths.setdefault(model, set()).update(entered)
        began = time.perf_counter()
        parts: dict[str, float] = {}
        laps = _LAPS.set([parts, began])
        settling = graphs._SETTLING.set(True)  # no torch op yields: only its boundaries do
        bypass = graphs.BYPASS.set(True)
        overlays = lora.LOOP.set(frozenset())  # its call armed none, and nothing may arm one now
        try:
            with ExitStack() as scopes:
                scopes.enter_context(torch.cuda.device(self.tensors[0].device))
                scopes.enter_context(torch.inference_mode(False))
                scopes.enter_context(torch.no_grad())
                for owner, method, components in self.entry:
                    scopes.enter_context(owner._cozy_scope(method, components))
                _lap("scope")
                _yield_point()
                _hold(self.scope, scopes)  # what an eager `after` evicted, as a replay refills it
                if any(_unresident(model, tuple(p)) for model, p in paths.items()):
                    return True  # its weights left the device: the next call queues again
                _lap("resident")
                _yield_point()
                with self.loops._lock:
                    measured = dict(self.known.measured.get(self.ident[1], {}))  # type: ignore[index]
                program = _record_whole(
                    self.loop, self.tensors, self.nwork, self.pre, self.body, self.post, self.ident,
                    measured,
                )
        finally:
            lora.LOOP.reset(overlays)
            graphs.BYPASS.reset(bypass)
            graphs._SETTLING.reset(settling)
            placements = (getattr(owner, "_cozy_residency", None) for owner, _, _ in self.entry)
            for placement in {id(p): p for p in placements if p is not None}.values():
                placement.release()  # the scopes' holds, as a request's end gives them back
            _lap("release")
            _LAPS.reset(laps)
        program.recorded = parts | {"total": round((time.perf_counter() - began) * 1000, 1)}
        self.loops._install(self.site, program, self.scope[0].memory)
        return True


def _used(seen: set[Any], scope: tuple[Model, ...]) -> dict[Model, tuple[str, ...]]:
    """Each scope model a call ran, with the block paths it entered ("" is common)."""
    paths: dict[Model, set[str]] = {}
    for item in seen:
        model = item if isinstance(item, Model) else getattr(item, "model", None)
        if model is not None and any(model is m for m in scope):
            paths.setdefault(model, set())
            if not isinstance(item, Model) and item.path:  # common comes with any binding
                paths[model].add(item.path)
    return {model: tuple(sorted(paths[model])) for model in scope if model in paths}


def _after(
    loop: _Loop, tensors: tuple[Any, ...], scope: tuple[Model, ...], known: _Site, shape: Hashable
) -> None:
    """`after` eagerly; a keyed call's site learns the blocks it ran for this layout (`shape`,
    None for a call without a key), and its working peak."""
    if loop.after is None:
        return
    if shape is None or shape in known.after:
        _once(loop.after, tensors)
        return
    with _entering() as seen, _measuring() as peaks:
        _part(loop.after, tensors, "after")
    _measured(known, shape, peaks)
    known.after[shape] = _used(seen, scope)


def _back(work: tuple[Any, ...], tensors: tuple[Any, ...]) -> None:
    """The program's copies' values into the caller's tensors (a no-op for the same tensors)."""
    for value, slot in zip(work, tensors, strict=False):
        if value is not slot:
            value.copy_(slot)


@contextmanager
def _owned(mutable: tuple[Any, ...]) -> Iterator[tuple[Any, ...]]:
    """Ordinary tensors for the loop to advance; an inference-mode one gets a copy and its
    final values back."""
    work = tuple(t.clone() if t.is_inference() else t for t in mutable)
    yield work
    with torch.inference_mode():
        for original, value in zip(mutable, work, strict=True):
            if value is not original:
                original.copy_(value)


@contextmanager
def _taken(lock: threading.Lock, ctx: Context) -> Iterator[None]:
    while not lock.acquire(timeout=0.05):
        ctx.raise_if_cancelled()
    try:
        yield
    finally:
        lock.release()


def _stale(program: _Program, ident: Hashable, *, checked: bool = False) -> str:
    """Why a kept program cannot serve this call ("" when it can). `checked`: also the
    definition a later call runs (`again`), an O(weights) walk, once per call after pinning."""
    if program.closed:
        return "released"
    if ident is None or program.ident != ident:
        return "another key or layout"
    for sequence in program.sequences():
        if sequence._closed:
            return "released"
    for stage in program.again.stages if checked and program.again is not None else ():
        if why := stage._invalid_reason():
            return str(why)
    return ""


def _close_idle(program: _Program) -> None:
    with program.lock:
        program.release()


@contextmanager
def _pinned(program: _Program) -> Iterator[str]:
    """Why the kept program cannot run now ("" when it can). For the run it is held against
    reclaim, its weights and pool back on the card; varena refusing them room before any of it
    runs, it runs eagerly. An unproven run keeps them (`completion_proven`): its executor
    retires."""
    if program.closed:
        yield "released"
        return
    held, why, sequences = ExitStack(), "", program.sequences()
    device = program.tensors[0].device
    try:
        if program.region is not None:  # a first call's rest, kept by no call
            held.enter_context(program.region.pin())
        stream = torch.cuda.current_stream(device)
        held.enter_context(graphs.replaying(stream, program.bindings()))
        sequences[0].map_pool()
    except varena.Exhausted as error:
        why = graphs.no_room(device.index, "its replay", program.pool_bytes, error)
        graphs.safeguard("no_room", why)
        why = f"{why}: it runs eagerly"
    except ResidencyError as error:  # released, or its weights changed
        why = f"{type(error).__name__}: {error}"[:300]
    if why:
        held.close()
        yield why
        return
    try:
        yield ""
    finally:
        if any(s._active is not None and not s._active.completion_proven() for s in sequences):
            _UNPROVEN.append(held.pop_all())
        held.close()


def _layouts(tensors: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in tensors)


def _phase_models() -> tuple[Model, ...]:
    """The placed components of every component scope the call is in, outermost first: a step
    reads any it holds (H3 Turbo's DiT, in its base's scope, and its overlay's factors)."""
    models: dict[int, Model] = {}
    for owner, _, components in _LOCAL_SCOPES.get():
        placement: Any = owner._cozy_residency
        for name in components:
            if placement is not None and name in placement.models:
                models.setdefault(id(placement.models[name]), placement.models[name])
    return tuple(models.values())


def _phase_configuration(
    loop: _Loop, indices: range, tensors: tuple[Any, ...], models: tuple[Model, ...]
) -> str | None:
    # The loop key declares its non-tensor inputs; unknown closures cannot train a reusable key.
    code = getattr(loop.step, "__code__", None)
    if loop.key is None or code is None or not models or stepcache._RUN.get() is not None:
        return None
    identities = tuple((model.content, model.wanted_identity()) for model in models)
    if any(identity is None for _, identity in identities):
        return None
    roots = {model.content: model.module for model in models}
    selected = attention.chosen(roots)
    artifacts = attention.observed(roots).artifacts or {}
    return working_phase.configuration((
        "denoise_loop", loop.step.__module__, loop.step.__qualname__,
        hashlib.sha256(marshal.dumps(code)).hexdigest(), loop.key, identities,
        tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device),
               t.storage_offset()) for t in tensors),
        (indices.start, indices.stop, indices.step), tuple(loop.label(i) for i in indices)
        if loop.labels is not None else None,
        selected.sites, selected.dense, selected.exact,
        tuple((name, row.get("artifact"), row.get("key"))
              for name, row in sorted(artifacts.items())),
        str(torch.get_default_dtype()), torch.is_autocast_enabled("cuda"),
        str(torch.get_autocast_dtype("cuda")), torch.get_float32_matmul_precision(),
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32, torch.are_deterministic_algorithms_enabled(),
    ))


def _graph_configuration(
    loop: _Loop,
    indices: range,
    tensors: tuple[Any, ...],
    models: tuple[Model, ...],
    role: str,
    site: tuple[str, int] | None = None,
) -> str | None:
    configuration = _phase_configuration(loop, indices, tensors, models)
    callback = {"before": loop.before, "after": loop.after}.get(role, loop.step)
    if callback is None:
        return None
    code = getattr(callback, "__code__", None)
    if configuration is None or code is None:
        return None
    return working_phase.configuration((
        "graph-stage/1", configuration, role, site,
        callback.__module__, callback.__qualname__, hashlib.sha256(marshal.dumps(code)).hexdigest(),
    ))


def _eligible(
    ctx: Context | None, steps: range, tensors: tuple[Any, ...]
) -> tuple[tuple[Model, ...], str]:
    """The step's placed components, or why the loop runs eagerly."""
    if ctx is None:
        return (), "outside a request"
    if ctx.device.type != "cuda" or any(not t.is_cuda for t in tensors):
        return (), "not on CUDA"
    if _execution is None:
        return (), "this Runtime has no native execution module"
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        return (), "several ranks"
    if len(steps) < 2:
        return (), "one step"
    if not _LOCAL_SCOPES.get():
        return (), ""  # no components: tensor work only
    for owner, _, components in _LOCAL_SCOPES.get():
        if skipped := [name for name in components if name in type(owner).__uncaptured__]:
            return (), f"its scope holds `@uncaptured` {', '.join(skipped)}"
    models = _phase_models()
    if not models:
        return (), "the scope's components are not placed on this device"
    return models, ""


#: each component's LoRA overlay hooks, which are permanent (`lora.Hook`)
_HOOKS: weakref.WeakKeyDictionary[Any, tuple[lora.Hook, ...]] = weakref.WeakKeyDictionary()


def _overlays(scope: tuple[Model, ...]) -> frozenset[int]:
    """The LoRA overlays armed in the scope's linears now, before the loop (`lora.Hook`), by
    identity: a kept program serves only calls that armed the same (`lora.LOOP`)."""
    armed: set[int] = set()
    for model in scope:
        if (hooks := _HOOKS.get(model.module)) is None:
            hooks = _HOOKS[model.module] = tuple(
                hook for sub in model.module.modules() for hook in sub._forward_hooks.values()
                if isinstance(hook, lora.Hook)
            )
        armed.update(id(factors) for hook in hooks if (factors := hook.armed()) is not None)
    return frozenset(armed)


def _unresident(model: Model, paths: tuple[str, ...] = ("",)) -> str:
    """Why these blocks of `model` (and its common weights) are not all on the device."""
    binding = model.retain(paths)
    try:
        return str(binding.unresident())
    finally:
        binding.close()


def _abandon(recording: _SequenceCapture, failure: BaseException | None = None) -> None:
    """A refused recording goes as its frames, then its stages and domain, then its pool, which
    varena is told is done (`vt.pool_close`): its pages are free for what runs next (SDXL's
    refused calls once left 12 pools, 9.26 GiB mapped: 34061-34065)."""
    if failure is not None:
        unwind(failure)
    pool = recording._pool
    device = None if recording._domain is None else recording._domain.device
    _drop_stages(recording)
    recording.close()
    if pool is not None and device is not None:
        graphs.close_pool(pool, device)


def _hold(models: Iterable[Model], stack: ExitStack) -> None:
    """Each of `models` not resident loaded and held in `stack`; `_NoRoom` where varena has no
    room for one, before anything is captured."""
    for model in models:
        if _unresident(model):
            try:
                hold = model.hold()
            except varena.Exhausted as error:
                raise _NoRoom(f"{model.component}'s weights: {error}") from error
            if hold is not None:
                stack.callback(hold.release)


def _bind(
    used: dict[Model, set[str]], attempt: ExitStack, bindings: list[Binding]
) -> None:
    """Each model a recording reads retained into `bindings` and its weights prepared in
    `attempt`: all resident, or the step cannot be recorded."""
    idle: set[int] = set()
    for model, paths in used.items():
        _checked(model)
        binding = model.retain(tuple(sorted(paths)))
        bindings.append(binding)
        if id(model.memory) not in idle:
            attempt.enter_context(model.memory.request(graphs.IDLE_REQUEST, idle=True))
            idle.add(id(model.memory))
        if why := binding.unresident():
            raise CapabilityError(
                f"capture requires the component's weights to be resident ({why})",
                code="prepared_graph_ineligible",
            )
        attempt.enter_context(binding.prepare())


def _admit(
    recording: _SequenceCapture,
    menu: list[tuple[int, bool]],
    used: Callable[[bool], dict[Model, set[str]]],
    preparing: ExitStack,
    bindings: list[Binding],
) -> bool:
    """The recording's mode, decided before anything is captured: the first of `menu` (`_menu`)
    whose weights and pool varena makes room for, with or without `after`. Its pool is prepared
    (`_working_bytes`), every capture of the recording takes it, and its weights stay prepared in
    `preparing`; `_NoRoom`, naming what each needed, when none has room."""
    if not menu:
        raise CapabilityError(
            "its parts' working memory was never measured", code="prepared_graph_ineligible"
        )
    pool, device = recording._pool.handle, recording._domain.device
    refused: list[str] = []
    for need, with_after in menu:
        attempt, held = ExitStack(), []
        try:
            _bind(used(with_after), attempt, held)
            vt.prepare_pool(pool, need, device)
        except BaseException as error:
            attempt.close()
            for binding in held:
                binding.close()
            vt.cancel_preparation(pool, device)
            if not isinstance(error, varena.Exhausted):
                raise
            what = "its whole call" if with_after else "its steps, `after` eagerly,"
            refused.append(graphs.no_room(device, what, need, error))
            continue
        preparing.enter_context(attempt)
        bindings.extend(held)
        recording._working_bytes = need
        return with_after
    raise _NoRoom("; ".join(refused))


def _drop_stages(recording: _SequenceCapture) -> None:
    """A refused attempt's stages go, before its recording closes."""
    with recording._submission:
        stages, recording.stages = recording.stages, []
        recording._areas.clear()
    for stage in stages:
        with stage._submission:
            stage._sequence_owner = None
        stage.close()


def _submit(
    program: _Program,
    sequence: PreparedSequence,
    indices: range,
    loop: _Loop,
    ctx: Context,
    row: dict[str, Any],
) -> None:
    """One native submission covering `indices`; no fallback once it is accepted."""
    finite = graphs.Finite(torch, program.tensors[: loop.latents])
    with torch.no_grad():
        program.counter.fill_(indices.start)
    row["at_ms"].setdefault("submitted", _now())
    _mark("submit")
    execution: Any = (
        sequence.submit(ctx)
        if ctx is None or ctx._submit_prepared is None
        else ctx._submit_prepared(sequence)
    )
    on_step, reported, seen = loop.on_step, 0, 0
    while True:
        if ctx is not None and ctx.cancelled:
            execution.cancel()
        progress = execution.poll()
        done = int(progress.completed_steps)
        if done > seen:
            seen = done
            if not finite.step():
                execution.cancel()
        if on_step is not None:
            for i in indices[reported:done]:
                with vt.external_callback(program.tensors[0].device.index):
                    on_step(i)
        reported = max(reported, done)
        if progress.done:
            break
        time.sleep(0.01)
    execution.wait()
    _mark(f"native {indices.start}-{indices[-1]}")
    row["at_ms"]["finished"] = _now()
    if ctx is not None:
        ctx.raise_if_cancelled()
    if not finite.last():
        raise _non_finite(indices.start - loop.steps.start + seen, len(loop.steps))
    facts = execution.facts()
    done = int(progress.completed_steps)
    if progress.canceled or done != len(indices):
        raise OutputError(
            f"the native loop stopped after {done} of {len(indices)} steps",
            code="incomplete_denoise",
        )
    if on_step is not None:
        for i in indices[reported:]:
            with vt.external_callback(program.tensors[0].device.index):
                on_step(i)
    row["native_steps"] = row.get("native_steps", 0) + done
    row["host_graph_launches"] = row.get("host_graph_launches", 0) + int(
        facts["host_graph_launches"]
    )
    if phases := facts.get("native_phases"):
        row["native_phases"] = row.get("native_phases", 0) + len(phases)


# ------------------------------------------------------------------ recording


def _record_resident(
    recording: _SequenceCapture,
    loop: _Loop,
    rest: range,
    tensors: tuple[Any, ...],
    mutable: tuple[Any, ...],
    counter: Any,
    body: dict[Model, tuple[str, ...]],
    pre: dict[Model, tuple[str, ...]] | None,
    post: dict[Model, tuple[str, ...]] | None,
    again: bool,
    menu: list[tuple[int, bool]],
) -> tuple[list[tuple[int, int]], tuple[Any, ...] | None, bool]:
    """One graph per distinct label, every component resident and held; a run of equal labels is
    one stage, its label's graph repeated natively. The parts are stage ranges, in order: `rest`
    (the recording call's own steps left, when any), then with `again` the reuse program over
    every step, `before` its first stage's prefix. With `pre`/`post` (the models `before`/`after`
    run, resident too), `after` is each part's last stage's suffix, when `menu`'s mode with it has
    room (`_admit`): whether it does comes back. Under a step cache, each step records in pieces
    at its verdicts (`_Graph.cut`); the counters its graphs write come back."""
    run = stepcache._RUN.get()
    plans = [indices for indices, wanted in ((rest, len(rest)), (loop.steps, again)) if wanted]
    runs = [_runs(indices, loop) for indices in plans]

    def used(with_after: bool) -> dict[Model, set[str]]:
        paths: dict[Model, set[str]] = {}
        for each in (body, pre or {}, (post or {}) if with_after else {}):
            for model, entered in each.items():
                paths.setdefault(model, set()).update(entered)
        return paths

    state = (*mutable, counter)
    bindings: list[Binding] = []
    provisional: list[_Graph] = []
    try:
        with ExitStack() as preparing:
            if loop.cache is not None and run is None:  # an idle recording's cache is its own
                run = preparing.enter_context(
                    stepcache.running(loop.cache, loop.steps, counter.device)
                )
            with_after = _admit(recording, menu, used, preparing, bindings)
            if not with_after:
                post = None
            held = tuple(used(with_after))
            _lap("bind")
            owners = tuple(bindings)
            landed = LANDED[0]
            bodies: dict[Any, _Graph] = {}
            armed: list[str] = []
            arming = lora.ARMED.set(armed)
            # Weights are checked by identity and version, never copied: a copy of each weight
            # buffer held SDXL's 2.7 GiB of int8 UNet twice through its recording (cr-264).
            weights = frozenset(i for binding in bindings for i in binding.immutable_buffer_ids())
            with _guarded(held, weights):
                for first, count, label in (run for each in runs for run in each):
                    if label in bodies:
                        continue
                    graph = _Graph(
                        recording, state, owners,
                        configuration=partial(
                            _graph_configuration, loop, range(first, first + count),
                            tensors, held, "step",
                        ),
                    )

                    def advance(label: Any = label, graph: _Graph = graph, run: Any = run) -> None:
                        if run is not None:
                            run.begin(counter, cut=graph.cut)
                        _call(loop.step, counter, tensors, label)
                        counter.add_(1)

                    provisional.append(graph)
                    try:
                        graph.record(advance)
                    finally:
                        if run is not None:
                            run.begin(None)
                    bodies[label] = graph
                    _lap("record_step")
                    _yield_point()
                prefix = suffix = None
                if loop.after is not None and post is not None:
                    suffix = _Graph(
                        recording, state, owners,
                        configuration=partial(
                            _graph_configuration, loop, loop.steps, tensors, held, "after",
                        ),
                    )
                    provisional.append(suffix)
                    suffix.record(lambda: loop.after(*tensors))  # type: ignore[misc]
                    _lap("record_after")
                    _yield_point()
                if again and loop.before is not None and pre is not None:
                    prefix = _Graph(
                        recording, state, owners,
                        configuration=partial(
                            _graph_configuration, loop, loop.steps, tensors, held, "before",
                        ),
                    )
                    provisional.append(prefix)
                    prefix.record(lambda: loop.before(*tensors))  # type: ignore[misc]
                    _lap("record_before")
            lora.ARMED.reset(arming)
            if armed:
                raise CapabilityError(
                    f"a LoRA overlay was armed on {armed[0]} as the loop recorded",
                    code="prepared_graph_ineligible",
                )
            _lap("record")
            _yield_point()
            parts: list[tuple[int, int]] = []
            for part, each in enumerate(runs):
                begin = len(recording.stages)
                for number, (_, count, label) in enumerate(each):
                    plan = bodies[label].plan(
                        landed,
                        count,
                        prefix=prefix if again and part == len(runs) - 1 and not number else None,
                        suffix=suffix if number == len(each) - 1 else None,
                    )
                    _own(plan, owners, recording)
                parts.append((begin, len(recording.stages)))
            for done in (*bodies.values(), prefix, suffix):
                if done is not None:
                    done.done()
            _lap("plan")
    except BaseException:
        # A later part can fail before any native stage owns the earlier captures.
        for graph in reversed(provisional):
            graph.reset()
        for binding in bindings:
            binding.close()
        raise
    for binding in bindings:  # the last stage closes the bindings every stage reads
        recording.stages[-1]._binding_cleanup.callback(binding.close)
    return parts, None if run is None else run.counted(), with_after


def _runs(indices: range, loop: _Loop) -> list[tuple[int, int, Any]]:
    """(first index, count, label) of each run of equal labels in `indices`."""
    runs: list[tuple[int, int, Any]] = []
    for i in indices:
        label = loop.label(i)
        if runs and runs[-1][2] == label:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1, label)
        else:
            runs.append((i, 1, label))
    return runs


def _checked(model: Model) -> None:
    placed = PLACED.get(model.module)
    if placed is not None and placed[0].lora_choices((placed[1],))[placed[1]][0] == "per_step":
        raise CapabilityError(
            "native loops need a fixed baked LoRA representation", code="prepared_graph_ineligible"
        )
    if why := graphs._hooked(model.module, overlays=True):
        raise CapabilityError(why, code="prepared_graph_ineligible")
    if not model.memory.cuda:
        raise CapabilityError("native loops need CUDA weights", code="prepared_graph_ineligible")


@contextmanager
def _guarded(models: tuple[Model, ...], immutable: frozenset[int]) -> Iterator[None]:
    """Recording leaves the modules' state as it found it, original operations recorded."""
    bypass = graphs.BYPASS.set(True)
    snapshots = [graphs._State(torch, m.module, immutable_buffers=immutable) for m in models]
    try:
        yield
    finally:
        changed = [change for snapshot in snapshots if (change := snapshot.restore())]
        graphs.BYPASS.reset(bypass)
    if changed:
        raise CapabilityError(
            "recording changed the model's state: " + ", ".join(changed),
            code="prepared_graph_ineligible",
        )


def _own(
    plan: PreparedRepeat,
    bindings: tuple[Binding, ...],
    recording: _SequenceCapture,
    cleanup: ExitStack | None = None,
) -> None:
    """The plan owns its bindings; residency retires it before releasing their weights."""
    plan._owned_bindings = bindings
    for binding in bindings:
        module = binding.model.module
        graphs.own_prepared(module, plan)
        if binding.model.retire_graphs is None:
            root = weakref.ref(module)

            def retire(root: Any = root) -> None:
                if (found := root()) is not None:
                    graphs.retire(torch, found)

            binding.model.retire_graphs = retire
    if cleanup is not None:
        plan._binding_cleanup = cleanup
    with recording._submission:
        plan._sequence_owner = weakref.ref(recording)
        recording.stages.append(plan)


class _Spans:
    """Byte ranges by device, merged and sorted: whether a tensor's storage overlaps any (a
    plan's held tensors against every output its graph made: O(n log n), where a scan of each
    against each took an RTX 4090's SDXL plan 320 ms)."""

    def __init__(self, areas: Iterable[tuple[Any, int, int]]) -> None:
        by: dict[Any, list[tuple[int, int]]] = {}
        for device, begin, end in areas:
            by.setdefault(device, []).append((begin, end))
        self.spans: dict[Any, tuple[list[int], list[int]]] = {}
        for device, ranges in by.items():
            starts: list[int] = []
            ends: list[int] = []
            for begin, end in sorted(ranges):
                if ends and begin <= ends[-1]:
                    ends[-1] = max(ends[-1], end)
                else:
                    starts.append(begin)
                    ends.append(end)
            self.spans[device] = (starts, ends)

    def overlaps(self, tensor: Any) -> bool:
        if not isinstance(tensor, torch.Tensor) or (found := self.spans.get(tensor.device)) is None:
            return False
        storage = tensor.untyped_storage()
        first, size = int(storage.data_ptr()), int(storage.nbytes())
        starts, ends = found
        i = bisect.bisect_left(starts, first + size) - 1
        return bool(size) and i >= 0 and ends[i] > first


class _Graph:
    """One CUDA graph recorded between begin() and end() into the recording's shared pool, in
    pieces when a step cache cuts it at its verdicts (`cut`).

    Every tensor the recorded operations read that they did not produce is retained by the
    native owner, except its weights: their Regions keep their addresses, and each replay takes
    their pages (`graphs.replaying`).
    """

    def __init__(
        self,
        recording: _SequenceCapture,
        state: tuple[Any, ...],
        bindings: tuple[Binding, ...],
        *,
        configuration: Callable[[], str | None] | None = None,
    ) -> None:
        self.recording, self.state, self.bindings = recording, state, bindings
        self.device = state[0].device
        self.domain = recording._domain
        self.held: dict[int, Any] = {id(t): t for t in state}
        self.produced: dict[int, Any] = {}
        self.generated: dict[int, tuple[Any, Any, Any, int, int]] = {}
        self.ops = 0
        self.graph: Any = None
        #: the finished pieces: the first, then per verdict its set and unset branches and the next
        self.pieces: list[Any] = []
        #: each verdict's flag (an int32's address), which picks its branch
        self.flags: list[int] = []
        self.index: Any = None
        self._contexts = ExitStack()
        self._piece = ExitStack()
        self._mode: Any = None
        self._caller: Any = None
        self._configuration = configuration
        self._configuration_before: str | None = None
        self._observation: working_phase.Observation | None = None
        self._replay_key: str | None = None

    def begin(self) -> None:
        """Capture on the recording's library stream; only the recorded region runs there."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("a loop stage begins outside capture")
        self._caller = torch.cuda.current_stream(self.device)
        self.index = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.held[id(self.index)] = self.index
        self._configuration_before = None if self._configuration is None else self._configuration()
        context = CURRENT.get()
        pause = context._pause_stall_watch if context is not None else None
        try:
            request = _REQUEST.get()
            self._observation = self._contexts.enter_context(working_phase.observing(
                self.device.index if request is not None else None, "capture",
                request.working_phases if request is not None else None,
            ))
            self._contexts.enter_context(pause() if pause is not None else nullcontext())
            self._contexts.enter_context(kernel_ready.recording())
            self.domain.stream.wait_stream(self._caller)
            self._open()
        except BaseException as error:
            self._contexts.__exit__(type(error), error, error.__traceback__)
            raise
        self._mode = _Owners(self)
        self._mode.__enter__()

    def _open(self) -> None:
        """A piece's capture, on the control stream; its work on the library stream, in the pool
        its recording's mode reserved (`_admit`)."""
        stream, control = self.domain.stream, self.domain.control
        control.wait_stream(stream)
        assert self.recording._working_bytes is not None
        self.graph = torch.cuda.CUDAGraph(keep_graph=True)
        try:
            self._piece.enter_context(vt.graph(
                self.graph, pool=self.recording._pool.handle, stream=control,
                reserve_bytes=self.recording._working_bytes, primed=(stream,),
            ))
            torch.cuda.set_stream(stream)
        except BaseException:
            torch.cuda.set_stream(self._caller)
            self.graph = None
            raise
        self.domain.fork.record(control)
        stream.wait_event(self.domain.fork)

    def _close(self, error: BaseException | None) -> None:
        """The piece's capture ends, joined first; a whole piece is kept."""
        stream, control = self.domain.stream, self.domain.control
        try:
            self.domain.join.record(stream)
            control.wait_event(self.domain.join)
        finally:
            try:
                # The owner ends actual CUDA capture even when the join was invalidated,
                # then completes routing and the Varena capture exclusion exactly once.
                self._piece.__exit__(
                    None if error is None else type(error), error,
                    None if error is None else error.__traceback__,
                )
            finally:
                torch.cuda.set_stream(self._caller)
        if error is None:
            self.pieces.append(self.graph)

    def cut(self, flag: Any) -> None:
        """A step cache's verdict (`stepcache`): this piece ends, and the next begins. After a
        `flag`, the branch it sets; then (None) the branch it leaves unset; then (None) the
        piece after both."""
        if (flag is not None) != (len(self.pieces) % 3 == 0):
            raise CapabilityError(
                "a step cache's branches do not close in its step", code="prepared_graph_ineligible"
            )
        self._close(None)
        if flag is not None:
            self.flags.append(flag.data_ptr())
        self._open()

    def end(self, error: BaseException | None = None) -> None:
        if self.graph is None:
            return
        try:
            if self._mode is not None:
                mode, self._mode = self._mode, None
                mode.__exit__(
                    None if error is None else type(error), error,
                    None if error is None else error.__traceback__,
                )
        finally:
            try:
                self._close(error)
            finally:
                self._contexts.__exit__(
                    None if error is None else type(error), error,
                    None if error is None else error.__traceback__,
                )
        if error is None and len(self.pieces) % 3 != 1:
            raise CapabilityError(
                "a step cache's branches do not close in its step", code="prepared_graph_ineligible"
            )
        if error is None:
            for piece in self.pieces:
                self.domain.graphs.add(piece)
            actual = None if self._configuration is None else self._configuration()
            observation = self._observation
            if (
                observation is not None and observation.result is not None
                and not observation.result.get("invalidated", True)
                and actual is not None and actual == self._configuration_before
            ):
                self._replay_key = working_phase.execution_key(actual, "runtime.graph/1")

    def abort(self, error: BaseException | None = None) -> None:
        """End a failed capture with its error so the owner cannot publish a partial graph. When
        the capture was invalidated, its join's failure (901) is raised instead, `error` its
        context: the step cannot be recorded (Sage2's launches on the legacy stream: run 33993)."""
        try:
            self.end(error)
        except BaseException as failure:
            if failure is error:
                self.reset()
                return
            try:
                self.reset()
            except BaseException as reset_error:
                failure.add_note(f"Graph reset also failed: {reset_error!r}")
            raise
        else:
            self.reset()

    def reset(self) -> None:
        for piece in {id(p): p for p in (*self.pieces, self.graph) if p is not None}.values():
            piece.reset()
        self.graph, self.pieces, self.flags = None, [], []
        self.held.clear()
        self.__dict__.pop("external", None)
        self._replay_key = None
        self._configuration = None

    def record(self, body: Callable[[], object]) -> None:
        self.begin()
        try:
            body()
        except BaseException as error:
            self.abort(error)
            raise
        self.end()

    @cached_property
    def external(self) -> dict[int, Any]:
        """What the recorded operations read from outside this graph and its weights."""
        inside = _Spans((
            *((d, b, e) for _, _, d, b, e in self.generated.values()),
            *(
                (self.device, region.ptr, region.ptr + region.allocated_bytes)
                for binding in self.bindings
                for region in binding.regions or ()
            ),
        ))
        return {key: tensor for key, tensor in self.held.items() if not inside.overlaps(tensor)}

    def plan(
        self,
        landed: int,
        steps: int,
        prefix: _Graph | None = None,
        suffix: _Graph | None = None,
    ) -> PreparedRepeat:
        """A native owner repeating this graph `steps` times, with one-shot `prefix` and `suffix`
        graphs of the same recording around the repeat; a verdict's pieces as its branches."""
        held: dict[int, Any] = {}
        for graph in (self, prefix, suffix):
            if graph is not None:
                held.update(graph.external)
        _lap("plan_held")
        raw = [piece.raw_cuda_graph() for piece in self.pieces]
        phases = {
            "prefix_graph": 0 if prefix is None else prefix.pieces[0].raw_cuda_graph(),
            "suffix_graph": 0 if suffix is None else suffix.pieces[0].raw_cuda_graph(),
            "branches": [
                (flag, *raw[3 * i + 1 : 3 * i + 4]) for i, flag in enumerate(self.flags)
            ],
        }
        extra = [graph.pieces[0] for graph in (prefix, suffix) if graph is not None]
        try:
            native = _execution.ResidentRepeat(
                raw[0],
                self.index.data_ptr(),
                self.device.index,
                steps,
                [
                    *self.pieces,
                    *extra,
                    self.domain.stream,
                    self.index,
                    *held.values(),
                    *(tensor.untyped_storage() for tensor in held.values()),
                    *self.bindings,
                    self.domain,
                    self.recording._pool,
                ],
                **phases,
            )
        except _execution.NativeExecutionError as error:
            self.reset()
            _native_error(error)
        except BaseException:
            self.reset()
            raise
        _lap("plan_native")
        if landed != LANDED[0]:
            native.close_owner()
            self.reset()
            raise CapabilityError(
                "kernel implementations changed during recording", code="prepared_graph_ineligible"
            )
        dependencies = tuple((weakref.ref(t), PreparedRepeat._layout(t)) for t in held.values())
        identities = tuple((weakref.ref(b), b.identity) for b in self.bindings)
        plan = PreparedRepeat(
            native, self.state, self.device.index, steps, dependencies, identities, landed
        )
        plan._capture_cache = {"recorded_ops": self.ops}
        parts = tuple(graph for graph in (prefix, self, suffix) if graph is not None)
        if all(graph._replay_key is not None for graph in parts):
            key = working_phase.configuration((
                "prepared-repeat/1", steps, tuple(graph._replay_key for graph in parts),
            ))
            assert key is not None
            plan._execution_key = "runtime.replay/1:" + hashlib.sha256(key.encode()).hexdigest()
        return plan

    def done(self) -> None:
        """Every native owner is made: drop this graph's Python references to what it read."""
        self.held.clear()
        self.__dict__.pop("external", None)
        self._configuration = None


class _Owners(TorchDispatchMode):  # type: ignore[misc]
    """What one recorded graph reads from outside itself, and the scratch it makes."""

    def __init__(self, graph: _Graph) -> None:
        super().__init__()
        self.g = graph

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        g, kwargs = self.g, kwargs or {}
        leaves = tree_leaves((args, kwargs))
        if torch.Tag.nondeterministic_seeded in func.tags and not _zero_dropout_attention(
            func, args, kwargs
        ):
            raise CapabilityError(
                "native loops take noise as an input, never a random draw",
                code="prepared_graph_ineligible",
            )
        inputs = set()
        for tensor in leaves:
            if isinstance(tensor, torch.Tensor):
                prior = g.produced.get(id(tensor))
                if prior is None or prior() is not tensor:
                    g.held[id(tensor)] = tensor
                inputs.add((tensor.device, tensor.untyped_storage().data_ptr()))
        result = func(*args, **kwargs)
        g.ops += 1
        for tensor in tree_leaves(result):
            if isinstance(tensor, torch.Tensor):
                g.produced[id(tensor)] = weakref.ref(tensor)
                storage = tensor.untyped_storage()
                pointer, size = int(storage.data_ptr()), int(storage.nbytes())
                if size and (tensor.device, pointer) not in inputs:
                    g.generated[id(tensor)] = (
                        weakref.ref(tensor),
                        weakref.ref(storage),
                        tensor.device,
                        pointer,
                        pointer + size,
                    )
        return result


def install() -> None:
    _loop.install(Loops())
