"""`author.denoise_loop` in the Runtime: eager, or one resident captured program.

A call runs its site's kept program natively from start to end, else eagerly, for real, on the
caller's stream (only recorded regions run on the recording's streams, which end with the program).
A keyed call whose blocks all stay resident gets its whole call (`before`, the steps and `after`)
recorded and kept for its call site, until a new key replaces it or residency retires it under
pressure. The first call of a key and layout runs `before` and its first step eagerly, warming
their kernels and naming their blocks, then records its whole call on copies of its tensors (every
scope model held and loaded, its `after` not yet run) and runs the rest natively from them (xs-134
F2): its eager steps cost more than the recording on a host-bound CPU (SDXL int8 on an RTX 4090:
173 ms eager against 79 native, a 0.9 s recording), and the next call runs natively whenever it
comes. Refused, its rest runs eagerly and the recording is queued for the idle period after it
(`_IdleRecording`, xs-134 F), in the call's component scope; a command that arrives preempts it
between its parts (never inside package code). A call that finds it unrecorded (no idle period
yet, or preempted) records the whole call before running any of it and runs it natively from its
start: one plan and no eager step. No idle period
opens between a call's prepare and its invoke (`Executor.awaiting_invoke`), so a call queued behind
another never waits for idle work. One that does not fit, or whose code cannot run outside its
request, is not tried again for that layout, and the calls say why. A program that no call keeps is
not recorded: the call runs eagerly. Resident, it costs no more per step than native execution saves
(SDXL, Qwen-Image measured). Streamed (its blocks do not all stay on the device), its replay pays
for neither its recording nor the extra reads the next request makes (Qwen-Image int8 1 MP on an RTX
3070: eager 45.6-46.8 s, native 49.3-50.7 s; MiniMax-H3 FL2VA 15 s on an RTX 5090: 716 vs 737 s).

A loop outside any component scope (tensor work only) records resident every call and is never kept:
`before` and the first step run eagerly, then the rest natively. A refusal before submission drops
that attempt and the rest runs eagerly; nothing runs twice. After submission nothing falls back.
"""

from __future__ import annotations

import bisect
import ctypes
import gc
import sys
import threading
import time
import weakref
from collections.abc import Callable, Hashable, Iterable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from tensorfs import varena
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from cozy_runtime import graphs
from cozy_runtime.author import _loop
from cozy_runtime.author._context import CURRENT, Context
from cozy_runtime.author._errors import CapabilityError, Fault, OutputError, fault, unwind
from cozy_runtime.author._model import _LOCAL_SCOPES, LANDED, PLACED, Building
from cozy_runtime.graphs import domain
from cozy_runtime.internal import kernel_ready
from cozy_runtime.internal.accel import ResidencyError
from cozy_runtime.internal.prepared import (
    PreparedRepeat,
    PreparedSequence,
    _native_error,
    _SequenceCapture,
    _zero_dropout_attention,
)
from cozy_runtime.internal.residency import (
    _CAPTURE_BLOCKS,
    _REQUEST,
    CUTTER,
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

#: The CUDA driver, for a recorded graph's node count (absent without an NVIDIA driver).
_DRIVER: Any
try:
    _DRIVER = ctypes.CDLL("libcuda.so.1")
except OSError:
    _DRIVER = None

#: Codes a preparation refusal carries: the loop continues in the next mode.
_REFUSED = (
    "prepared_graph_ineligible",
    "prepared_load_unavailable",
    "prepared_load_events_unavailable",
)
#: cudaError_t stream-capture codes (900-908): an operation the step runs cannot be recorded.
_CAPTURE_ERRORS = range(900, 909)
#: request id -> its loops, for the run record (`execution.loops`)
ROWS: dict[str, list[dict[str, Any]]] = {}


def _call(step: Callable[..., object], index: Any, tensors: tuple[Any, ...], phase: Any) -> None:
    if phase is _NONE:
        step(index, *tensors)
    else:
        step(index, *tensors, phase=phase)


_NONE = object()
#: The recording call's phases, for the run record (`parts_ms`): [{phase: ms}, last lap's clock].
_LAPS: ContextVar[list[Any] | None] = ContextVar("cozy_loop_laps", default=None)


@contextmanager
def _laps(row: dict[str, Any]) -> Iterator[None]:
    """Time a recording call's phases into its row (`parts_ms`)."""
    token = _LAPS.set([row.setdefault("parts_ms", {}), time.perf_counter()])
    try:
        yield
    finally:
        _LAPS.reset(token)


def _lap(name: str) -> None:
    """Close the current phase of a recording call as `name` (milliseconds since the last lap)."""
    laps = _LAPS.get()
    if laps is not None:
        now = time.perf_counter()
        laps[0][name] = round(laps[0].get(name, 0.0) + (now - laps[1]) * 1000, 1)
        laps[1] = now


def _now() -> float:
    return round(time.time() * 1000, 1)


def _refusal(error: BaseException) -> str | None:
    """The reason a preparation error lets the loop continue in another mode, else None."""
    if isinstance(error, CapabilityError):
        return str(error) if error.code in _REFUSED else None
    if isinstance(error, Building):
        return f"capture waits for kernel readiness: {error}"
    if isinstance(error, torch.OutOfMemoryError):
        return f"preparation exceeded capacity: {error}"
    if isinstance(error, ResidencyError):
        return f"residency refused preparation: {error}"
    code = getattr(error, "error_code", None)
    if isinstance(code, int) and code in _CAPTURE_ERRORS:
        return f"the step cannot be recorded: {error}"
    if isinstance(error, Exception) and fault(error) is Fault.OTHER:
        return f"recording failed: {type(error).__name__}: {error}"
    return None


@dataclass
class _Program:
    """A recorded native loop and the tensors its graphs read.

    `first` runs the steps after the eager first one (and `after` when it is recorded); `again`,
    when recorded, runs `before`, every step and `after` for a later call, with no eager step.
    Several runs of equal phases are several parts, submitted in order.
    """

    first: tuple[tuple[PreparedSequence, int, int], ...]
    again: PreparedSequence | None
    mode: str
    tensors: tuple[Any, ...]
    counter: Any
    ident: Hashable
    stages: int
    blocks: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: a kept program's cache handle: residency releases it under pressure while it is idle
    region: Any = None
    #: its idle recording's parts and total (ms)
    recorded: dict[str, float] | None = None

    def sequences(self) -> tuple[PreparedSequence, ...]:
        rest = () if self.again is None else (self.again,)
        return (*(sequence for sequence, _, _ in self.first), *rest)

    def close(self) -> None:
        for sequence in self.sequences():
            sequence.close()

    def release(self) -> None:
        if self.region is None:
            self.close()
        else:
            self.region.close()  # calls close unless a reclaim already did


@dataclass
class _Site:
    """What a call site's keyed calls learned, per layout: the blocks an eager `after` ran, the
    whole call queued to record at idle, and the whole calls ruled out (and why)."""

    after: dict[Hashable, dict[Model, tuple[str, ...]]] = field(default_factory=dict)
    unfit: set[Hashable] = field(default_factory=set)
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
        phases: tuple[Hashable, ...] | None,
        key: Hashable | None,
        on_step: Callable[[int], object] | None,
        before: Callable[..., object] | None,
        after: Callable[..., object] | None,
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
        bypass = graphs.BYPASS.set(True)
        try:
            self._run(step, steps, state, carried, inputs, phases, key, on_step, before, after, row)
        finally:
            graphs.BYPASS.reset(bypass)
            row["at_ms"]["returned"] = _now()

    def _run(
        self,
        step: Callable[..., object],
        steps: range,
        state: tuple[Any, ...],
        carried: tuple[Any, ...],
        inputs: tuple[Any, ...],
        phases: tuple[Hashable, ...] | None,
        key: Hashable | None,
        on_step: Callable[[int], object] | None,
        before: Callable[..., object] | None,
        after: Callable[..., object] | None,
        row: dict[str, Any],
    ) -> None:
        ctx = CURRENT.get()
        loop = _Loop(step, steps, phases, on_step, before, after)
        mutable = state + carried
        with torch.inference_mode(False), torch.no_grad(), _owned(mutable) as work:
            tensors = work + inputs
            scope, why = _eligible(ctx, steps, tensors)
            if why:
                row["refusals"].append(why)
                loop.eager_all(tensors, ctx)
                return
            assert ctx is not None
            site = (step.__code__, tuple(id(model) for model in scope))
            # What a call site learns (`_Site`) depends on a keyed call's layout, not its key.
            shape = None if key is None else (_layouts(tensors), len(steps), phases)
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
                with _taken(kept.lock, ctx), _pinned(kept) as pinned:
                    why = _stale(kept, ident, checked=True) if pinned else "released"
                    if not why:
                        self._reuse(kept, loop, tensors, work, ctx, row)
                        return
                row["dropped"] = why
                with self._lock:
                    if self._kept.get(site) is kept:
                        del self._kept[site]
                _close_idle(kept)
            job = None if ident is None else known.pending.get(ident)
            if scope and job is None:
                self._first(site, known, loop, tensors, work, scope, ident, shape, ctx, row)
                return
            if job is not None:
                self._take_over(site, known, job, loop, tensors, work, scope, ctx, row)
                return
            program = None
            try:
                with _laps(row):
                    program = _record(loop, tensors, work, key, ctx, row)
                if program is None:
                    return
                for sequence, first, count in program.first:
                    _submit(program, sequence, range(first, first + count), loop, ctx, row)
            except BaseException:
                if program is not None:
                    program.close()
                raise
            # No call keeps a componentless program; it closes before the work after it (an
            # eager decode) needs its pool.
            program.close()
            _after(loop, program.tensors, scope, known, shape)
            _back(work, program.tensors)

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
        """A component call its site has no program for: `before` and its first step eagerly,
        warming their kernels and naming their blocks. A keyed whole call then records on its
        program's own copies and runs the rest natively from its second step; the program is
        kept for the site. Otherwise, or refused, the rest runs eagerly and the recording is
        queued for the idle period after it."""
        with _laps(row):
            pre, body = _warm(loop, tensors, scope, ctx)
            program, refused, began = None, "", time.perf_counter()
            if not _unkept(known, loop, ident, shape, body):
                program, refused = _record_first(loop, tensors, len(work), scope, pre, body, ident)
        if program is not None:
            row.update(
                mode=program.mode,
                stages=program.stages,
                blocks=program.blocks,
                whole_call="recorded on its first call, after its first step",
                capture_ms=round((time.perf_counter() - began) * 1000, 3),
            )
            self._install(site, program, scope[0].memory)
            with _taken(program.lock, ctx), _pinned(program) as pinned:
                if pinned:
                    ((rest, first, count),) = program.first
                    try:
                        _submit(program, rest, range(first, first + count), loop, ctx, row)
                    except BaseException:
                        program.close()
                        raise
                    rest.close()  # later calls run `again`
                    program.first = ()
                    _back(work, program.tensors)
                    return
            row["dropped"] = "released"
        if refused:
            row["refusals"].append(f"whole call on its first call: {refused}")
        loop.eager(loop.steps[1:], tensors, ctx)
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
    ) -> None:
        """Its idle recording has not run (no idle period yet) or was preempted: this call
        records the whole call before running any of it, from what its first call learned,
        and runs it natively from its start (one plan, no eager step)."""
        shape = job.ident[1]  # type: ignore[index]
        with self._lock:
            known.pending.pop(job.ident, None)
        program, began = None, time.perf_counter()
        try:
            with _laps(row):
                program = _record_whole(
                    loop, job.tensors, job.nwork, job.pre, job.body, job.post, job.ident
                )
        except Exception as error:
            if isinstance(error, Building) or isinstance(error.__cause__, Building):
                with self._lock:
                    known.pending[job.ident] = job  # a kernel still opens: the idle period
                graphs.IDLE.add(job)
                row["refusals"].append("whole call: a kernel still opens")
            elif (why := _refusal(error)) is not None:
                row["refusals"].append(f"whole call: {why}")
                with self._lock:
                    known.unfit.add(shape)
                    known.why[shape] = why[:400]
            else:
                raise
        if program is None:
            loop.eager_all(tensors, ctx)
            return
        row.update(
            whole_call="recorded on this call, its idle recording not run",
            capture_ms=round((time.perf_counter() - began) * 1000, 3),
        )
        self._install(site, program, scope[0].memory)
        with _taken(program.lock, ctx), _pinned(program) as pinned:
            if pinned and not _stale(program, job.ident, checked=True):
                self._reuse(program, loop, tensors, work, ctx, row)
                return
        row["dropped"] = "released"
        loop.eager_all(tensors, ctx)

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
    ) -> str:
        """Why the call ran eagerly: its whole call queued to record at idle, or why none does."""
        if why := _unkept(known, loop, ident, shape, body):
            return why
        with self._lock:
            if ident not in known.pending:
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
                    _LOCAL_SCOPES.get()[-1],
                    tuple(model.wanted_identity() for model in scope),
                )
                known.pending[ident] = job
                graphs.IDLE.add(job)
        return "resident: the whole call records at idle"

    def _install(self, site: Hashable, program: _Program, memory: Any) -> None:
        program.region = memory.region("denoise_loop", "program", release=program.close)
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
        row.update(mode=kept.mode, reused=True, stages=kept.stages, blocks=kept.blocks)
        if kept.recorded is not None:
            row["recorded_idle_ms"] = kept.recorded
        try:
            for slot, value in zip(kept.tensors, tensors, strict=True):
                slot.copy_(value)
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
    phases: tuple[Hashable, ...] | None
    on_step: Callable[[int], object] | None
    before: Callable[..., object] | None
    after: Callable[..., object] | None

    def phase(self, i: int) -> Any:
        return _NONE if self.phases is None else self.phases[i - self.steps.start]

    def eager(self, indices: range, tensors: tuple[Any, ...], ctx: Context | None) -> None:
        """The steps in order, each reported (`on_step(i)`) once queued; the call returns once the
        last has run on the device. Returning with it still queued put its time into the caller's
        next stage (Qwen-Image's last 0.9 s step into its decode)."""
        if not len(indices):
            return
        index = torch.zeros(1, dtype=torch.int64, device=tensors[0].device)
        for i in indices:
            if ctx is not None:
                ctx.raise_if_cancelled()
            index.fill_(i)
            _call(self.step, index, tensors, self.phase(i))
            if self.on_step is not None:
                self.on_step(i)
        if index.is_cuda:
            torch.cuda.current_stream(index.device).synchronize()

    def eager_all(self, tensors: tuple[Any, ...], ctx: Context | None) -> None:
        _once(self.before, tensors)
        self.eager(self.steps, tensors, ctx)
        _once(self.after, tensors)


def _once(fn: Callable[..., object] | None, tensors: tuple[Any, ...]) -> None:
    if fn is not None:
        fn(*tensors)


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
) -> tuple[dict[Model, tuple[str, ...]], dict[Model, tuple[str, ...]]]:
    """A component call's `before` and first step eagerly, on the caller's tensors: the blocks
    each ran."""
    with _entering() as seen:
        _once(loop.before, tensors)
    _lap("before")
    pre = _used(seen, scope)
    with _entering() as seen:
        loop.eager(loop.steps[:1], tensors, ctx)
    _lap("first_step")
    return pre, _used(seen, scope) or {model: ("",) for model in scope}


def _unkept(
    known: _Site,
    loop: _Loop,
    ident: Hashable,
    shape: Hashable,
    body: dict[Model, tuple[str, ...]],
) -> str:
    """Why no call keeps this call's whole program ("" when one does)."""
    resident = all(not _unresident(model, paths) for model, paths in body.items())
    if ident is None or loop.after is None or loop.phases is not None or not resident:
        return f"{'resident' if resident else 'streamed'}, and no call keeps the program"
    if shape in known.unfit:
        return f"no whole call: {known.why.get(shape, 'it did not fit')}"
    return ""


def _record_first(
    loop: _Loop,
    tensors: tuple[Any, ...],
    nwork: int,
    scope: tuple[Model, ...],
    pre: dict[Model, tuple[str, ...]],
    body: dict[Model, tuple[str, ...]],
    ident: Hashable,
) -> tuple[_Program | None, str]:
    """A first call's whole program, recorded after its first step on copies of its tensors:
    `first` runs its rest (from its second step, `after` included), `again` a later call. Its
    `after` has not run, so every scope model is held, and loaded first; (None, why) when
    refused, its attempt gone."""
    own = tuple(value.clone() for value in tensors)
    mutable = own[:nwork]
    counter = torch.zeros(1, dtype=torch.int64, device=own[0].device)
    every: dict[Model, tuple[str, ...]] = {model: ("",) for model in scope}
    recording = PreparedSequence.record()
    recording.__enter__()
    failure: BaseException | None = None
    try:
        with ExitStack() as loading:
            for model in scope:
                if _unresident(model) and (hold := model.hold()) is not None:
                    loading.callback(hold.release)  # the recording's bindings retain them
            _lap("load")
            recording._prepare((*mutable, counter))
            torch.cuda.empty_cache()
            _lap("prepare")
            with recording.capturing():
                parts, blocks = _record_resident(
                    recording, loop, loop.steps[1:], own, mutable, counter, body, pre, every, True
                )
            made = recording.finish_parts(
                outputs=mutable, parts=[(b, e, passes) for b, e, passes, _, _ in parts]
            )
        _lap("finish")
    except BaseException as error:
        failure = error
    if failure is not None:
        unwind(failure)
        _drop_stages(recording)
        gc.collect()
        torch.cuda.empty_cache()
        recording.close()
        if (why := _refusal(failure)) is None:
            raise failure
        return None, why
    rest, again = made  # one pass of equal steps (no phases): its rest, then the reuse program
    begin, end, _, first, count = parts[0]
    first_part = ((rest, first, count),)
    return _Program(first_part, again, "resident", own, counter, ident, end - begin, blocks), ""


def _record(
    loop: _Loop,
    tensors: tuple[Any, ...],
    work: tuple[Any, ...],
    key: Hashable | None,
    ctx: Context,
    row: dict[str, Any],
) -> _Program | None:
    """A componentless loop: `before` and the first step eagerly, then the rest recorded; None
    when the recording was refused, after the loop ran eagerly to its end."""
    if key is not None:  # the program owns copies; this call's values go in and out
        tensors = tuple(value.clone() for value in tensors)
    mutable = tensors[: len(work)]
    counter = torch.zeros(1, dtype=torch.int64, device=tensors[0].device)
    steps = loop.steps
    _once(loop.before, tensors)
    _lap("before")
    loop.eager(steps[:1], tensors, ctx)
    _lap("first_step")
    rest = steps[1:]
    recording = PreparedSequence.record()
    recording.__enter__()
    try:
        why = None
        try:
            recording._prepare((*mutable, counter))
            torch.cuda.empty_cache()
            _lap("prepare")
            began = time.perf_counter()
            ctx.raise_if_cancelled()
            with recording.capturing():
                parts, blocks = _record_resident(
                    recording, loop, rest, tensors, mutable, counter, {}, None, None, False
                )
            made = recording.finish_parts(
                outputs=mutable, parts=[(b, e, passes) for b, e, passes, _, _ in parts]
            )
            _lap("finish")
        except BaseException as error:
            if (why := _refusal(error)) is None:
                raise
        if why is not None:
            # Outside the handler: its traceback no longer holds the attempt's graphs, so their
            # pool returns before the eager rest.
            row["refusals"].append(f"resident: {why}")
            _drop_stages(recording)
            gc.collect()
            torch.cuda.empty_cache()
            _back(work, tensors)  # the eager rest runs on the caller's tensors
            loop.eager(rest, work + tensors[len(work) :], ctx)
            _once(loop.after, work + tensors[len(work) :])
            return None
        first = tuple(
            (sequence, start, count)
            for sequence, (_, _, _, start, count) in zip(made, parts, strict=True)
        )
        stages = sum(end - begin for begin, end, _, _, _ in parts)
        row.update(
            mode="resident",
            stages=stages,
            blocks=blocks,
            capture_ms=round((time.perf_counter() - began) * 1000, 3),
            after_native=False,
        )
        return _Program(first, None, "resident", tensors, counter, None, stages, blocks)
    finally:
        if not recording._finished:
            recording.close()


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
) -> _Program:
    """The whole call (`before`, every step and `after`) recorded on `tensors` as one program
    a later call runs natively; it yields to a command between its parts (`_yield_point`). A
    failed or preempted attempt goes as a refused one does: its frames first, then its stages and
    the tensors of its pool, then its domain (a preemption that closed the domain under its live
    frames crashed the executor)."""
    mutable = tensors[:nwork]
    counter = torch.zeros(1, dtype=torch.int64, device=tensors[0].device)
    recording = PreparedSequence.record()
    recording.__enter__()
    failure: BaseException | None = None
    try:
        recording._prepare((*mutable, counter))
        torch.cuda.empty_cache()
        _lap("prepare")
        _yield_point()
        with recording.capturing():
            parts, blocks = _record_resident(
                recording,
                loop,
                loop.steps[1:],
                tensors,
                mutable,
                counter,
                body,
                pre,
                post,
                True,
                remaining=False,
            )
        if len(parts) != 1:
            raise CapabilityError(
                "the whole call is not one pass of equal steps", code="prepared_graph_ineligible"
            )
        (again,) = recording.finish_parts(
            outputs=mutable, parts=[(b, e, passes) for b, e, passes, _, _ in parts]
        )
        _lap("finish")
    except BaseException as error:
        failure = error
    if failure is not None:
        unwind(failure)
        _drop_stages(recording)
        gc.collect()
        torch.cuda.empty_cache()
        recording.close()
        raise failure
    stages = parts[0][1] - parts[0][0]
    return _Program((), again, "resident", tensors, counter, ident, stages, blocks)


@dataclass(eq=False)
class _IdleRecording:
    """A keyed call's whole program, recorded at an idle period (`graphs.IDLE`) on copies of its
    tensors, in the call's component scope, and kept for its site. A command preempts it at its
    next torch op inside the recorded code; it records again at the next idle. One whose models
    changed fill or left the device goes (the next call queues again); one that cannot be
    recorded (it does not fit, or its code cannot run outside its request) is ruled out for its
    layout, and the next calls say why."""

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
    #: the call's innermost component scope: its owner, method and components
    entry: tuple[Any, str, tuple[str, ...]]
    identities: tuple[Any, ...]
    #: it yields between its parts (`_yield_point`), not per torch op (`graphs.IDLE`): under the
    #: per-op mode an RTX 4090's SDXL plan took 3.4 s instead of 0.46 s
    yields_per_op = False

    def __call__(self) -> bool:
        shape = self.ident[1]  # type: ignore[index]
        with self.loops._lock:
            if self.known.pending.get(self.ident) is not self:
                return True
        try:
            done = self._record()
        except Exception as error:
            if isinstance(error, graphs.Yielded) or fault(error) in (Fault.STICKY, Fault.DEVICE):
                raise
            if isinstance(error, Building) or isinstance(error.__cause__, Building):
                return False  # a kernel still opens: a later idle period
            why = _refusal(error) or f"{type(error).__name__}: {error}"
            with self.loops._lock:
                self.known.unfit.add(shape)
                self.known.why[shape] = f"its idle recording failed: {why}"[:400]
            graphs.say(f"loop:{why}", f"[loop] a whole call is not recorded: {why}"[:400])
            done = True
        if done:
            with self.loops._lock:
                if self.known.pending.get(self.ident) is self:
                    del self.known.pending[self.ident]
        return done

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
        owner, method, components = self.entry
        began = time.perf_counter()
        parts: dict[str, float] = {}
        laps = _LAPS.set([parts, began])
        settling = graphs._SETTLING.set(True)  # no torch op yields: only its boundaries do
        bypass = graphs.BYPASS.set(True)
        try:
            with (
                torch.cuda.device(self.tensors[0].device),
                torch.inference_mode(False),
                torch.no_grad(),
                owner._cozy_scope(method, components),
            ):
                _lap("scope")
                _yield_point()
                if any(_unresident(model, tuple(p)) for model, p in paths.items()):
                    return True  # its weights left the device: the next call queues again
                _lap("resident")
                _yield_point()
                program = _record_whole(
                    self.loop, self.tensors, self.nwork, self.pre, self.body, self.post, self.ident
                )
        finally:
            graphs.BYPASS.reset(bypass)
            graphs._SETTLING.reset(settling)
            if (placement := getattr(owner, "_cozy_residency", None)) is not None:
                placement.release()  # the scope's holds, as a request's end gives them back
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
    None for a call without a key)."""
    if loop.after is None:
        return
    if shape is None or shape in known.after:
        _once(loop.after, tensors)
        return
    with _entering() as seen:
        _once(loop.after, tensors)
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
def _pinned(program: _Program) -> Iterator[bool]:
    """Whether the kept program is still there; held against reclaim while it runs."""
    try:
        pin = program.region.pin()
        pin.__enter__()
    except ResidencyError:  # closed, or a reclaim is releasing it now
        yield False
        return
    try:
        yield True
    finally:
        pin.__exit__(None, None, None)


def _layouts(tensors: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in tensors)


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
    scopes = _LOCAL_SCOPES.get()
    if not scopes:
        return (), ""  # no components: tensor work only
    owner, _, components = scopes[-1]
    placement: Any = owner._cozy_residency
    models = tuple(
        placement.models[name]
        for name in components
        if placement is not None and name in placement.models
    )
    if not models:
        return (), "the scope's components are not placed on this device"
    return models, ""


def _unresident(model: Model, paths: tuple[str, ...] = ("",)) -> str:
    """Why these blocks of `model` (and its common weights) are not all on the device."""
    binding = model.retain(paths)
    try:
        return str(binding.unresident())
    finally:
        binding.close()


def _drop_stages(recording: _SequenceCapture) -> None:
    """A refused attempt's stages go; the recording's domain and pool stay for the next."""
    with recording._submission:
        stages, recording.stages = recording.stages, []
        recording._areas.clear()
    for stage in stages:
        with stage._submission:
            stage._sequence_owner = None
        stage.close()
    # A failed capture leaves its pool's bookkeeping behind; the next attempt takes a new one.
    recording._pool = domain.Pool(torch.cuda.graph_pool_handle())


def _submit(
    program: _Program,
    sequence: PreparedSequence,
    indices: range,
    loop: _Loop,
    ctx: Context,
    row: dict[str, Any],
) -> None:
    """One native submission covering `indices`; no fallback once it is accepted."""
    with torch.no_grad():
        program.counter.fill_(indices.start)
    row["at_ms"].setdefault("submitted", _now())
    execution: Any = (
        sequence.submit(ctx) if ctx._submit_prepared is None else ctx._submit_prepared(sequence)
    )
    streamed = program.mode == "streamed"
    on_step, reported = loop.on_step, 0
    while True:
        if ctx.cancelled:
            execution.cancel()
        progress = execution.poll()
        done = (
            int(execution.facts()["sequence"]["completed_passes"])
            if streamed
            else int(progress.completed_steps)
        )
        if on_step is not None:
            for i in indices[reported:done]:
                on_step(i)
        reported = max(reported, done)
        if progress.done:
            break
        time.sleep(0.01)
    execution.wait()
    row["at_ms"]["finished"] = _now()
    ctx.raise_if_cancelled()
    facts = execution.facts()
    done = int(facts["sequence"]["completed_passes"]) if streamed else int(progress.completed_steps)
    if progress.canceled or done != len(indices):
        raise OutputError(
            f"the native loop stopped after {done} of {len(indices)} steps",
            code="incomplete_denoise",
        )
    if on_step is not None:
        for i in indices[reported:]:
            on_step(i)
    row["native_steps"] = row.get("native_steps", 0) + done
    row["host_graph_launches"] = row.get("host_graph_launches", 0) + int(
        facts["host_graph_launches"]
    )
    reads = [int(stage.get("read_bytes", 0)) for stage in facts.get("stage_inputs", ())]
    if reads:
        row["streamed_bytes"] = row.get("streamed_bytes", 0) + sum(reads)
    for pair in facts.get("stage_prefetch_admission", ()):
        admitted = row.setdefault("prefetch", {})
        admitted[str(pair.get("state"))] = admitted.get(str(pair.get("state")), 0) + 1
    if phases := facts.get("native_phases"):
        row["native_phases"] = row.get("native_phases", 0) + len(phases)


# ------------------------------------------------------------------ recording


#: A recorded part: (first stage, end stage, passes, first step index or -1 for the reuse
#: program, steps).
_Part = tuple[int, int, int, int, int]


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
    *,
    remaining: bool = True,
) -> tuple[list[_Part], int]:
    """One WHILE graph per run of equal phases, every component resident and held. With
    `pre`/`post` (the models `before`/`after` run, resident too), `after` is recorded as the
    last graph's suffix, and `again` also makes the reuse program: `before`, every step and
    `after` over the same graphs. `remaining` plans the recording call's own rest too."""
    runs = _runs(rest, loop.phases, loop.steps)
    paths: dict[Model, set[str]] = {}
    for used in (body, pre or {}, post or {}):
        for model, entered in used.items():
            paths.setdefault(model, set()).update(entered)
    held = tuple(paths)
    state = (*mutable, counter)
    bindings: list[Binding] = []
    try:
        with ExitStack() as preparing:
            idle: set[int] = set()
            for model in held:
                _checked(model)
                binding = model.retain(tuple(sorted(paths[model])))
                bindings.append(binding)
                if id(model.memory) not in idle:
                    preparing.enter_context(model.memory.request(graphs.IDLE_REQUEST, idle=True))
                    idle.add(id(model.memory))
                if why := binding.unresident():
                    raise CapabilityError(
                        f"capture requires the component's weights to be resident ({why})",
                        code="prepared_graph_ineligible",
                    )
                preparing.enter_context(binding.prepare())
            _lap("bind")
            owners = tuple(bindings)
            landed = LANDED[0]
            graphs_: list[tuple[_Graph, int, int]] = []
            with _guarded(held, frozenset()):
                for first, count, phase in runs:

                    def advance(phase: Any = phase) -> None:
                        _call(loop.step, counter, tensors, phase)
                        counter.add_(1)

                    graph = _Graph(recording, state, count, owners)
                    graph.record(advance)
                    graphs_.append((graph, first, count))
                    _lap("record_step")
                    _yield_point()
                prefix = suffix = None
                if loop.after is not None and post is not None:
                    suffix = _Graph(recording, state, 1, owners)
                    suffix.record(lambda: loop.after(*tensors))  # type: ignore[misc]
                    _lap("record_after")
                    _yield_point()
                if again and loop.before is not None and pre is not None:
                    prefix = _Graph(recording, state, 1, owners)
                    prefix.record(lambda: loop.before(*tensors))  # type: ignore[misc]
                    _lap("record_before")
            _lap("record")
            _yield_point()
            parts: list[_Part] = []
            for number, (graph, first, count) in enumerate(graphs_ if remaining else ()):
                last = number == len(graphs_) - 1
                plan = graph.plan(landed, suffix=suffix if last else None)
                _own(plan, owners, recording)
                parts.append((len(recording.stages) - 1, len(recording.stages), 1, first, count))
            if again and len(graphs_) == 1:
                graph, first, count = graphs_[0]
                plan = graph.plan(landed, count + 1, prefix=prefix, suffix=suffix)
                _own(plan, owners, recording)
                parts.append((len(recording.stages) - 1, len(recording.stages), 1, -1, count + 1))
            for done in (*(g for g, _, _ in graphs_), prefix, suffix):
                if done is not None:
                    done.done()
            _lap("plan")
    except BaseException:
        for binding in bindings:
            binding.close()
        raise
    for binding in bindings:  # the last stage closes the bindings every stage reads
        recording.stages[-1]._binding_cleanup.callback(binding.close)
    return parts, 0


def _record_streamed(
    recording: _SequenceCapture,
    loop: _Loop,
    rest: range,
    tensors: tuple[Any, ...],
    mutable: tuple[Any, ...],
    counter: Any,
    models: tuple[Model, ...],
) -> tuple[list[_Part], int]:
    """Per run of equal phases, one pass cut into stages at the component's block boundaries;
    its sequence repeats that pass natively."""
    if len(models) != 1:
        raise CapabilityError(
            "streamed native runs one component", code="prepared_graph_ineligible"
        )
    (model,) = models
    if not model.blocks:
        raise CapabilityError(
            f"{model.component} has no blocks to stream", code="prepared_graph_ineligible"
        )
    parts: list[_Part] = []
    blocks = 0
    for first, count, phase in _runs(rest, loop.phases, loop.steps):

        def body(phase: Any = phase) -> None:
            _call(loop.step, counter, tensors, phase)
            counter.add_(1)

        begin = len(recording.stages)
        cutter = _Cutter(recording, model, (*mutable, counter))
        cutter.record(body)
        parts.append((begin, len(recording.stages), count, first, count))
        blocks = max(blocks, cutter.blocks)
    return parts, blocks


def _runs(
    rest: range, phases: tuple[Hashable, ...] | None, steps: range
) -> list[tuple[int, int, Any]]:
    """(first index, count, phase) of each run of equal phases in `rest`."""
    if phases is None:
        return [(rest.start, len(rest), _NONE)]
    runs: list[tuple[int, int, Any]] = []
    for i in rest:
        phase = phases[i - steps.start]
        if runs and runs[-1][2] == phase:
            first, count, _ = runs[-1]
            runs[-1] = (first, count + 1, phase)
        else:
            runs.append((i, 1, phase))
    return runs


def _checked(model: Model) -> None:
    placed = PLACED.get(model.module)
    if placed is not None and placed[0].lora_choices((placed[1],))[placed[1]][0] == "per_step":
        raise CapabilityError(
            "native loops need a fixed baked LoRA representation", code="prepared_graph_ineligible"
        )
    if why := graphs._hooked(model.module):
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
    """One CUDA graph recorded between begin() and end() into the recording's shared pool.

    Every tensor the recorded operations read that they did not produce is retained by the
    native owner, except the loaded inputs of a streamed stage and the scratch of the same pass,
    which the pool and the replay order keep valid.
    """

    def __init__(
        self,
        recording: _SequenceCapture,
        state: tuple[Any, ...],
        steps: int,
        bindings: tuple[Binding, ...],
        *,
        loaded: tuple[Any, ...] = (),
        scratch: list[tuple[Any, int, int]] | None = None,
        unbound: frozenset[int] = frozenset(),
        undeclared: tuple[tuple[Any, int, int], ...] = (),
    ) -> None:
        self.recording, self.state, self.steps, self.bindings = recording, state, steps, bindings
        self.device = state[0].device
        self.domain = recording._domain
        self.ranges = tuple(
            dict.fromkeys(
                (t.device, int(s.data_ptr()), int(s.data_ptr()) + s.nbytes())
                for t in loaded
                if (s := t.untyped_storage()).nbytes()
            )
        )
        self.loaded = bool(loaded)
        self.scratch = [] if scratch is None else scratch
        self.unbound, self.undeclared = unbound, undeclared
        self.held: dict[int, Any] = {id(t): t for t in state}
        self.produced: dict[int, Any] = {}
        self.generated: dict[int, tuple[Any, Any, Any, int, int]] = {}
        self.ops = 0
        self.graph: Any = None
        self.index: Any = None
        self._contexts = ExitStack()
        self._mode: Any = None
        self._caller: Any = None

    def _overlaps(self, tensor: Any, areas: Any) -> bool:
        if not isinstance(tensor, torch.Tensor):
            return False
        storage = tensor.untyped_storage()
        first, size = int(storage.data_ptr()), int(storage.nbytes())
        return bool(size) and any(
            tensor.device == device and first < end and begin < first + size
            for device, begin, end in areas
        )

    def begin(self) -> None:
        """Capture on the recording's library stream; only the recorded region runs there."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("a loop stage begins outside capture")
        stream, control = self.domain.stream, self.domain.control
        self._caller = torch.cuda.current_stream(self.device)
        self.index = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.held[id(self.index)] = self.index
        self.graph = torch.cuda.CUDAGraph(keep_graph=True)
        context = CURRENT.get()
        pause = context._pause_stall_watch if context is not None else None
        try:
            self._contexts.enter_context(pause() if pause is not None else nullcontext())
            self._contexts.enter_context(kernel_ready.recording())
            self._contexts.enter_context(varena.capture(self.device.index))
            stream.wait_stream(self._caller)
            control.wait_stream(stream)
            torch.cuda.set_stream(control)
            try:
                self.graph.capture_begin(
                    pool=self.recording._pool.handle, capture_error_mode="global"
                )
            finally:
                torch.cuda.set_stream(stream)
        except BaseException:
            torch.cuda.set_stream(self._caller)
            self._contexts.close()
            self.graph = None
            raise
        self.domain.fork.record(control)
        stream.wait_event(self.domain.fork)
        self._mode = _Owners(self)
        self._mode.__enter__()

    def end(self) -> None:
        stream, control = self.domain.stream, self.domain.control
        if self.graph is None:
            return
        try:
            if self._mode is not None:
                mode, self._mode = self._mode, None
                mode.__exit__(None, None, None)
        finally:
            try:
                try:
                    self.domain.join.record(stream)
                    control.wait_event(self.domain.join)
                finally:
                    # An invalidated capture refuses the join; only its end takes both streams
                    # out of capture, so the caller's next CUDA call is not inside it (cr-227).
                    torch.cuda.set_stream(control)
                    try:
                        self.graph.capture_end()
                    finally:
                        torch.cuda.set_stream(self._caller)
            finally:
                self._contexts.close()
        control.synchronize()
        self.domain.graphs.add(self.graph)

    def abort(self) -> None:
        """After a failure inside the recorded region: end the capture, keep nothing."""
        with suppress(BaseException):
            self.end()
        self.reset()

    def nodes(self) -> int:
        """The recorded graph's node count: what it launches, whatever launched it."""
        count = ctypes.c_size_t(0)
        if _DRIVER.cuGraphGetNodes(
            ctypes.c_void_p(self.graph.raw_cuda_graph()), None, ctypes.byref(count)
        ):
            raise RuntimeError("cuGraphGetNodes failed")
        return int(count.value)

    def reset(self) -> None:
        if self.graph is not None:
            self.graph.reset()
            self.graph = None
        self.held.clear()

    def record(self, body: Callable[[], object]) -> None:
        self.begin()
        try:
            body()
        except BaseException:
            self.abort()
            raise
        self.end()

    def external(self) -> dict[int, Any]:
        """What the recorded operations read from outside this graph and its pass's scratch."""
        inside = _Spans((*self.scratch, *((d, b, e) for _, _, d, b, e in self.generated.values())))
        loaded = _Spans(self.ranges if self.loaded else ())
        return {
            key: tensor
            for key, tensor in self.held.items()
            if not inside.overlaps(tensor) and not loaded.overlaps(tensor)
        }

    def plan(
        self,
        landed: int,
        steps: int | None = None,
        prefix: _Graph | None = None,
        suffix: _Graph | None = None,
    ) -> PreparedRepeat:
        """A native owner repeating this graph `steps` times (its own count by default), with
        one-shot `prefix` and `suffix` graphs of the same recording around the repeat."""
        steps = self.steps if steps is None else steps
        held: dict[int, Any] = {}
        for graph in (self, prefix, suffix):
            if graph is not None:
                held.update(graph.external())
        _lap("plan_held")
        owners = () if self.loaded else self.bindings
        phases = {
            "prefix_graph": 0 if prefix is None else prefix.graph.raw_cuda_graph(),
            "suffix_graph": 0 if suffix is None else suffix.graph.raw_cuda_graph(),
        }
        extra = [graph.graph for graph in (prefix, suffix) if graph is not None]
        try:
            native = _execution.ResidentRepeat(
                self.graph.raw_cuda_graph(),
                self.index.data_ptr(),
                self.device.index,
                steps,
                [
                    self.graph,
                    *extra,
                    self.domain.stream,
                    self.index,
                    *held.values(),
                    *(tensor.untyped_storage() for tensor in held.values()),
                    *owners,
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
        return plan

    def done(self) -> None:
        """Every native owner is made: drop this graph's Python references to what it read."""
        self.held.clear()


class _Owners(TorchDispatchMode):  # type: ignore[misc]
    """What one recorded graph reads from outside itself, and the scratch it makes."""

    def __init__(self, graph: _Graph) -> None:
        super().__init__()
        self.g = graph

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        g, kwargs = self.g, kwargs or {}
        leaves = tree_leaves((args, kwargs))
        if g.unbound or g.undeclared:
            for tensor in leaves:
                if id(tensor) in g.unbound or g._overlaps(tensor, g.undeclared):
                    raise CapabilityError(
                        "a stage reads weights outside its block", code="prepared_graph_ineligible"
                    )
        if torch.Tag.nondeterministic_seeded in func.tags and not _zero_dropout_attention(
            func, args, kwargs
        ):
            raise CapabilityError(
                "native loops take noise as an input, never a random draw",
                code="prepared_graph_ineligible",
            )
        if g.ranges:
            for position, argument in enumerate(func._schema.arguments):
                if argument.alias_info is None or not argument.alias_info.is_write:
                    continue
                value = args[position] if position < len(args) else kwargs.get(argument.name)
                if any(g._overlaps(tensor, g.ranges) for tensor in tree_leaves(value)):
                    raise CapabilityError(
                        "the step writes loaded weight storage", code="prepared_graph_ineligible"
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


class _Cutter:
    """One pass of the step recorded as stages cut at a component's block boundaries.

    Every stage loads its raw weights at their own addresses: a block's stage its block and the
    common weights, the stages between blocks the common weights. Scratch that crosses a cut
    stays in the shared pool: later stages read it at the address the earlier one wrote, as
    replay runs them in order. A block the cut cannot take cleanly (nested blocks, weights read
    across a cut, a derived layout) refuses the whole pass.
    """

    def __init__(self, recording: _SequenceCapture, model: Model, state: tuple[Any, ...]) -> None:
        self.recording, self.model, self.state = recording, model, state
        self.common = [model.common] if model.common is not None else []
        self.graph: _Graph | None = None
        self.block: Any = None
        self.binding: Binding | None = None
        self.load: Any = None
        self.blocks = 0
        self.failed = False
        self.landed = LANDED[0]
        self.scratch: list[tuple[Any, int, int]] = []
        self.immutable: frozenset[int] = frozenset()
        self._preparing = ExitStack()
        self._declared: Any = None

    def record(self, body: Callable[[], object]) -> None:
        _checked(self.model)
        census = _binding(self.model, [])
        try:
            self.immutable = census.immutable_buffer_ids()
        finally:
            census.close()
        with ExitStack() as preparing:
            # The scope's open pass lends its common block to the stages that load it.
            preparing.enter_context(self.model.lending())
            preparing.enter_context(_guarded((self.model,), self.immutable))
            token = CUTTER.set(self)
            preparing.callback(CUTTER.reset, token)
            self._open(None)
            try:
                body()
            except BaseException:
                self.failed = True
                self._abort()
                raise
            self._close()
        if self.blocks == 0:
            raise CapabilityError(
                "the step entered none of the component's blocks", code="prepared_graph_ineligible"
            )

    # Called by residency's block hooks while CUTTER is set.
    def enter(self, block: Any) -> None:
        if self.failed or block.model is not self.model:
            if block.model is not self.model and not self.failed:
                raise CapabilityError(
                    "the step runs another component's block", code="prepared_graph_ineligible"
                )
            return
        if self.block is not None:
            raise CapabilityError("nested blocks cannot be cut", code="prepared_graph_ineligible")
        self._close()
        self._open(block)

    def exit(self, block: Any) -> None:
        if self.failed or block.model is not self.model:
            return
        if sys.exc_info()[1] is not None:  # the forward failed: unwind, record nothing more
            self.failed = True
            self._abort()
            return
        if block is not self.block:
            raise CapabilityError(
                "a block cut closed out of order", code="prepared_graph_ineligible"
            )
        self._close()
        self._open(None)

    def _open(self, block: Any) -> None:
        """A stage over `block` (None: the common weights between blocks), loaded natively."""
        groups = [*self.common, *(() if block is None else (block,))]
        binding = load = None
        if groups:
            binding = _binding(self.model, groups)
            try:
                load = binding.prepare_load(stage=True)
            except BaseException:
                binding.close()
                raise
            if load is None:
                binding.close()
                raise CapabilityError(
                    f"{block.path if block is not None else 'common weights'} cannot be "
                    "detached as a stage",
                    code="prepared_graph_ineligible",
                )
        self.block, self.binding, self.load = block, binding, load
        self._preparing = ExitStack()
        try:
            if binding is not None:
                self._preparing.enter_context(binding.prepare())
            self.graph = _Graph(
                self.recording,
                self.state,
                1,
                () if binding is None else (binding,),
                loaded=() if binding is None else tuple(binding.immutable_inputs()),
                scratch=self.scratch,
                unbound=self.immutable
                - (frozenset() if binding is None else binding.immutable_buffer_ids(selected=True)),
                undeclared=() if binding is None else binding.undeclared_input_ranges(),
            )
            self._declared = _CAPTURE_BLOCKS.set(frozenset(id(b) for b in groups))
            self.graph.begin()
        except BaseException:
            self._abort()
            raise

    def _close(self) -> None:
        """End the current stage: keep it, or drop it when it recorded nothing."""
        graph, self.graph = self.graph, None
        if graph is None:
            return
        try:
            graph.end()
        finally:
            _CAPTURE_BLOCKS.reset(self._declared)
            self._declared = None
        self.scratch.extend(
            (device, start, end) for _, _, device, start, end in graph.generated.values()
        )
        block, binding, load = self.block, self.binding, self.load
        self.block = self.binding = self.load = None
        if block is None and not graph.nodes():  # nothing ran between two blocks
            graph.reset()
            self._preparing.close()
            if binding is not None:
                binding.close()  # abandons its load slot; resident inputs keep their cache
            return
        try:
            if load is not None:
                load.seal()
            self._preparing.close()  # parameter slots back before any capture-only view drops
            plan = graph.plan(self.landed)
            graph.done()
            if binding is None:
                _own(plan, (), self.recording)
                return
            try:
                source = binding.detach_stage()
            except BaseException as error:
                plan.close()
                if isinstance(error, Exception) and not isinstance(error, ResidencyError):
                    _native_error(error)
                raise
        except BaseException:
            graph.reset()
            if binding is not None:
                binding.close()
            raise
        events = bool(getattr(varena.LoadSlot, "supports_event_input", False)) and bool(
            getattr(_execution.ResidentRepeat, "supports_event_input", False)
        )
        plan._load = source
        plan._load_ordering_requested = "events"
        plan._load_ordering = "events" if events else "checked"
        cleanup = ExitStack()
        cleanup.callback(binding.close)
        _own(plan, (binding,), self.recording, cleanup)
        self.blocks += block is not None

    def _abort(self) -> None:
        graph, self.graph = self.graph, None
        if graph is not None:
            graph.abort()
        if self._declared is not None:
            _CAPTURE_BLOCKS.reset(self._declared)
            self._declared = None
        self._preparing.close()
        if self.binding is not None:
            self.binding.close()
        self.block = self.binding = self.load = None


def _binding(model: Model, blocks: list[Any]) -> Binding:
    """A binding over exactly `blocks` (residency's retain always adds the common block)."""
    model._claim()
    try:
        model.memory._declare(model)
        return Binding(model, blocks)
    finally:
        model._release()


def install() -> None:
    _loop.install(Loops())
