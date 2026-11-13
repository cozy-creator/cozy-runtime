"""CUDA graphs of whole components over retained virtual addresses.

Kernel choice is shipped. Runtime prepares memory, captures process-local addresses and
replays; it neither times competing paths nor compiles kernels. A component's region is its
root call; nothing inside a root is a region, so a streamed component runs as written. A root
has one graph per signature: the path its call takes, keyed by its inputs' layout, the values
of the attributes its package declares steer its forward (`Model.__graph_keys__`), its modules'
classes and hooks, the LoRA overlays its hooks arm, the plain context variables a package set,
and the Runtime's choices (the kernels serving, `LANDED`; the attention layout's band). A
graph also holds the module attributes its capture read (`_reading`): it replays only while
each still has the value it had, so one path may keep a few graphs, one per such state. A
signature's first warm call runs as written and is measured (`_first`), its next with the device
held back while the host submits it (`_probe`), which measures its launch overhead; a later one
records in the call when that overhead over the calls ahead beats the capture's cost
(`_capture_call`): the graph replays once on that call's inputs, the call runs as
written and serves, and the graph is kept when the two match bit for bit (outputs, the inputs
it writes and the state it writes outside the call) and the replay is faster; every later call
replays. What the capture reads outside its inputs, pool and weights (`observe`) must still be
there at each replay, or the graph goes and the call records again. Recording changes no state
of the model. An idle period records the signatures seen once, under their call's context
(`capture_idle`), as the backup.
Every graph has its own static buffers in its region's capture pool (`domain`), opened only for
a capture that may keep it. A cached binding owns virtual addresses only; its physical lease
covers preparation through output copies, and a completion event orders reuse across streams.
"""

from __future__ import annotations

import contextvars
import functools
import math
import sys
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, cast

import tensorfs.varena.torch as vt
from tensorfs import varena

from cozy_runtime.accel import lora, stepcache
from cozy_runtime.accel.attention import (
    AttentionLayout,
    active_attention_layout,
    attention_scope,
    producer_sites,
)
from cozy_runtime.author import _request_path as request_path
from cozy_runtime.author._context import CURRENT
from cozy_runtime.author._context import device as placed_device
from cozy_runtime.author._errors import (
    CAPTURE_ERRORS,
    Fault,
    OutputError,
    fault,
    is_capacity,
    unwind,
)
from cozy_runtime.author._model import LANDED, PLACED, WARNINGS, Building, Yielded, recovered
from cozy_runtime.graphs import domain, observe
from cozy_runtime.internal import attention_ulysses, kernel_ready
from cozy_runtime.internal.accel import StaleBinding

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Protocol

    from torch import Generator, Tensor
    from torch import device as TorchDevice
    from torch import dtype as TorchDtype
    from torch.cuda import CUDAGraph, Event, Stream
    from torch.nn import Module
    from torch.utils._pytree import TreeSpec

    from cozy_runtime.author._model import Residency, WarningRow
    from cozy_runtime.internal.attention_ulysses import Seen
    from cozy_runtime.internal.residency import Binding, Memory, Placeholder

    class _Call(Protocol):
        """A root's own `__call__`, unbound: the module, then its arguments as written."""

        def __call__(self, module: Module, /, *args: object, **kwargs: object) -> object: ...

    class _Op(Protocol):
        """A torch function a mode intercepts: any operands."""

        def __call__(self, *args: object, **kwargs: object) -> object: ...

    #: a placed component's residency and its name (`PLACED`)
    type _Placed = tuple[Residency, str]
    #: a root call's arguments, as `tree_flatten((args, kwargs))` reads them
    type _Arguments = tuple[tuple[object, ...], dict[str, object]]
    #: a counter snapshot's eager or replayed timing (`_Timing.counts`)
    type _Counts = tuple[int, int, float, float, float]

BYPASS: contextvars.ContextVar[bool] = contextvars.ContextVar("cozy_graphs_bypass", default=False)
#: Set while a region warms up its capture: the regions under it run as written.
_WARMING = contextvars.ContextVar("cozy_graphs_warming", default=False)
#: The idle period recording on this thread (`capture_idle`): each region boundary checks it.
_IDLE: contextvars.ContextVar[_Idle | None] = contextvars.ContextVar("cozy_graphs_idle")
#: an idle job's cleanup (its model's state put back): ops there never yield
_SETTLING: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "cozy_graphs_settling", default=False
)
_PLAIN = (type(None), bool, int, float, str)
#: graph owners beside the regions (the LM engine's decode steps), each its (snapshot, release)
#: of a component root, registered by its module when a package imports it: a Runtime serving no
#: LM never imports Transformers for them (v7: 2.8 s of executor imports)
DECODERS: list[
    tuple[
        Callable[[Module], tuple[dict[str, object], list[domain.Pool], domain.Domain | None]],
        Callable[[Module], None],
    ]
] = []
_SAID: set[str] = set()
#: the isolated request used for idle warm-up and capture observations
IDLE_REQUEST = "graphs-idle"
#: what a run's record counts (`execution.safeguards`): each a recovered failure (`recovered`), said
#: once quietly, never a warning
SAFEGUARDS = {
    "capture_refused": "a CUDA graph was refused at capture",
    "replay_mismatch": "a CUDA graph differed from eager and was discarded",
    "scratch_declined": "a CUDA graph's pool stays mapped while idle",
    "no_room": "a kept CUDA graph had no room and ran eagerly",
}


def say(key: str, line: str) -> None:
    if key not in _SAID:
        _SAID.add(key)
        print(line, file=sys.stderr)


def safeguard(code: str, why: str) -> None:
    """One for this attempt's record, or the next attempt's when raised at idle."""
    WARNINGS.append(recovered(SAFEGUARDS[code], code=code, why=why.splitlines()[0][:300]))


def no_room(device: int, what: str, need: int, error: BaseException) -> str:
    """Why there was no room (varena refused it, or an allocation failed), for the record."""
    room = vt.room(device)

    def gib(n: int) -> str:
        return f"{n / (1 << 30):.1f} GiB"

    need = need or getattr(error, "requested_bytes", 0)
    return (
        f"{what} {f'needs {gib(need)}' if need else 'found no room'} beside "
        f"{gib(room['working_live'])} live and {gib(room['pinned'])} of weights in use, of "
        f"{gib(room['capacity'])} ({error})"
    )


def safeguards(warnings: list[WarningRow]) -> dict[str, list[str]]:
    """An attempt's reasons, by `SAFEGUARDS` code."""
    found: dict[str, list[str]] = {}
    for _, fields in warnings:
        if fields.get("code") in SAFEGUARDS:
            found.setdefault(cast("str", fields["code"]), []).append(str(fields["why"]))
    return found


class Finite:
    """Latents checked in passing: a step's sums go to pinned host memory as they are made, behind
    an event, and the next step reads them there, so the host is at most a step ahead and waits on
    no step it just sent, only on the copy of the one before. A non-finite step fails the call a
    step later at worst; `last` reads the last step's. A sum holds no tensor of latents' size and
    carries any NaN or infinity to its end. A native loop's reads are unordered with its steps,
    which only write finite values over finite ones. A tensor not finite on entry is not latents
    yet (SDXL's `image`, which `after` fills): its entry sum is read with the first step's."""

    #: the last step's queued sums
    sums: tuple[Tensor, Event | None] | None

    def __init__(self, torch: ModuleType, tensors: tuple[Tensor, ...] = ()) -> None:
        self.torch = torch
        self.tensors = tuple(t for t in tensors if t.is_floating_point())
        self.entry, self.sums = self._queue(self.tensors), None
        self.kept: tuple[bool, ...] | None = None

    def _queue(self, tensors: tuple[Tensor, ...]) -> tuple[Tensor, Event | None] | None:
        """Their sums, on the host once the event says so (None off CUDA: read at once)."""
        if not tensors:
            return None
        torch = self.torch
        sums = torch.stack([tensor.sum(dtype=torch.float32) for tensor in tensors])
        if not sums.is_cuda:
            return sums, None
        host = torch.empty(sums.shape, dtype=torch.float32, pin_memory=True)
        host.copy_(sums, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        return host, event

    @staticmethod
    def _read(queued: tuple[Tensor, Event | None] | None) -> list[float]:
        if queued is None:
            return []
        values, event = queued
        if event is not None:
            event.synchronize()
        return cast("list[float]", values.tolist())  # cast: a 1-D float32 tensor's

    def _sound(self, queued: tuple[Tensor, Event | None] | None, latents: bool = True) -> bool:
        found = self._read(queued)
        if latents:
            if self.kept is None:
                self.kept = tuple(math.isfinite(value) for value in self._read(self.entry))
            found = [value for value, kept in zip(found, self.kept, strict=False) if kept]
        return all(math.isfinite(value) for value in found)

    def step(self, tensors: tuple[Tensor, ...] | None = None) -> bool:
        """Queues the sums of what this step wrote; whether the step before's were finite."""
        queued, self.sums = self.sums, self._queue(self.tensors if tensors is None else tensors)
        return self._sound(queued, latents=tensors is None)

    def last(self) -> bool:
        """After the last step's completion."""
        return self._sound(self.sums) and self._sound(self._queue(self.tensors))


def _same_shaped(
    torch: ModuleType, args: tuple[object, ...], kwargs: dict[str, object], result: object
) -> tuple[Tensor, ...]:
    """A root's float inputs and outputs of one shape: a denoiser's latents and what it returns of
    them. Both: a quantized forward turns NaN into integers, so NaN latents can return finite."""

    def floats(value: object) -> list[Tensor]:
        leaves = torch.utils._pytree.tree_leaves(value)
        return [v for v in leaves if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dim()]

    inputs, outputs = floats((args, kwargs)), floats(result)
    shared = {tuple(t.shape) for t in inputs} & {tuple(t.shape) for t in outputs}
    return tuple(tensor for tensor in (*inputs, *outputs) if tuple(tensor.shape) in shared)


@contextmanager
def _measured(device: int) -> Iterator[list[int]]:
    """The working peak varena charged what runs inside, once it ran: none where another
    observation owns the device."""
    peak: list[int] = []
    try:
        token = vt.phase_enter(device, None, mode="preparation")
    except vt.PhaseOverlapError:
        yield peak
        return
    ran = False
    try:
        yield peak
        ran = True
    finally:
        record = vt.phase_leave(device, token, completed=ran)
    if ran and record.get("peak_bytes") is not None:
        peak.append(int(record["peak_bytes"]))


def close_pool(pool: domain.Pool, device: int) -> None:
    """A pool none of its owner's graphs use any more, told varena (`vt.pool_close`): its pages
    go at the next reclaim."""
    vt.pool_close(pool.handle, device)


@request_path.forbid("cuda graph capture")
def capture(
    torch: ModuleType,
    step: Callable[[], object],
    pool: domain.Pool,
    owner: domain.Domain,
    reserve: int,
    *,
    alone: bool,
) -> tuple[CUDAGraph | None, str]:
    """Record a pure computation into Varena's capture pool, `reserve` bytes of it prepared first:
    refused there (`varena.Exhausted`), nothing is captured.

    `reserve` is a guess from an eager pass's peak, which sizes no layout in one run: the capture
    packs the same blocks there, and a small block left in the hole two freed large ones made has
    the next large one find no run (cr-285: a 16 MiB `repeat_kv` at a 2,048-token window). A
    capture that runs short in its pool records again with room for the block it lacked too
    (`vt.pool_short`), until it fits or that room is refused: it never ends short while the card
    has room. `alone`: the pool holds none of its owner's graphs, so a new one takes the short
    one's place. `pool.prepared` is what the last attempt was prepared with. An allocation the
    capture finds no room for outside its pool fails it, as a device fault does.

    Bindings and outputs remain caller-owned; `step` runs again after a short attempt.
    """
    from cozy_runtime.internal.execution.catch import CURRENT as REQUEST

    mapped = -1  # the pool's bytes before this attempt
    while True:
        pool.prepared = reserve
        try:
            total = vt.prepare_pool(pool.handle, reserve, owner.device)
        except BaseException:
            vt.cancel_preparation(pool.handle, owner.device)
            raise
        refused = vt.ooms(owner.device)
        try:
            return _capture(torch, step, pool, owner, reserve)
        except Exception as exc:
            short = 0
            if is_capacity(exc) and not isinstance(exc, varena.Exhausted):
                short = vt.pool_short(pool.handle, owner.device)
            if not short:
                raise
            unwind(exc)  # its tensors leave the pool before it is prepared again
        if (request := REQUEST.get()) is not None:  # answered here, for the contract's count
            request.regrown += vt.ooms(owner.device) - refused
        if alone:
            close_pool(pool, owner.device)
            pool.handle = torch.cuda.graph_pool_handle()
        elif total == mapped:  # no new run, so the same room: past every run the pool has
            reserve = total
        reserve, mapped = reserve + short, total


def _capture(
    torch: ModuleType,
    step: Callable[[], object],
    pool: domain.Pool,
    owner: domain.Domain,
    reserve: int,
) -> tuple[CUDAGraph | None, str]:
    """One capture into `pool`, prepared with `reserve`."""
    graph = domain.cuda_graph()
    current: Stream = torch.cuda.current_stream()
    # cast: an initialized domain's streams and events (`Domain.initialize`), as each capture's
    stream, work = cast("Stream", owner.control), cast("Stream", owner.stream)
    stream.wait_stream(current)
    stream.wait_stream(work)
    failure: BaseException | None = None
    try:
        with (
            kernel_ready.recording(),
            torch.utils._python_dispatch._disable_current_modes(),
            vt.graph(graph, pool=pool.handle, stream=stream, reserve_bytes=reserve, primed=(work,)),
        ):
            cast("Event", owner.fork).record(stream)
            work.wait_event(cast("Event", owner.fork))
            with torch.cuda.stream(work):
                try:
                    step()
                finally:
                    cast("Event", owner.join).record(work)
                    stream.wait_event(cast("Event", owner.join))
    except Exception as exc:
        failure = exc
    current.wait_stream(stream)
    current.wait_stream(work)
    if failure is not None:
        vt.cancel_preparation(pool.handle, owner.device)
        try:
            graph.reset()
        except Exception as cleanup:
            if fault(failure) in (Fault.STICKY, Fault.DEVICE):
                failure.add_note(f"discarding its failed capture also failed: {cleanup}")
                raise failure from None
            error = torch.AcceleratorError(
                f"failed capture could not be discarded safely: {cleanup}"
            )
            error.error_code = 999
            raise error from failure
    if isinstance(failure, Building | Yielded) or (failure is not None and is_capacity(failure)):
        raise failure
    if failure is not None:
        if fault(failure) in (Fault.STICKY, Fault.DEVICE):
            if getattr(failure, "error_code", None) not in CAPTURE_ERRORS:
                raise failure
        return None, f"{type(failure).__name__}: {failure}".splitlines()[0][:300]
    owner.graphs.add(graph)
    return graph, ""


class _Recording:
    """One signature's graph and its static buffers. A replay copies the caller's inputs into
    its static inputs and its static outputs into caller-owned tensors (`output`), so nothing
    of the graph's own memory leaves it."""

    graph: CUDAGraph | None
    pool: domain.Pool | None
    inputs: list[Tensor | None]
    outputs: object

    def __init__(
        self,
        torch: ModuleType,
        graph: CUDAGraph,
        pool: domain.Pool,
        inputs: list[Tensor | None],
        outputs: object,
        layout: object,
        served: Seen,
        owner: domain.Domain,
        observed: observe.Observed,
    ) -> None:
        self.torch, self.graph, self.pool, self.owner = torch, graph, pool, owner
        self.inputs, self.outputs, self.layout = inputs, outputs, layout
        self.served = [(b, i) for b, impls in served.items() for i in impls]
        self.refused = dict(served.refused)
        self.bands = [(site, band, b) for (site, band), held in served.bands.items() for b in held]
        #: the storages outside the call it reads (`observe`): each must live for it to replay
        self.foreign = observed.foreign
        #: the static inputs it writes, copied back to the caller's tensors after each replay
        self.arg_writes = sorted(observed.arg_writes)
        #: the foreign views it writes, held only until its check ends (`_capture_call`)
        self.writes = list(observed.writes.values())
        #: its region's attempt count at its last replay: the least recent goes first (`_admit`)
        self.used = 0
        #: the module attributes its capture read (`_reading`): (module, name, value, plain)
        self.reads: tuple[tuple[Module, str, object, bool], ...] = ()
        #: the region generation its read guards last held at
        self.guarded = -1
        #: its replays in a row that found weights gone and refilled them (`_REFILLS`)
        self.refilled = 0
        #: bytes the last replay copied in and out
        self.copied = 0
        self.done: Event | None = None
        self.stream: Stream | None = None
        self.fenced = True
        # Nothing reads a pool between replays, so the core may take its pages back when idle.
        held = [t for t in inputs if t is not None]
        held += [t for t in torch.utils._pytree.tree_leaves(outputs) if isinstance(t, torch.Tensor)]
        try:
            vt.pool_scratch(pool.handle, held, owner.device)
        except (RuntimeError, ValueError) as exc:  # An escaped alias: its pool just stays mapped.
            safeguard("scratch_declined", str(exc))

    def output(self) -> object:
        """Caller-owned outputs, allocated before residency admission or graph launch."""
        torch = self.torch
        return torch.utils._pytree.tree_map(
            lambda v: torch.empty_like(v) if isinstance(v, torch.Tensor) else v, self.outputs
        )

    def live(self) -> bool:
        """Whether every storage outside the call it reads is still the one it captured."""
        return not any(ref.expired() for ref in self.foreign)

    def __call__(
        self, leaves: Sequence[object], out: object, *, written: list[Tensor] | None = None
    ) -> object:
        """Replays, copying its outputs into `out` and what it wrote into its static inputs
        back into the caller's (the call as written wrote those); with `written`, into that
        list instead (its check, which runs the call as written on the caller's next). Nothing
        of the pool is read once it returns: its pages may go while it is idle."""
        stream: Stream
        torch, stream = self.torch, self.torch.cuda.current_stream()
        if self.done is None:
            self.done = torch.cuda.Event()
            self.done.record(stream)  # instantiate before any replay can be enqueued
        else:
            stream.wait_event(self.done)
        if self.stream is not None and self.stream != stream:
            stream.wait_stream(self.stream)
        self.stream, self.fenced = stream, False
        copied = 0
        try:
            # cast: a recording is called only until it closes
            vt.prepare_pool(cast("domain.Pool", self.pool).handle, 0, self.owner.device)
            for static, value in zip(self.inputs, leaves, strict=True):
                # Always: a pool's contents need not survive between replays.
                if static is not None:
                    static.copy_(cast("Tensor", value))  # cast: a tensor leaf has a static input
                    copied += static.nbytes
            with self.owner.recovery_blocked(replay=True):
                vt.replay(cast("CUDAGraph", self.graph), stream=stream)
            for backend, implementation in self.served:
                attention_ulysses.record(backend, implementation)
            for (why, took), calls in self.refused.items():
                attention_ulysses.refused(why, calls, took)
            for site, band, backend in self.bands:
                attention_ulysses.banded(site, band, backend)
            source = torch.utils._pytree.tree_leaves(self.outputs)
            for dst, src in zip(torch.utils._pytree.tree_leaves(out), source, strict=True):
                if isinstance(src, torch.Tensor):
                    dst.copy_(src)
                    copied += src.nbytes
            for index in self.arg_writes:
                static = cast("Tensor", self.inputs[index])  # cast: a written leaf is a tensor
                if written is None:
                    cast("Tensor", leaves[index]).copy_(static)
                else:
                    written.append(static.clone())
                copied += static.nbytes
            return out
        finally:
            self.copied = copied
            cast("Event", self.done).record(stream)  # cast: made above
            vt.record_pool_stream(cast("domain.Pool", self.pool).handle, stream)
            self.fenced = True

    def close(self) -> None:
        """After last use: release static tensors and executable before their pool."""
        self.inputs, self.outputs, self.writes, self.foreign, self.reads = [], None, [], [], ()
        cast("CUDAGraph", self.graph).reset()  # cast: it closes once
        self.graph = self.pool = None

    def idle(self) -> bool:
        return self.fenced and (self.done is None or self.done.query())

    def finish(self) -> None:
        """Wait for its last replay and output copies."""
        if not self.fenced and self.stream is not None:
            self.stream.synchronize()
        if self.done is not None:
            self.done.synchronize()


class _Timing:
    """One way a region runs (eager, replayed, or as written for a reason): its calls, the host
    time each took, of which its pass submitted for `submitted_ms` (before its synchronize,
    `passed_at`), and its device time between two CUDA events around it, read once the device
    has passed them (never by waiting; a later call or the reply settles them)."""

    def __init__(self) -> None:
        self.calls = self.timed = 0
        self.host_ms = self.device_ms = self.submitted_ms = 0.0
        #: the last call's observed pass: (submit ms, working peak bytes), None when none
        self.passed: tuple[float, int] | None = None
        #: the first call's host and device times apart (a cold run's first eager call:
        #: autotune, library initialization, page-in)
        self.first_ms: float | None = None
        self.first_device_ms = 0.0
        self.pending: deque[tuple[Event, Event]] = deque()
        self.free: list[tuple[Event, Event]] = []

    def run[T](self, torch: ModuleType, call: Callable[[], T]) -> T:
        stream = torch.cuda.current_stream()
        start, end = (
            self.free.pop()
            if self.free
            else (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        )
        start.record(stream)
        _passed()
        began = time.perf_counter()
        try:
            return call()
        finally:
            took = (time.perf_counter() - began) * 1e3
            self.passed = passed = _passed()
            self.submitted_ms += took if passed is None else passed[0]
            self.host_ms += took
            self.first_ms = took if self.first_ms is None else self.first_ms
            end.record(stream)
            self.calls += 1
            self.pending.append((start, end))
            self.settle()

    def settle(self) -> None:
        while self.pending and self.pending[0][1].query():
            start, end = self.pending.popleft()
            took = start.elapsed_time(end)
            self.first_device_ms = took if not self.timed else self.first_device_ms
            self.device_ms += took
            self.timed += 1
            self.free.append((start, end))

    def submit_ms(self) -> float | None:
        """Host submission per call beyond the first (a cold first call autotunes and pages
        in): None until a second call."""
        if self.calls < 2:
            return None
        return (self.host_ms - (self.first_ms or 0.0)) / (self.calls - 1)

    def wall_ms(self) -> float | None:
        """Device time per call between its events, beyond the first, the same way."""
        self.settle()
        if self.timed < 2:
            return None
        return (self.device_ms - self.first_device_ms) / (self.timed - 1)

    def counts(self) -> _Counts:
        self.settle()
        return self.calls, self.timed, self.host_ms, self.device_ms, self.submitted_ms

    def document(self, base: _Counts = (0, 0, 0.0, 0.0, 0.0)) -> dict[str, int | float]:
        calls, timed, host, device, submit = (
            now - then for now, then in zip(self.counts(), base, strict=True)
        )
        return {
            "calls": calls,
            "first_ms": round(self.first_ms or 0.0, 3),
            "host_ms": round(host, 3),
            "submit_ms": round(submit, 3),
            "device_ms": round(device, 3),
            "device_timed": timed,
        }


#: a recorder's counters a receipt reports per attempt
_COUNTERS = (
    "attempts",
    "replays",
    "expired",
    "capture_attempts",
    "captures",
    "retired",
    "copy_bytes",
    "capture_ms",
    "call_ms",
    "prepare_ms",
)


class _Recorder:
    """A component root's region: its graphs, one per signature, and how its calls ran."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        #: the attribute paths the package declares steer its forward (`Model.__graph_keys__`)
        self.keys = keys
        self.lock = threading.Lock()
        #: its graphs by handle (`key`, serial), and each key's handles (`_find`)
        self.recordings: dict[tuple[object, ...], _Recording] = {}
        self.paths: dict[tuple[object, ...], list[tuple[object, ...]]] = {}
        self.serial = 0
        #: each signature waiting for an idle period to record (`capture_idle`)
        self.queued: dict[tuple[object, ...], _Pending] = {}
        #: why a signature runs as written until its graphs are dropped (a refusal)
        self.failed: dict[tuple[object, ...], str] = {}
        #: why one runs as written for the process: measured, it would save nothing
        #: (device-bound, or a replay no faster than its eager call)
        self.judged: dict[tuple[object, ...], str] = {}
        #: per signature, its idle recordings in a row that found no room (`_short`)
        self.short: dict[tuple[object, ...], int] = {}
        self.waiting: dict[tuple[object, ...], int] = {}
        #: per signature its idle measure: host submission, the warm-up's host time and its
        #: waits on the device, and (recorded) replay ms a call
        self.bound: dict[tuple[object, ...], tuple[float, float, float, float | None]] = {}
        #: per signature, its last call as written (`_first`), which decides a capture in a call
        self.seen: dict[tuple[object, ...], _Seen] = {}
        #: per signature, its launch-overhead probe (`_probe`), and its calls this attempt
        self.probes: dict[tuple[object, ...], _Probe] = {}
        self.sights: dict[tuple[object, ...], int] = {}
        #: per signature, why its idle recording failed: a call may still record it
        self.idle_failed: dict[tuple[object, ...], str] = {}
        #: its modules' classes and hooks (`_structure`), read at each attempt's first call and
        #: whenever a hook was registered since; and why a package hook keeps it as written
        self.structure: tuple[int, tuple[int, int], str] | None = None
        #: its LoRA overlay hooks (`_loras`), found with its structure
        self.loras = _Loras((), ())
        #: bumped by an attribute set on any of its modules (`_watching`), which a graph's
        #: read guards are checked again for; `structural`, by a class or submodule set
        self.generation = 0
        self.structural = 0
        #: the last module attribute whose value kept a graph from replaying
        self.moved = ""
        self.binding: Binding | None = None
        #: its graphs' capture pool (`domain.Pool`), made at its first capture
        self.pool: domain.Pool | None = None
        self.replays = 0
        #: graphs dropped because storage they read outside the call was gone
        self.expired = 0
        self.attempts = 0
        self.capture_attempts = 0
        self.captures = 0
        self.retired = 0
        self.copy_bytes = 0
        #: why its last call ran as written beside its graphs (a refusal), "" when none
        self.gap = ""
        self.landed = LANDED[0]
        self.domain: domain.Domain | None = None
        #: host time preparing and capturing at idle (warm-up and instantiation included),
        #: each capture's apart, and each way it ran
        self.capture_ms = 0.0
        self.capture_each: list[float] = []
        #: host time of every call through the region (its checks, preparation, replay or eager
        #: run) and of its replays' preparation: what a call costs beside the work it runs
        self.call_ms = self.prepare_ms = 0.0
        #: its previous call's key, and what changed in its key between calls (a key that changes
        #: every call records again every call)
        self.last: tuple[object, ...] | None = None
        self.changes: dict[str, int] = {}
        self.eager: _Timing = _Timing()
        self.replayed: _Timing = _Timing()
        #: calls a guard ran as written, by reason (`_region_call`'s `why`)
        self.written: dict[str, _Timing] = {}
        #: how each call ran (`OUTCOMES`) and the last reason for each; the current call's
        self.outcomes: dict[str, int] = {}
        self.reasons: dict[str, str] = {}
        self.now: tuple[str, str] = ("first_occurrence", "")
        #: why a signature runs as written for good, by `OUTCOMES` code (`failed`, `judged`)
        self.codes: dict[tuple[object, ...], str] = {}
        #: the counters when the current attempt opened (`mark`): receipts are per request
        self.base: dict[str, object] = {}

    def snapshot(self) -> dict[str, object]:
        return {
            **{name: getattr(self, name) for name in _COUNTERS},
            "changes": dict(self.changes),
            "eager": self.eager.counts(),
            "replayed": self.replayed.counts(),
            "written": {why: timing.counts() for why, timing in self.written.items()},
            "outcomes": dict(self.outcomes),
        }

    def since(self, name: str) -> float:
        """Counter `name` since the attempt opened."""
        return cast("float", getattr(self, name) - self.base.get(name, 0))  # cast: a counter's

    def forget(self) -> bool:
        if any(not entry.idle() for entry in self.recordings.values()):
            return False
        self.drop()
        return True

    def finish(self) -> None:
        """Wait for every replay's GPU work and output copies."""
        for entry in self.recordings.values():
            entry.finish()

    def drop(self) -> None:
        self.retired += len(self.recordings)
        for entry in self.recordings.values():
            entry.close()
        self.recordings.clear()
        self.paths.clear()
        if self.binding is not None:
            self.binding.close()
            self.binding = None
        self.pool = None
        self.queued.clear()
        self.failed.clear()
        self.idle_failed.clear()
        self.short.clear()
        self.waiting.clear()
        self.seen.clear()
        self.probes.clear()
        self.codes.clear()
        self.landed = LANDED[0]


#: each installed component root's region
_RECORDERS: weakref.WeakKeyDictionary[Module, _Recorder] = weakref.WeakKeyDictionary()
#: each component root's plain attributes at its first call, until its second compares them
_FIRST: weakref.WeakKeyDictionary[Module, dict[str, object] | None] = weakref.WeakKeyDictionary()
#: per component, the undeclared plain attributes of its root that changed between its first two
#: calls: each would key its graphs or steer its forward unseen (`task conform` fails on it)
UNDECLARED: dict[str, set[str]] = {}
_CLASSES: dict[type[Module], type[Module]] = {}


#: an idle job's ops still queued on the device as its host reaches its next one (`_Idle.meter`):
#: a command that arrives waits for at most these there
_QUEUED = 2
#: the host's poll while idle work waits on the device to finish (`_settled`)
_POLL_S = 0.0002


class _Idle:
    """Work for the device's idle periods, in the order it queued (`capture_idle`): a job answers
    whether it is done; one that is not (its region measured once, its pages not resident, a
    kernel building) waits for the next period. `check`, before each torch op of a job (`_yielding`)
    and at each region boundary, yields to a command that arrived: the job ends there, a capture
    discarded, first again next period. The longest stretch between checks is the longest a
    command waited for the host (v17: 386 ms when it was the whole UNet's capture); `meter`
    keeps the device as close (a DiT forward queued ahead of the host made a 4090's Anima
    request wait 0.4 s for it after the yield)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: deque[Callable[[], bool]] = deque()
        self.stop: Callable[[], bool] = lambda: False
        self.last = 0.0
        self.periods = self.yields = self.failures = 0
        self.longest_ms = 0.0
        #: the running job's ops still queued on the device: each one's event and host time
        self.queued: deque[tuple[Event, float]] = deque()
        self.spare: list[Event] = []
        #: host ms the jobs' ops waited for the device (`meter`)
        self.metered_ms = 0.0

    def meter(self) -> None:
        """Before a job's next op on the device: wait, yielding, while `_QUEUED` of its ops are
        queued there. Its waits are the device's time, which says whether a call is
        device-bound (`_record`), so it only yields the CPU between polls: a 0.2 ms sleep per wait
        counted its oversleep as the device's (SDXL's UNet: 96 to 163 ms of waits a call, against
        a 78 ms replay)."""
        self._settle()
        if len(self.queued) < _QUEUED:
            return
        began = time.perf_counter()
        while len(self.queued) >= _QUEUED:
            self.check()
            time.sleep(0)
            self._settle()
        self.metered_ms += (time.perf_counter() - began) * 1e3

    def mark(self, torch: ModuleType) -> None:
        """A job's op was queued on the current stream."""
        event = self.spare.pop() if self.spare else torch.cuda.Event()
        event.record()
        self.queued.append((event, time.perf_counter()))

    def _settle(self) -> None:
        while self.queued and self.queued[0][0].query():
            self.spare.append(self.queued.popleft()[0])

    def add(self, job: Callable[[], bool]) -> None:
        with self.lock:
            self.jobs.append(job)

    def check(self) -> None:
        now = time.perf_counter()
        self.longest_ms = max(self.longest_ms, (now - self.last) * 1e3)
        self.last = now
        if self.stop():
            raise Yielded

    def run(self, stop: Callable[[], bool]) -> bool:
        """Each queued job once, while `stop()` stays false; False when it turned true (the job
        it interrupted goes first next period)."""
        self.stop, self.last, later = stop, time.perf_counter(), []
        self.periods += 1
        torch = sys.modules.get("torch")  # a job's ops yield only where torch runs them
        token = _IDLE.set(self)
        try:
            while True:
                with self.lock:
                    if not self.jobs:
                        return True
                    job = self.jobs.popleft()
                try:
                    self.check()
                    # A job that yields only at its own boundaries runs without the per-op
                    # check: every torch call under a Python mode pays its dispatch.
                    ops = torch is not None and getattr(job, "yields_per_op", True)
                    with _yielding(torch)() if ops else nullcontext():
                        done = job()
                    if not done:
                        later.append(job)
                except Yielded:
                    self.yields += 1
                    with self.lock:
                        self.jobs.appendleft(job)
                    return False
                except Exception as exc:  # one job's failure never ends the period (v16)
                    if fault(exc) in (Fault.STICKY, Fault.DEVICE):
                        raise  # the device itself failed: the executor ends (`_idle`)
                    self.failures += 1
                    say(
                        f"idle:{type(exc).__name__}:{exc}",
                        f"[graphs] an idle job failed and goes: {type(exc).__name__}: {exc}"[:400],
                    )
        finally:
            _IDLE.reset(token)
            self.queued.clear()
            self.longest_ms = max(self.longest_ms, (time.perf_counter() - self.last) * 1e3)
            with self.lock:
                self.jobs.extend(later)

    def document(self) -> dict[str, int | float]:
        return {
            "queued": len(self.jobs),
            "periods": self.periods,
            "yields": self.yields,
            "failures": self.failures,
            "longest_block_ms": round(self.longest_ms, 3),
        }


IDLE = _Idle()


@functools.cache
def _yielding(torch: ModuleType) -> Callable[[], AbstractContextManager[object]]:
    """The torch function mode an idle job runs under: before each op, a command that arrived
    ends the job (`_Idle.check`), so a whole component's capture yields within one op, and the
    device is at most `_QUEUED` ops behind (`_Idle.meter`; nothing runs while a stream
    captures). Its cleanup (`_SETTLING`) never yields."""

    class Yielding(torch.overrides.TorchFunctionMode):  # type: ignore[misc, name-defined]  # torch is passed in: a module
        def __torch_function__(
            self,
            func: _Op,
            types: tuple[type, ...],
            args: tuple[object, ...] = (),
            kwargs: dict[str, object] | None = None,
        ) -> object:
            if _SETTLING.get() or (idle := _IDLE.get(None)) is None:
                return func(*args, **(kwargs or {}))
            idle.check()
            metered = torch.cuda.is_initialized() and not torch.cuda.is_current_stream_capturing()
            if metered:
                idle.meter()
            out = func(*args, **(kwargs or {}))
            if metered and isinstance(out, torch.Tensor) and out.is_cuda:
                idle.mark(torch)
            return out

    return Yielding


def capture_idle(stop: Callable[[], bool]) -> bool:
    """Record the queued signatures now: call only while no request runs on the device. A
    region boundary where `stop()` is true (a command arrived) ends it at once, its capture
    discarded; False then."""
    return IDLE.run(stop)


class _Leaf(NamedTuple):
    """A tensor argument's layout, for the inputs an idle capture records on."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: TorchDtype
    device: TorchDevice


@dataclass(eq=False)
class _Pending:
    """A signature whose first call ran as written, queued to record at idle."""

    torch: ModuleType
    module: weakref.ref[Module]
    call: _Call
    key: tuple[object, ...]
    spec: TreeSpec
    #: each argument leaf: a tensor's layout, or the plain value
    leaves: list[object]
    #: the package's plain context variables its call ran under (`_context`)
    context: tuple[tuple[contextvars.ContextVar[object], object], ...]

    def __call__(self) -> bool:
        tokens = [(var, var.set(value)) for var, value in self.context]
        try:
            return _capture_pending(self)
        finally:
            for var, token in reversed(tokens):
                var.reset(token)


def install(torch: ModuleType, model: object, roots: Mapping[str, Module]) -> list[str]:
    """Each component root is a region, keyed by the attributes the model declares steer it
    (`__graph_keys__`: component -> attribute paths). Nothing inside a root is a region: a
    streamed component runs as written."""
    _watching(torch)
    skipped = set(getattr(type(model), "__uncaptured__", ()))
    declared: Mapping[str, tuple[str, ...]] = getattr(type(model), "__graph_keys__", {})
    installed = []
    for name, root in roots.items():
        if name in skipped or not isinstance(root, torch.nn.Module):
            continue
        if root not in _RECORDERS:
            _RECORDERS[root] = _Recorder(tuple(declared.get(name, ())))
        elif not _RECORDERS[root].recordings and _RECORDERS[root].pool is None:
            _RECORDERS[root].keys = tuple(declared.get(name, ()))
        if not getattr(type(root), "_cozy_graphed", False):
            root.__class__ = _graphed(torch, type(root))
        installed.append(name)
    return installed


def _graphed(torch: ModuleType, cls: type[Module]) -> type[Module]:
    made = _CLASSES.get(cls)
    if made is None:
        call = cls.__call__

        def __call__(self: Module, *args: object, **kwargs: object) -> object:
            return _dispatch(torch, self, call, args, kwargs)

        made = _CLASSES[cls] = cast("type[Module]", type(
            cls.__name__,
            (cls,),
            {
                "__call__": __call__,
                "_cozy_graphed": True,
                "_cozy_original_call": call,
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
            },
        ))
    return made


def retire(torch: ModuleType, root: Module) -> None:
    """Drain `root`'s graphs and drop them, keeping its region: its calls run as written until
    an idle period records them again (an idle offload, a changed fill)."""
    recorder = _RECORDERS.get(root)
    if recorder is None or (not recorder.recordings and recorder.pool is None):
        return
    scope = recorder.domain.using() if recorder.domain is not None else nullcontext()
    with scope, recorder.lock:
        recorder.finish()
        recorder.drop()


def retire_stale(torch: ModuleType, *roots: Module) -> None:
    """Before a scope binds `roots`' weights: graphs recorded against another fill (a LoRA mode
    or bake recipe this attempt changed) go first, so nothing replays over a refill."""
    for root in roots:
        recorder = _RECORDERS.get(root)
        if recorder is not None and recorder.binding is not None and not recorder.binding.valid():
            retire(torch, root)


def release(torch: ModuleType, root: Module) -> None:
    """A generation's explicit teardown: finish its output copies before releasing addresses."""
    retire(torch, root)
    _RECORDERS.pop(root, None)
    domain.release(root)


def mark(root: Module) -> None:
    """A new attempt of `root`'s component opens: its receipts count from here."""
    sweep(root)
    recorder = _RECORDERS.get(root)
    if recorder is not None:
        with recorder.lock:
            recorder.base = recorder.snapshot()
            recorder.structure = None  # a request may swap classes or hooks before its loop
            recorder.sights.clear()


def capture_domain(torch: ModuleType, root: Module, memory: Memory | None = None) -> domain.Domain:
    """The component owns hidden library scratch independently of its disposable graphs."""
    return domain.acquire(torch, root, memory)


def capture_device(torch: ModuleType, memory: Memory) -> domain.Domain:
    """A placed component's graphs share the device's domain with its other components'."""
    return domain.device(torch, memory)


#: a timing's counts before any call
_ZERO: _Counts = (0, 0, 0.0, 0.0, 0.0)


def describe(root: Module) -> dict[str, object]:
    """One final-reply snapshot of `root`'s region, never an aggregate scan on the replay hot
    path. Counts are this attempt's, since the Placement opened it (`mark`); an unplaced
    component's are its generation's. Pool bytes count each live pool once (a device's shared
    pool under the first component it served); driver/context and pending native frees remain
    ledger facts."""
    pools: dict[tuple[int, int], domain.Pool] = {}
    domains: dict[int, domain.Domain] = {}
    answer: dict[str, object] = dict.fromkeys(
        ("active_graphs", "region_calls", "replays", "capture_attempts", "captures"), 0
    )
    answer.update(retired_graphs=0, replay_copy_bytes=0, capture_ms=0.0)
    recorder = _RECORDERS.get(root)
    if recorder is not None:
        with recorder.lock:
            since, base = recorder.since, recorder.base
            before = cast("dict[str, int]", base.get("changes", {}))  # cast: `snapshot`'s
            written = cast("dict[str, _Counts]", base.get("written", {}))
            answer.update(
                active_graphs=len(recorder.recordings),
                region_calls=since("attempts"),
                replays=since("replays"),
                capture_attempts=since("capture_attempts"),
                captures=since("captures"),
                retired_graphs=since("retired"),
                expired_graphs=since("expired"),
                replay_copy_bytes=since("copy_bytes"),
                capture_ms=round(since("capture_ms"), 3),
                call_ms=round(since("call_ms"), 3),
                prepare_ms=round(since("prepare_ms"), 3),
                key_changes={
                    what: n - before.get(what, 0)
                    for what, n in recorder.changes.items()
                    if n > before.get(what, 0)
                },
                graph=_receipt(recorder),
                eager=recorder.eager.document(cast("_Counts", base.get("eager", _ZERO))),
                replay=recorder.replayed.document(cast("_Counts", base.get("replayed", _ZERO))),
                as_written={
                    why: timing.document(written.get(why, _ZERO))
                    for why, timing in recorder.written.items()
                    if timing.calls > written.get(why, _ZERO)[0]
                },
                eager_gap=recorder.gap,
                calls=_since(recorder.outcomes, cast("dict[str, int]", base.get("outcomes", {}))),
                why={
                    code: why
                    for code, why in recorder.reasons.items()
                    if why and code in recorder.outcomes
                },
            )
            # A device's shared pool and domain count once.
            if recorder.pool is not None and recorder.recordings:
                pools[recorder.pool.handle] = recorder.pool
            if recorder.domain is not None:
                domains[id(recorder.domain)] = recorder.domain
    shared = domain.existing(root)
    if shared is not None:
        domains[id(shared)] = shared
    for snapshot, _ in DECODERS:
        decode, decode_pools, decode_domain = snapshot(root)
        pools.update((pool.handle, pool) for pool in decode_pools)
        if decode_domain is not None:
            domains[id(decode_domain)] = decode_domain
        if decode:
            answer["decode"] = decode
            for key in (
                "active_graphs",
                "capture_attempts",
                "captures",
                "retired_graphs",
                "replays",
            ):
                answer[key] = cast("int", answer[key]) + cast("int", decode[key])  # cast: counts
    answer.update(
        idle=IDLE.document(),
        graph_pools=len(pools),
        graph_pool_bytes=None,  # stock Torch does not expose a public per-pool byte counter
        domains=len(domains),
    )
    return answer


def _receipt(recorder: _Recorder) -> dict[str, object]:
    """A region's graph economics so far: each idle capture's cost (warm-up, capture and
    instantiation), what each signature measured, a replay's saving over an eager
    call (device time between their events, beyond each way's first call) and the calls that
    repay the captures. A device-bound signature never records."""
    eager, replay = recorder.eager.wall_ms(), recorder.replayed.wall_ms()
    saving = eager - replay if eager is not None and replay is not None else None
    cost = sum(recorder.capture_each)
    return {
        "capture_each_ms": list(recorder.capture_each),
        "measured": [
            {
                "submit_ms": round(submit, 3),
                "warm_up_ms": round(warm_up, 3),
                "waited_ms": round(waited, 3),
                "replay_ms": None if replayed is None else round(replayed, 3),
            }
            for submit, warm_up, waited, replayed in recorder.bound.values()
        ],
        "saving_ms": None if saving is None else round(saving, 3),
        "payback_calls": math.ceil(cost / saving) if saving and saving > 0 else None,
        "eager": sorted(
            {
                *recorder.failed.values(),
                *recorder.judged.values(),
                *(f"at idle: {why}" for why in recorder.idle_failed.values()),
            }
        ),
    }


def _dispatch(
    torch: ModuleType, module: Module, call: _Call, args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    context = CURRENT.get()
    if (
        context is not None
        and not (torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing())
        and not (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)
    ):
        context.raise_if_cancelled()
    if (idle := _IDLE.get(None)) is not None:
        idle.check()
    recorder = _RECORDERS.get(module)
    if recorder is not None:
        _watch(module, recorder)
    capturing = torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()
    if recorder is None or _WARMING.get() or capturing:
        return call(module, *args, **kwargs)  # a copy of a root, or inside a recording
    bypass = (
        "graphs are off for this call"
        if BYPASS.get()
        else "under a step cache"  # its verdicts are host reads in the forward
        if stepcache._RUN.get() is not None
        else "off CUDA"
        if placed_device().type != "cuda"
        else ""
    )
    if bypass:
        recorder.outcomes["guard"] = recorder.outcomes.get("guard", 0) + 1
        recorder.reasons["guard"] = bypass
        return call(module, *args, **kwargs)
    began = time.perf_counter()
    try:
        result = _region_call(torch, module, recorder, call, args, kwargs)
    finally:
        recorder.call_ms += (time.perf_counter() - began) * 1e3
    if context is not None:
        # A loop of the package's own: each root call is a step, and any next one reads it.
        if context._finite is None:
            context._finite = Finite(torch)
        if not context._finite.step(_same_shaped(torch, args, kwargs, result)):
            raise OutputError(
                f"the latents went non-finite before this call of {_name(module)}",
                code="non_finite_latents",
            )
    return result


def _watch(root: Module, recorder: _Recorder) -> None:
    """At a component root's second call: the undeclared plain attributes that changed since
    its first (`UNDECLARED`), said once. A step counter set on the root each call keys every
    call apart once declared, or steers a graph unseen when not: the package decides."""
    if root in _FIRST and _FIRST[root] is None:
        return
    now = {name: value for name, value in vars(root).items() if type(value) in _PLAIN}
    first = _FIRST.get(root)
    _FIRST[root] = now if first is None else None
    if first is None:
        return
    keys = {path.split(".")[0] for path in recorder.keys}
    changed = {n for n in first.keys() | now.keys() if first.get(n) != now.get(n)} - keys
    if changed:
        name = _name(root)
        UNDECLARED.setdefault(name, set()).update(changed)
        say(
            f"undeclared:{id(root)}",
            f"[graphs] component {name}'s root attributes {sorted(changed)} changed between its "
            "first two calls: declare each that steers its forward in Model.__graph_keys__",
        )
        WARNINGS.append(
            recovered(
                "a component's undeclared state changed between calls",
                component=name,
                attributes=", ".join(sorted(changed)),
            )
        )


def _name(root: Module) -> str:
    placed = PLACED.get(root)
    return placed[1] if placed is not None else type(root).__name__


#: how a region call ran, as the record counts it (`describe`'s `calls`): replayed, recorded
#: (this call captured a graph that serves the next; the call as written served it), or as
#: written for a reason
OUTCOMES = {
    "replayed": "its graph replayed",
    "recorded": "it captured a graph, checked against the call as written, which served",
    "first_occurrence": "its path's first call: measured for its next",
    "measuring": "its last call's timing was still on the device",
    "device_bound": "the device, not the host, sets its pace: a graph saves nothing",
    "no_gain": "a checked replay was no faster",
    "no_room": "no room for its graph's pool or weights",
    "refused": "the code can't be recorded (a capture refusal, state it changes)",
    "capture_refused": "its capture failed",
    "replay_mismatch": "its replay differed from the call as written; the graph went",
    "building": "a kernel is still building, or its weights moved: a later call records",
    "retired": "its graphs went (a kernel landed, its weights changed)",
    "waiting": "its graph waits for room after a refusal this attempt",
    "guard": "a guard keeps it as written",
}


def _note(recorder: _Recorder, code: str, why: str = "") -> None:
    """How the current call runs (`OUTCOMES`), and why."""
    recorder.now = (code, why.splitlines()[0][:300] if why else "")


def _since(now: Mapping[str, int], base: Mapping[str, int]) -> dict[str, int]:
    return {code: n - base.get(code, 0) for code, n in now.items() if n > base.get(code, 0)}


def _region_call(
    torch: ModuleType,
    module: Module,
    recorder: _Recorder,
    call: _Call,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """A region call, counted by how it ran (`OUTCOMES`)."""
    recorder.now = ("first_occurrence", "")
    try:
        return _region_path(torch, module, recorder, call, args, kwargs)
    finally:
        code, why = recorder.now
        recorder.outcomes[code] = recorder.outcomes.get(code, 0) + 1
        if why:
            recorder.reasons[code] = why


def _region_path(
    torch: ModuleType,
    module: Module,
    recorder: _Recorder,
    call: _Call,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """The call replayed when its signature has a graph; recorded in the call at its signature's
    second sight when its first was launch-bound (`_capture_call`); otherwise as written, with
    nothing around it but its key and two timing events."""
    placed = PLACED.get(module)
    leaves: list[object]
    spec: TreeSpec
    leaves, spec = torch.utils._pytree.tree_flatten((args, kwargs))
    layout = active_attention_layout()
    structure, hooked = _structure(recorder, module)
    why = (
        "with gradients"
        if torch.is_grad_enabled()
        else "under custom tensor dispatch"
        if torch.utils._python_dispatch._get_current_dispatch_mode() is not None
        else "across GPUs"
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        # A dev-only attention capture sees every call with its own inputs and step.
        else "under an attention capture"
        if attention_ulysses.CAPTURE.get() is not None
        # A sparse step picks its key blocks from the values, and a package's producer may
        # keep request state at a call: a replay would freeze one call's.
        else "under a sparse attention layout"
        if layout is not None and layout.sparse
        else "under an attention layout its producer keeps state for"
        if layout is not None and _stateful(module, layout)
        # A package hook runs Python at every call, which a replay would skip.
        else hooked
        or _unplain(torch, leaves)
    )
    state: tuple[object, ...] = ()
    if not why:  # a guarded call reads no overlays
        state, why = _state(torch, recorder, module)
    if why:
        recorder.gap = why
        _note(recorder, "guard", why)
        if placed is not None:
            placed[0].note(f"{placed[1]}.graphs", f"eager {why}")
        timing = recorder.written.get(why) or recorder.written.setdefault(why, _Timing())
        return timing.run(torch, lambda: call(module, *args, **kwargs))
    addresses = (
        ()
        if placed is not None
        else tuple(
            (value.data_ptr(), tuple(value.shape), value.dtype, value.stride())
            for value in (*module.parameters(), *module.buffers())
        )
    )
    autocast = torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda")
    declared = tuple(functools.reduce(getattr, p.split("."), module) for p in recorder.keys)
    key = (
        *_signature(torch, leaves, spec),
        _Parts(declared, addresses, autocast, _dense(layout), structure, state, _context()),
    )
    if layout is not None:  # an idle recording runs outside the caller's scope: under its own
        call = functools.partial(_under, layout, call)
    if recorder.last is not None and recorder.last != key:
        _changed(torch, module, recorder, key, (args, kwargs), len(leaves))
    recorder.last = key
    recorder.attempts += 1
    if key in recorder.paths:
        result = _replay_call(torch, module, recorder, key, leaves, placed)
        if result is _EAGER:
            return recorder.eager.run(torch, lambda: call(module, *args, **kwargs))
        if result is not _STALE:
            return result
        # No graph of this path holds now (what it read outside the call is gone, or a module
        # attribute it read has another value): this call records one.
    recorder.sights[key] = recorder.sights.get(key, 0) + 1
    if (seen := recorder.seen.get(key)) is not None and key not in recorder.failed:
        if key not in recorder.judged:
            result = _capture_call(
                torch, module, recorder, call, key, spec, leaves, (args, kwargs), placed, seen
            )
            if result is _PROBE:
                return _probe(torch, recorder, key, seen, lambda: call(module, *args, **kwargs))
            if result is not _EAGER:
                return result
    elif not any(key in held for held in _settled_by(recorder)):
        _queue(torch, module, recorder, call, key, spec, leaves, placed)
    return _first(torch, recorder, key, lambda: call(module, *args, **kwargs))


def _settled_by(recorder: _Recorder) -> tuple[Mapping[tuple[object, ...], object], ...]:
    """Where a signature already has its fate: queued, refused, judged, refused at idle."""
    return recorder.queued, recorder.failed, recorder.judged, recorder.idle_failed


class _Parts(NamedTuple):
    """A signature's parts beside its arguments, by name (`_changed` says which moved)."""

    declared: tuple[object, ...]
    addresses: tuple[object, ...]
    autocast: tuple[bool, object]
    attention: tuple[object, ...] | None
    structure: int
    #: the LoRA overlays its hooks arm now (`_overlays`)
    state: tuple[object, ...]
    context: tuple[tuple[str, object], ...]


@dataclass(eq=False, slots=True)
class _Seen:
    """A signature's last call as written: its host submission (before its pass's synchronize,
    `execution.catch.passing`), all of its host time, its device time between two events, and
    its working peak (`passed_at`), which sizes its capture pool."""

    submit_ms: float
    host_ms: float
    start: Event
    end: Event
    peak: int | None
    #: its region's first call: autotuning, library setup and page-in make it no measure
    cold: bool = False

    def device_ms(self) -> float | None:
        """Read once the device passed its end; None before (never a wait)."""
        return float(self.start.elapsed_time(self.end)) if self.end.query() else None


#: this thread's last observed component pass (`execution.catch.passing`): its host submission
#: (ms, before the pass's synchronize) and working peak (bytes)
_PASSED = threading.local()


def passed_at(submit_ms: float, peak: int) -> None:
    """A component pass on this thread ended: what `_first` measures its call by."""
    _PASSED.last = (submit_ms, peak)


def _passed() -> tuple[float, int] | None:
    last: tuple[float, int] | None = getattr(_PASSED, "last", None)
    _PASSED.last = None
    return last


def _first[T](
    torch: ModuleType, recorder: _Recorder, key: tuple[object, ...], run: Callable[[], T]
) -> T:
    """The call as written, measured for its signature's next sight (`_Seen`). A signature
    refused for good says its reason (`codes`)."""
    if recorder.now[0] == "first_occurrence" and (code := recorder.codes.get(key)) is not None:
        _note(recorder, code, recorder.failed.get(key) or recorder.judged.get(key, ""))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    began = time.perf_counter()
    result = recorder.eager.run(torch, run)
    host = (time.perf_counter() - began) * 1e3
    end.record()
    passed = recorder.eager.passed
    submit, peak = passed if passed is not None else (host, None)
    last = recorder.seen.get(key)
    if peak is None and last is not None:
        peak = last.peak
    elif peak is not None and last is not None and last.peak is not None:
        peak = max(peak, last.peak)
    recorder.seen[key] = _Seen(submit, host, start, end, peak, cold=recorder.eager.calls == 1)
    return result


#: the longest call a probe measures (`_probe`): past it, launch overhead (microseconds a kernel)
#: can't be a share worth capturing in the call
_PROBE_MS = 1000.0
#: the least share of a call its launch overhead must be for a capture in the call
_SAVING = 0.05
#: `_capture_call`'s answer when the call should run as its probe
_PROBE = object()


class _Probe:
    """A path's call run with the device held back while the host submitted it (`_probe`): its
    device time between its events is the work alone, as a graph would run it."""

    __slots__ = ("end", "start")

    def __init__(self, start: Event, end: Event) -> None:
        self.start, self.end = start, end

    def busy_ms(self) -> float | None:
        return float(self.start.elapsed_time(self.end)) if self.end.query() else None


def _probe[T](
    torch: ModuleType,
    recorder: _Recorder,
    key: tuple[object, ...],
    seen: _Seen,
    run: Callable[[], T],
) -> T:
    """The call as written, measured for its launch overhead: a spin kernel holds the device
    for longer than the host takes to submit the call (`seen`'s submission), so every kernel is
    queued before the device reaches it and runs back to back. Its device time is the work
    alone; the path's ordinary device time less it is what a replay saves. It costs this call
    the hold, once per path."""
    stream = torch.cuda.current_stream()
    props = torch.cuda.get_device_properties(stream.device)
    torch.cuda._sleep(int(seen.submit_ms * 1.25 * props.clock_rate))  # clock_rate: kHz
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    result = recorder.eager.run(torch, run)
    end.record(stream)
    recorder.probes[key] = _Probe(start, end)
    _note(recorder, "measuring", "its launch overhead (probe)")
    return result


def _ahead(recorder: _Recorder, key: tuple[object, ...]) -> int:
    """The calls of `key` likely still to come in this attempt: what a loop's progress says is
    left, else as many as it ran already (a loop that ran k times runs about k more)."""
    context = CURRENT.get()
    step = getattr(context, "_step", None) if context is not None else None
    if isinstance(step, tuple) and len(step) == 2 and step[1] > step[0]:
        return int(step[1] - step[0])
    return recorder.sights.get(key, 1)


def _capture_cost(recorder: _Recorder, submit_ms: float, busy_ms: float) -> float:
    """What capturing and checking a call costs this one: its region's last such cost, else a
    capture's submission, a replay and the call as written (SDXL 4090/5090: 0.6-0.7 s)."""
    if recorder.capture_each:
        return recorder.capture_each[-1]
    return 3 * submit_ms + 2 * busy_ms + 250


def _capture_call(
    torch: ModuleType,
    module: Module,
    recorder: _Recorder,
    call: _Call,
    key: tuple[object, ...],
    spec: TreeSpec,
    leaves: list[object],
    arguments: _Arguments,
    placed: _Placed | None,
    seen: _Seen,
) -> object:
    """Record `key`'s signature in this call, replay it once on this call's inputs, then run the
    call as written, which serves, and keep the graph when the two match bit for bit and the
    replay costs less than the call did; `_EAGER` when it doesn't record now. A capture that
    finds no room or a kernel still building runs as written and tries again at a later call;
    one the code refuses, or a mismatch, runs as written for the process, and the run says why.
    Never an error a call as written would not raise."""
    device = seen.device_ms()
    if device is None:
        _note(recorder, "measuring")
        return _EAGER  # its last call's events are still queued: the next call decides
    if seen.cold:  # a region's first call autotunes and pages in: measure it once more
        _note(recorder, "measuring", "its region's first call")
        return _EAGER
    if device > _PROBE_MS:
        # Launch overhead is a few microseconds a kernel: a call this long can't owe it a
        # share worth a capture in the call. Idle capture, off the critical path, decides it.
        recorder.gap = f"device-bound: a {device:.0f} ms call"
        _note(recorder, "device_bound", recorder.gap)
        return _EAGER
    probe = recorder.probes.get(key)
    if probe is None:
        return _PROBE
    busy = probe.busy_ms()
    if busy is None:
        _note(recorder, "measuring", "its probe is still on the device")
        return _EAGER
    saving, ahead = max(0.0, device - busy), _ahead(recorder, key)
    cost = _capture_cost(recorder, seen.submit_ms, busy)
    if saving < _SAVING * device or saving * ahead <= cost:
        recorder.gap = (
            f"launch overhead {saving:.1f} of {device:.1f} ms a call, {ahead} calls ahead, "
            f"against ~{cost:.0f} ms to capture and check"
        )
        _note(recorder, "device_bound", recorder.gap)
        return _EAGER
    if seen.peak is None:  # its pass was observed inside another's: a later call may be
        recorder.gap = "its calls' working memory is not measured"
        _note(recorder, "building", recorder.gap)
        return _EAGER
    if len(recorder.paths.get(key, ())) >= _VARIANTS:
        why = f"a module attribute it reads changes between calls: {recorder.moved or '?'}"
        with recorder.lock:
            recorder.judged[key] = recorder.gap = why
            recorder.codes[key] = "refused"
        _note(recorder, "refused", why)
        return _EAGER
    owner = _owner(torch, module, recorder, placed)
    recorder.capture_attempts += 1
    began = time.perf_counter()
    entry = _capture_verified(torch, module, recorder, call, key, spec, leaves, placed, seen, owner)
    if entry is None:
        recorder.capture_ms += (time.perf_counter() - began) * 1e3
        return _EAGER
    recording, out, replayed = entry
    # As written, last: what serves the call is eager's, whatever the check finds.
    args, kwargs = arguments
    result = recorder.eager.run(torch, lambda: call(module, *args, **kwargs))
    kept = False
    try:
        written = [cast("Tensor", leaves[i]) for i in recording.arg_writes]
        differs = vt.replay_differs(result, out)
        if not differs and (state := vt.replay_differs(recording.writes, replayed.wrote)):
            differs = f"the state it writes outside the call: {state}"
        if not differs and (inputs := vt.replay_differs(written, replayed.inputs)):
            differs = f"the inputs it writes: {inputs}"
        if differs:
            if not written:  # a second run would write its caller's inputs twice
                served = [t.clone() for t in recording.writes]
                for tensor, saved in zip(recording.writes, replayed.saved, strict=True):
                    tensor.copy_(saved)  # as the call found it
                if vt.replay_differs(result, call(module, *args, **kwargs)):
                    differs += "; the call as written differs run to run too"
                for tensor, saved in zip(recording.writes, served, strict=True):
                    tensor.copy_(saved)  # as the call that serves left it
            with recorder.lock:
                _failed(module, recorder, key, differs, "replay_mismatch")
            return result
        replayed.synchronize()
        replay = max(replayed.host_ms, replayed.device_ms())
        cost = max(seen.host_ms, device)
        with recorder.lock:
            recorder.bound[key] = (seen.submit_ms, seen.host_ms, device, replay)
            if cost - replay <= max(_NOISE * cost, 0.05):
                recorder.judged[key] = recorder.gap = (
                    f"no gain: replay {replay:.3f} ms a call against {cost:.3f} ms as written"
                )
                recorder.codes[key] = "no_gain"
                _note(recorder, "no_gain", recorder.gap)
                return result
            if recorder.landed != LANDED[0] or _find(recorder, key) is not None:
                return result  # a kernel landed meanwhile: its graphs go
            _admit(recorder, key, recording)
            kept = True
            recorder.queued.pop(key, None)
            recorder.short.pop(key, None)
            recorder.captures += 1
            recorder.gap = ""
            recorder.capture_each.append(round((time.perf_counter() - began) * 1e3, 3))
            _note(recorder, "recorded", f"capture and check {recorder.capture_each[-1]:.0f} ms")
        return result
    finally:
        recorder.capture_ms += (time.perf_counter() - began) * 1e3
        recording.writes, replayed.saved, replayed.wrote, replayed.inputs = [], [], [], []
        if not kept:
            with recorder.lock:
                recording.finish()
                recording.close()
                _drop_unused(recorder, owner)


class _Replayed:
    """A verifying replay's host time and the events around it on the device."""

    def __init__(self, torch: ModuleType, stream: Stream) -> None:
        self.stream = stream
        self.first = torch.cuda.Event(enable_timing=True)
        self.last = torch.cuda.Event(enable_timing=True)
        self.host_ms = 0.0
        #: the foreign state it writes before and after the replay, and its written inputs
        self.saved: list[Tensor] = []
        self.wrote: list[Tensor] = []
        self.inputs: list[Tensor] = []

    def synchronize(self) -> None:
        self.last.synchronize()

    def device_ms(self) -> float:
        return float(self.first.elapsed_time(self.last))


def _capture_verified(
    torch: ModuleType,
    module: Module,
    recorder: _Recorder,
    call: _Call,
    key: tuple[object, ...],
    spec: TreeSpec,
    leaves: list[object],
    placed: _Placed | None,
    seen: _Seen,
    owner: domain.Domain,
) -> tuple[_Recording, object, _Replayed] | None:
    """`key`'s graph recorded in its region's pool on this call's argument layout, and replayed
    once on its inputs: the graph, its outputs and the replay's timing, or None (its reason
    recorded, or a later call tries again)."""
    current: Stream = torch.cuda.current_stream()
    scope = placed[0].graph_pass(placed[1], "") if placed is not None else nullcontext()
    entry: _Recording | None = None
    found: tuple[_Recording, object, _Replayed] | None = None
    try:
        with (
            scope,
            owner.using(),
            recorder.lock,
            request_path.known("cozy_runtime.graphs._record"),
        ):
            if recorder.landed != LANDED[0] or (
                recorder.binding is not None and not recorder.binding.valid()
            ):
                recorder.forget()
                return None
            if placed is not None and not placed[0].graph_accounted():
                _failed(module, recorder, key, "driver/context memory is unmeasured")
                return None
            owner.initialize()
            owner.prime_blas()
            if placed is not None and recorder.binding is None:
                recorder.binding = placed[0].retain(placed[1], ("",))
            weights = (
                recorder.binding.immutable_buffer_ids()
                if recorder.binding is not None
                else frozenset()
            )
            layout = [
                _Leaf(tuple(v.shape), v.stride(), v.dtype, v.device)
                if isinstance(v, torch.Tensor)
                else v
                for v in leaves
            ]
            with ExitStack() as held:
                if recorder.binding is not None:
                    held.enter_context(recorder.binding.prepare(current))
                state = held.enter_context(
                    _capture_state(torch, module, cast("Stream", owner.stream), weights)
                )
                served = held.enter_context(attention_ulysses.observing())
                entry = _build(
                    torch, module, recorder, key, owner, placed, call, spec, layout,
                    vt.pool_bytes(cast("int", seen.peak)), state, served,
                )
            if entry is None:
                return None
            out = entry.output()
            replayed = _Replayed(torch, current)
            # The state it writes outside the call, as the call found it: put back after the
            # check's replay, so the call as written starts where the replay did.
            replayed.saved = [t.clone() for t in entry.writes]
            with _lease(torch, placed, recorder.binding):
                replayed.first.record(current)
                began = time.perf_counter()
                entry(leaves, out, written=replayed.inputs)
                replayed.host_ms = (time.perf_counter() - began) * 1e3
                replayed.last.record(current)
            replayed.wrote = [t.clone() for t in entry.writes]
            for tensor, saved in zip(entry.writes, replayed.saved, strict=True):
                tensor.copy_(saved)
            found = (entry, out, replayed)
            return found
    except Exception as exc:
        if isinstance(exc, Building | StaleBinding):
            _note(recorder, "building", str(exc))
            return None  # a kernel still building, or its weights left: a later call
        if is_capacity(exc):
            need = recorder.pool.prepared if recorder.pool is not None else 0
            with recorder.lock:
                _short(module, recorder, key, no_room(owner.device, "its capture", need, exc))
            return None
        if fault(exc) in (Fault.STICKY, Fault.DEVICE):
            raise  # the device itself failed: no call as written runs either
        # Anything else is the recording's, never the call's: it runs as written, and the run
        # says why.
        with recorder.lock:
            _failed(module, recorder, key, f"{type(exc).__name__}: {exc}".splitlines()[0][:300])
        return None
    finally:
        if found is None:
            with recorder.lock:
                if entry is not None:
                    entry.close()
                _drop_unused(recorder, owner)


def _drop_unused(recorder: _Recorder, owner: domain.Domain) -> None:
    """A region with no graph keeps no pool and no Regions (held: its lock)."""
    if recorder.recordings:
        return
    if recorder.pool is not None:
        close_pool(recorder.pool, owner.device)
    recorder.pool = None
    if recorder.binding is not None:
        recorder.binding.close()
        recorder.binding = None


#: a replay is kept when it saves more than this share of the call as written
_NOISE = 0.02


def _owner(
    torch: ModuleType, module: Module, recorder: _Recorder, placed: _Placed | None
) -> domain.Domain:
    """The region's graph domain: its placement's device domain, else its own."""
    with recorder.lock:
        if recorder.domain is None:
            recorder.domain = (
                placed[0].graph_domain(placed[1])
                if placed is not None
                else capture_domain(torch, module)
            )
        return recorder.domain


def _structure(recorder: _Recorder, module: Module) -> tuple[int, str]:
    """Its modules' classes and forward hooks as one number, and why a package hook keeps it as
    written ("" when none). Read at an attempt's first call and whenever a hook was registered
    since (`RemovableHandle.next_id`): HiDiffusion swaps SDXL's block classes and adds hooks
    per request, and a graph of the base UNet must never serve it."""
    from torch.utils.hooks import RemovableHandle

    registered = RemovableHandle.next_id, recorder.structural
    held = recorder.structure
    if held is not None and held[1] == registered:
        return held[0], held[2]
    watch = weakref.ref(recorder)
    for sub in module.modules():
        sub.__dict__[_WATCH] = watch  # its sets move the region's generation (`_watching`)
    number = hash(
        tuple(
            (type(sub), id(sub), tuple(sub._forward_pre_hooks), tuple(sub._forward_hooks))
            for sub in module.modules()
        )
    )
    why = _hooked(module)
    recorder.structure = (number, registered, why)
    recorder.loras = _loras(module)
    return number, why


#: a module's own bookkeeping, which steers no forward (hooks are the structure's)
_INTERNAL = frozenset(
    {
        "_parameters", "_buffers", "_non_persistent_buffers_set", "_modules",
        "_backward_pre_hooks", "_backward_hooks", "_is_full_backward_hook",
        "_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called",
        "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs", "_state_dict_hooks",
        "_state_dict_pre_hooks", "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks",
    }
)
#: a watched module's region (a weak reference), in its `__dict__`
_WATCH = "_cozy_graph_region"


@functools.cache
def _watching(torch: ModuleType) -> None:
    """Every attribute set on a module of a region moves the region's generation: its graphs'
    read guards are checked again (`_holds`). A class or submodule set moves its structure too
    (`_structure`). Tensors are the observer's (`observe`), and the Runtime's own `_cozy`
    attributes steer no path."""
    module = torch.nn.Module
    original = module.__setattr__

    def __setattr__(self: Module, name: str, value: object) -> None:
        original(self, name, value)
        watch = self.__dict__.get(_WATCH)
        if watch is None or name.startswith("_cozy") or isinstance(value, torch.Tensor):
            return
        recorder = watch()
        if recorder is not None:
            recorder.generation += 1
            if name == "__class__" or isinstance(value, module):
                recorder.structural += 1

    module.__setattr__ = __setattr__  # one patch per process


#: each thread whose capture records the module attributes it reads (`_reading`): its reads
_READER: dict[int, dict[tuple[int, str], tuple[Module, str, object, bool]]] = {}
_READING = threading.Lock()
_MISSING = object()


@contextmanager
def _reading(
    torch: ModuleType,
) -> Iterator[dict[tuple[int, str], tuple[Module, str, object, bool]]]:
    """The module attributes this thread's code reads inside, by (module, name): plain values by
    value, other objects (a turbo head armed, a processor) by identity. Submodules are the
    structure's and tensors the observer's. Only a capture runs under it: while one is open,
    every module attribute read in the process goes through a Python function."""
    reads: dict[tuple[int, str], tuple[Module, str, object, bool]] = {}
    me = threading.get_ident()
    with _READING:
        _READER[me] = reads
        if len(_READER) == 1:
            torch.nn.Module.__getattribute__ = _reader(torch)  # removed below
    try:
        yield reads
    finally:
        with _READING:
            del _READER[me]
            if not _READER:
                del torch.nn.Module.__getattribute__


@functools.cache
def _reader(torch: ModuleType) -> Callable[[Module, str], object]:
    module, tensor = torch.nn.Module, torch.Tensor
    get, ident = object.__getattribute__, threading.get_ident

    def __getattribute__(self: Module, name: str) -> object:
        value = get(self, "__dict__").get(name, _MISSING)
        if (
            value is not _MISSING
            and (reads := _READER.get(ident())) is not None
            and name not in _INTERNAL
            and not name.startswith(("_cozy", "__"))
        ):
            if type(value) in _PLAIN:
                reads.setdefault((id(self), name), (self, name, value, True))
            elif not isinstance(value, tensor | module):
                reads.setdefault((id(self), name), (self, name, value, False))
        return get(self, name)

    return __getattribute__


def _holds(recorder: _Recorder, entry: _Recording) -> bool:
    """Whether every module attribute `entry`'s capture read still has its value: checked again
    only when an attribute was set on the region's modules since it last held."""
    if entry.guarded == recorder.generation:
        return True
    for module, name, value, plain in entry.reads:
        now = module.__dict__.get(name, _MISSING)
        if now is value or (plain and type(now) is type(value) and now == value):
            continue
        recorder.moved = f"{type(module).__name__}.{name}"
        what = f"module state {recorder.moved}"
        recorder.changes[what] = recorder.changes.get(what, 0) + 1
        return False
    entry.guarded = recorder.generation
    return True


def _find(
    recorder: _Recorder, key: tuple[object, ...]
) -> tuple[tuple[object, ...], _Recording] | None:
    """`key`'s graph whose read guards hold now, and its handle."""
    for handle in recorder.paths.get(key, ()):
        entry = recorder.recordings[handle]
        if _holds(recorder, entry):
            return handle, entry
    return None


#: graphs one path keeps, one per state of the module attributes it reads; a path that needs
#: another runs as written for the process (a step counter its forward reads)
_VARIANTS = 4


def _state(
    torch: ModuleType, recorder: _Recorder, module: Module
) -> tuple[tuple[object, ...], str]:
    """The key's overlays part, and why the call runs as written ("" when none)."""
    return _overlays(recorder.loras)


def _context() -> tuple[tuple[str, object], ...]:
    """The plain values of the context variables a package set: a forward may steer by one
    (H3's `minimax_h3_checked_loop`), and a call under another value takes another path. The
    Runtime's own (`cozy_*`) are in the key where they steer, or never do."""
    return tuple(
        sorted(
            (var.name, value)
            for var, value in contextvars.copy_context().items()
            if type(value) in _PLAIN and not var.name.startswith("cozy")
        )
    )


def _dense(layout: AttentionLayout | None) -> tuple[object, ...] | None:
    """What a dense step's kernels read of its layout: every field but the step itself."""
    if layout is None:
        return None
    return (
        layout.live_tokens,
        layout.protected_prefix,
        layout.step >= layout.dense_until_step,
        layout.dense_paths,
        layout.exact_paths,
        layout.tau,
    )


def _under(
    layout: AttentionLayout, call: _Call, *args: *tuple[Module, *tuple[object, ...]], **kwargs: object
) -> object:
    with attention_scope(layout):
        return call(*args, **kwargs)


def _stateful(module: Module, layout: AttentionLayout) -> bool:
    """Whether a package's attention producer serving a site under `module` now does not state
    this call stateless (`AttentionProducer.stateless`): its `produce` must run."""
    producers = {site.producer.name: site.producer for site in producer_sites(module)}
    return any(p.stateless is None or not p.stateless(layout) for p in producers.values())


def _queue(
    torch: ModuleType,
    module: Module,
    recorder: _Recorder,
    call: _Call,
    key: tuple[object, ...],
    spec: TreeSpec,
    leaves: list[object],
    placed: _Placed | None,
) -> None:
    """A new signature waits for an idle period to record it (`capture_idle`), with the plain
    context variables its call ran under (`_context`), which the recording runs under too."""
    layout: list[object] = [
        _Leaf(tuple(v.shape), v.stride(), v.dtype, v.device) if isinstance(v, torch.Tensor) else v
        for v in leaves
    ]
    context = tuple(
        (var, value)
        for var, value in contextvars.copy_context().items()
        if type(value) in _PLAIN and not var.name.startswith("cozy")
    )
    _owner(torch, module, recorder, placed)
    with recorder.lock:
        if key in recorder.queued or key in recorder.failed or key in recorder.judged:
            return
        pending = recorder.queued[key] = _Pending(
            torch, weakref.ref(module), call, key, spec, layout, context
        )
    IDLE.add(pending)


#: `_replay`'s answer when the call runs as written
_EAGER = object()
#: `_replay_call`'s answer when its graph read storage outside the call that is gone
_STALE = object()
#: graphs a region keeps; one more drops the least recently replayed (`_admit`)
_MAX_GRAPHS = 16
#: replays in a row that refill weights before a path runs as written for the process
_REFILLS = 3


def _remove(recorder: _Recorder, handle: tuple[object, ...]) -> None:
    """A graph goes, its executable and static buffers with it (held: the region's lock)."""
    entry = recorder.recordings.pop(handle)
    key = cast("tuple[object, ...]", handle[0])  # a handle is (key, serial)
    held = recorder.paths.get(key)
    if held is not None:
        held.remove(handle)
        if not held:
            del recorder.paths[key]
    entry.finish()
    entry.close()
    recorder.retired += 1


def _expire(recorder: _Recorder, handle: tuple[object, ...], owner: domain.Domain) -> None:
    """A graph goes: storage it read outside the call is gone (held: the region's lock)."""
    _remove(recorder, handle)
    recorder.expired += 1
    _drop_unused(recorder, owner)


def _admit(recorder: _Recorder, key: tuple[object, ...], entry: _Recording) -> None:
    """`entry` serves `key` while its read guards hold; past `_MAX_GRAPHS` the least recently
    replayed goes (held: lock)."""
    entry.used, entry.guarded = recorder.attempts, recorder.generation
    while len(recorder.recordings) >= _MAX_GRAPHS:
        _remove(recorder, min(recorder.recordings, key=lambda h: recorder.recordings[h].used))
    recorder.serial += 1
    handle = (key, recorder.serial)
    recorder.recordings[handle] = entry
    recorder.paths.setdefault(key, []).append(handle)


def sweep(root: Module) -> None:
    """At an attempt's start: graphs whose outside storage is gone go now, their static inputs
    with them (a loop's graphs, once its tables are freed)."""
    recorder = _RECORDERS.get(root)
    if recorder is None or recorder.domain is None or not recorder.recordings:
        return
    with recorder.domain.using(), recorder.lock:
        for handle in [h for h, entry in recorder.recordings.items() if not entry.live()]:
            _expire(recorder, handle, recorder.domain)


def _replay_call(
    torch: ModuleType,
    module: Module,
    recorder: _Recorder,
    key: tuple[object, ...],
    leaves: list[object],
    placed: _Placed | None,
) -> object:
    """The call replayed in its component's pass, serialized on the device, or `_EAGER`: its
    graphs retired (a kernel landed, a changed fill), or no room to prepare its weights."""
    owner = recorder.domain
    assert owner is not None
    scope = placed[0].graph_pass(placed[1], "") if placed is not None else nullcontext()
    with scope, owner.using(), recorder.lock:
        stale = recorder.landed != LANDED[0] or (
            placed is not None and recorder.binding is not None and not recorder.binding.valid()
        )
        if stale:
            recorder.forget()  # in flight, it goes at a later call; as written meanwhile
            _note(recorder, "retired", "a kernel landed or its weights changed")
            return _EAGER
        attempt = placed[0].attempts if placed is not None else recorder.attempts
        if recorder.waiting.get(key) == attempt:
            _note(recorder, "waiting")
            return _EAGER
        found = _find(recorder, key)
        if found is None:
            return _STALE  # no graph of this path holds now: this call records one
        handle, entry = found
        if not entry.live():
            _expire(recorder, handle, owner)
            return _STALE
        entry.used = recorder.attempts
        if placed is not None and recorder.binding is None:
            recorder.binding = placed[0].retain(placed[1], ("",))
        return _replay(torch, leaves, placed, recorder, key, entry, attempt)


def _replay(
    torch: ModuleType,
    leaves: list[object],
    placed: _Placed | None,
    recorder: _Recorder,
    key: tuple[object, ...],
    entry: _Recording,
    attempt: int,
) -> object:
    """The call replayed, or `_EAGER`. Its results leave the component (a package's loop may
    keep them past its next call): they are copied out, allocated before the physical lease so
    admission sees their segments. varena refusing its weights room, this call runs as written
    and its graphs stay; weights that changed retire them at the next call (`_replay_call`)."""
    out = entry.output()
    missed: list[int] = []
    try:
        began = time.perf_counter()
        lease = _lease(torch, placed, recorder.binding, missed)
        lease.__enter__()
        recorder.prepare_ms += (time.perf_counter() - began) * 1e3
    except StaleBinding as exc:
        _note(recorder, "building", str(exc))
        return _EAGER
    except varena.Exhausted as exc:  # before the replay
        why = no_room(entry.owner.device, "its weights", 0, exc)
        _refused(recorder, placed, why)
        safeguard("no_room", why)
        _note(recorder, "no_room", why)
        return _EAGER
    try:
        result = recorder.replayed.run(torch, lambda: entry(leaves, out))
    finally:
        lease.__exit__(None, None, None)
    recorder.gap = ""
    recorder.replays += 1
    _note(recorder, "replayed")
    recorder.copy_bytes += entry.copied
    entry.refilled = entry.refilled + 1 if missed and missed[0] else 0
    if entry.refilled >= _REFILLS:
        # Its weights leave between its calls (another component takes their room each step): a
        # replay refills all of them before it starts, where the call as written streams them
        # behind its compute. The path runs as written for the process.
        why = f"its weights leave between calls: {entry.refilled} replays in a row refilled them"
        recorder.judged[key] = recorder.gap = why
        recorder.codes[key] = "no_room"
        for handle in list(recorder.paths.get(key, ())):
            _remove(recorder, handle)
        _drop_unused(recorder, entry.owner)
    return result


def _capture_pending(pending: _Pending) -> bool:
    """An idle period's job: `pending`'s signature recorded, or ruled out (`_record`). False
    waits for a later period: only its first call measured, or a placed region's pages not
    resident. No room queues it again at its next call (`_short`); anything else that fails it
    ends it, and the run says why: never retried."""
    torch, module = pending.torch, pending.module()
    recorder = _RECORDERS.get(module) if module is not None else None
    if recorder is None or recorder.queued.get(pending.key) is not pending:
        return True  # released, or its graphs dropped since it queued: its next call queues
    submit = recorder.eager.submit_ms()
    if submit is None:
        return False
    seen = recorder.seen.get(pending.key)
    if seen is not None and (took := seen.device_ms()) is not None and took > _PROBE_MS:
        # A warm-up would hold the idle device for the whole call (an H3 DiT step: 34 s) to
        # find what its length already says: launch overhead is no share of it worth a graph.
        recorder.idle_failed[pending.key] = f"device-bound: a {took:.0f} ms call"
        recorder.queued.pop(pending.key, None)
        return True
    placed, owner = PLACED.get(module), recorder.domain
    assert owner is not None
    device = next(leaf.device for leaf in pending.leaves if isinstance(leaf, _Leaf))
    with torch.cuda.device(device), owner.using(), recorder.lock:
        if recorder.landed != LANDED[0] or (
            recorder.binding is not None and not recorder.binding.valid()
        ):
            return recorder.forget()  # its queue went with its graphs
        if placed is not None:
            if recorder.binding is None:
                recorder.binding = placed[0].retain(placed[1], ("",))
            if not placed[0].graph_resident(recorder.binding):
                return False
        idle = placed[0].graph_idle() if placed is not None else nullcontext()
        try:
            # Each step that runs no torch op of its own looks for a command first: handles,
            # a capture, a replay check and its timing hold the host where no op yields.
            IDLE.check()
            owner.initialize()
            owner.prime_blas()  # before the idle request: Torch owns the shared workspaces
            with idle:
                # cast: its recorder is the live module's (above)
                done = _record(torch, cast("Module", module), pending, recorder, placed, submit)
        except Building:
            return False  # a cold probe postpones capture without consuming the signature
        except Exception as exc:  # the signature runs as written; the period goes on
            if isinstance(exc, Yielded) or fault(exc) in (Fault.STICKY, Fault.DEVICE):
                raise
            why = f"{type(exc).__name__}: {exc}"[:300]
            if is_capacity(exc):
                done = _short(cast("Module", module), recorder, pending.key, why, idle=True)
            else:
                _failed(cast("Module", module), recorder, pending.key, why, idle=True)
                done = True
        if done:
            recorder.queued.pop(pending.key, None)
        return done


def _record(
    torch: ModuleType,
    module: Module,
    pending: _Pending,
    recorder: _Recorder,
    placed: _Placed | None,
    submit: float,
) -> bool:
    """Records `pending`'s signature at idle on seeded inputs of its layout, in its region's
    capture pool: a warm-up run, measured, then the capture unless the warm-up's host waited on
    the device half its time (device-bound: a graph saves nothing). The replay decides the rest:
    bit for bit equal to eager, and faster than its eager calls (`_no_gain`). Its pool gets one
    free run of `vt.pool_bytes` of the warm-up's peak beside what the pool holds, more if the
    capture takes more (`capture`). No room for its weights, warm-up, pool or capture: it runs as
    written until a call records it, the run says what it needed, and its next call queues it
    again (`_short`). False tries again at a later period (a kernel building, its weights left)."""
    key, call, owner = pending.key, pending.call, recorder.domain
    assert owner is not None
    current: Stream = torch.cuda.current_stream()
    entry: _Recording | None = None
    recorder.capture_attempts += 1
    began = time.perf_counter()
    stage, need = "its weights", 0
    try:
        if why := _hooked(module):
            recorder.idle_failed[key] = why
            return True
        parts, why = _state(torch, recorder, module)
        if why or parts != cast("_Parts", key[-1]).state:
            recorder.idle_failed[key] = why or "its modules or overlay at idle are not its call's"
            return True
        if placed is not None and not placed[0].graph_accounted():
            _failed(module, recorder, key, "driver/context memory is unmeasured", idle=True)
            return True
        scope = placed[0].graph_pass(placed[1], "") if placed is not None else nullcontext()
        lease = recorder.binding.prepare(current) if recorder.binding is not None else nullcontext()
        with ExitStack() as entered:
            entered.enter_context(scope)
            entered.enter_context(lease)
            stage = "its warm-up"
            # Its weights are inputs, checked by identity and version: a copy of each held
            # beside them took the room a capped card's warm-up needed (SDXL at 6 GiB).
            weights = (
                recorder.binding.immutable_buffer_ids()
                if recorder.binding is not None
                else frozenset()
            )
            # cast: `_capture_pending` initialized the domain
            stream = cast("Stream", owner.stream)
            state = entered.enter_context(_capture_state(torch, module, stream, weights))
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                # Warm-up runs with ordinary Torch allocations, before graph-private storage.
                seeded = torch.Generator(device=current.device).manual_seed(_SEED)
                warming = _WARMING.set(True)
                try:
                    with attention_ulysses.observing() as served:
                        with _measured(owner.device) as peak:  # what its capture allocates
                            inputs: list[Tensor | None] = [
                                _input(torch, v, seeded) if isinstance(v, _Leaf) else None
                                for v in pending.leaves
                            ]
                            cargs, ckwargs = _arguments(torch, pending.spec, pending.leaves, inputs)
                            took, metered = time.perf_counter(), IDLE.metered_ms
                            eager = call(module, *cargs, **ckwargs)
                            took = (time.perf_counter() - took) * 1e3
                        if not _clonable(torch, eager):
                            _failed(module, recorder, key, "output is not a tensor tree", None)
                            return True  # a property of the code: no call records it either
                finally:
                    _WARMING.reset(warming)
                waited = IDLE.metered_ms - metered
                recorder.bound[key] = (submit, took, waited, None)
                if 2 * waited >= took:  # the device, not the host, set its pace
                    recorder.idle_failed[key] = (
                        f"device-bound: its warm-up's host waited {waited:.3f} of {took:.3f} ms "
                        "on the device"
                    )
                    return True
                if not peak:
                    recorder.idle_failed[key] = "its warm-up's working memory was not measured"
                    return True
                IDLE.check()
                stage, need = "its capture", vt.pool_bytes(peak[0])
                entry = _build(
                    torch, module, recorder, key, owner, placed, call, pending.spec,
                    pending.leaves, need, state, served, idle=True,
                )
                stage, need = "its replay check", 0
                if entry is None:
                    return True
                IDLE.check()
                replayed = entry(inputs, entry.output())
                if differs := vt.replay_differs(eager, replayed):
                    _failed(module, recorder, key, differs, "replay_mismatch", idle=True)
                    return True
                IDLE.check()
                why, replay_ms = _no_gain(torch, entry, recorder.eager)
                recorder.bound[key] = (submit, took, waited, replay_ms)
                if why:
                    recorder.idle_failed[key] = why
                    return True
        _admit(recorder, key, entry)
        entry = None
        recorder.short.pop(key, None)
        recorder.captures += 1
        recorder.capture_each.append(round((time.perf_counter() - began) * 1e3, 3))
        return True
    except Exception as exc:
        if isinstance(exc, Building | StaleBinding):
            return False  # a kernel still building, or its weights left: a later period
        if is_capacity(exc):
            if stage == "its capture" and recorder.pool is not None:
                need = recorder.pool.prepared
            return _short(module, recorder, key, no_room(owner.device, stage, need, exc), idle=True)
        if not isinstance(exc, domain.Unavailable):
            raise
        why = f"{type(exc).__name__}: {exc}".splitlines()[0][:300]
        _failed(module, recorder, key, why, idle=True)
        return True
    finally:
        recorder.capture_ms += (time.perf_counter() - began) * 1e3
        if entry is not None:
            entry.close()
        if not recorder.recordings:
            if recorder.pool is not None:
                close_pool(recorder.pool, owner.device)
            recorder.pool = None
            if recorder.binding is not None:
                # A rejected warm-up keeps no Regions when no executable reads them.
                recorder.binding.close()
                recorder.binding = None


def _build(
    torch: ModuleType,
    module: Module,
    recorder: _Recorder,
    key: tuple[object, ...],
    owner: domain.Domain,
    placed: _Placed | None,
    call: _Call,
    spec: TreeSpec,
    layout: list[object],
    reserve: int,
    state: _State,
    served: Seen,
    *,
    idle: bool = False,
) -> _Recording | None:
    """`call` recorded on static inputs of `layout` in its region's pool, `reserve` bytes of it
    prepared first; None when it can't be kept, its reason recorded: the capture refused it
    (`capture_refused`), recording changed the model's state, or its output is no tensor tree.
    The caller holds the region's weights, and `state` is the model's before recording."""
    pool = recorder.pool
    if pool is None:
        pool = recorder.pool = (
            placed[0].graph_pool(placed[1])
            if placed is not None
            else domain.Pool(torch.cuda.graph_pool_handle())
        )
    # Its static inputs are the pool's, made in the capture (an allocation launches nothing),
    # so their storage stays owned with the graph.
    statics: list[Tensor | None] = []
    outputs: list[object] = []
    observed: list[observe.Observed] = []
    reads: list[tuple[Module, str, object, bool]] = []
    weights = _weights(torch, module, recorder.binding)

    def step() -> None:
        try:
            statics.extend(_static(torch, v) if isinstance(v, _Leaf) else None for v in layout)
            sargs, skwargs = _arguments(torch, spec, layout, statics)
            mode, seen = observe.observer(torch, statics, weights)
            observed[:] = [seen]
            with cast("AbstractContextManager[object]", mode), _reading(torch) as read:
                outputs.append(call(module, *sargs, **skwargs))
            reads[:] = read.values()
        except BaseException:  # a short capture's go before it runs again
            statics.clear()
            outputs.clear()
            raise

    graph, why = capture(torch, step, pool, owner, reserve, alone=not recorder.recordings)
    if graph is None:
        _failed(module, recorder, key, why, idle=idle)
        return None
    if changed := state.restore():
        graph.reset()
        recorder.judged[key] = recorder.gap = f"recording it changed the model's state: {changed}"
        recorder.codes[key] = "refused"
        _note(recorder, "refused", recorder.gap)
        return None
    if not _clonable(torch, outputs[0]):
        graph.reset()
        _failed(module, recorder, key, "captured output is not a tensor tree", None, idle=idle)
        return None
    seen = observed[0]
    why = seen.refusal() or (
        "it writes its inputs or state outside its call: a call records it"
        if idle and (seen.arg_writes or seen.writes)
        else ""
    )
    if why:
        graph.reset()
        (recorder.idle_failed if idle else recorder.judged)[key] = recorder.gap = why
        if not idle:
            recorder.codes[key] = "refused"
            _note(recorder, "refused", why)
        return None
    entry = _Recording(torch, graph, pool, statics, outputs[0], key, served, owner, seen)
    entry.reads = tuple(reads)
    return entry


def _weights(
    torch: ModuleType, module: Module, binding: Binding | None
) -> list[tuple[int, int]]:
    """The address ranges of what a capture of `module` may read as its weights: its binding's
    Regions and computed tensors, and its own parameters and buffers bound now."""
    spans = observe.ranges((*module.parameters(), *module.buffers()))
    if binding is not None:
        spans += observe.ranges(binding.computed)
        spans += [(r.ptr, r.ptr + r.allocated_bytes) for r in binding.regions or () if r.ptr]
    return observe.merge(spans)


@contextmanager
def _capture_state(
    torch: ModuleType, module: Module, stream: Stream, weights: frozenset[int]
) -> Iterator[_State]:
    """Restore warm-up state and join its stream before the prepared weight lease ends."""
    current: Stream
    state: _State | None
    current, state = torch.cuda.current_stream(), None
    try:
        state = _State(torch, module, immutable_buffers=weights)
        yield state
    finally:
        current.wait_stream(stream)
        settling = _SETTLING.set(True)
        try:
            if state is not None:
                state.restore()
        finally:
            _SETTLING.reset(settling)


def _hooked(module: Module) -> str:
    """Why a component's forward is no graph: a package hook on one of its modules runs Python
    at every call, which a replay would skip (Runtime's binding and scope hooks are served). A
    LoRA overlay's hook (`lora.Hook`, `weights.LoRASource`) is the key's (`_loras`)."""
    for sub in module.modules():
        for hooks in (sub._forward_pre_hooks, sub._forward_hooks):
            for hook in hooks.values():
                if callable(getattr(hook, "lora_factors", None)):
                    continue
                if not getattr(hook, "__module__", "").startswith(("tensorfs.", "cozy_runtime.")):
                    return f"a package forward hook on {type(sub).__name__}"
    return ""


class _Loras(NamedTuple):
    """A component's LoRA overlay hooks: `lora.Sites` (each asked once a call) and the rest."""

    sites: tuple[tuple[lora.Sites, tuple[object, ...]], ...]
    hooks: tuple[tuple[str, object], ...]


def _loras(module: Module) -> _Loras:
    sites: dict[int, tuple[lora.Sites, list[object]]] = {}
    hooks: list[tuple[str, object]] = []
    for path, sub in module.named_modules():
        for hook in sub._forward_hooks.values():
            if not callable(getattr(hook, "lora_factors", None)):
                continue
            held = getattr(hook, "sites", None)
            if isinstance(held, lora.Sites):
                sites.setdefault(id(held), (held, []))[1].append(hook)
            else:
                hooks.append((path, hook))
    return _Loras(tuple((s, tuple(h)) for s, h in sites.values()), tuple(hooks))


def _armed_now(hook: object) -> object | None:
    """The factors a LoRA hook adds now (None: none)."""
    armed = getattr(hook, "armed", None)
    if callable(armed):
        return cast("object | None", armed())
    answer = hook.lora_factors()  # type: ignore[attr-defined]  # `_loras` kept LoRA sources
    return None if answer is None else cast("tuple[object, ...]", answer)[0]


def _overlays(loras: _Loras) -> tuple[tuple[object, ...], str]:
    """What the LoRA hooks arm now, as a key part, and why the call runs as written: factors a
    hook adds per call read another component's pages, which a replay does not hold. Baked
    factors are in the call's own pages, under the overlay they were filled with, so a graph
    keyed by them reads the right ones."""
    part: list[object] = []
    for sites, hooks in loras.sites:
        if sites.armed is None:
            continue
        for hook in hooks:
            if _armed_now(hook) is not None and not getattr(hook, "_cozy_baked", False):
                return (), "under a LoRA overlay applied per call"
        part.append((id(sites), id(sites.armed)))
    for path, hook in loras.hooks:
        factors = _armed_now(hook)
        if factors is None:
            continue
        if not getattr(hook, "_cozy_baked", False):
            return (), f"under a LoRA overlay applied per call at {path}"
        part.append((path, id(factors)))
    return tuple(part), ""


class _State:
    """What an idle recording must leave as it found it: the component's modules' plain
    attributes, its buffers (contents and version counters) and the RNG. Package
    code may keep a counter or a table in them; recording runs the forward twice and captures
    it once, so a change means the forward has state a replay would not keep (run 11226:
    measuring must never change a later request)."""

    def __init__(
        self, torch: ModuleType, module: Module, *, immutable_buffers: frozenset[int] = frozenset()
    ) -> None:
        self.torch = torch
        self.modules = list(module.modules())
        self.plain = [{n: v for n, v in vars(m).items() if type(v) in _PLAIN} for m in self.modules]
        self.buffers = [
            (name, tensor, tensor._version, tensor.clone())
            for m in self.modules
            for name, tensor in m._buffers.items()
            if tensor is not None and id(tensor) not in immutable_buffers
        ]
        #: each immutable buffer: its table, name, tensor, version, saved self and geometry
        self.immutable: list[
            tuple[dict[str, Tensor | None], str, Tensor, int, Tensor, tuple[object, ...]]
        ] = []
        for m in self.modules:
            for name, tensor in m._buffers.items():
                if tensor is None or id(tensor) not in immutable_buffers:
                    continue
                read = getattr(tensor, "_varena_read", None)
                saved: Tensor
                if read is not None:  # residency's unbound weight: keep its reader, not data
                    # cast: only residency's Placeholder has a `_varena_read`
                    saved = cast("type[Placeholder]", type(tensor))(
                        tensor.shape, tensor.dtype, tensor.device
                    )
                    saved._varena_read = read
                else:
                    saved = tensor.detach()
                geometry = (
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.dtype,
                    tensor.device,
                    tensor.storage_offset(),
                    tensor.data_ptr(),
                )
                self.immutable.append((m._buffers, name, tensor, tensor._version, saved, geometry))
        self.rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
        self.changed: str | None = None

    def restore(self) -> str:
        """Everything put back; the names that had changed ("" when none)."""
        if self.changed is None:
            changed = set()
            for module, plain in zip(self.modules, self.plain, strict=True):
                for name, value in plain.items():
                    if vars(module).get(name, _EAGER) != value:
                        changed.add(name)
                        vars(module)[name] = value
            for table, name, tensor, version, saved, geometry in self.immutable:
                current = (
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.dtype,
                    tensor.device,
                    tensor.storage_offset(),
                    tensor.data_ptr(),
                )
                if (
                    table.get(name) is not tensor
                    or tensor._version != version
                    or current != geometry
                ):
                    changed.add(name)
                    if current != geometry:
                        tensor.data = saved
                    table[name] = tensor
            for name, tensor, version, saved in self.buffers:
                if tensor._version != version:
                    changed.add(name)
                    tensor.copy_(saved)
            self.torch.random.set_rng_state(self.rng[0])
            self.torch.cuda.set_rng_state(self.rng[1])
            self.changed = ", ".join(sorted(changed))
        return self.changed


def _no_gain(torch: ModuleType, entry: _Recording, eager: _Timing) -> tuple[str, float | None]:
    """Why a verified graph is dropped, and a replay's measured ms: a replay as a request makes
    it (its inputs copied in, its outputs copied out), timed twice at idle, costs no less than
    an eager call by more than the two differ by (v14/v15: CLIP's tiny regions recorded, then
    replayed no faster: the copies and the launch were their whole cost). A call costs the
    longer of its host submission and its device time between events: a launch-bound call
    queued behind other work shows its kernels' time alone on the device, its cost on the host
    (v17's two-stream MLP). "" when it gains, or with no eager call measured past the first."""
    submit, device = eager.submit_ms(), eager.wall_ms()
    if submit is None or device is None:
        return "", None
    leaves = [t.clone() if t is not None else None for t in entry.inputs]
    costs = []
    for _ in range(2):
        out = entry.output()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        began = time.perf_counter()
        entry(leaves, out)
        host = (time.perf_counter() - began) * 1e3
        end.record()
        _settled(end)
        costs.append(max(host, float(start.elapsed_time(end))))
    replay, called = sum(costs) / len(costs), max(submit, device)
    if called - replay > abs(costs[0] - costs[1]):
        return "", replay
    return f"no gain: replay {replay:.3f} ms a call against eager {called:.3f} ms", replay


#: the idle inputs' seed: any fixed one (the verify needs the same inputs twice, not these)
_SEED = 0


def _arguments(
    torch: ModuleType, spec: TreeSpec, layout: Sequence[object], tensors: Sequence[Tensor | None]
) -> _Arguments:
    """The call's arguments with `tensors` at its leaves' places (None keeps the leaf)."""
    leaves = [t if t is not None else v for t, v in zip(tensors, layout, strict=True)]
    args: tuple[object, ...]
    kwargs: dict[str, object]
    args, kwargs = torch.utils._pytree.tree_unflatten(leaves, spec)
    return args, kwargs


def _static(torch: ModuleType, leaf: _Leaf) -> Tensor:
    """A static input of `leaf`'s layout, uninitialized: made inside a capture, from its pool."""
    span = 1 + sum((n - 1) * s for n, s in zip(leaf.shape, leaf.stride, strict=True))
    flat: Tensor = torch.empty(span if all(leaf.shape) else 0, dtype=leaf.dtype, device=leaf.device)
    return flat.as_strided(leaf.shape, leaf.stride)


def _input(torch: ModuleType, leaf: _Leaf, generator: Generator) -> Tensor:
    """A static input of `leaf`'s layout, seeded: floats standard normal (zeros would hide a
    replay that drops work), bools fair coins, integers zero (a valid index into any table)."""
    span = 1 + sum((n - 1) * s for n, s in zip(leaf.shape, leaf.stride, strict=True))
    span = span if all(leaf.shape) else 0
    flat: Tensor = torch.zeros(span, dtype=leaf.dtype, device=leaf.device)
    if flat.is_floating_point() or flat.is_complex():  # drawn wide, cast (fp8 draws nothing)
        wide = torch.complex64 if flat.is_complex() else torch.float32
        flat.copy_(torch.randn(span, generator=generator, device=leaf.device, dtype=wide))
    elif flat.dtype == torch.bool:
        flat.copy_(torch.randint(0, 2, (span,), generator=generator, device=leaf.device))
    return flat.as_strided(leaf.shape, leaf.stride)


def _settled(event: Event) -> None:
    """Wait for `event`, yielding to a command that arrives meanwhile (`_Idle.check`)."""
    while not event.query():
        if (idle := _IDLE.get(None)) is not None:
            idle.check()
        time.sleep(_POLL_S)


@contextmanager
def replaying(
    stream: Stream, bindings: Iterable[Binding], missed: list[int] | None = None
) -> Iterator[None]:
    """A graph's weights for one replay, back at their addresses if they left, and no longer. All
    of what is still here is pinned before any fill, which can then evict nothing the replay
    needs; what left comes back in the core's reclaim order. It never reassigns a module slot.
    `missed` gets how many Regions had left."""
    with ExitStack() as held:
        taken = [held.enter_context(binding.prepare(stream, replay=True)) for binding in bindings]
        left = 0
        for uses in taken:
            for region, use in uses.items():
                if use is None:
                    left += 1
                    uses[region] = region.use(stream.cuda_stream)
        if missed is not None:
            missed.append(left)
        yield


def _lease(
    torch: ModuleType,
    placed: _Placed | None,
    binding: Binding | None,
    missed: list[int] | None = None,
) -> AbstractContextManager[None]:
    if placed is None or binding is None:
        return nullcontext()
    stream: Stream
    stream, held = torch.cuda.current_stream(), placed[0].graph_held(placed[1])
    if held is not None and held != stream:
        stream.wait_stream(held)
    return replaying(stream, (binding,), missed)


def _refused(recorder: _Recorder, placed: _Placed | None, why: str) -> None:
    """A root whose preparation the memory refused runs as written; the run says so once."""
    why = why.splitlines()[0][:300]
    recorder.gap = why
    if placed is not None:
        placed[0].note(f"{placed[1]}.graphs", f"eager {why}")
    say(f"refused:{why}", f"[graphs] a component runs as written: {why}")


def _failed(
    module: Module,
    recorder: _Recorder,
    key: tuple[object, ...],
    why: str,
    code: str | None = "capture_refused",
    *,
    idle: bool = False,
) -> None:
    """A capture refused for good: the signature runs as written; the run says so, unless
    `code` is None: a property of the code, not a run going wrong, kept in the record alone.
    Refused at idle (`idle`), a call may still record it: an idle recording runs on seeded
    inputs outside its request."""
    (recorder.idle_failed if idle else recorder.failed)[key] = why
    recorder.gap = why
    if not idle:
        recorder.codes[key] = code or "refused"
        _note(recorder, code or "refused", why)
    if code is not None:
        safeguard(code, f"{_name(module)}: {why}")


def _short(
    module: Module, recorder: _Recorder, key: tuple[object, ...], why: str, *, idle: bool = False
) -> bool:
    """A recording found no room: the signature runs as written and a later call records or
    queues it again, since each request moves what holds the card (a component's weights leave,
    a peer ends). `_ROOM_TRIES` such recordings in a row and it is refused for good (`_failed`)."""
    tries = recorder.short[key] = recorder.short.get(key, 0) + 1
    if tries >= _ROOM_TRIES:
        _failed(module, recorder, key, why, "no_room", idle=idle)
    else:
        recorder.gap = why
        if not idle:
            _note(recorder, "no_room", why)
    return True


#: idle recordings of one signature in a row that may find no room before it is refused
_ROOM_TRIES = 3


def _changed(
    torch: ModuleType,
    module: Module,
    recorder: _Recorder,
    key: tuple[object, ...],
    arguments: _Arguments,
    count: int,
) -> None:
    """Count what in the region's key changed since its previous call. A plain argument whose
    value changes on consecutive calls (a timestep or a guidance scale passed as a Python number)
    keys every call apart, so the region records again every call: the run says so once, by name."""
    last = recorder.last
    assert last is not None
    if last[0] != key[0]:
        what = "argument layout"
    elif (at := next((i for i in range(1, count + 1) if last[i] != key[i]), None)) is not None:
        paths = torch.utils._pytree.tree_flatten_with_path(arguments)[0]
        name = torch.utils._pytree.keystr(paths[at - 1][0])
        plain = len(cast("tuple[object, ...]", key[at])) == 2
        what = f"{'value' if plain else 'tensor'} {name}"
        if plain and recorder.changes.get(what):
            component = _name(module)
            say(
                f"scalar:{id(recorder)}:{name}",
                f"[graphs] {component} records again every call: its argument {name} is a "
                "Python number that changes between calls (pass it as a tensor)",
            )
            WARNINGS.append(
                recovered(
                    "a graph region records again every call", region=component, argument=name
                )
            )
    else:
        before, now = cast("_Parts", last[-1]), cast("_Parts", key[-1])
        what = next(n for n in _Parts._fields if getattr(before, n) != getattr(now, n))
        if what == "state":
            what = "LoRA overlay"
        elif what == "context":
            moved = sorted(set(before.context) ^ set(now.context))
            what = f"context {moved[0][0]}"
    recorder.changes[what] = recorder.changes.get(what, 0) + 1


def _unplain(torch: ModuleType, leaves: list[object]) -> str:
    tensors = [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]
    if any(type(t) not in (torch.Tensor, torch.nn.Parameter) for t in tensors):
        return "with custom tensor dispatch"
    if not tensors or not all(t.is_cuda for t in tensors):
        return "without CUDA tensor arguments"
    if any(
        not isinstance(v, (torch.Tensor, torch.dtype, torch.device)) and type(v) not in _PLAIN
        for v in leaves
    ):
        return "with opaque arguments"
    return ""


def _clonable(torch: ModuleType, out: object) -> bool:
    leaves = torch.utils._pytree.tree_leaves(out)
    return any(isinstance(v, torch.Tensor) for v in leaves) and all(
        type(v) in (torch.Tensor, torch.nn.Parameter) or type(v) in _PLAIN for v in leaves
    )


def _signature(torch: ModuleType, leaves: list[object], spec: TreeSpec) -> tuple[object, ...]:
    return (
        str(spec),
        *(
            (tuple(v.shape), v.dtype, v.device, v.stride())
            if isinstance(v, torch.Tensor)
            else (type(v), v)
            for v in leaves
        ),
    )


__all__ = [
    "BYPASS",
]
