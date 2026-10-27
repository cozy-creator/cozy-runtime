"""CUDA graphs of whole components over retained virtual addresses.

Kernel choice is shipped. Runtime prepares memory, captures process-local addresses and
replays; it neither times competing paths nor compiles kernels. A component's region is its
root call; nothing inside a root is a region, so a streamed component runs as written. A root
has one graph per signature, keyed by its inputs' layout, the values of the attributes its
package declares steer its forward (`Model.__graph_keys__`), and the kernels serving
(`LANDED`). Until a signature has a graph, its calls run as written with nothing around them
but their key and two timing events. Nothing records on a request's path: an idle period
records a signature on seeded inputs of its layout if its eager calls were launch-bound, its
replay matches eager bit for bit and saves time, and recording changed no state of the model
(`capture_idle`); the next call replays. Every graph has its own static buffers in the device's
one capture pool (`domain`), opened only for a capture that may keep it. A cached binding owns
virtual addresses only; its physical lease covers preparation through output copies, and a
completion event orders reuse across execution streams.
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
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

from cozy_runtime.author import _request_path as request_path
from cozy_runtime.author._context import CURRENT
from cozy_runtime.author._context import device as placed_device
from cozy_runtime.author._errors import Fault, fault, is_capacity
from cozy_runtime.author._model import LANDED, PLACED, WARNINGS, Building
from cozy_runtime.graphs import domain
from cozy_runtime.internal import attention_ulysses, kernel_ready

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
    tuple[Callable[[Any], tuple[dict[str, int], list[Any], Any]], Callable[[Any], None]]
] = []
_SAID: set[str] = set()
#: the isolated request used for idle warm-up and capture observations
IDLE_REQUEST = "graphs-idle"


def say(key: str, line: str) -> None:
    if key not in _SAID:
        _SAID.add(key)
        print(line, file=sys.stderr)


@request_path.forbid("cuda graph capture")
def capture(
    torch: Any, step: Callable[[], object], pool: domain.Pool, owner: domain.Domain
) -> tuple[Any, str]:
    """Record a caller-owned capture-safe computation in its already prepared pool.

    The caller owns purity, residency, static-data lifetime and all consuming streams. No
    weight staging or eviction is allowed inside capture. Torch owns the capture pool and its
    allocation bookkeeping; a failed capture is reset before its caller may try eager execution.
    """
    graph = torch.cuda.CUDAGraph()
    current = torch.cuda.current_stream()
    stream, work = owner.control, owner.stream
    stream.wait_stream(current)
    stream.wait_stream(work)
    failure: BaseException | None = None
    try:
        with (
            kernel_ready.recording(),
            owner.recovery_blocked(),
            torch.utils._python_dispatch._disable_current_modes(),
            torch.cuda.stream(stream),
        ):
            try:
                graph.capture_begin(pool=pool.handle, capture_error_mode="thread_local")
            except Exception as exc:
                failure = exc
            else:
                try:
                    owner.fork.record(stream)
                    work.wait_event(owner.fork)
                    with torch.cuda.stream(work):
                        try:
                            step()
                        finally:
                            owner.join.record(work)
                            stream.wait_event(owner.join)
                except Exception as exc:
                    failure = exc
                try:
                    graph.capture_end()
                except Exception as exc:
                    failure = failure or exc
    except Building as exc:
        failure = exc
    current.wait_stream(stream)
    current.wait_stream(work)
    if failure is not None:
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
    if isinstance(failure, Building | Yielded):
        raise failure
    if failure is not None:
        if fault(failure) in (Fault.STICKY, Fault.DEVICE):
            code = getattr(failure, "error_code", None)
            if code not in (900, 901):  # unsupported or invalidated capture, already ended
                raise failure
        return None, f"{type(failure).__name__}: {failure}".splitlines()[0][:300]
    owner.graphs.add(graph)
    return graph, ""


class _Recording:
    """One signature's graph and its static buffers. A replay copies the caller's inputs into
    its static inputs and its static outputs into caller-owned tensors (`output`), so nothing
    of the graph's own memory leaves it."""

    def __init__(
        self,
        torch: Any,
        graph: Any,
        pool: Any,
        inputs: list[Any],
        outputs: Any,
        layout: object,
        served: Any,
        owner: domain.Domain,
    ) -> None:
        self.torch, self.graph, self.pool, self.owner = torch, graph, pool, owner
        self.inputs, self.outputs, self.layout = inputs, outputs, layout
        self.served = [(b, i) for b, impls in served.items() for i in impls]
        #: bytes the last replay copied in and out
        self.copied = 0
        self.done: Any = None
        self.stream: Any = None
        self.fenced = True

    def output(self) -> Any:
        """Caller-owned outputs, allocated before residency admission or graph launch."""
        torch = self.torch
        return torch.utils._pytree.tree_map(
            lambda v: torch.empty_like(v) if isinstance(v, torch.Tensor) else v, self.outputs
        )

    def __call__(self, leaves: list[Any], out: Any) -> Any:
        """Replays, copying its outputs into `out`."""
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
            for static, value in zip(self.inputs, leaves, strict=True):
                if static is not None and not _same(static, value):
                    static.copy_(value)
                    copied += static.nbytes
            with self.owner.recovery_blocked(replay=True):
                self.graph.replay()
            for backend, implementation in self.served:
                attention_ulysses.record(backend, implementation)
            source = torch.utils._pytree.tree_leaves(self.outputs)
            for dst, src in zip(torch.utils._pytree.tree_leaves(out), source, strict=True):
                if isinstance(src, torch.Tensor):
                    dst.copy_(src)
                    copied += src.nbytes
            return out
        finally:
            self.copied = copied
            self.done.record(stream)
            self.fenced = True

    def close(self) -> None:
        """After last use: release static tensors and executable before their pool."""
        self.inputs, self.outputs = [], None
        self.graph.reset()
        self.graph = self.pool = None

    def idle(self) -> bool:
        return self.fenced and (self.done is None or self.done.query())

    def finish(self) -> None:
        """Wait for its last replay and output copies."""
        if not self.fenced and self.stream is not None:
            self.stream.synchronize()
        if self.done is not None:
            self.done.synchronize()


def _same(static: Any, value: Any) -> bool:
    """`value` is `static`'s memory as it is laid out: reading it in place needs no copy."""
    return value is static or (
        value.data_ptr() == static.data_ptr()
        and value.shape == static.shape
        and value.stride() == static.stride()
        and value.dtype == static.dtype
    )


class _Timing:
    """One way a region runs (eager or replayed): its calls, the host time each took to submit,
    and its device time between two CUDA events around it, read once the device has passed them
    (never by waiting; a later call or the reply settles them)."""

    def __init__(self) -> None:
        self.calls = self.timed = 0
        self.host_ms = self.device_ms = 0.0
        #: the first call's host and device times apart (a cold run's first eager call:
        #: autotune, library initialization, page-in)
        self.first_ms: float | None = None
        self.first_device_ms = 0.0
        self.pending: deque[tuple[Any, Any]] = deque()
        self.free: list[tuple[Any, Any]] = []

    def run[T](self, torch: Any, call: Callable[[], T]) -> T:
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
        began = time.perf_counter()
        try:
            return call()
        finally:
            took = (time.perf_counter() - began) * 1e3
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

    def counts(self) -> tuple[int, int, float, float]:
        self.settle()
        return self.calls, self.timed, self.host_ms, self.device_ms

    def document(self, base: tuple[int, int, float, float] = (0, 0, 0.0, 0.0)) -> dict[str, Any]:
        calls, timed, host, device = (
            now - then for now, then in zip(self.counts(), base, strict=True)
        )
        return {
            "calls": calls,
            "first_ms": round(self.first_ms or 0.0, 3),
            "host_ms": round(host, 3),
            "device_ms": round(device, 3),
            "device_timed": timed,
        }


#: a recorder's counters a receipt reports per attempt
_COUNTERS = (
    "attempts",
    "replays",
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
        self.recordings: dict[tuple[object, ...], _Recording] = {}
        #: each signature waiting for an idle period to record (`capture_idle`)
        self.queued: dict[tuple[object, ...], _Pending] = {}
        #: why a signature runs as written until its graphs are dropped (a refusal)
        self.failed: dict[tuple[object, ...], str] = {}
        #: why one runs as written for the process: measured, it would save nothing
        #: (device-bound, or a replay no faster than its eager call)
        self.judged: dict[tuple[object, ...], str] = {}
        self.waiting: dict[tuple[object, ...], int] = {}
        #: per signature its idle measure: host submission, device-busy and (recorded) replay
        #: ms a call
        self.bound: dict[tuple[object, ...], tuple[float, float, float | None]] = {}
        self.binding: Any = None
        #: Native prepared definitions owned by this same component retirement boundary.
        self.prepared: weakref.WeakSet[Any] = weakref.WeakSet()
        #: its graphs' capture pool (`domain.Pool`), made at its first capture
        self.pool: domain.Pool | None = None
        self.replays = 0
        self.attempts = 0
        self.capture_attempts = 0
        self.captures = 0
        self.retired = 0
        self.copy_bytes = 0
        #: why its last call ran as written beside its graphs (a refusal), "" when none
        self.gap = ""
        self.landed = LANDED[0]
        self.domain: domain.Domain | None = None
        #: host time preparing and capturing at idle (warm-up, its busy measure and the
        #: instantiation included), each capture's apart, and each way it ran
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
        #: the counters when the current attempt opened (`mark`): receipts are per request
        self.base: dict[str, Any] = {}

    def snapshot(self) -> dict[str, Any]:
        return {
            **{name: getattr(self, name) for name in _COUNTERS},
            "changes": dict(self.changes),
            "eager": self.eager.counts(),
            "replayed": self.replayed.counts(),
        }

    def since(self, name: str) -> Any:
        """Counter `name` since the attempt opened."""
        return getattr(self, name) - self.base.get(name, 0)

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
        if self.binding is not None:
            self.binding.close()
            self.binding = None
        self.pool = None
        self.queued.clear()
        self.failed.clear()
        self.waiting.clear()
        self.landed = LANDED[0]


#: each installed component root's region
_RECORDERS: weakref.WeakKeyDictionary[Any, _Recorder] = weakref.WeakKeyDictionary()
#: each component root's plain attributes at its first call, until its second compares them
_FIRST: weakref.WeakKeyDictionary[Any, dict[str, object] | None] = weakref.WeakKeyDictionary()
#: per component, the undeclared plain attributes of its root that changed between its first two
#: calls: each would key its graphs or steer its forward unseen (`task conform` fails on it)
UNDECLARED: dict[str, set[str]] = {}
_CLASSES: dict[type, type] = {}


class Yielded(Exception):
    """An idle capture gives the device back at a region boundary: a command arrived."""


class _Idle:
    """Work for the device's idle periods, in the order it queued (`capture_idle`): a job answers
    whether it is done; one that is not (its region measured once, its pages not resident, no
    free room) waits for the next period. `check`, before each torch op of a job (`_yielding`)
    and at each region boundary, yields to a command that arrived: the job ends there, a capture
    discarded, first again next period. The longest stretch between checks is the longest a
    command waited (v17: 386 ms when it was the whole UNet's capture)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: deque[Callable[[], bool]] = deque()
        self.stop: Callable[[], bool] = lambda: False
        self.last = 0.0
        self.periods = self.yields = self.failures = 0
        self.longest_ms = 0.0

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
                    with _yielding(torch)() if torch is not None else nullcontext():
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
            self.longest_ms = max(self.longest_ms, (time.perf_counter() - self.last) * 1e3)
            with self.lock:
                self.jobs.extend(later)

    def document(self) -> dict[str, Any]:
        return {
            "queued": len(self.jobs),
            "periods": self.periods,
            "yields": self.yields,
            "failures": self.failures,
            "longest_block_ms": round(self.longest_ms, 3),
        }


IDLE = _Idle()


@functools.cache
def _yielding(torch: Any) -> Any:
    """The torch function mode an idle job runs under: before each op, a command that arrived
    ends the job (`_Idle.check`), so a whole component's capture yields within one op. Its
    cleanup (`_SETTLING`) never yields; a package that catches the raise meets it at its next op."""

    class Yielding(torch.overrides.TorchFunctionMode):  # type: ignore[misc]
        def __torch_function__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            if not _SETTLING.get() and (idle := _IDLE.get(None)) is not None:
                idle.check()
            return func(*args, **(kwargs or {}))

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
    dtype: Any
    device: Any


@dataclass(eq=False)
class _Pending:
    """A signature whose first call ran as written, queued to record at idle."""

    torch: Any
    module: weakref.ref[Any]
    call: Callable[..., Any]
    key: tuple[object, ...]
    spec: Any
    #: each argument leaf: a tensor's layout, or the plain value
    leaves: list[object]

    def __call__(self) -> bool:
        return _capture_pending(self)


def install(torch: Any, model: object, roots: Mapping[str, Any]) -> list[str]:
    """Each component root is a region, keyed by the attributes the model declares steer it
    (`__graph_keys__`: component -> attribute paths). Nothing inside a root is a region: a
    streamed component runs as written."""
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


def _graphed(torch: Any, cls: type) -> type:
    made = _CLASSES.get(cls)
    if made is None:
        call = cls.__call__

        def __call__(self: Any, *args: Any, **kwargs: Any) -> Any:
            return _dispatch(torch, self, call, args, kwargs)

        made = _CLASSES[cls] = type(
            cls.__name__,
            (cls,),
            {
                "__call__": __call__,
                "_cozy_graphed": True,
                "_cozy_original_call": call,
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
            },
        )
    return made


def retire(torch: Any, root: Any) -> None:
    """Drain `root`'s graphs and drop them, keeping its region: its calls run as written until
    an idle period records them again (an idle offload, a changed fill)."""
    recorder = _RECORDERS.get(root)
    if recorder is None:
        return
    with recorder.lock:
        prepared = tuple(recorder.prepared)
    # Native close can join CUDA and invoke Python owner destructors; never hold this lock.
    for definition in prepared:
        definition._retire_if_idle()
    if not recorder.recordings and recorder.pool is None:
        return
    scope = recorder.domain.using() if recorder.domain is not None else nullcontext()
    with scope, recorder.lock:
        recorder.finish()
        recorder.drop()


def retire_stale(torch: Any, root: Any) -> None:
    """Before a scope binds `root`'s weights: graphs recorded against another fill (a LoRA mode
    or bake recipe this attempt changed) go first, so nothing replays over a refill."""
    recorder = _RECORDERS.get(root)
    if recorder is not None:
        with recorder.lock:
            prepared = tuple(recorder.prepared)
        for definition in prepared:
            if not definition.valid():
                definition._retire_if_idle()
    if recorder is not None and recorder.binding is not None and not recorder.binding.valid():
        retire(torch, root)


def own_prepared(root: Any, definition: Any) -> None:
    """Attach a native definition to the existing component owner, without graph interception."""
    recorder = _RECORDERS.get(root)
    if recorder is None:
        recorder = _RECORDERS[root] = _Recorder(())
    with recorder.lock:
        recorder.prepared.add(definition)


def forget_prepared(root: Any, definition: Any) -> None:
    recorder = _RECORDERS.get(root)
    if recorder is not None:
        with recorder.lock:
            recorder.prepared.discard(definition)


def release(torch: Any, root: Any) -> None:
    """A generation's explicit teardown: finish its output copies before releasing addresses."""
    retire(torch, root)
    _RECORDERS.pop(root, None)
    domain.release(root)


def mark(root: Any) -> None:
    """A new attempt of `root`'s component opens: its receipts count from here."""
    recorder = _RECORDERS.get(root)
    if recorder is not None:
        with recorder.lock:
            recorder.base = recorder.snapshot()


def capture_domain(torch: Any, root: Any, memory: Any = None) -> domain.Domain:
    """The component owns hidden library scratch independently of its disposable graphs."""
    return domain.acquire(torch, root, memory)


def capture_device(torch: Any, memory: Any) -> domain.Domain:
    """A placed component's graphs share the device's domain with its other components'."""
    return domain.device(torch, memory)


def describe(root: Any) -> dict[str, Any]:
    """One final-reply snapshot of `root`'s region, never an aggregate scan on the replay hot
    path. Counts are this attempt's, since the Placement opened it (`mark`); an unplaced
    component's are its generation's. Pool bytes count each live pool once (a device's shared
    pool under the first component it served); driver/context and pending native frees remain
    ledger facts."""
    pools: dict[tuple[int, int], Any] = {}
    domains: dict[int, domain.Domain] = {}
    answer: dict[str, Any] = dict.fromkeys(
        ("active_graphs", "region_calls", "replays", "capture_attempts", "captures"), 0
    )
    answer.update(retired_graphs=0, replay_copy_bytes=0, capture_ms=0.0)
    recorder = _RECORDERS.get(root)
    if recorder is not None:
        with recorder.lock:
            since, base = recorder.since, recorder.base
            before = base.get("changes", {})
            answer.update(
                active_graphs=len(recorder.recordings),
                region_calls=since("attempts"),
                replays=since("replays"),
                capture_attempts=since("capture_attempts"),
                captures=since("captures"),
                retired_graphs=since("retired"),
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
                eager=recorder.eager.document(base.get("eager", (0, 0, 0.0, 0.0))),
                replay=recorder.replayed.document(base.get("replayed", (0, 0, 0.0, 0.0))),
                eager_gap=recorder.gap,
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
                answer[key] += decode[key]
    answer.update(
        idle=IDLE.document(),
        graph_pools=len(pools),
        graph_pool_bytes=None,  # stock Torch does not expose a public per-pool byte counter
        domains=len(domains),
    )
    return answer


def _receipt(recorder: _Recorder) -> dict[str, Any]:
    """A region's graph economics so far: each idle capture's cost (warm-up, busy measure,
    capture and instantiation), what each signature measured, a replay's saving over an eager
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
                "busy_ms": round(busy, 3),
                "replay_ms": None if replayed is None else round(replayed, 3),
            }
            for submit, busy, replayed in recorder.bound.values()
        ],
        "saving_ms": None if saving is None else round(saving, 3),
        "payback_calls": math.ceil(cost / saving) if saving and saving > 0 else None,
        "eager": sorted({*recorder.failed.values(), *recorder.judged.values()}),
    }


def _unprepared(exc: BaseException) -> bool:
    return is_capacity(exc) or getattr(exc, "code", None) == "STALE_BINDING"


def _dispatch(torch: Any, module: Any, call: Callable[..., Any], args: Any, kwargs: Any) -> Any:
    if (
        (context := CURRENT.get()) is not None
        and not (torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing())
        and not (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)
    ):
        context.raise_if_cancelled()
    if (idle := _IDLE.get(None)) is not None:
        idle.check()
    recorder = _RECORDERS.get(module)
    if recorder is not None:
        _watch(module, recorder)
    if (
        recorder is None  # a copy of a root: no region of its own
        or BYPASS.get()
        or _WARMING.get()
        or placed_device().type != "cuda"
        or torch.cuda.is_current_stream_capturing()
    ):
        return call(module, *args, **kwargs)
    began = time.perf_counter()
    try:
        return _region_call(torch, module, recorder, call, args, kwargs)
    finally:
        recorder.call_ms += (time.perf_counter() - began) * 1e3


def _watch(root: Any, recorder: _Recorder) -> None:
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
            (
                "a component's undeclared state changed between calls",
                {"component": name, "attributes": ", ".join(sorted(changed))},
            )
        )


def _name(root: Any) -> str:
    placed = PLACED.get(root)
    return placed[1] if placed is not None else type(root).__name__


def _region_call(
    torch: Any, module: Any, recorder: _Recorder, call: Callable[..., Any], args: Any, kwargs: Any
) -> Any:
    """The call replayed when its signature has a graph; otherwise as written, with nothing
    around it but its key and two timing events: no lock, pass or preparation until a graph
    exists (an idle period records it)."""
    placed = PLACED.get(module)
    leaves, spec = torch.utils._pytree.tree_flatten((args, kwargs))
    why = (
        "with gradients"
        if torch.is_grad_enabled()
        else "under custom tensor dispatch"
        if torch.utils._python_dispatch._get_current_dispatch_mode() is not None
        else "across GPUs"
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        else _unplain(torch, leaves)
    )
    if why:
        recorder.gap = why
        if placed is not None:
            placed[0].note(f"{placed[1]}.graphs", f"eager {why}")
        return call(module, *args, **kwargs)
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
    key = (*_signature(torch, leaves, spec), declared, addresses, autocast)
    if recorder.last is not None and recorder.last != key:
        _changed(torch, module, recorder, key, (args, kwargs), len(leaves))
    recorder.last = key
    recorder.attempts += 1
    if key in recorder.recordings:
        result = _replay_call(torch, module, recorder, key, leaves, placed)
        if result is not _EAGER:
            return result
    elif key not in recorder.queued and key not in recorder.failed and key not in recorder.judged:
        _queue(torch, module, recorder, call, key, spec, leaves, placed)
    return recorder.eager.run(torch, lambda: call(module, *args, **kwargs))


def _queue(
    torch: Any,
    module: Any,
    recorder: _Recorder,
    call: Callable[..., Any],
    key: tuple[object, ...],
    spec: Any,
    leaves: list[Any],
    placed: Any,
) -> None:
    """A new signature waits for an idle period to record it (`capture_idle`)."""
    layout = [
        _Leaf(tuple(v.shape), v.stride(), v.dtype, v.device) if isinstance(v, torch.Tensor) else v
        for v in leaves
    ]
    with recorder.lock:
        if recorder.domain is None:
            recorder.domain = (
                placed[0].graph_domain(placed[1])
                if placed is not None
                else capture_domain(torch, module)
            )
        if key in recorder.queued or key in recorder.failed or key in recorder.judged:
            return
        pending = recorder.queued[key] = _Pending(
            torch, weakref.ref(module), call, key, spec, layout
        )
    IDLE.add(pending)


#: `_replay`'s answer when the call runs as written
_EAGER = object()


def _replay_call(
    torch: Any,
    module: Any,
    recorder: _Recorder,
    key: tuple[object, ...],
    leaves: list[Any],
    placed: Any,
) -> Any:
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
            return _EAGER
        entry = recorder.recordings.get(key)
        attempt = placed[0].attempts if placed is not None else recorder.attempts
        if entry is None or recorder.waiting.get(key) == attempt:
            return _EAGER
        if placed is not None and recorder.binding is None:
            recorder.binding = placed[0].retain(placed[1], ("",))
        return _replay(torch, leaves, placed, recorder, key, entry, attempt)


def _replay(
    torch: Any,
    leaves: list[Any],
    placed: Any,
    recorder: _Recorder,
    key: tuple[object, ...],
    entry: _Recording,
    attempt: int,
) -> Any:
    """The call replayed, or `_EAGER`. Its results leave the component (a package's loop may
    keep them past its next call): they are copied out, allocated before the physical lease so
    admission sees their segments."""
    try:
        out = entry.output()
    except Exception as exc:
        if not _unprepared(exc):
            raise
        # No replay launched. The cached private pool can itself be why ordinary outputs
        # do not fit, so drain and retire it before the eager path allocates those outputs.
        recorder.finish()
        recorder.drop()
        torch.cuda.empty_cache()
        return _EAGER
    try:
        began = time.perf_counter()
        lease = _lease(torch, placed, recorder.binding)
        lease.__enter__()
        recorder.prepare_ms += (time.perf_counter() - began) * 1e3
    except Exception as exc:
        if not _unprepared(exc):
            raise
        del out
        recorder.finish()
        recorder.drop()
        torch.cuda.empty_cache()
        _refused(recorder, placed, f"its weights cannot all be resident at once ({exc})")
        return _EAGER
    try:
        result = recorder.replayed.run(torch, lambda: entry(leaves, out))
    finally:
        lease.__exit__(None, None, None)
    recorder.gap = ""
    recorder.replays += 1
    recorder.copy_bytes += entry.copied
    return result


def _capture_pending(pending: _Pending) -> bool:
    """An idle period's job: `pending`'s signature recorded, or ruled out (`_record`). False
    waits for a later period: only its first call measured, or a placed region's pages not
    resident or no free room for it."""
    torch, module = pending.torch, pending.module()
    recorder = _RECORDERS.get(module) if module is not None else None
    if recorder is None or recorder.queued.get(pending.key) is not pending:
        return True  # released, or its graphs dropped since it queued: its next call queues
    submit = recorder.eager.submit_ms()
    if submit is None:
        return False
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
            owner.initialize()
            owner.prime_blas()  # before the idle request: Torch owns the shared workspaces
            with idle:
                done = _record(torch, module, pending, recorder, placed, submit)
        except Building:
            return False  # a cold probe postpones capture without consuming the signature
        except Exception as exc:  # the signature runs as written; the period goes on
            if isinstance(exc, Yielded) or fault(exc) in (Fault.STICKY, Fault.DEVICE):
                raise
            if is_capacity(exc):
                return False
            _failed(module, recorder, pending.key, f"{type(exc).__name__}: {exc}"[:300])
            done = True
        if done:
            recorder.queued.pop(pending.key, None)
        return done


def _record(
    torch: Any,
    module: Any,
    pending: _Pending,
    recorder: _Recorder,
    placed: Any,
    submit: float,
) -> bool:
    """Records `pending`'s signature at idle on seeded inputs of its layout, in the device's
    capture pool: a warm-up run, its device-busy time (`_busy`), then the capture only when its
    eager calls were launch-bound: their host submission took longer than the device's work, so
    the device sat idle inside them. False tries again at a later period (no free room, a kernel
    building)."""
    # This path is restricted to the shipped pure contracts. Warmup discovers capacity in
    # the exact capture pool; it does not time or select kernels, or run arbitrary package code.
    key, call, owner = pending.key, pending.call, recorder.domain
    assert owner is not None
    current = torch.cuda.current_stream()
    graph = None
    recorder.capture_attempts += 1
    began = time.perf_counter()
    try:
        if why := _hooked(module):
            recorder.judged[key] = why
            return True
        if placed is not None and not placed[0].graph_accounted():
            _failed(module, recorder, key, "driver/context memory is unmeasured")
            return True
        scope = placed[0].graph_pass(placed[1], "") if placed is not None else nullcontext()
        # Discovery owns the complete range set and then the admitted free headroom.
        lease = recorder.binding.prepare(current) if recorder.binding is not None else nullcontext()
        with scope, lease, _capture_state(torch, module, owner.stream) as state:
            stream = owner.stream
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                # Warm-up runs with ordinary Torch allocations, before graph-private storage.
                seeded = torch.Generator(device=current.device).manual_seed(_SEED)
                inputs = [
                    _input(torch, v, seeded) if isinstance(v, _Leaf) else None
                    for v in pending.leaves
                ]
                cargs, ckwargs = _arguments(torch, pending, inputs)
                warming = _WARMING.set(True)
                try:
                    with attention_ulysses.observing() as served:
                        took = time.perf_counter()
                        eager = call(module, *cargs, **ckwargs)
                        took = (time.perf_counter() - took) * 1e3
                        if not _clonable(torch, eager):
                            _failed(module, recorder, key, "output is not a tensor tree")
                            return True
                        busy = _busy(torch, lambda: call(module, *cargs, **ckwargs), took)
                finally:
                    _WARMING.reset(warming)
                recorder.bound[key] = (submit, busy, None)
                if why := _device_bound(submit, busy):
                    recorder.judged[key] = why
                    return True
                if recorder.pool is None:
                    recorder.pool = (
                        placed[0].graph_pool(placed[1])
                        if placed is not None
                        else domain.Pool(torch.cuda.graph_pool_handle())
                    )
                pool = recorder.pool
                # Its static inputs are the pool's, made in the capture (an allocation launches
                # nothing), so their storage stays owned with the graph.
                statics: list[Any] = []
                outputs: list[Any] = []

                def step() -> None:
                    statics.extend(
                        _static(torch, v) if isinstance(v, _Leaf) else None for v in pending.leaves
                    )
                    sargs, skwargs = _arguments(torch, pending, statics)
                    outputs.append(call(module, *sargs, **skwargs))

                took = time.perf_counter_ns()
                graph, why = capture(torch, step, pool, owner)
                if graph is None:
                    _failed(module, recorder, key, why)
                    return True
                if changed := state.restore():
                    recorder.judged[key] = f"recording it changed the model's state: {changed}"
                    graph.reset()
                    graph = None
                    return True
                if not _clonable(torch, outputs[0]):
                    _failed(module, recorder, key, "captured output is not a tensor tree")
                    return True
                for target, value in zip(statics, inputs, strict=True):
                    if target is not None:
                        target.copy_(value)
                with owner.recovery_blocked(replay=True):
                    graph.replay()  # served only when it reproduces the eager run bit for bit
                pairs = zip(*map(torch.utils._pytree.tree_leaves, (outputs[0], eager)), strict=True)
                if not all(torch.equal(a, b) for a, b in pairs if isinstance(a, torch.Tensor)):
                    why = "its replay differs from the eager run on the same inputs"
                    _failed(module, recorder, key, why)
                    WARNINGS.append(("a CUDA graph was not used", {"why": why}))
                    return True
                entry = _Recording(torch, graph, pool, statics, outputs[0], key, served, owner)
                why, replayed = _no_gain(torch, entry, recorder.eager)
                recorder.bound[key] = (submit, busy, replayed)
                if why:
                    recorder.judged[key] = why
                    entry.close()
                    graph = None
                    return True
        recorder.recordings[key] = entry
        recorder.captures += 1
        recorder.capture_each.append(round((time.perf_counter() - began) * 1e3, 3))
        return True
    except Exception as exc:
        if is_capacity(exc) or _unprepared(exc) or isinstance(exc, Building):
            return False
        if not isinstance(exc, domain.Unavailable):
            raise
        _failed(module, recorder, key, f"{type(exc).__name__}: {exc}".splitlines()[0][:300])
        return True
    finally:
        recorder.capture_ms += (time.perf_counter() - began) * 1e3
        if key not in recorder.recordings:
            if graph is not None:
                graph.reset()
            if not recorder.recordings:
                recorder.pool = None
                if recorder.binding is not None:
                    # A stock binding owns weight storage once prepared. A rejected warm-up
                    # must not retain a whole root when no executable uses those addresses.
                    recorder.binding.close()
                    recorder.binding = None


@contextmanager
def _capture_state(torch: Any, module: Any, stream: Any) -> Iterator[_State]:
    """Restore warm-up state and join its stream before the prepared weight lease ends."""
    current, state = torch.cuda.current_stream(), None
    try:
        state = _State(torch, module)
        yield state
    finally:
        current.wait_stream(stream)
        settling = _SETTLING.set(True)
        try:
            if state is not None:
                state.restore()
        finally:
            _SETTLING.reset(settling)


def _hooked(module: Any) -> str:
    """Why a component's forward is no graph: a package hook on one of its modules runs Python
    at every call, which a replay would skip (Runtime's binding and scope hooks are served). A
    LoRA overlay hook (`lora.Hook`) asks the package at every call which overlay is armed, and
    the bound pages follow that answer: a replay would keep the captured one."""
    for sub in module.modules():
        for hooks in (sub._forward_pre_hooks, sub._forward_hooks):
            for hook in hooks.values():
                if callable(getattr(hook, "lora_factors", None)):
                    return f"a LoRA overlay hook on {type(sub).__name__}"
                if not getattr(hook, "__module__", "").startswith(("tensorfs.", "cozy_runtime.")):
                    return f"a package forward hook on {type(sub).__name__}"
    return ""


class _State:
    """What an idle recording must leave as it found it: the component's modules' plain
    attributes, its buffers (contents and version counters) and the RNG. Package
    code may keep a counter or a table in them; recording runs the forward twice and captures
    it once, so a change means the forward has state a replay would not keep (run 11226:
    measuring must never change a later request)."""

    def __init__(
        self, torch: Any, module: Any, *, immutable_buffers: frozenset[int] = frozenset()
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
        from cozy_runtime.internal.residency import Placeholder

        self.immutable = []
        for m in self.modules:
            for name, tensor in m._buffers.items():
                if tensor is None or id(tensor) not in immutable_buffers:
                    continue
                if isinstance(tensor, Placeholder):
                    saved = Placeholder(tensor.shape, tensor.dtype, tensor.device)
                    saved._varena_read = tensor._varena_read
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


def _no_gain(torch: Any, entry: _Recording, eager: _Timing) -> tuple[str, float | None]:
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
        end.synchronize()
        costs.append(max(host, float(start.elapsed_time(end))))
    replay, called = sum(costs) / len(costs), max(submit, device)
    if called - replay > abs(costs[0] - costs[1]):
        return "", replay
    return f"no gain: replay {replay:.3f} ms a call against eager {called:.3f} ms", replay


def _device_bound(submit: float, busy: float) -> str:
    """Why a signature never records: the device's work a call takes at least as long as the
    host's submission of it, so the device never waited on the host inside it and a graph
    saves nothing. "" when launch-bound."""
    if busy < submit:
        return ""
    return f"device-bound: {busy:.3f} ms of device work a call, {submit:.3f} ms to submit it"


#: the idle inputs' seed: any fixed one (the verify needs the same inputs twice, not these)
_SEED = 0


def _arguments(torch: Any, pending: Any, tensors: list[Any]) -> tuple[Any, Any]:
    """The call's arguments with `tensors` at its leaves' places (None keeps the leaf)."""
    leaves = [t if t is not None else v for t, v in zip(tensors, pending.leaves, strict=True)]
    args, kwargs = torch.utils._pytree.tree_unflatten(leaves, pending.spec)
    return args, kwargs


def _static(torch: Any, leaf: _Leaf) -> Any:
    """A static input of `leaf`'s layout, uninitialized: made inside a capture, from its pool."""
    span = 1 + sum((n - 1) * s for n, s in zip(leaf.shape, leaf.stride, strict=True))
    flat = torch.empty(span if all(leaf.shape) else 0, dtype=leaf.dtype, device=leaf.device)
    return flat.as_strided(leaf.shape, leaf.stride)


def _input(torch: Any, leaf: _Leaf, generator: Any) -> Any:
    """A static input of `leaf`'s layout, seeded: floats standard normal (zeros would hide a
    replay that drops work), bools fair coins, integers zero (a valid index into any table)."""
    span = 1 + sum((n - 1) * s for n, s in zip(leaf.shape, leaf.stride, strict=True))
    span = span if all(leaf.shape) else 0
    flat = torch.zeros(span, dtype=leaf.dtype, device=leaf.device)
    if flat.is_floating_point() or flat.is_complex():  # drawn wide, cast (fp8 draws nothing)
        wide = torch.complex64 if flat.is_complex() else torch.float32
        flat.copy_(torch.randn(span, generator=generator, device=leaf.device, dtype=wide))
    elif flat.dtype == torch.bool:
        flat.copy_(torch.randint(0, 2, (span,), generator=generator, device=leaf.device))
    return flat.as_strided(leaf.shape, leaf.stride)


#: per device, its sleep kernel's cycles a millisecond (`_busy`'s gate)
_CYCLES: dict[int, float] = {}
#: cycles of the calibrating sleep: ~2 ms at 2 GHz, far above the events' 0.5 us resolution
_CALIBRATE = 1 << 22


def _busy(torch: Any, run: Callable[[], object], gate_ms: float) -> float:
    """Device-busy ms of one `run` on the current stream: its launches queue behind a sleep as
    long as its last host submission (`gate_ms`), so the device runs them back to back. A host
    slower than the gate leaves the device waiting inside: the time is then an upper bound,
    which only ever keeps a region eager."""
    device, stream = torch.cuda.current_device(), torch.cuda.current_stream()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    if device not in _CYCLES:
        start.record(stream)
        torch.cuda._sleep(_CALIBRATE)
        end.record(stream)
        end.synchronize()
        _CYCLES[device] = _CALIBRATE / start.elapsed_time(end)
    torch.cuda._sleep(int(gate_ms * _CYCLES[device]))
    start.record(stream)
    run()
    end.record(stream)
    end.synchronize()
    return float(start.elapsed_time(end))


def _lease(torch: Any, placed: Any, binding: Any) -> Any:
    """A replay uses already-retained ordinary storage; it never reassigns module slots."""
    if placed is None or binding is None:
        return nullcontext()
    stream, held = torch.cuda.current_stream(), placed[0].graph_held(placed[1])
    if held is not None and held != stream:
        stream.wait_stream(held)
    return binding.replay_use(stream)


def _refused(recorder: _Recorder, placed: Any, why: str) -> None:
    """A root whose preparation the memory refused runs as written; the run says so once."""
    why = why.splitlines()[0][:300]
    recorder.gap = why
    if placed is not None:
        placed[0].note(f"{placed[1]}.graphs", f"eager {why}")
    say(f"refused:{why}", f"[graphs] a component runs as written: {why}")


def _failed(module: Any, recorder: _Recorder, key: tuple[object, ...], why: str) -> None:
    """An idle capture refused for good: the signature runs as written; the run says so once."""
    recorder.failed[key] = why
    recorder.gap = why
    say(f"graph:{id(recorder)}:{why}", f"[graphs] {_name(module)} remains eager: {why}")


def _changed(
    torch: Any,
    module: Any,
    recorder: _Recorder,
    key: tuple[object, ...],
    arguments: Any,
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
                ("a graph region records again every call", {"region": component, "argument": name})
            )
    else:
        names = ("declared keys", "weight addresses", "autocast")
        what = next(n for n, i in zip(names, range(-3, 0), strict=True) if last[i] != key[i])
    recorder.changes[what] = recorder.changes.get(what, 0) + 1


def _unplain(torch: Any, leaves: list[Any]) -> str:
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


def _clonable(torch: Any, out: Any) -> bool:
    leaves = torch.utils._pytree.tree_leaves(out)
    return any(isinstance(v, torch.Tensor) for v in leaves) and all(
        type(v) in (torch.Tensor, torch.nn.Parameter) or type(v) in _PLAIN for v in leaves
    )


def _signature(torch: Any, leaves: list[Any], spec: Any) -> tuple[object, ...]:
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
    "DECODERS",
    "IDLE",
    "Yielded",
    "capture",
    "capture_device",
    "capture_domain",
    "capture_idle",
    "describe",
    "install",
    "mark",
    "release",
    "say",
]
