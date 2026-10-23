"""CUDA graphs of whole components over retained virtual addresses.

Kernel choice is shipped. Runtime prepares memory, captures process-local addresses and
replays; it neither times competing paths nor compiles kernels. A component's region is its
root call: one graph per signature, keyed by its inputs' layout, the values of the attributes
its package declares steer its forward (`Model.__graph_keys__`), and the kernels serving
(`LANDED`). A placed component that streams (no step hold) runs its root as written and graphs
each declared block instead. Nothing records on a request's path: a signature's calls run as
written until the device is idle, when it records on seeded inputs of its layout if its eager
calls were launch-bound, its replay matches eager bit for bit and saves time, and recording
changed no state of the model (`capture_idle`); the next request replays. Every graph has its
own static buffers in the device's one capture pool (`domain`). A cached binding owns virtual
addresses only; its physical lease covers preparation through output copies, and a completion
event orders reuse across execution streams.
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
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

from cozy_runtime.author._capture import _capture_clock
from cozy_runtime.author._errors import Fault, fault, is_capacity
from cozy_runtime.author._model import LANDED, PLACED, WARNINGS, Building
from cozy_runtime.graphs import domain
from cozy_runtime.internal import attention_ulysses

BYPASS: contextvars.ContextVar[bool] = contextvars.ContextVar("cozy_graphs_bypass", default=False)
#: Set while a region warms up its capture: the regions under it run as written.
_WARMING = contextvars.ContextVar("cozy_graphs_warming", default=False)
#: The root of a streamed component running as written: its blocks are its regions meanwhile.
_STREAMING: contextvars.ContextVar[object | None] = contextvars.ContextVar("cozy_graphs_streaming")
#: The idle period recording on this thread (`capture_idle`): each region boundary checks it.
_IDLE: contextvars.ContextVar[_Idle | None] = contextvars.ContextVar("cozy_graphs_idle")
_PLAIN = (type(None), bool, int, float, str)
#: graph owners beside the regions (the LM engine's decode steps), each its (snapshot, release)
#: of a component root, registered by its module when a package imports it: a Runtime serving no
#: LM never imports Transformers for them (v7: 2.8 s of executor imports)
DECODERS: list[
    tuple[Callable[[Any], tuple[dict[str, int], list[Any], Any]], Callable[[Any], None]]
] = []
_SAID: set[str] = set()


def say(key: str, line: str) -> None:
    if key not in _SAID:
        _SAID.add(key)
        print(line, file=sys.stderr)


def capture(
    torch: Any, step: Callable[[], object], pool: Any, owner: domain.Domain
) -> tuple[Any, str]:
    """Record a caller-owned capture-safe computation in its already prepared pool.

    The caller owns purity, residency, static-data lifetime and all consuming streams. No
    allocation recovery or chunk selection is allowed inside capture. While it is open the
    memory knows (`Memory.capturing`): torch skips its release-and-retry during any capture on
    the device, so another thread's refused allocation makes room at once (A1).
    """
    graph = torch.cuda.CUDAGraph()
    current = torch.cuda.current_stream()
    stream, work = owner.control, owner.stream
    stream.wait_stream(current)
    stream.wait_stream(work)
    failure: BaseException | None = None
    with (
        owner.memory.capturing() if owner.memory is not None else nullcontext(),
        torch.utils._python_dispatch._disable_current_modes(),
        torch.cuda.stream(stream),
    ):
        try:
            graph.capture_begin(pool=pool, capture_error_mode="thread_local")
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
    current.wait_stream(stream)
    current.wait_stream(work)
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
    """One signature's graph and its static buffers.

    Its outputs are lent: a replay returns aliases of the graph's own output tensors. Their
    storage lives as long as any alias does, and the next replay runs only once none is alive
    (`ready`; until then a call runs as written) and after every reader's GPU work: the stream
    it replayed on, and each stream a reader declared (`record_stream`).
    """

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
        self.statics = [
            t for t in torch.utils._pytree.tree_leaves(outputs) if isinstance(t, torch.Tensor)
        ]
        self.served = [(b, i) for b, impls in served.items() for i in impls]
        #: each output storage's use count with no alias lent, read before the first replay
        self.held: list[int] | None = None
        #: bytes the last replay copied into its static inputs
        self.copied = 0
        #: other streams that read the last replay's lent outputs (`record_stream`)
        self.readers: dict[Any, None] = {}
        self.done: Any = None
        self.stream: Any = None
        self.fenced = True

    def ready(self) -> bool:
        """False while a lent alias of its outputs lives."""
        uses = [_uses(self.torch, t) for t in self.statics]
        if self.held is None:
            self.held = uses
        return uses == self.held

    def output(self) -> Any:
        """Caller-owned outputs for a component root's call, whose results leave the
        component: allocated before residency admission or graph launch."""
        torch = self.torch
        return torch.utils._pytree.tree_map(
            lambda v: torch.empty_like(v) if isinstance(v, torch.Tensor) else v, self.outputs
        )

    def __call__(self, leaves: list[Any], out: Any = None) -> Any:
        """Replays; returns lent aliases, or copies into `out` (a component root's call)."""
        torch, stream = self.torch, self.torch.cuda.current_stream()
        if self.done is None:
            self.done = torch.cuda.Event()
            self.done.record(stream)  # instantiate before any replay can be enqueued
        else:
            stream.wait_event(self.done)
        if self.stream is not None and self.stream != stream:
            stream.wait_stream(self.stream)  # readers of the last replay's lent outputs
        for reader in self.readers:
            stream.wait_stream(reader)
        self.readers.clear()
        self.stream, self.fenced = stream, False
        copied = 0
        try:
            for static, value in zip(self.inputs, leaves, strict=True):
                if static is not None and not _same(static, value):
                    static.copy_(value)
                    copied += static.nbytes
            self.graph.replay()
            for backend, implementation in self.served:
                attention_ulysses.record(backend, implementation)
            if out is None:
                return torch.utils._pytree.tree_map(
                    lambda v: v.detach() if isinstance(v, torch.Tensor) else v, self.outputs
                )
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
        for tensor in self.statics:
            found = _LENT.get(tensor.data_ptr())
            if found is not None and found[0]() is self:
                del _LENT[tensor.data_ptr()]
        self.inputs, self.outputs, self.statics = [], None, []
        self.graph.reset()
        self.graph = self.pool = None

    def idle(self) -> bool:
        if not self.fenced or (self.done is not None and not self.done.query()):
            return False
        for reader in self.readers:
            passed = self.torch.cuda.Event()
            passed.record(reader)
            if not passed.query():
                return False
        return True


def _uses(torch: Any, tensor: Any) -> int:
    """Tensors on its storage: ours, and every alias or view of one a caller still holds."""
    return int(torch._C._storage_Use_Count(tensor.untyped_storage()._cdata))


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
    "held",
    "capture_ms",
    "call_ms",
    "prepare_ms",
)


class _Recorder:
    def __init__(self) -> None:
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
        #: per signature its idle measure: host submission and device-busy ms a call
        self.bound: dict[tuple[object, ...], tuple[float, float]] = {}
        self.binding: Any = None
        self.replays = 0
        self.attempts = 0
        self.capture_attempts = 0
        self.captures = 0
        self.retired = 0
        self.copy_bytes = 0
        #: calls run as written because a lent output was still alive
        self.held = 0
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

    def forget(self, torch: Any) -> bool:
        if any(not entry.idle() for entry in self.recordings.values()):
            return False
        self.drop()
        return True

    def drop(self) -> None:
        self.retired += len(self.recordings)
        for entry in self.recordings.values():
            entry.close()
        self.recordings.clear()
        if self.binding is not None:
            self.binding.close()
            self.binding = None
        self.queued.clear()
        self.failed.clear()
        self.waiting.clear()
        self.landed = LANDED[0]


_RECORDERS: weakref.WeakKeyDictionary[Any, _Recorder] = weakref.WeakKeyDictionary()
#: each live graph output's address -> (its recording, its index among the outputs)
_LENT: dict[int, tuple[weakref.ref[_Recording], int]] = {}
_REGIONS: weakref.WeakKeyDictionary[Any, tuple[Any, str]] = weakref.WeakKeyDictionary()
_GAPS: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()
#: each component root's declared control-flow attributes (`Model.__graph_keys__`)
_KEYS: weakref.WeakKeyDictionary[Any, tuple[str, ...]] = weakref.WeakKeyDictionary()
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
    free room) waits for the next period. `check`, at each region boundary in a job, yields to a
    command that arrived; the longest stretch between checks is the longest a command waited."""

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
        token = _IDLE.set(self)
        try:
            while True:
                with self.lock:
                    if not self.jobs:
                        return True
                    job = self.jobs.popleft()
                try:
                    self.check()
                    if not job():
                        later.append(job)
                except Yielded:
                    self.yields += 1
                    with self.lock:
                        self.jobs.appendleft(job)
                    return False
                except Exception as exc:  # one job's failure never ends the period (v16)
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
    (`__graph_keys__`: component -> attribute paths); a placed component's declared blocks are
    regions while it streams."""
    skipped = set(getattr(type(model), "__uncaptured__", ()))
    declared: Mapping[str, tuple[str, ...]] = getattr(type(model), "__graph_keys__", {})
    installed = []
    for name, root in roots.items():
        if name in skipped or not isinstance(root, torch.nn.Module):
            continue
        _KEYS[root] = tuple(declared.get(name, ()))
        placed = PLACED.get(root)
        blocks = placed[0].graph_segments(placed[1]) if placed is not None else ()
        for path, module in (("", root), *((p, root.get_submodule(p)) for p in blocks)):
            _REGIONS[module] = (weakref.ref(root), path)
            if not getattr(type(module), "_cozy_graphed", False):
                module.__class__ = _graphed(torch, type(module))
            installed.append(f"{name}.{path}" if path else name)
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


def _placed(module: Any) -> tuple[Any, str] | None:
    region = _REGIONS.get(module)
    return PLACED.get(region[0]()) if region is not None else PLACED.get(module)


def _shed(torch: Any, owner: domain.Domain, _want: int = 0) -> int:
    """Varena's ask: every graph on the device goes now, with their shared pool. It never asks
    a held pool, so no call reads, captures or replays a recording of it; a call that took its
    hold after varena decided finds it held, and nothing goes. Only the pool's own lock is
    taken: a region call may hold the domain while its bind waits on varena (review 7, K1)."""
    with owner.pool_lock:
        if owner.held:
            return 0
        nbytes = owner.capture_bytes()
        for recorder in list(owner.recorders):
            recorder.drop()
        owner.drop_captures()
        return nbytes


def release(torch: Any, root: Any) -> None:
    """A generation's explicit teardown: finish its output copies before releasing addresses."""
    modules = [module for module, (owner, _) in _REGIONS.items() if owner() is root]
    shared: set[domain.Domain] = set()
    for module in modules:
        recorder = _RECORDERS.pop(module, None)
        if recorder is not None:
            if recorder.domain is not None:
                shared.add(recorder.domain)
                recorder.domain.recorders.discard(recorder)
            with recorder.lock:
                for entry in recorder.recordings.values():
                    if not entry.fenced:
                        entry.stream.synchronize()
                        entry.fenced = True
                    elif entry.done is not None:
                        entry.done.synchronize()
                recorder.forget(torch)
        _REGIONS.pop(module, None)
        _GAPS.pop(module, None)
    for owner in shared:
        if not owner.recorders:  # the device's last component: its pool goes with it
            owner.drop_captures()
            owner.close_cache()
    domain.release(root)


def mark(root: Any) -> None:
    """A new attempt of `root`'s component opens: its receipts count from here."""
    for module, (owner, _) in list(_REGIONS.items()):
        if owner() is root and (recorder := _RECORDERS.get(module)) is not None:
            with recorder.lock:
                recorder.base = recorder.snapshot()


def capture_domain(torch: Any, root: Any, memory: Any = None) -> domain.Domain:
    """The component owns hidden library scratch independently of its disposable graphs."""
    return domain.acquire(torch, root, memory)


def capture_device(torch: Any, memory: Any) -> domain.Domain:
    """A placed component's graphs share the device's domain with its other components'."""
    return domain.device(torch, memory)


def describe(root: Any) -> dict[str, Any]:
    """One final-reply snapshot, never an aggregate scan on the replay hot path.

    Counts are this attempt's, since the Placement opened it (`mark`); an unplaced component's
    are its generation's. Region call counts are not a
    percentage of whole-model work: eager parents may enclose captured children. Pool bytes
    count each live pool once (a device's shared pool under the first component it served);
    driver/context and pending native frees remain ledger facts.
    """
    pools: dict[tuple[int, int], Any] = {}
    domains: dict[int, domain.Domain] = {}
    answer: dict[str, Any] = dict.fromkeys(
        (
            "declared_regions",
            "active_graphs",
            "region_calls",
            "replays",
            "capture_attempts",
            "captures",
            "retired_graphs",
            "replay_copy_bytes",
            "held_calls",
            "capture_ms",
        ),
        0,
    )
    captured, gaps, timed = [], {}, {}
    for module, (owner, path) in list(_REGIONS.items()):
        if owner() is not root:
            continue
        answer["declared_regions"] += 1
        if module in _GAPS:
            gaps[path] = _GAPS[module]
        recorder = _RECORDERS.get(module)
        if recorder is None:
            continue
        with recorder.lock:
            since, base = recorder.since, recorder.base
            answer["active_graphs"] += len(recorder.recordings)
            answer["region_calls"] += since("attempts")
            answer["replays"] += since("replays")
            answer["capture_attempts"] += since("capture_attempts")
            answer["captures"] += since("captures")
            answer["retired_graphs"] += since("retired")
            answer["replay_copy_bytes"] += since("copy_bytes")
            answer["held_calls"] += since("held")
            answer["capture_ms"] += round(since("capture_ms"), 3)
            if recorder.recordings:
                captured.append(path)
            if since("attempts"):
                before = base.get("changes", {})
                timed[path] = {
                    "capture_attempts": since("capture_attempts"),
                    "captures": since("captures"),
                    "capture_ms": round(since("capture_ms"), 3),
                    "graph": _receipt(recorder),
                    "key_changes": {
                        what: n - before.get(what, 0)
                        for what, n in recorder.changes.items()
                        if n > before.get(what, 0)
                    },
                    "call_ms": round(since("call_ms"), 3),
                    "prepare_ms": round(since("prepare_ms"), 3),
                    "held": since("held"),
                    "eager": recorder.eager.document(base.get("eager", (0, 0, 0.0, 0.0))),
                    "replay": recorder.replayed.document(base.get("replayed", (0, 0, 0.0, 0.0))),
                }
            # A device's shared pool and domain count once, under the first component served.
            first = recorder.domain is None or not recorder.domain.served
            if not first and recorder.domain is not None:
                first = recorder.domain.served[0]() is root
            if first:
                pools.update((entry.pool.id, entry.pool) for entry in recorder.recordings.values())
                if recorder.domain is not None:
                    domains[id(recorder.domain)] = recorder.domain
    shared = domain.existing(root)
    if shared is not None:
        domains[id(shared)] = shared
    for snapshot, _ in DECODERS:
        decode, decode_pools, decode_domain = snapshot(root)
        pools.update((pool.id, pool) for pool in decode_pools)
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
        captured_regions=sorted(captured),
        regions=dict(sorted(timed.items())),
        eager_gaps=gaps,
        graph_pools=len(pools),
        graph_pool_bytes=sum(
            segment["total_size"]
            for pool in pools.values()
            for segment in pool.snapshot(include_traces=False)
        ),
        domains=len(domains),
        domain_workspace_bytes=sum(d.nbytes() for d in domains.values()),
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
            {"submit_ms": round(submit, 3), "busy_ms": round(busy, 3)}
            for submit, busy in recorder.bound.values()
        ],
        "saving_ms": None if saving is None else round(saving, 3),
        "payback_calls": math.ceil(cost / saving) if saving and saving > 0 else None,
        "eager": sorted({*recorder.failed.values(), *recorder.judged.values()}),
    }


def _unprepared(exc: BaseException) -> bool:
    return is_capacity(exc) or getattr(exc, "code", None) == "STALE_BINDING"


def _dispatch(torch: Any, module: Any, call: Callable[..., Any], args: Any, kwargs: Any) -> Any:
    if (idle := _IDLE.get(None)) is not None:
        idle.check()
    region = _REGIONS.get(module)
    if region is not None and not region[1]:
        _watch(module)
    if (
        region is None  # a copy of a region's module: no region of its own
        or BYPASS.get()
        or _WARMING.get()
        or not torch.cuda.is_available()
        or torch.cuda.is_current_stream_capturing()
    ):
        return call(module, *args, **kwargs)
    placed = _placed(module)
    owner, path = region
    if path and _STREAMING.get(None) is not owner():
        return call(module, *args, **kwargs)  # its root's graph or eager run holds it
    if (
        not path
        and placed is not None
        and placed[0].graph_held(placed[1]) is None
        and placed[0].graph_segments(placed[1])
    ):  # it streams: it runs as written, its blocks replay
        token = _STREAMING.set(module)
        try:
            return call(module, *args, **kwargs)
        finally:
            _STREAMING.reset(token)
    scope = placed[0].graph_pass(placed[1], path) if placed is not None else nullcontext()
    began = time.perf_counter()
    try:
        with scope:
            return _region_call(torch, module, call, args, kwargs, placed)
    finally:
        if (recorder := _RECORDERS.get(module)) is not None:
            recorder.call_ms += (time.perf_counter() - began) * 1e3


def _watch(root: Any) -> None:
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
    keys = {path.split(".")[0] for path in _KEYS.get(root, ())}
    changed = {n for n in first.keys() | now.keys() if first.get(n) != now.get(n)} - keys
    if changed:
        placed = PLACED.get(root)
        name = placed[1] if placed is not None else type(root).__name__
        UNDECLARED.setdefault(name, set()).update(changed)
        say(
            f"undeclared:{id(root)}",
            f"[graphs] component {name}'s root attributes {sorted(changed)} changed between its "
            "first two calls: declare each that steers its forward in Model.__graph_keys__",
        )
        WARNINGS.append(
            ("a component's undeclared state changed between calls", {"component": name})
        )


def _region_call(
    torch: Any, module: Any, call: Callable[..., Any], args: Any, kwargs: Any, placed: Any
) -> Any:
    leaves, spec = torch.utils._pytree.tree_flatten((args, kwargs))
    why = (
        "with gradients"
        if torch.is_grad_enabled()
        else "under activation capture"
        if _capture_clock.get() is not None
        else "under custom tensor dispatch"
        if torch.utils._python_dispatch._get_current_dispatch_mode() is not None
        else "across GPUs"
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        else _unplain(torch, leaves)
    )
    if why:
        _GAPS[module] = why
        if placed is not None:
            placed[0].note(f"{placed[1]}.graphs", f"eager {why}")
        return call(module, *args, **kwargs)
    recorder = _RECORDERS.get(module)
    if recorder is None:
        recorder = _RECORDERS.setdefault(module, _Recorder())
    levels = placed[0].graph_levels() if placed is not None else ()
    addresses = (
        ()
        if placed is not None
        else tuple(
            (value.data_ptr(), tuple(value.shape), value.dtype, value.stride())
            for value in (*module.parameters(), *module.buffers())
        )
    )
    autocast = torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda")
    root = _REGIONS[module][0]()
    declared = tuple(functools.reduce(getattr, p.split("."), root) for p in _KEYS.get(root, ()))
    key = (*_signature(torch, leaves, spec), declared, levels, addresses, autocast)
    if recorder.last is not None and recorder.last != key:
        _changed(torch, module, recorder, key, (args, kwargs), len(leaves))
    recorder.last = key
    if recorder.domain is None:
        recorder.domain = (
            placed[0].graph_domain(placed[1])
            if placed is not None
            else capture_domain(torch, _REGIONS[module][0]())
        )
        recorder.domain.recorders.add(recorder)

    def eager() -> Any:
        return recorder.eager.run(torch, lambda: call(module, *args, **kwargs))

    with recorder.domain.using(), recorder.lock:
        if recorder.landed != LANDED[0] and not recorder.forget(torch):
            return eager()
        recorder.attempts += 1
        attempt = placed[0].attempts if placed is not None else recorder.attempts
        entry = recorder.recordings.get(key)
        if entry is None:  # as written; an idle period records it
            pending = recorder.queued.get(key)
            if pending is None and key not in recorder.failed and key not in recorder.judged:
                pending = recorder.queued[key] = _Pending(
                    torch,
                    weakref.ref(module),
                    call,
                    key,
                    spec,
                    [
                        _Leaf(tuple(v.shape), v.stride(), v.dtype, v.device)
                        if isinstance(v, torch.Tensor)
                        else v
                        for v in leaves
                    ],
                )
                IDLE.add(pending)
            return eager()
        if recorder.waiting.get(key) == attempt:
            return eager()
        if placed is not None:
            if (
                recorder.binding is not None
                and not recorder.binding.valid()
                and not recorder.forget(torch)
            ):
                return eager()
            if recorder.binding is None:
                recorder.binding = placed[0].retain(placed[1], (_REGIONS[module][1],))
        # The pool is held from reading its recording through preparation and the replay until
        # the device completes it: varena never shrinks the graph a call is using.
        with recorder.domain.holding():
            result = _replay(torch, module, leaves, placed, recorder, key, entry)
        return eager() if result is _EAGER else result


#: `_replay`'s answer when the call runs as written
_EAGER = object()


def _replay(
    torch: Any,
    module: Any,
    leaves: list[Any],
    placed: Any,
    recorder: _Recorder,
    key: tuple[object, ...],
    entry: _Recording,
) -> Any:
    """The call replayed, or `_EAGER`."""
    owner = recorder.domain
    assert owner is not None
    attempt = placed[0].attempts if placed is not None else recorder.attempts
    binding = recorder.binding
    if not entry.ready():
        recorder.held += 1
        return _EAGER
    # A component root's results leave the component (a package's loop may keep them past its
    # next call): they are copied out, allocated before the physical lease so admission sees
    # their segments. Regions inside the call lend theirs.
    try:
        out = entry.output() if _REGIONS[module][0]() is module else None
    except Exception as exc:
        if not _unprepared(exc):
            raise
        recorder.waiting[key] = attempt
        return _EAGER
    try:
        began = time.perf_counter()
        lease = _lease(torch, placed, binding)
        lease.__enter__()
        recorder.prepare_ms += (time.perf_counter() - began) * 1e3
    except Exception as exc:
        if not _unprepared(exc):
            raise
        del out
        recorder.waiting[key] = attempt
        _refused(module, placed, f"its weights cannot all be resident at once ({exc})")
        return _EAGER
    try:
        result = recorder.replayed.run(torch, lambda: entry(leaves, out))
        _GAPS.pop(module, None)
        recorder.replays += 1
        recorder.copy_bytes += entry.copied
        return result
    finally:
        lease.__exit__(None, None, None)


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
    placed, owner = _placed(module), recorder.domain
    assert owner is not None
    device = next(leaf.device for leaf in pending.leaves if isinstance(leaf, _Leaf))
    with torch.cuda.device(device), owner.using(), recorder.lock:
        if recorder.landed != LANDED[0] or (
            recorder.binding is not None and not recorder.binding.valid()
        ):
            return recorder.forget(torch)  # its queue went with its graphs
        if placed is not None:
            if recorder.binding is None:
                recorder.binding = placed[0].retain(placed[1], (_REGIONS[module][1],))
            if not placed[0].graph_resident(recorder.binding):
                return False
        idle = placed[0].graph_idle() if placed is not None else nullcontext()
        try:
            with idle, owner.holding():
                done = _record(torch, module, pending, recorder, placed, submit)
        except Yielded:
            raise
        except Exception as exc:  # the signature runs as written; the period goes on
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
    current, stream, state = torch.cuda.current_stream(), None, None
    recorder.capture_attempts += 1
    began = time.perf_counter()
    try:
        if why := _hooked(module):
            recorder.judged[key] = why
            return True
        if placed is not None and not placed[0].graph_accounted():
            _failed(module, recorder, key, "driver/context memory is unmeasured")
            return True
        scope = (
            placed[0].graph_pass(placed[1], _REGIONS[module][1])
            if placed is not None
            else nullcontext()
        )
        # Discovery owns the complete range set and then the admitted free headroom.
        lease = recorder.binding.prepare(current) if recorder.binding is not None else nullcontext()
        with scope, lease:
            owner.initialize()
            stream = owner.stream
            stream.wait_stream(current)
            # Library handles cache scratch by stream. Initialize under domain ownership before
            # any region pool becomes active; every executable sharing that scratch is serialized.
            owner.prime_blas()
            pool = owner.capture_pool()
            allocating = (
                placed[0].graph_allocating(pool)
                if placed is not None
                else torch.cuda.use_mem_pool(pool)
            )
            state = _State(torch, module)
            with torch.cuda.stream(stream):
                with allocating:
                    seeded = torch.Generator(device=current.device).manual_seed(_SEED)
                    inputs = [
                        _input(torch, v, seeded) if isinstance(v, _Leaf) else None
                        for v in pending.leaves
                    ]
                    static = [
                        s if s is not None else v
                        for s, v in zip(inputs, pending.leaves, strict=True)
                    ]
                    cargs, ckwargs = torch.utils._pytree.tree_unflatten(static, pending.spec)
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
                recorder.bound[key] = (submit, busy)
                if why := _device_bound(submit, busy):
                    recorder.judged[key] = why
                    return True
                capturing = (
                    placed[0].graph_allocating(pool, route=False)
                    if placed is not None
                    else nullcontext()
                )
                with capturing:
                    outputs: list[Any] = []
                    graph, why = capture(
                        torch, lambda: outputs.append(call(module, *cargs, **ckwargs)), pool, owner
                    )
                if graph is None:
                    _failed(module, recorder, key, why)
                    return True
                if changed := state.restore():
                    recorder.judged[key] = f"recording it changed the model's state: {changed}"
                    graph.reset()
                    return True
                if not _clonable(torch, outputs[0]):
                    _failed(module, recorder, key, "captured output is not a tensor tree")
                    return True
                graph.replay()  # served only when it reproduces the eager run bit for bit
                pairs = zip(*map(torch.utils._pytree.tree_leaves, (outputs[0], eager)), strict=True)
                if not all(torch.equal(a, b) for a, b in pairs if isinstance(a, torch.Tensor)):
                    why = "its replay differs from the eager run on the same inputs"
                    _failed(module, recorder, key, why)
                    WARNINGS.append(("a CUDA graph was not used", {"why": why}))
                    return True
                entry = _Recording(torch, graph, pool, inputs, outputs[0], key, served, owner)
                root = _REGIONS[module][0]() is module
                if why := _no_gain(torch, entry, root, recorder.eager.wall_ms()):
                    recorder.judged[key] = why
                    entry.close()
                    return True
        recorder.recordings[key] = entry
        recorder.captures += 1
        recorder.capture_each.append(round((time.perf_counter() - began) * 1e3, 3))
        for index, tensor in enumerate(entry.statics):
            _LENT[tensor.data_ptr()] = (weakref.ref(entry), index)
        joined = owner.serve(_REGIONS[module][0]())
        if placed is not None and (joined or owner.cache is None):
            # One cache for the device's pool, naming every component recorded into it (its next
            # use is the nearest of theirs): registered again when one joins.
            if owner.cache is not None:
                owner.cache.close()
            served = [root for root in (ref() for ref in owner.served) if root is not None]
            owner.cache = placed[0].cache("graphs", functools.partial(_shed, torch, owner), served)
        if owner.cache is not None:
            owner.cache.cached(owner.capture_bytes())
        return True
    except Exception as exc:
        if is_capacity(exc) or _unprepared(exc) or isinstance(exc, Building):
            return False
        if not isinstance(exc, domain.Unavailable):
            raise
        _failed(module, recorder, key, f"{type(exc).__name__}: {exc}".splitlines()[0][:300])
        return True
    finally:
        if state is not None:
            state.restore()
        recorder.capture_ms += (time.perf_counter() - began) * 1e3
        if stream is not None:
            # Warmup can queue reads before refusing an allocation. Join even the failure
            # path before the lease records its completion event.
            current.wait_stream(stream)


def _hooked(module: Any) -> str:
    """Why a component's forward is no graph: a package hook on one of its modules runs Python
    at every call, which a replay would skip (varena's binds and Runtime's own are served)."""
    for sub in module.modules():
        for hooks in (sub._forward_pre_hooks, sub._forward_hooks):
            for hook in hooks.values():
                if not getattr(hook, "__module__", "").startswith(("tensorfs.", "cozy_runtime.")):
                    return f"a package forward hook on {type(sub).__name__}"
    return ""


class _State:
    """What an idle recording must leave as it found it: the component's modules' plain
    attributes, its buffers (contents and version counters) and the RNG. Package
    code may keep a counter or a table in them; recording runs the forward twice and captures
    it once, so a change means the forward has state a replay would not keep (run 11226:
    measuring must never change a later request)."""

    def __init__(self, torch: Any, module: Any) -> None:
        self.torch = torch
        self.modules = list(module.modules())
        self.plain = [{n: v for n, v in vars(m).items() if type(v) in _PLAIN} for m in self.modules]
        self.buffers = [
            (name, tensor, tensor._version, tensor.clone())
            for m in self.modules
            for name, tensor in m._buffers.items()
            if tensor is not None
        ]
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
            for name, tensor, version, saved in self.buffers:
                if tensor._version != version:
                    changed.add(name)
                    tensor.copy_(saved)
            self.torch.random.set_rng_state(self.rng[0])
            self.torch.cuda.set_rng_state(self.rng[1])
            self.changed = ", ".join(sorted(changed))
        return self.changed


def _no_gain(torch: Any, entry: _Recording, root: bool, eager: float | None) -> str:
    """Why a verified graph is dropped: a replay as a request makes it (its inputs copied in, a
    root's outputs copied out), timed twice at idle, saves no more than the two differ by
    against its eager calls' device time a call (v14/v15: CLIP's tiny regions recorded, then
    replayed no faster: the copies and the launch were their whole cost). "" when it gains,
    or with no eager wall measured."""
    leaves = [t.clone() if t is not None else None for t in entry.inputs]
    walls = []
    for _ in range(2):
        out = entry.output() if root else None
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        entry(leaves, out)
        end.record()
        end.synchronize()
        walls.append(float(start.elapsed_time(end)))
    replay = sum(walls) / len(walls)
    if eager is None or eager - replay > abs(walls[0] - walls[1]):
        return ""
    return f"no gain: replay {replay:.3f} ms a call against eager {eager:.3f} ms"


def _device_bound(submit: float, busy: float) -> str:
    """Why a signature never records: the device's work a call takes at least as long as the
    host's submission of it, so the device never waited on the host inside it and a graph
    saves nothing. "" when launch-bound."""
    if busy < submit:
        return ""
    return f"device-bound: {busy:.3f} ms of device work a call, {submit:.3f} ms to submit it"


#: the idle inputs' seed: any fixed one (the verify needs the same inputs twice, not these)
_SEED = 0


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


def _lent(torch: Any, leaf: Any) -> _Recording | None:
    """The live recording whose static output `leaf` is an alias of."""
    found = _LENT.get(leaf.data_ptr()) if isinstance(leaf, torch.Tensor) else None
    entry = found[0]() if found is not None else None
    if found is None or entry is None or entry.graph is None:
        return None
    return entry if _same(entry.statics[found[1]], leaf) else None


def record_stream(torch: Any, tensor: Any, stream: Any) -> None:
    """`tensor`, a graph's lent output, is read on `stream`: the graph's next replay waits for
    the work queued there (torch's `Tensor.record_stream`, for a lent output's reuse)."""
    found = _lent(torch, tensor)
    if found is not None:
        found.readers[stream] = None


def _lease(torch: Any, placed: Any, binding: Any) -> Any:
    """A replay's residency: its region's blocks prepared around it, unless its component's
    blocks are held for the plan's step (`Placement.using`; v7: a CLIP replay bound and unbound
    its whole block, ~6 ms a call)."""
    if placed is None or binding is None:
        return nullcontext()
    stream, held = torch.cuda.current_stream(), placed[0].graph_held(placed[1])
    if held is None:
        return binding.prepare(stream)
    if held != stream:
        stream.wait_stream(held)
    return nullcontext()


def _refused(module: Any, placed: Any, why: str) -> None:
    """A region whose preparation the memory refused runs as written this request, its finer
    regions replaying under it; the run says so once."""
    why = why.splitlines()[0][:300]
    _GAPS[module] = why
    if placed is not None:
        placed[0].note(f"{placed[1]}.graphs", f"eager {why}")
    say(f"refused:{why}", f"[graphs] a region runs as written, its finer regions replay: {why}")


def _failed(module: Any, recorder: _Recorder, key: tuple[object, ...], why: str) -> None:
    """An idle capture refused for good: the signature runs as written; the run says so once."""
    recorder.failed[key] = why
    _GAPS[module] = why
    say(f"graph:{id(recorder)}:{why}", f"[graphs] pure region remains eager: {why}")


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
            path = _REGIONS[module][1] or "<root>"
            say(
                f"scalar:{id(recorder)}:{name}",
                f"[graphs] region {path} records again every call: its argument {name} is a "
                "Python number that changes between calls (pass it as a tensor)",
            )
            WARNINGS.append(
                ("a graph region records again every call", {"region": path, "argument": name})
            )
    else:
        names = ("declared keys", "cut levels", "weight addresses", "autocast")
        what = next(n for n, i in zip(names, range(-4, 0), strict=True) if last[i] != key[i])
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
    "record_stream",
    "release",
    "say",
]
