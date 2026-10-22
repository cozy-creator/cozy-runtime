"""CUDA graphs of declared-pure regions over retained virtual addresses.

Kernel choice is shipped. Runtime prepares memory, captures process-local addresses and
replays; it neither times competing paths nor compiles kernels. Unsupported Python stays
eager around safe regions. A placed component's regions are its pure subtrees that hold
declared blocks, coarsest first (`regions.segments`): static inputs and outputs sit at those
boundaries only, and a boundary between two graphs copies nothing (`_Recording`). Every graph
has its own static buffers in the component's one capture pool (`domain`). A cached binding
owns virtual addresses only; its physical lease covers preparation through output copies, and
a completion event orders reuse across execution streams.
"""

from __future__ import annotations

import contextvars
import enum
import functools
import sys
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import Any, cast

from cozy_runtime.author._capture import _capture_clock
from cozy_runtime.author._errors import Fault, fault, is_capacity
from cozy_runtime.author._model import LANDED, PLACED, WARNINGS, Building
from cozy_runtime.graphs import domain
from cozy_runtime.graphs.regions import WRAPPERS, contract, reads, regions, safe, segments
from cozy_runtime.internal import attention_ulysses

BYPASS: contextvars.ContextVar[bool] = contextvars.ContextVar("cozy_graphs_bypass", default=False)
#: Set while a region warms up its capture: the regions under it run as written.
_WARMING = contextvars.ContextVar("cozy_graphs_warming", default=False)
#: The component root call in flight: a region's purity and structure are read once per pass.
_PASS: contextvars.ContextVar[object | None] = contextvars.ContextVar("cozy_graphs_pass")
_PLAIN = (type(None), bool, int, float, str)
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
    allocation recovery or chunk selection is allowed inside capture.
    """
    graph = torch.cuda.CUDAGraph()
    current = torch.cuda.current_stream()
    stream, work = owner.control, owner.stream
    stream.wait_stream(current)
    stream.wait_stream(work)
    failure: BaseException | None = None
    with torch.utils._python_dispatch._disable_current_modes(), torch.cuda.stream(stream):
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
    if isinstance(failure, Building):
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
    it replayed on, and each stream a reader declared (`record_stream`). A static input that is
    another recording's output (`chained`, a segment boundary) is read in place and never
    written, so a chain of segments copies nothing between them.
    """

    def __init__(
        self,
        torch: Any,
        graph: Any,
        pool: Any,
        inputs: list[Any],
        chained: list[_Recording | None],
        outputs: Any,
        layout: object,
        served: Any,
        owner: domain.Domain,
    ) -> None:
        self.torch, self.graph, self.pool, self.owner = torch, graph, pool, owner
        self.inputs, self.chained, self.outputs, self.layout = inputs, chained, outputs, layout
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

    def ready(self, leaves: list[Any]) -> str:
        """ "" to replay; "held" while a lent alias lives or a chained input is not its producer's
        output; "stale" once that producer is gone."""
        uses = [_uses(self.torch, t) for t in self.statics]
        if self.held is None:
            self.held = uses
        elif uses != self.held:
            return "held"
        for static, value, producer in zip(self.inputs, leaves, self.chained, strict=True):
            if producer is not None and not _same(static, value):
                return "stale" if producer.graph is None else "held"
        return ""

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
        #: the first call's wall time apart (a cold run's first eager call: autotune, library
        #: initialization, page-in)
        self.first_ms: float | None = None
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
            self.device_ms += start.elapsed_time(end)
            self.timed += 1
            self.free.append((start, end))

    def document(self) -> dict[str, Any]:
        self.settle()
        return {
            "calls": self.calls,
            "first_ms": round(self.first_ms or 0.0, 3),
            "host_ms": round(self.host_ms, 3),
            "device_ms": round(self.device_ms, 3),
            "device_timed": self.timed,
        }


class _Recorder:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.recordings: dict[tuple[object, ...], _Recording] = {}
        self.seen: set[tuple[object, ...]] = set()
        #: per key, the static buffers its graph would hold (measured on its first, eager call)
        self.statics: dict[tuple[object, ...], int] = {}
        self.failed: dict[tuple[object, ...], str] = {}
        self.waiting: dict[tuple[object, ...], int] = {}
        self.binding: Any = None
        self.replays = 0
        self.attempts = 0
        self.capture_attempts = 0
        self.captures = 0
        self.retired = 0
        self.copy_bytes = 0
        #: calls run as written because a lent output was still alive or a chained input was not
        #: its producer's output
        self.held = 0
        self.landed = LANDED[0]
        self.domain: domain.Domain | None = None
        #: host time preparing and capturing (warmup included), each capture's apart, and each
        #: way it ran
        self.capture_ms = 0.0
        self.capture_each: list[float] = []
        #: host time of every call through the region (its checks, preparation, replay or eager
        #: run) and of its replays' preparation: what a call costs beside the work it runs
        self.call_ms = self.prepare_ms = 0.0
        #: its previous call's key, and what changed in its key between calls (a key that changes
        #: every call records again every call)
        self.last: tuple[object, ...] | None = None
        self.changes: dict[str, int] = {}
        self.eager, self.replayed = _Timing(), _Timing()

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
        self.seen.clear()
        self.statics.clear()
        self.failed.clear()
        self.waiting.clear()
        self.landed = LANDED[0]


_RECORDERS: weakref.WeakKeyDictionary[Any, _Recorder] = weakref.WeakKeyDictionary()
#: each live graph output's address -> (its recording, its index among the outputs)
_LENT: dict[int, tuple[weakref.ref[_Recording], int]] = {}
_REGIONS: weakref.WeakKeyDictionary[Any, tuple[Any, str]] = weakref.WeakKeyDictionary()
_DECLARED: weakref.WeakSet[Any] = weakref.WeakSet()
_GAPS: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()
#: each region's purity and structure, with the root call (pass) they were read in
_CHECKED: weakref.WeakKeyDictionary[Any, tuple[object, bool, tuple[object, ...]]] = (
    weakref.WeakKeyDictionary()
)
_CLASSES: dict[type, type] = {}


def install(torch: Any, model: object, roots: Mapping[str, Any]) -> list[str]:
    """Install shipped pure regions, including the blocks of a streamed component."""
    skipped = set(getattr(type(model), "__uncaptured__", ()))
    installed = []
    for name, root in roots.items():
        if name in skipped or not isinstance(root, torch.nn.Module):
            continue
        placed = PLACED.get(root)
        blocks = placed[0].graph_segments(placed[1]) if placed is not None else ()
        candidates = segments(torch, root, blocks) if blocks else regions(torch, root)
        for path, module in candidates:
            _DECLARED.add(module)
            _REGIONS[module] = (weakref.ref(root), path)
            if not getattr(type(module), "_cozy_graphed", False):
                module.__class__ = _graphed(torch, type(module))
            installed.append(f"{name}.{path}" if path else name)
        if root not in _REGIONS:
            # The root reports its explicit eager gap; eligible children still replay.
            _REGIONS[root] = (weakref.ref(root), "")
            if not getattr(type(root), "_cozy_graphed", False):
                root.__class__ = _graphed(torch, type(root))
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
        WRAPPERS[made] = cls
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
        owner.unhold()
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
        _DECLARED.discard(module)
        _GAPS.pop(module, None)
        _CHECKED.pop(module, None)
    for owner in shared:
        owner.unhold(root)
        if not owner.recorders:  # the device's last component: its pool goes with it
            owner.drop_captures()
            owner.close_cache()
    domain.release(root)


def capture_domain(torch: Any, root: Any, memory: Any = None) -> domain.Domain:
    """The component owns hidden library scratch independently of its disposable graphs."""
    return domain.acquire(torch, root, memory)


def capture_device(torch: Any, memory: Any) -> domain.Domain:
    """A placed component's graphs share the device's domain with its other components'."""
    return domain.device(torch, memory)


def describe(root: Any) -> dict[str, Any]:
    """One final-reply snapshot, never an aggregate scan on the replay hot path.

    Counts are cumulative for this component generation. Region call counts are not a
    percentage of whole-model work: eager parents may enclose captured children. Pool bytes
    count each live pool once (a device's shared pool under the first component it served);
    driver/context and pending native frees remain ledger facts.
    """
    from cozy_runtime import lm

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
        answer["declared_regions"] += module in _DECLARED
        if module in _GAPS:
            gaps[path] = _GAPS[module]
        recorder = _RECORDERS.get(module)
        if recorder is None:
            continue
        with recorder.lock:
            answer["active_graphs"] += len(recorder.recordings)
            answer["region_calls"] += recorder.attempts
            answer["replays"] += recorder.replays
            answer["capture_attempts"] += recorder.capture_attempts
            answer["captures"] += recorder.captures
            answer["retired_graphs"] += recorder.retired
            answer["replay_copy_bytes"] += recorder.copy_bytes
            answer["held_calls"] += recorder.held
            answer["capture_ms"] += round(recorder.capture_ms, 3)
            if recorder.recordings:
                captured.append(path)
            if recorder.attempts:
                timed[path] = {
                    "capture_attempts": recorder.capture_attempts,
                    "captures": recorder.captures,
                    "capture_ms": round(recorder.capture_ms, 3),
                    "capture_each_ms": list(recorder.capture_each),
                    "key_changes": dict(recorder.changes),
                    "call_ms": round(recorder.call_ms, 3),
                    "prepare_ms": round(recorder.prepare_ms, 3),
                    "held": recorder.held,
                    "eager": recorder.eager.document(),
                    "replay": recorder.replayed.document(),
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
    decode, decode_pools, decode_domain = lm._graph_snapshot(root)
    pools.update((pool.id, pool) for pool in decode_pools)
    if decode_domain is not None:
        domains[id(decode_domain)] = decode_domain
    if decode:
        answer["decode"] = decode
        for key in ("active_graphs", "capture_attempts", "captures", "retired_graphs", "replays"):
            answer[key] += decode[key]
    answer.update(
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


def _unprepared(exc: BaseException) -> bool:
    return is_capacity(exc) or getattr(exc, "code", None) == "STALE_BINDING"


def _dispatch(torch: Any, module: Any, call: Callable[..., Any], args: Any, kwargs: Any) -> Any:
    if (
        BYPASS.get()
        or _WARMING.get()
        or not torch.cuda.is_available()
        or torch.cuda.is_current_stream_capturing()
    ):
        return call(module, *args, **kwargs)
    region = _REGIONS.get(module)
    if region is None:  # a copy of a region's module: no region of its own
        return call(module, *args, **kwargs)
    placed = _placed(module)
    owner, path = region
    scope = (
        placed[0].graph_pass(placed[1], path, args, kwargs) if placed is not None else nullcontext()
    )
    token = _PASS.set(object()) if owner() is module else None
    began = time.perf_counter()
    try:
        with scope:
            return _region_call(torch, module, call, args, kwargs, placed)
    finally:
        if token is not None:
            _PASS.reset(token)
        if (recorder := _RECORDERS.get(module)) is not None:
            recorder.call_ms += (time.perf_counter() - began) * 1e3


def _region_call(
    torch: Any, module: Any, call: Callable[..., Any], args: Any, kwargs: Any, placed: Any
) -> Any:
    leaves, spec = torch.utils._pytree.tree_flatten((args, kwargs))
    stamp = _PASS.get(None)
    checked = _CHECKED.get(module)
    if checked is None or stamp is None or checked[0] is not stamp:
        pure = safe(torch, module)
        checked = _CHECKED[module] = (stamp, pure, _structure(torch, module) if pure else ())
    _, pure, structure = checked
    why = (
        "with gradients"
        if torch.is_grad_enabled()
        else "under activation capture"
        if _capture_clock.get() is not None
        else "under custom tensor dispatch"
        if torch.utils._python_dispatch._get_current_dispatch_mode() is not None
        else "across GPUs"
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        else "outside its purity contract"
        if not pure
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
    key = (*_signature(torch, leaves, spec), structure, levels, addresses, autocast)
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
        if key in recorder.failed or recorder.waiting.get(key) == attempt:
            return eager()
        if key not in recorder.seen:
            recorder.seen.add(key)
            out = eager()
            # Its graph's static buffers: the inputs it would copy in and the outputs it keeps.
            recorder.statics[key] = _nbytes(torch, leaves) + _nbytes(
                torch, torch.utils._pytree.tree_leaves(out)
            )
            return out
        if placed is not None:
            if (
                recorder.binding is not None
                and not recorder.binding.valid()
                and not recorder.forget(torch)
            ):
                return eager()
            if recorder.binding is None:
                recorder.binding = placed[0].retain(placed[1], (_REGIONS[module][1],))
        # The pool is held from reading its recording through preparation, a capture and the
        # replay until the device completes it: varena never shrinks the graph a call is using.
        with recorder.domain.holding():
            result = _replay(torch, module, call, leaves, spec, placed, recorder, key)
        return eager() if result is _EAGER else result


#: `_replay`'s answer when the call runs as written
_EAGER = object()


def _replay(
    torch: Any,
    module: Any,
    call: Callable[..., Any],
    leaves: list[Any],
    spec: Any,
    placed: Any,
    recorder: _Recorder,
    key: tuple[object, ...],
) -> Any:
    """The call replayed (captured first when it has no recording), or `_EAGER`."""
    owner = recorder.domain
    assert owner is not None
    attempt = placed[0].attempts if placed is not None else recorder.attempts
    binding = recorder.binding
    entry = recorder.recordings.get(key)
    if entry is None:
        if placed is not None and not placed[0].graph_accounted():
            why = _GAPS[module] = "driver/context memory is unmeasured"
            _failed(recorder, key, placed, why)
            return _EAGER
        # Admitted only beside the step's working set: its static buffers must fit in what the
        # request could get without evicting W or a pinned page (the pool's growth past that is
        # refused by the allocator, never taken from W).
        spare = placed[0].graph_spare() if placed is not None else None
        if spare is not None and spare < recorder.statics.get(key, 0):
            recorder.waiting[key] = attempt
            _refused(module, placed, f"graph refused: room ({spare} bytes spare)")
            return _EAGER
        # Discovery owns the complete range set and then the admitted free headroom.
        # Only the declared pure region runs here; this does not select/tune kernels.
        lease = (
            binding.prepare(torch.cuda.current_stream(), workspace=None)
            if binding is not None
            else nullcontext()
        )
        try:
            lease.__enter__()
        except Exception as exc:
            if not _unprepared(exc):
                raise
            recorder.waiting[key] = attempt
            _refused(module, placed, f"its weights cannot all be resident at once ({exc})")
            return _EAGER
        began = time.perf_counter()
        try:
            entry = _record(torch, module, call, leaves, spec, recorder, key, placed)
        finally:
            lease.__exit__(None, None, None)
            took = (time.perf_counter() - began) * 1e3
            recorder.capture_ms += took
            recorder.capture_each.append(round(took, 3))
        if entry is None:
            return _EAGER
    verdict = entry.ready(leaves)
    if verdict:
        recorder.held += 1
        if verdict == "stale" and entry.idle():
            recorder.recordings.pop(key).close()
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
        root, path = _REGIONS[module][0](), _REGIONS[module][1]
        whole = path == "" or (placed is not None and placed[0].graph_whole(placed[1], binding))
        began = time.perf_counter()
        lease = _lease(torch, owner, placed, binding, attempt, root, whole)
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


def _record(
    torch: Any,
    module: Any,
    call: Callable[..., Any],
    leaves: list[Any],
    spec: Any,
    recorder: _Recorder,
    key: tuple[object, ...],
    placed: Any,
) -> _Recording | None:
    # This path is restricted to the shipped pure contracts. Warmup discovers capacity in
    # the exact capture pool; it does not time or select kernels, or run arbitrary package code.
    current, stream = torch.cuda.current_stream(), None
    recorder.capture_attempts += 1
    try:
        assert recorder.domain is not None
        recorder.domain.initialize()
        stream = recorder.domain.stream
        stream.wait_stream(current)
        # Library handles cache scratch by stream. Initialize under domain ownership before
        # any region pool becomes active; every executable sharing that scratch is serialized.
        recorder.domain.prime_blas()
        pool = recorder.domain.capture_pool()
        allocating = (
            placed[0].graph_allocating(pool)
            if placed is not None
            else torch.cuda.use_mem_pool(pool)
        )
        with torch.cuda.stream(stream):
            with allocating:
                found = [_lent(torch, leaf, recorder.domain) for leaf in leaves]
                chained = [f[0] if f is not None else None for f in found]
                inputs = [
                    f[1]
                    if f is not None
                    else leaf.clone()
                    if isinstance(leaf, torch.Tensor)
                    else None
                    for leaf, f in zip(leaves, found, strict=True)
                ]
                static = [s if s is not None else v for s, v in zip(inputs, leaves, strict=True)]
                cargs, ckwargs = torch.utils._pytree.tree_unflatten(static, spec)
                warming = _WARMING.set(True)
                try:
                    with attention_ulysses.observing() as served:
                        warm = call(module, *cargs, **ckwargs)
                finally:
                    _WARMING.reset(warming)
                if not _clonable(torch, warm):
                    recorder.failed[key] = "output is not a tensor tree"
                    return None
            del warm  # preseeded capacity remains in this pool; no reference execution/oracle
            capturing = (
                placed[0].graph_allocating(pool, route=False)
                if placed is not None
                else nullcontext()
            )
            with capturing:
                outputs: list[Any] = []
                graph, why = capture(
                    torch,
                    lambda: outputs.append(call(module, *cargs, **ckwargs)),
                    pool,
                    recorder.domain,
                )
            if graph is None:
                _failed(recorder, key, placed, why)
                return None
            if not _clonable(torch, outputs[0]):
                recorder.failed[key] = "captured output is not a tensor tree"
                return None
        if placed is not None:
            placed[0].keep(
                *(
                    t
                    for t in [*inputs, *torch.utils._pytree.tree_leaves(outputs[0])]
                    if isinstance(t, torch.Tensor)
                )
            )
        owner = recorder.domain
        entry = _Recording(torch, graph, pool, inputs, chained, outputs[0], key, served, owner)
        recorder.recordings[key] = entry
        recorder.captures += 1
        for index, tensor in enumerate(entry.statics):
            _LENT[tensor.data_ptr()] = (weakref.ref(entry), index)
        if placed is not None and (owner.serve(_REGIONS[module][0]()) or owner.cache is None):
            # One cache for the pool, serving every component recorded into it (its class is
            # the most protective of theirs): registered again when one joins.
            if owner.cache is not None:
                owner.cache.close()
            roots = [r for r in (ref() for ref in owner.served) if r is not None]
            owner.cache = placed[0].cache("graphs", functools.partial(_shed, torch, owner), roots)
        if owner.cache is not None:
            owner.cache.cached(owner.capture_bytes())
        return entry
    except Exception as exc:
        if not is_capacity(exc) and not isinstance(exc, Building | domain.Unavailable):
            raise
        why = f"{type(exc).__name__}: {exc}".splitlines()[0][:300]
        _failed(recorder, key, placed, f"graph refused: room ({why})" if is_capacity(exc) else why)
        return None
    finally:
        if stream is not None:
            # Warmup can queue reads before refusing an allocation. Join even the failure
            # path before the caller's physical lease records its completion event.
            current.wait_stream(stream)


def _lent(
    torch: Any, leaf: Any, owner: domain.Domain | None = None
) -> tuple[_Recording, Any] | None:
    """The live recording and static output `leaf` is an alias of (of `owner`'s graphs)."""
    found = _LENT.get(leaf.data_ptr()) if isinstance(leaf, torch.Tensor) else None
    entry = found[0]() if found is not None else None
    if found is None or entry is None or entry.graph is None:
        return None
    static = entry.statics[found[1]]
    if not _same(static, leaf) or (owner is not None and entry.owner is not owner):
        return None
    return entry, static


def record_stream(torch: Any, tensor: Any, stream: Any) -> None:
    """`tensor`, a graph's lent output, is read on `stream`: the graph's next replay waits for
    the work queued there (torch's `Tensor.record_stream`, for a lent output's reuse)."""
    found = _lent(torch, tensor)
    if found is not None:
        found[0].readers[stream] = None


def _lease(
    torch: Any,
    owner: domain.Domain,
    placed: Any,
    binding: Any,
    attempt: int,
    root: Any,
    whole: bool,
) -> Any:
    """A replay's residency: its region's blocks prepared around it. A region whose blocks are
    the whole component (its root, or any region of a one-block component: a text encoder's
    97 linears) instead holds every block bound for the rest of the request from its first
    replay (design §5: warm resident segments keep their backing), and while that hold stands no
    region of the component prepares again (v7: a CLIP replay bound and unbound its whole block,
    ~6 ms a call). The hold goes at the next request, when the graphs are shed, or when the fill
    changes under it. A component that does not fit refuses the hold as it would the lease."""
    if placed is None or binding is None:
        return nullcontext()
    stream = torch.cuda.current_stream()
    held = owner.holds.get(root)
    if held is not None and held.attempt != attempt:
        owner.unhold(root)
        held = None
    if held is not None:
        if held.stream != stream:
            stream.wait_stream(held.stream)
        return nullcontext()
    if not whole:
        return binding.prepare(stream)
    hold = placed[0].graph_hold(placed[1], functools.partial(owner.unhold, root))
    owner.holds[root] = domain.Hold(hold, attempt, stream)
    return nullcontext()


def _refused(module: Any, placed: Any, why: str) -> None:
    """A region whose preparation the memory refused runs as written this request, its finer
    regions replaying under it; the run says so once."""
    why = why.splitlines()[0][:300]
    _GAPS[module] = why
    if placed is not None:
        placed[0].note(f"{placed[1]}.graphs", f"eager {why}")
    say(f"refused:{why}", f"[graphs] a region runs as written, its finer regions replay: {why}")


def _failed(recorder: _Recorder, key: tuple[object, ...], placed: Any, why: str) -> None:
    recorder.waiting[key] = placed[0].attempts if placed is not None else recorder.attempts
    say(f"graph:{key!r}:{why}", f"[graphs] pure region remains eager: {why}")
    WARNINGS.append(("a CUDA graph was not used", {"why": why}))


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
        names = ("structure", "cut levels", "weight addresses", "autocast")
        what = next(n for n, i in zip(names, range(-4, 0), strict=True) if last[i] != key[i])
    recorder.changes[what] = recorder.changes.get(what, 0) + 1


def _nbytes(torch: Any, leaves: list[Any]) -> int:
    return sum(v.numel() * v.element_size() for v in leaves if isinstance(v, torch.Tensor))


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


def _structure(torch: Any, module: Any) -> tuple[object, ...]:
    """The code each module runs and the plain values its contract can read (the names the
    region's contract code and attention processors name), in attribute order (a reordering
    only records again). A value only package code names (a step counter) keys nothing."""
    kinds = (torch.dtype, torch.device, enum.Enum)
    modules = list(module.modules())
    processors = [p for m in modules if (p := vars(m).get("processor")) is not None]
    stop = torch.nn.Module
    names = frozenset().union(
        *(reads(c, stop) for c in {contract(type(m)) for m in modules}),
        *(reads(c, object) for c in {type(p) for p in processors}),
    )

    def plain(owner: object) -> tuple[object, ...]:
        return tuple(
            value
            for name, value in vars(owner).items()
            if name in names
            and (
                type(value) in _PLAIN
                or isinstance(value, kinds)
                or (type(value) is tuple and all(type(item) in _PLAIN for item in value))
            )
        )

    return tuple(
        (
            id(type(m).forward),
            id(vars(m).get("forward")),
            id(vars(m).get("_cozy_forward")),
            plain(m),
            (type(p), plain(p)) if (p := vars(m).get("processor")) is not None else None,
        )
        for m in modules
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
    "capture",
    "capture_device",
    "capture_domain",
    "describe",
    "install",
    "record_stream",
    "release",
    "say",
]
