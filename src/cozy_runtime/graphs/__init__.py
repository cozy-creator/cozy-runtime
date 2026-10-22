"""CUDA graphs over a component's weights at their home addresses, on by default.

Runtime records each placed component's calls and replays them (`install`, run by the executor
when it constructs a model); a package declares a component `@uncaptured` to opt out. A graph
reads weights at their addresses, so its component is pinned for as long as its recordings live
(`Residency.pin`): wholly bound at home, counted as a cache. When memory needs room, or the
component's LoRAs change, the pin's shed drops every recording and its pool, and the weights go.

The first call of each signature runs as written; one whose GPU work outlasts its launches
(`HOST_BOUND`) runs as written for good, since a graph saves only launches. At the second, the
component is pinned, the call is recorded into the component's one memory pool with its tensor
inputs in static buffers, and the first replay is checked against the call run as written: on a
mismatch that signature runs as written from then on, with a run warning, and a recording that
is not faster (`PAYS`) is dropped with its memory. Every later call copies its inputs in,
replays, and returns clones of the outputs (a multistep scheduler keeps earlier ones).
Recordings outlive the request while the pin holds and the module runs the same code
(`_structure`); otherwise it records again.

A signature is the tensors' shapes, dtypes, devices and strides and every other argument by
value; a recording also keys on the code each submodule runs (an attention kernel switch, a
forward swapped between requests). A call with an argument
that is not a tensor or a plain value (a cache object), with gradients, on a CPU, inside another
capture, or under `BYPASS` (`cozy_runtime.lm` runs its own) runs as written, and so does one in
an attention layout's scope (sparse attention reads its step), under an activation capture,
across GPUs (every GPU's peers must replay alike), whose blocks do not all fit at home (a
component that streams), or whose capture failed in `TRIES` requests (a run warning: in a freshly
started process other device work can end a capture, so a failed one waits for the next
request).

`capture(torch, step, pool, stream)` is the recipe: on a side stream, in thread-local mode so the
plane's own threads keep working while it records, and outside Runtime's dispatch modes, whose
out-of-memory recovery copies what an op writes into first, which a graph would then copy on
every replay. `torch` is the torch module, passed in: this module imports no torch, so a session
without a GPU, whose executor imports it too, never does.
"""

from __future__ import annotations

import contextvars
import functools
import sys
import time
import weakref
from collections.abc import Callable, Mapping
from typing import Any

from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT
from cozy_runtime.author._capture import _capture_clock
from cozy_runtime.author._errors import is_capacity
from cozy_runtime.author._model import LANDED, PLACED, WARNINGS, Building
from cozy_runtime.internal import attention_ulysses, module_state, streams

#: Set while a mechanism runs its own graphs over a component (`cozy_runtime.lm`): its calls
#: pass straight through.
BYPASS: contextvars.ContextVar[bool] = contextvars.ContextVar("cozy_graphs_bypass", default=False)
#: The first replay's largest relative L2 distance from the call run as written. A capture may
#: take other kernels for the same math: CLIP-G's replay on an RTX 4090 starts exact at its
#: embeddings and drifts 1e-3 per layer to 2.4e-2 at its 32nd; a replay reading wrong state
#: is off from its first output.
TOLERANCE = 5e-2
#: A signature records only when its first call's launches took at least this share of its time
#: (a GPU that outruns the launches gains nothing from a graph), and keeps its recording only
#: when a replay takes at most `PAYS` of the call run as written.
HOST_BOUND, PAYS = 0.5, 0.9
#: Requests in which a signature's capture may fail before it runs as written for good.
TRIES = 4
_PLAIN = (type(None), bool, int, float, str)
_SAID: set[str] = set()


def say(key: str, line: str) -> None:
    """`line` on stderr, once per process for `key`."""
    if key not in _SAID:
        _SAID.add(key)
        print(line, file=sys.stderr)


def capture(torch: Any, step: Callable[[], object], pool: Any, stream: Any) -> tuple[Any, str]:
    """`step` recorded as a CUDA graph into `pool` on `stream`: (graph, "") or (None, why).
    After a failure the pool is spent: the allocator keeps a released pool until its blocks
    are free, and a capture into it again trips its use count, so the next try takes a new one
    (`graph_pool_handle()`). An out-of-memory is said as `OutOfMemoryError`: the caller can run
    the step eagerly, whose recovery makes room, and try again."""
    graph = torch.cuda.CUDAGraph()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    failure: BaseException | None = None
    with torch.utils._python_dispatch._disable_current_modes(), torch.cuda.stream(stream):
        try:
            graph.capture_begin(pool=pool, capture_error_mode="thread_local")
        except Exception as exc:
            failure = exc
        else:
            try:
                step()
            except Exception as exc:  # the first cause, not the capture's end it spoils
                failure = exc
            try:
                graph.capture_end()
            except Exception as exc:
                failure = failure or exc
    current.wait_stream(stream)
    if isinstance(failure, Building):
        raise failure
    if failure is not None:
        return None, f"{type(failure).__name__}: {failure}".splitlines()[0][:300]
    return graph, ""


class _Recording:
    """One signature's graph, its static inputs and outputs, the layout it was recorded at, and
    the attention implementations it serves (a replay reports them as a call would)."""

    def __init__(
        self, torch: Any, graph: Any, inputs: list[Any], outputs: Any, layout: object, served: Any
    ) -> None:
        self.torch, self.graph, self.inputs, self.outputs = torch, graph, inputs, outputs
        self.layout = layout
        self.served = [(b, i) for b, impls in served.items() for i in impls]

    def __call__(self, leaves: list[Any]) -> Any:
        for static, value in zip(self.inputs, leaves, strict=True):
            if static is not None:
                static.copy_(value)
        self.graph.replay()
        for backend, implementation in self.served:
            attention_ulysses.record(backend, implementation)
        tensor = self.torch.Tensor
        return self.torch.utils._pytree.tree_map(
            lambda v: v.clone() if isinstance(v, tensor) else v, self.outputs
        )


class _Recorder:
    """A component's recordings, kept across requests in one memory pool."""

    def __init__(self) -> None:
        self.seen: set[tuple[object, ...]] = set()
        #: signatures that run as written for good, and captures that failed so far
        self.failed: set[tuple[object, ...]] = set()
        self.tries: dict[tuple[object, ...], int] = {}
        #: signatures whose capture failed, by the attempt it failed in: tried in the next one
        self.waiting: dict[tuple[object, ...], int] = {}
        self.recordings: dict[tuple[object, ...], _Recording] = {}
        self.pool: Any = None
        self.stream: Any = None
        #: the pin its structure was read under, and that structure
        self.structure: tuple[object, tuple[object, ...]] = (None, ())
        self.pool_bytes = self.replays = 0
        #: signatures that run as written by choice, by why, and the latest one's detail
        self.chose: dict[str, int] = {}
        self.detail = ""
        self.record_s = 0.0
        #: `LANDED` when it last started over: a kernel landing since changes what calls run
        self.landed = LANDED[0]

    def forget(self) -> None:
        """Start over: what was recorded or decided ran code a kernel landing replaced."""
        self.landed = LANDED[0]
        self.seen.clear()
        self.failed.clear()
        self.tries.clear()
        self.waiting.clear()
        self.recordings.clear()
        self.pool, self.pool_bytes, self.chose = None, 0, {}


#: Every graphed component's recorder in this process.
_RECORDERS: weakref.WeakKeyDictionary[Any, _Recorder] = weakref.WeakKeyDictionary()
_CLASSES: dict[type, type] = {}


def install(torch: Any, model: object, roots: Mapping[str, Any]) -> list[str]:
    """Record and replay the calls of each of `model`'s component roots (`roots`, name -> module)
    the Model class does not declare `@uncaptured`. Returns the components installed."""
    skipped = set(getattr(type(model), "__uncaptured__", ()))
    installed = []
    for name, root in roots.items():
        if name in skipped or not isinstance(root, torch.nn.Module):
            continue
        if not getattr(type(root), "_cozy_graphed", False):
            root.__class__ = _graphed(torch, type(root))
        installed.append(name)
    return installed


def _graphed(torch: Any, cls: type) -> type:
    """`cls` with a `__call__` that replays recordings: the same name, a subclass of it."""
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
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
            },
        )
    return made


def _shed(torch: Any, recorder: _Recorder) -> int:
    """Drop a component's recordings and their pool (its pin's shed); the pool's bytes."""
    freed = recorder.pool_bytes
    recorder.recordings.clear()  # each holds its pin: the last one dropped lets it go
    recorder.structure = (None, ())
    recorder.pool, recorder.pool_bytes = None, 0
    torch.cuda.empty_cache()
    return freed


def _dispatch(torch: Any, module: Any, call: Callable[..., Any], args: Any, kwargs: Any) -> Any:
    if BYPASS.get() or not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
        return call(module, *args, **kwargs)
    leaves, spec = torch.utils._pytree.tree_flatten((args, kwargs))
    why = (
        "with gradients"
        if torch.is_grad_enabled()
        else "in an attention layout's scope"
        if _ACTIVE_LAYOUT.get() is not None
        else "under an activation capture"
        if _capture_clock.get() is not None
        else "across GPUs"
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        else _unplain(torch, leaves)
    )
    if why:
        placed = PLACED.get(module)
        if placed is not None:
            placed[0].note(f"{placed[1]}.graphs", f"runs as written {why}")
        return call(module, *args, **kwargs)
    recorder = _RECORDERS.get(module)
    if recorder is None:
        recorder = _RECORDERS[module] = _Recorder()
    if recorder.landed != LANDED[0]:
        recorder.forget()
    key = _signature(torch, leaves, spec)
    placed = PLACED.get(module)
    if key in recorder.failed:
        if placed is not None:
            placed[0].note(f"{placed[1]}.graphs", _facts(recorder))
        return call(module, *args, **kwargs)
    if key not in recorder.seen:
        # The first call of a signature: as written.
        recorder.seen.add(key)
        torch.cuda.synchronize()
        started = time.perf_counter()
        state = module_state.snapshot(module)
        out = call(module, *args, **kwargs)
        launched = time.perf_counter() - started
        if module_state.changed(state, module_state.snapshot(module)):
            # A replay runs none of its Python: one that changes its own attributes (HiDiffusion's
            # step counters) runs as written, and is never run again to measure it.
            _chose(recorder, key, "its state", "it changes its own attributes")
        torch.cuda.synchronize()
        took = time.perf_counter() - started
        if not _clonable(torch, out):
            _chose(recorder, key, "its output", f"a {type(out).__name__}")
        elif launched < HOST_BOUND * took:
            _chose(recorder, key, "GPU-bound", f"launched in {launched:.3f} of {took:.3f} s")
        if placed and key in recorder.failed:
            placed[0].note(f"{placed[1]}.graphs", _facts(recorder))
        return out
    pin: object = None
    if placed is not None:
        residency, name = placed
        if recorder.waiting.get(key) == residency.attempts:
            return call(module, *args, **kwargs)
        pin = residency.pin(name, f"graphs:{name}", functools.partial(_shed, torch, recorder))
        if pin is None:
            return call(module, *args, **kwargs)
    if pin is None or recorder.structure[0] is not pin:  # read once per pin
        recorder.structure = (pin, _structure(module))
    layout = (pin, recorder.structure[1])
    entry = recorder.recordings.get(key)
    if entry is not None and entry.layout != layout:
        del recorder.recordings[key]  # its weights moved, or its code: it records again
        entry = None
        if not recorder.recordings:  # its pool went with its last graph
            recorder.pool = None
    if entry is None:
        entry = _record(torch, module, call, leaves, spec, recorder, key, layout, placed)
        if entry is None:
            if not recorder.recordings:
                _shed(torch, recorder)
                if placed is not None:
                    placed[0].unpin(placed[1], f"graphs:{placed[1]}", pin)
                # Both locals retain the token: let its event-ordered hold go before eager
                # execution. Other callers' recordings still retain their own references.
                pin = layout = None
            return call(module, *args, **kwargs)
    recorder.replays += 1
    if placed is not None:
        placed[0].note(f"{placed[1]}.graphs", _facts(recorder))
    return entry(leaves)


def _record(
    torch: Any,
    module: Any,
    call: Callable[..., Any],
    leaves: list[Any],
    spec: Any,
    recorder: _Recorder,
    key: tuple[object, ...],
    layout: object,
    placed: Any,
) -> _Recording | None:
    """Record the call at `key` into the component's pool, and check its first replay against
    the call run as written. None, with a run warning, when it cannot be."""
    started, landed = time.perf_counter(), LANDED[0]
    try:
        if recorder.pool is None:
            recorder.pool, recorder.stream = torch.cuda.graph_pool_handle(), streams.stream(torch)
        reserved = torch.cuda.memory_reserved()
        inputs = [leaf.clone() if isinstance(leaf, torch.Tensor) else None for leaf in leaves]
        static = [s if s is not None else v for s, v in zip(inputs, leaves, strict=True)]
        cargs, ckwargs = torch.utils._pytree.tree_unflatten(static, spec)
        outputs: list[Any] = []
        pool = recorder.pool
        # Measuring restores the attributes and generators advanced by its passes.
        state = module_state.snapshot(module)
        cpu, cuda = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    except Exception as exc:
        # Preparation has not invoked package code. Never extend this catch around capture,
        # replay or validation: the model may already have executed by then.
        if not is_capacity(exc):
            raise
        why = f"{type(exc).__name__}: {exc}".splitlines()[0][:300]
        return _failed(torch, recorder, key, placed, why, type(module).__name__)
    try:
        graph, why = capture(
            torch, lambda: outputs.append(call(module, *cargs, **ckwargs)), pool, recorder.stream
        )
    except Building as building:  # as written until a kernel lands: `forget` tries again
        graph, why, recorder.pool = None, "", None  # the aborted capture spent its pool
        _chose(recorder, key, "kernel building", f"{building} builds; it records once one lands")
    if graph is not None and not _clonable(torch, outputs[0]):
        graph, why = None, f"its output {type(outputs[0]).__name__} is not a tree of tensors"
    if graph is not None:
        graph.replay()
        replayed = _floats(torch, outputs[0])
        written: list[Any] = []
        with attention_ulysses.observing() as served:
            written_s = _seconds(torch, lambda: written.append(call(module, *cargs, **ckwargs)))
        written = _floats(torch, written[0])
        error = _distance(replayed, written)
        if LANDED[0] != landed:
            graph, why = None, ""  # a kernel landed meanwhile: the next call starts over
            _chose(recorder, key, "kernel landed", "")
        elif len(replayed) != len(written) or not error <= TOLERANCE:
            each = " ".join(
                f"{_distance([a], [b]):.0e}" for a, b in zip(replayed, written, strict=False)
            )
            why = (
                f"its first replay is {error:.2e} from the call run as written (by output: {each})"
            )
            graph = None
        elif (replay_s := _seconds(torch, graph.replay)) > PAYS * written_s:
            graph, why = None, ""  # as fast as written: not worth its memory
            _chose(
                recorder, key, "no faster", f"replay {replay_s:.4f} s, {written_s:.4f} s as written"
            )
    module_state.restore(module, state)
    torch.random.set_rng_state(cpu)
    torch.cuda.set_rng_state(cuda)
    recorder.record_s += time.perf_counter() - started
    if graph is None:
        outputs.clear()
        return _failed(torch, recorder, key, placed, why, type(module).__name__)
    entry = _Recording(torch, graph, inputs, outputs[0], layout, served)
    if recorder.pool is pool:  # else the plane released the pool for room meanwhile
        recorder.pool_bytes += max(torch.cuda.memory_reserved() - reserved, 0)
        recorder.recordings[key] = entry
    return entry


def _failed(
    torch: Any,
    recorder: _Recorder,
    key: tuple[object, ...],
    placed: Any,
    why: str,
    name: str,
) -> None:
    """Retire an unsuccessful recording and defer this signature before eager execution."""
    recorder.tries[key] = tries = recorder.tries.get(key, 0) + 1
    if not why or why.startswith("its ") or tries >= TRIES:
        recorder.failed.add(key)
    elif placed is not None:
        recorder.waiting[key] = placed[0].attempts
    if why or not recorder.recordings:
        recorder.pool = None
        torch.cuda.empty_cache()
    if why:
        line = f"{name} runs as written at this signature: {why}"
        say(f"graphs:{name}:{tries}:{why[:40]}", f"[graphs] {line}")
        if key in recorder.failed:
            WARNINGS.append(("a CUDA graph was not used", {"module": name, "why": why}))


def _chose(recorder: _Recorder, key: tuple[object, ...], why: str, detail: str) -> None:
    recorder.failed.add(key)
    recorder.chose[why] = recorder.chose.get(why, 0) + 1
    recorder.detail = f"{why}: {detail}" if detail else why


def _facts(recorder: _Recorder) -> str:
    chose = ", ".join(f"{n} {why}" for why, n in recorder.chose.items())
    return (
        f"{len(recorder.recordings)} recorded in {recorder.record_s:.2f} s, "
        f"{recorder.replays} replays, pool {recorder.pool_bytes >> 20} MiB"
        + (f"; as written: {chose} ({recorder.detail})" if chose else "")
    )


def _seconds(torch: Any, step: Callable[[], object]) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    step()
    torch.cuda.synchronize()
    return time.perf_counter() - started


def _unplain(torch: Any, leaves: list[Any]) -> str:
    """Why a call's arguments cannot be replayed into, or ""."""
    tensors = [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]
    if not tensors:
        return "with no tensor argument"
    if not all(tensor.is_cuda for tensor in tensors):
        return "with a tensor argument off the GPU"
    for leaf in leaves:
        if (
            not isinstance(leaf, (torch.Tensor, torch.dtype, torch.device))
            and type(leaf) not in _PLAIN
        ):
            return f"with a {type(leaf).__name__} argument"
    return ""


def _floats(torch: Any, out: Any) -> list[Any]:
    return [v.float() for v in torch.utils._pytree.tree_leaves(out) if isinstance(v, torch.Tensor)]


def _distance(a: list[Any], b: list[Any]) -> float:
    """The largest relative L2 distance between corresponding tensors."""
    return max(
        (float((x - y).norm() / y.norm().clamp_min(1e-12)) for x, y in zip(a, b, strict=False)),
        default=0.0,
    )


def _clonable(torch: Any, out: Any) -> bool:
    """Whether every leaf of `out` is a tensor or a plain value, so a replay can hand clones."""
    leaves, _ = torch.utils._pytree.tree_flatten(out)
    return any(isinstance(v, torch.Tensor) for v in leaves) and all(
        isinstance(v, torch.Tensor) or type(v) in _PLAIN for v in leaves
    )


def _structure(module: Any) -> tuple[object, ...]:
    """Which code each submodule runs: its class, any forward set on the instance (SDXL's
    hidiffusion swaps them between requests), its attention processor and the kernel that
    processor calls (a request's pin, or the compiled one replacing SDPA)."""
    rows = []
    for m in module.modules():
        processor = getattr(m, "processor", None)
        backend = getattr(processor, "_attention_backend", None)
        rows.append((id(type(m)), id(m.__dict__.get("forward")), id(processor), backend))
    return tuple(rows)


def _signature(torch: Any, leaves: list[Any], spec: Any) -> tuple[object, ...]:
    """The tensors' geometry and every other argument by value."""
    key: list[object] = [str(spec)]
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            key.append((tuple(leaf.shape), leaf.dtype, leaf.device, leaf.stride()))
        else:
            key.append((type(leaf), leaf))
    return tuple(key)


__all__ = [
    "BYPASS",
    "HOST_BOUND",
    "PAYS",
    "TOLERANCE",
    "TRIES",
    "capture",
    "install",
    "native",
    "say",
]
