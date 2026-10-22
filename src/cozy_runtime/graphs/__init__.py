"""CUDA graphs of declared-pure regions over retained virtual addresses.

Kernel choice is shipped. Runtime prepares memory, captures process-local addresses and
replays; it neither times competing paths nor compiles kernels. Unsupported Python stays
eager around safe regions. Every graph has its own static buffers and allocator pool.
A cached binding owns virtual addresses only; its physical lease covers preparation through
output copies, and a completion event orders reuse across execution streams.
"""

from __future__ import annotations

import contextvars
import functools
import sys
import threading
import weakref
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import Any

from cozy_runtime.author._capture import _capture_clock
from cozy_runtime.author._errors import Fault, fault, is_capacity
from cozy_runtime.author._model import LANDED, PLACED, WARNINGS, Building
from cozy_runtime.graphs.regions import regions, safe
from cozy_runtime.internal import attention_ulysses

BYPASS: contextvars.ContextVar[bool] = contextvars.ContextVar("cozy_graphs_bypass", default=False)
_PLAIN = (type(None), bool, int, float, str)
_SAID: set[str] = set()


def say(key: str, line: str) -> None:
    if key not in _SAID:
        _SAID.add(key)
        print(line, file=sys.stderr)


def capture(torch: Any, step: Callable[[], object], pool: Any, stream: Any) -> tuple[Any, str]:
    """Record a caller-owned capture-safe computation in its already prepared pool.

    The caller owns purity, residency, static-data lifetime and all consuming streams. No
    allocation recovery or chunk selection is allowed inside capture.
    """
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
            except Exception as exc:
                failure = exc
            try:
                graph.capture_end()
            except Exception as exc:
                failure = failure or exc
    current.wait_stream(stream)
    if isinstance(failure, Building):
        raise failure
    if failure is not None:
        if fault(failure) in (Fault.STICKY, Fault.DEVICE):
            code = getattr(failure, "error_code", None)
            if code not in (900, 901):  # unsupported or invalidated capture, already ended
                raise failure
        return None, f"{type(failure).__name__}: {failure}".splitlines()[0][:300]
    return graph, ""


class _Recording:
    def __init__(
        self,
        torch: Any,
        graph: Any,
        pool: Any,
        inputs: list[Any],
        outputs: Any,
        layout: object,
        served: Any,
        nbytes: int,
    ) -> None:
        self.torch, self.graph, self.pool = torch, graph, pool
        self.inputs, self.outputs, self.layout = inputs, outputs, layout
        self.served = [(b, i) for b, impls in served.items() for i in impls]
        self.nbytes = nbytes
        self.done: Any = None

    def output(self) -> Any:
        """Caller-owned outputs allocated before residency admission or graph launch."""
        torch = self.torch
        return torch.utils._pytree.tree_map(
            lambda v: torch.empty_like(v) if isinstance(v, torch.Tensor) else v, self.outputs
        )

    def __call__(self, leaves: list[Any], out: Any) -> Any:
        torch, stream = self.torch, self.torch.cuda.current_stream()
        if self.done is not None:
            stream.wait_event(self.done)
        for static, value in zip(self.inputs, leaves, strict=True):
            if static is not None:
                static.copy_(value)
        self.graph.replay()
        source = torch.utils._pytree.tree_leaves(self.outputs)
        for dst, src in zip(torch.utils._pytree.tree_leaves(out), source, strict=True):
            if isinstance(src, torch.Tensor):
                dst.copy_(src)
        for backend, implementation in self.served:
            attention_ulysses.record(backend, implementation)
        self.done = torch.cuda.Event()
        self.done.record(stream)
        return out

    def idle(self) -> bool:
        return self.done is None or bool(self.done.query())


class _Recorder:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.recordings: dict[tuple[object, ...], _Recording] = {}
        self.seen: set[tuple[object, ...]] = set()
        self.failed: dict[tuple[object, ...], str] = {}
        self.waiting: dict[tuple[object, ...], int] = {}
        self.binding: Any = None
        self.cache: Any = None
        self.replays = 0
        self.attempts = 0
        self.landed = LANDED[0]

    def forget(self, torch: Any) -> bool:
        if any(not entry.idle() for entry in self.recordings.values()):
            return False
        self.recordings.clear()
        if self.binding is not None:
            self.binding.close()
            self.binding = None
        if self.cache is not None:
            self.cache.close()
            self.cache = None
        self.seen.clear()
        self.failed.clear()
        self.waiting.clear()
        self.landed = LANDED[0]
        torch.cuda.empty_cache()
        return True


_RECORDERS: weakref.WeakKeyDictionary[Any, _Recorder] = weakref.WeakKeyDictionary()
_REGIONS: weakref.WeakKeyDictionary[Any, tuple[Any, str]] = weakref.WeakKeyDictionary()
_CLASSES: dict[type, type] = {}


def install(torch: Any, model: object, roots: Mapping[str, Any]) -> list[str]:
    """Install shipped pure regions, including the blocks of a streamed component."""
    skipped = set(getattr(type(model), "__uncaptured__", ()))
    installed = []
    for name, root in roots.items():
        if name in skipped or not isinstance(root, torch.nn.Module):
            continue
        placed = PLACED.get(root)
        blocks = placed[0].graph_segments(placed[1]) if placed is not None else ("",)
        candidates = (
            candidate
            for block in blocks
            for candidate in regions(torch, root.get_submodule(block) if block else root, block)
        )
        for path, module in candidates:
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
    return made


def _placed(module: Any) -> tuple[Any, str] | None:
    region = _REGIONS.get(module)
    return PLACED.get(region[0]()) if region is not None else PLACED.get(module)


def _shed(torch: Any, recorder: _Recorder, _want: int = 0) -> int:
    if not recorder.lock.acquire(blocking=False):
        return 0
    try:
        nbytes = sum(entry.nbytes for entry in recorder.recordings.values())
        return nbytes if recorder.forget(torch) else 0
    finally:
        recorder.lock.release()


def release(torch: Any, root: Any) -> None:
    """A generation's explicit teardown: finish its output copies before releasing addresses."""
    modules = [module for module, (owner, _) in _REGIONS.items() if owner() is root]
    for module in modules:
        recorder = _RECORDERS.pop(module, None)
        if recorder is not None:
            with recorder.lock:
                for entry in recorder.recordings.values():
                    if entry.done is not None:
                        entry.done.synchronize()
                recorder.forget(torch)
        _REGIONS.pop(module, None)


def _unprepared(exc: BaseException) -> bool:
    return is_capacity(exc) or getattr(exc, "code", None) == "STALE_BINDING"


def _dispatch(torch: Any, module: Any, call: Callable[..., Any], args: Any, kwargs: Any) -> Any:
    if BYPASS.get() or not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
        return call(module, *args, **kwargs)
    leaves, spec = torch.utils._pytree.tree_flatten((args, kwargs))
    placed = _placed(module)
    why = (
        "with gradients"
        if torch.is_grad_enabled()
        else "under activation capture"
        if _capture_clock.get() is not None
        else "across GPUs"
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        else "outside its purity contract"
        if not safe(torch, module)
        else _unplain(torch, leaves)
    )
    if why:
        if placed is not None:
            placed[0].note(f"{placed[1]}.graphs", f"eager {why}")
        return call(module, *args, **kwargs)
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
    key = (*_signature(torch, leaves, spec), _structure(torch, module), levels, addresses, autocast)
    with recorder.lock:
        if recorder.landed != LANDED[0] and not recorder.forget(torch):
            return call(module, *args, **kwargs)
        recorder.attempts += 1
        attempt = placed[0].attempts if placed is not None else recorder.attempts
        if key in recorder.failed or recorder.waiting.get(key) == attempt:
            return call(module, *args, **kwargs)
        if key not in recorder.seen:
            recorder.seen.add(key)
            return call(module, *args, **kwargs)
        if placed is not None:
            if (
                recorder.binding is not None
                and not recorder.binding.valid()
                and not recorder.forget(torch)
            ):
                return call(module, *args, **kwargs)
            if recorder.binding is None:
                path = _REGIONS[module][1]
                recorder.binding = placed[0].retain(placed[1], (path,))
                recorder.cache = placed[0].cache(
                    f"graphs:{placed[1]}:{path}", functools.partial(_shed, torch, recorder)
                )
        binding = recorder.binding
        entry = recorder.recordings.get(key)
        if entry is None:
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
                return call(module, *args, **kwargs)
            try:
                entry = _record(torch, module, call, leaves, spec, recorder, key, placed)
            finally:
                lease.__exit__(None, None, None)
            if entry is None:
                return call(module, *args, **kwargs)
        # Outputs are real allocations before the physical lease: admission sees their actual
        # allocator segments. No output allocation can fail after graph launch and cause replay.
        try:
            out = entry.output()
        except Exception as exc:
            if not _unprepared(exc):
                raise
            recorder.waiting[key] = attempt
            return call(module, *args, **kwargs)
        lease = (
            binding.prepare(torch.cuda.current_stream()) if binding is not None else nullcontext()
        )
        try:
            lease.__enter__()
        except Exception as exc:
            if not _unprepared(exc):
                raise
            del out
            recorder.waiting[key] = attempt
            return call(module, *args, **kwargs)
        try:
            result = entry(leaves, out)
            recorder.replays += 1
            if placed is not None:
                placed[0].note(f"{placed[1]}.graphs", _facts(recorder))
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
    try:
        pool = placed[0].graph_pool() if placed is not None else torch.cuda.MemPool()
        stream = torch.cuda.Stream()
        stream.wait_stream(current)
        allocating = (
            placed[0].graph_allocating(pool)
            if placed is not None
            else torch.cuda.use_mem_pool(pool)
        )
        with allocating, torch.cuda.stream(stream):
            inputs = [leaf.clone() if isinstance(leaf, torch.Tensor) else None for leaf in leaves]
            static = [s if s is not None else v for s, v in zip(inputs, leaves, strict=True)]
            cargs, ckwargs = torch.utils._pytree.tree_unflatten(static, spec)
            with attention_ulysses.observing() as served:
                warm = call(module, *cargs, **ckwargs)
            if not _clonable(torch, warm):
                recorder.failed[key] = "output is not a tensor tree"
                return None
            # Keep an independent reference for correctness, outside the capture pool's
            # device bytes. The copy completes before its source can be reused by capture.
            reference = torch.utils._pytree.tree_map(
                lambda t: t.cpu() if isinstance(t, torch.Tensor) else t, warm
            )
            del warm
            outputs: list[Any] = []
            graph, why = capture(
                torch, lambda: outputs.append(call(module, *cargs, **ckwargs)), pool, stream
            )
            if graph is None:
                _failed(recorder, key, placed, why)
                return None
            if not _clonable(torch, outputs[0]):
                recorder.failed[key] = "captured output is not a tensor tree"
                return None
            graph.replay()
            for got, want in zip(
                torch.utils._pytree.tree_leaves(outputs[0]),
                torch.utils._pytree.tree_leaves(reference),
                strict=True,
            ):
                if isinstance(got, torch.Tensor):
                    try:
                        torch.testing.assert_close(got.cpu(), want, rtol=5e-2, atol=1e-5)
                    except AssertionError:
                        recorder.failed[key] = "captured result differs from the pure call"
                        _failed(recorder, key, placed, recorder.failed[key])
                        return None
            current.wait_stream(stream)
        if placed is not None:
            placed[0].keep(
                *(
                    t
                    for t in [*inputs, *torch.utils._pytree.tree_leaves(outputs[0])]
                    if isinstance(t, torch.Tensor)
                )
            )
        nbytes = sum(segment["total_size"] for segment in pool.snapshot())
        entry = _Recording(torch, graph, pool, inputs, outputs[0], key, served, nbytes)
        recorder.recordings[key] = entry
        if recorder.cache is not None:
            recorder.cache.cached(sum(e.nbytes for e in recorder.recordings.values()))
        return entry
    except Exception as exc:
        if not is_capacity(exc) and not isinstance(exc, Building):
            raise
        _failed(recorder, key, placed, f"{type(exc).__name__}: {exc}".splitlines()[0][:300])
        return None
    finally:
        if stream is not None:
            # Warmup can queue reads before refusing an allocation. Join even the failure
            # path before the caller's physical lease records its completion event.
            current.wait_stream(stream)


def _failed(recorder: _Recorder, key: tuple[object, ...], placed: Any, why: str) -> None:
    recorder.waiting[key] = placed[0].attempts if placed is not None else recorder.attempts
    say(f"graph:{key!r}:{why}", f"[graphs] pure region remains eager: {why}")
    WARNINGS.append(("a CUDA graph was not used", {"why": why}))


def _facts(recorder: _Recorder) -> str:
    held = sum(e.nbytes for e in recorder.recordings.values())
    return f"{len(recorder.recordings)} regions, {recorder.replays} replays, {held >> 20} MiB pools"


def _unplain(torch: Any, leaves: list[Any]) -> str:
    tensors = [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]
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
        isinstance(v, torch.Tensor) or type(v) in _PLAIN for v in leaves
    )


def _structure(torch: Any, module: Any) -> tuple[object, ...]:
    return tuple(
        (
            id(type(m).forward),
            id(vars(m).get("forward")),
            id(getattr(m, "processor", None)),
            id(getattr(m, "_cozy_forward", None)),
            tuple(
                sorted(
                    (name, value)
                    for name, value in vars(m).items()
                    if type(value) in _PLAIN
                    or isinstance(value, (torch.dtype, torch.device))
                    or (isinstance(value, tuple) and all(type(item) in _PLAIN for item in value))
                )
            ),
        )
        for m in module.modules()
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


__all__ = ["BYPASS", "capture", "install", "native", "release", "say"]
