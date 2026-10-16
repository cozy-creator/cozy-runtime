"""CUDA graphs over a component's weights at their plane homes: one capture recipe for Runtime.

    from cozy_runtime import graphs

    with graphs.graphed(unet) as call:
        noise = call(latents, t, encoder_hidden_states=text, return_dict=False)[0]

A weight set keeps one device address range for its life, and a refill maps a region back at the
same address, so a graph that reads weights stays valid while they are bound there: no pinning.
The first call of each input signature runs eagerly through the paging hooks, which record the
blocks it reads. From the second on, `graphed` leases those blocks at their homes for the `with`
(`Residency.held`), records the call with its tensor inputs in static buffers once, and replays
it: each call copies its inputs in, replays, and returns clones of the outputs (a multistep
scheduler keeps earlier ones). Recordings outlive the `with` and the request: a later request at
the same signature replays from its first call, once its weights are bound where they were when
recorded (`Residency.layout`); moved, it records again. A signature is the tensors' shapes,
dtypes, devices and strides, every other argument by value, and the module's attention
processors (a kernel switch records again). Weights with LoRA baked in are not held, so those
calls run eagerly. Where the blocks do not fit, on a CPU, or when a capture fails, calls run
eagerly as they would have. The plane drops every recording before it unmaps weights for room
(`release`).

`capture(step, pool, stream)` is the recipe (`cozy_runtime.lm` uses it too): on a side stream,
in thread-local mode so the plane's own threads keep working while it records, and outside
Runtime's dispatch modes, whose out-of-memory recovery copies what an op writes into first,
which a graph would then copy on every replay.
"""

from __future__ import annotations

import sys
import weakref
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

from cozy_runtime.author._model import PLACED, RELEASERS

try:
    import torch
    from torch.utils import _pytree as pytree
    from torch.utils._python_dispatch import _disable_current_modes
except ImportError:  # pragma: no cover - the base package is torch-free
    torch = None

_SAID: set[str] = set()


def say(key: str, line: str) -> None:
    """`line` on stderr, once per process for `key`."""
    if key not in _SAID:
        _SAID.add(key)
        print(line, file=sys.stderr)


def capture(step: Callable[[], object], pool: Any, stream: Any) -> tuple[Any, str]:
    """`step` recorded as a CUDA graph into `pool` on `stream`: (graph, "") or (None, why).
    After a failure the pool is spent: the allocator keeps a released pool until its blocks
    are free, and a capture into it again trips its use count, so the next try takes a new one
    (`graph_pool_handle()`). An out-of-memory is said as `OutOfMemoryError`: the caller can run
    the step eagerly, whose recovery makes room, and try again."""
    graph = torch.cuda.CUDAGraph()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    failure: BaseException | None = None
    with _disable_current_modes(), torch.cuda.stream(stream):
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
    if failure is not None:
        return None, f"{type(failure).__name__}: {failure}".splitlines()[0][:300]
    return graph, ""


class _Recording:
    """One signature's graph, its static inputs and outputs, and where the weights it reads were
    bound when it was recorded: it replays only while they still are."""

    def __init__(self, graph: Any, inputs: list[Any], outputs: Any, layout: object) -> None:
        self.graph, self.inputs, self.outputs, self.layout = graph, inputs, outputs, layout

    def __call__(self, leaves: list[Any]) -> Any:
        for static, value in zip(self.inputs, leaves, strict=True):
            if static is not None:
                static.copy_(value)
        self.graph.replay()
        return pytree.tree_map(
            lambda v: v.clone() if isinstance(v, torch.Tensor) else v, self.outputs
        )


class _Recorder:
    """A module's recordings, kept across requests: the blocks its calls read (as the first one
    ran them through the paging hooks), and a recording per signature."""

    def __init__(self) -> None:
        self.blocks: dict[str, set[Any]] = {}
        self.seen: set[tuple[object, ...]] = set()
        self.failed: set[tuple[object, ...]] = set()
        self.recordings: dict[tuple[object, ...], _Recording] = {}


#: Every module's recordings in this process, until the module goes or memory is wanted.
_RECORDERS: weakref.WeakKeyDictionary[Any, _Recorder] = weakref.WeakKeyDictionary()


def release() -> bool:
    """Drop every recording (their graphs' memory pools and static buffers): what the plane
    calls before it unmaps weights for room. Returns whether there was any."""
    had = any(recorder.recordings for recorder in _RECORDERS.values())
    for recorder in _RECORDERS.values():
        recorder.recordings.clear()
    return had


RELEASERS.append(release)


def _clonable(out: Any) -> bool:
    """Whether every leaf of `out` is a tensor or a plain value, so a replay can hand clones."""
    leaves, _ = pytree.tree_flatten(out)
    return any(isinstance(v, torch.Tensor) for v in leaves) and all(
        v is None or isinstance(v, (torch.Tensor, int, float, bool, str)) for v in leaves
    )


def _structure(module: Any) -> tuple[object, ...]:
    """Which code each submodule runs: its class and any forward set on the instance. A package
    that swaps a block's forward between requests (SDXL's hidiffusion) records again."""
    return tuple((id(type(m)), id(m.__dict__.get("forward"))) for m in module.modules())


def _signature(module: Any, leaves: list[Any], spec: Any) -> tuple[object, ...] | None:
    """The tensors' geometry, every other argument by value, and the attention processors the
    module serves with now (a kernel switch records again)."""
    key: list[object] = [str(spec)]
    processors = getattr(module, "attn_processors", None)
    if isinstance(processors, dict):
        key.append(tuple(sorted((name, id(p)) for name, p in processors.items())))
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            key.append((tuple(leaf.shape), leaf.dtype, leaf.device, leaf.stride()))
            continue
        try:
            hash(leaf)
        except TypeError:
            return None
        key.append((type(leaf), leaf))
    return tuple(key)


@contextmanager
def graphed(module: Any) -> Iterator[Callable[..., Any]]:
    """`module`'s calls for the body, replayed from CUDA graphs recorded at each signature's
    second call and kept across requests. Open it in the `@uses_components` method that calls
    the module (once per step is fine: the recordings outlive the `with`)."""
    placed = PLACED.get(module)
    recorder = _RECORDERS.get(module)
    if recorder is None:
        recorder = _RECORDERS[module] = _Recorder()
    with ExitStack() as stack:
        state: dict[str, Any] = {"held": None, "layout": None}

        def hold() -> bool:
            if state["held"] is None:
                held: bool = True
                layout: tuple[object, ...] = ()
                if placed is not None:
                    residency, name = placed
                    blocks = recorder.blocks.get(name, set())
                    held = stack.enter_context(residency.held(name, blocks))
                    layout = residency.layout(name, blocks) if held else ()
                state["held"] = held
                state["layout"] = (layout, _structure(module)) if held else None
                state["stream"] = torch.cuda.Stream()
            return bool(state["held"])

        def call(*args: Any, **kwargs: Any) -> Any:
            leaves, spec = pytree.tree_flatten((args, kwargs))
            tensors = [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]
            key = (
                None if not tensors or not tensors[0].is_cuda else _signature(module, leaves, spec)
            )
            if key is None or key in recorder.failed:
                return module(*args, **kwargs)
            if key not in recorder.seen:
                # The first call of a signature: eager, through the hooks, which record its blocks.
                recorder.seen.add(key)
                if placed is not None and state["held"] is None:
                    with placed[0].recording() as read:
                        out = module(*args, **kwargs)
                    for name, blocks in read.items():
                        recorder.blocks.setdefault(name, set()).update(blocks)
                    return out
                return module(*args, **kwargs)
            if not hold():
                return module(*args, **kwargs)
            entry = recorder.recordings.get(key)
            if entry is not None and entry.layout != state["layout"]:
                del recorder.recordings[key]  # its weights moved: it is recorded again
                torch.cuda.empty_cache()  # into a new pool: the spent one's memory goes back
                entry = None
            if entry is None:
                inputs = [v.clone() if isinstance(v, torch.Tensor) else None for v in leaves]
                static = [s if s is not None else v for s, v in zip(inputs, leaves, strict=True)]
                cargs, ckwargs = pytree.tree_unflatten(static, spec)
                outputs: list[Any] = []
                graph, why = capture(
                    lambda: outputs.append(module(*cargs, **ckwargs)), None, state["stream"]
                )
                if graph is not None and not _clonable(outputs[0]):
                    graph, why = (
                        None,
                        f"its output {type(outputs[0]).__name__} is not a tree of tensors",
                    )
                if graph is None:
                    recorder.failed.add(key)
                    say(f"graphed:{type(module).__name__}", f"[graphs] eager: {why}")
                    return module(*args, **kwargs)
                entry = recorder.recordings[key] = _Recording(
                    graph, inputs, outputs[0], state["layout"]
                )
            return entry(leaves)

        yield call


__all__ = ["capture", "graphed", "release", "say"]
