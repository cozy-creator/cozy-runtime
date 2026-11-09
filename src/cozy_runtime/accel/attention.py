"""Sparse attention layouts and a package's own attention producers. Opt-in: a call with no
layout attends dense."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from cozy_runtime.accel.kernels import PreQuantized


@dataclass(frozen=True, slots=True)
class AttentionLayout:
    """One live document and its protected prefix, supplied by model inference.

    Dense paths are component-relative attention module prefixes. They protect
    refiners or sensitive layers without making a generic backend infer model
    semantics from tensor values. Exact paths are the layers whose dense rows take
    the exact kernel when the dense kernel quantizes (a layer whose keys a coarse
    scale cannot carry): Kitchen INT8 where the route has it, else BF16 SDPA. Runtime
    adds every layer whose key RMSNorm gain has outlier channels
    (`attention_sol.outlier_keys`), so a model names only the rest. Step indices are
    zero-based. `tau` is the sparse steps' block-selection threshold (Sol-Attn's: higher
    keeps fewer key blocks; NVIDIA's Regenerate presets run 1.0 and 1.75), read by every
    sparse route. `sparse` says which steps those are. This describes inference only; it
    does not authorize a backend or select a kernel.
    """

    live_tokens: int
    protected_prefix: int
    step: int
    dense_until_step: int = 0
    dense_paths: tuple[str, ...] = ()
    exact_paths: tuple[str, ...] = ()
    tau: float = 1.0

    def __post_init__(self) -> None:
        for name in ("live_tokens", "protected_prefix", "step", "dense_until_step"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.live_tokens or self.protected_prefix > self.live_tokens:
            raise ValueError("attention layout requires a live document containing its prefix")
        if type(self.tau) is not float or not 0.0 <= self.tau < float("inf"):
            raise ValueError("tau must be a finite nonnegative float")
        for name in ("dense_paths", "exact_paths"):
            paths = getattr(self, name)
            if not isinstance(paths, tuple) or any(
                not isinstance(path, str) or not path or path.startswith(".") or path.endswith(".")
                for path in paths
            ):
                raise ValueError(f"{name} must be a tuple of nonempty module prefixes")

    @property
    def sparse(self) -> bool:
        """Whether this step attends sparsely: past the dense steps, unless the protected prefix
        leaves half the score matrix exact (its rows against every key, every other row against
        its keys: a prefix of 29% or more). That half alone costs a dense step on 8-bit kernels,
        and routing costs more: an H3 plate shot (prefix 51%) ran its Sol steps 1.7x its dense
        ones on an RTX 4090 (xs-142). Such a step runs dense, on the kernel its route gives the
        protected rows."""
        n, p = self.live_tokens, self.protected_prefix
        return self.step >= self.dense_until_step and 2 * (2 * n * p - p * p) < n * n


_ACTIVE_LAYOUT: ContextVar[AttentionLayout | None] = ContextVar(
    "cozy_attention_layout", default=None
)


@contextmanager
def attention_scope(layout: AttentionLayout) -> Iterator[None]:
    """Apply `layout` to the model calls inside; restore the previous one even on failure."""
    if not isinstance(layout, AttentionLayout):
        raise TypeError("attention_scope requires an AttentionLayout")
    token = _ACTIVE_LAYOUT.set(layout)
    try:
        yield
    finally:
        _ACTIVE_LAYOUT.reset(token)


def active_attention_layout() -> AttentionLayout | None:
    """The layout the model supplied for the inference call in progress, or None."""
    return _ACTIVE_LAYOUT.get()


#: What the runtime does with an early length statement; the executor's attention installs it.
_EXPECTED: list[Callable[[int, Any], None]] = []


def expect_attention(live_tokens: int, module: Any) -> None:
    """State, before inference reaches it, the live document length `module`'s attention sites
    will see, so a kernel specialized on that length can start preparing now (h3a-087).

    Advisory only: the layout each forward supplies stays the authority, and a wrong length
    costs a wasted background compile, never an output."""
    if type(live_tokens) is int and live_tokens > 0:
        for listener in _EXPECTED:
            listener(live_tokens, module)


@dataclass(frozen=True, slots=True)
class ProducerCall:
    """One call of an attention module at a site Runtime selected a producer for."""

    module: Any
    component: str
    path: str
    layout: AttentionLayout
    kernel: Any
    """The base backend's imported kernel module (e.g. `comfy_kitchen`)."""
    state: Any
    """The producer's `state()` for this request on this GPU."""
    group: Any
    """The Ulysses process group under context parallelism; None on one GPU."""
    original: Callable[..., Any]
    """The site's own processor: dense steps and dense paths, served by the base backend."""
    quantize: Callable[[Any], PreQuantized]
    """The `fp8.rowwise` leaf's activation quantizer: one operand all three projections share."""
    exchange: Callable[[Any, Any, Any, Callable[[Any, Any, Any], Any]], Any]
    """`exchange(q, k, v, compute)`: the Ulysses all-to-all to whole sequences per head
    subset, `compute` there, and back."""
    served: Callable[[str], None]
    """Record a call the producer served sparse: its implementation, and the protected
    prefix the layout recomputed dense."""


@dataclass(frozen=True, slots=True)
class AttentionProducer:
    """A package's own attention call, selectable by `name` like any Runtime backend.

    Runtime serves it through `base`, one of its sparse (Sol-family) backends: the base's
    device admission, compiled build and dense kernel, with the model's `AttentionLayout`.
    Wherever the base is in a site's cell, the producer is tried just before it; a site it
    validates is a sparse site (its model supplies a layout there). At
    each site it is selected for, Runtime wraps the attention module's processor so the
    module's call reaches `produce(call, *args, **kwargs)`; `call.original` is the site's own
    processor. `validate(module)` raises ValueError, before any change, for a module it does
    not serve. `requires` names the callables it needs in the base's kernel module; one the
    build lacks makes the producer unavailable there, as a failed build does. `state()`
    makes one object per request on each GPU (`request_state`), closed (its `close()`, when
    it has one) when the request ends; what it holds for one loop the package frees after it.
    `context_parallel` admits Ulysses sequence parallelism;
    one without it steps to the next kernel under a group.

    `stateless(layout)`: whether, for a component call under `layout` made now, `produce`
    only passes each site's call on to `call.original` and keeps nothing (a dense step that
    seeds no statistics). The Runtime then records the component call as a CUDA graph with the
    sites' own processors and replays it, skipping `produce`. It is asked at each call, in the
    caller's context. Without it every call under a layout at the producer's sites runs as
    written.
    """

    name: str
    base: str
    produce: Callable[..., Any]
    validate: Callable[[Any], None]
    requires: tuple[str, ...] = ()
    state: Callable[[], Any] = dict
    context_parallel: bool = False
    stateless: Callable[[AttentionLayout], bool] | None = None

    def request_state(self) -> Any | None:
        """This request's state on this GPU, made on first use; None outside a request."""
        store = _REQUEST.get()
        if store is None:
            return None
        if self.name not in store:
            store[self.name] = self.state()
        return store[self.name]


@dataclass(frozen=True, slots=True)
class ProducerSite:
    """An attention module whose calls the Runtime routes to a producer."""

    module: Any
    component: str
    path: str
    producer: AttentionProducer


def producer_sites(root: Any) -> tuple[ProducerSite, ...]:
    """The attention modules under `root` a producer serves now (their site's backend is the
    producer's), in module order: where a package finds a producer's per-site state."""
    sites = []
    for module in root.modules():
        processor = getattr(module, "processor", None)
        producer = getattr(processor, "serving", None)
        if processor is not None and isinstance(producer, AttentionProducer):
            sites.append(ProducerSite(module, processor.component, processor.path, producer))
    return tuple(sites)


#: Producers this process's packages registered, by name.
_PRODUCERS: dict[str, AttentionProducer] = {}


def register_attention(producer: AttentionProducer) -> AttentionProducer:
    """Make `producer.name` selectable in this process (its base's cells, a request's
    `--attention-kernel`). A package registers at import; its name takes precedence."""
    _PRODUCERS[producer.name] = producer
    return producer


_REQUEST: ContextVar[dict[str, Any] | None] = ContextVar("cozy_attention_request", default=None)


@contextmanager
def _request(held: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Runtime's: one request's producer states on this GPU. A fresh store closes them when
    the scope ends; a `held` one (a follower's attempt, across its calls) by `_close`."""
    store = {} if held is None else held
    token = _REQUEST.set(store)
    try:
        yield store
    finally:
        _REQUEST.reset(token)
        if held is None:
            _close(store)


def _close(store: dict[str, Any]) -> None:
    states = list(store.values())
    store.clear()
    for state in states:
        close = getattr(state, "close", None)
        if callable(close):
            close()
