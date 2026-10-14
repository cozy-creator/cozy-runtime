"""A model's attention selection policy, and the attention calls a package brings."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from cozy_runtime.author._attention_scope import AttentionLayout
from cozy_runtime.author.kernels import PreQuantized


@dataclass(frozen=True, slots=True)
class AttentionContext:
    """One attention site's eligible choices, supplied before model warm-up.

    Eligibility covers installed dependencies, hardware, site dtype/head size and
    parallelism support. It is not a model-quality approval. Returning an explicit
    backend still requires Runtime's kernel validation; unavailable choices refuse.
    Activation dtype comes from projection output contracts, with an ordinary floating
    Linear fallback (no external autocast). Unknown/mismatched contracts are empty.
    Processors without per-site dispatch have empty eligible/recommended values.
    Return None to leave them unchanged; an explicit choice refuses.
    Weight storage precision is deliberately separate from attention activation dtype.
    """

    component: str
    module_path: str
    device_kind: str
    device_name: str
    sm: int
    activation_dtype: str
    head_dim: int
    degree: int
    eligible: tuple[str, ...]
    recommended: str
    recommendation_reason: str
    #: Eligible kernels with a compatible Runtime-provided wheel. Others may build from source.
    prebuilt: tuple[str, ...] = ()


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
    """The `fp8-rowwise` leaf's activation quantizer: one operand all three projections share."""
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
    device admission, compiled build and dense kernel, with the model's `AttentionLayout`. At
    each site it is selected for, Runtime wraps the attention module's processor so the
    module's call reaches `produce(call, *args, **kwargs)`; `call.original` is the site's own
    processor. `validate(module)` raises ValueError, before any change, for a module it does
    not serve. `requires` names the callables it needs in the base's kernel module; one the
    build lacks makes the producer unavailable there, as a failed build does. `state()`
    makes one object per request on each GPU (`request_state`), closed (its `close()`, when
    it has one) when the request ends. `context_parallel` admits Ulysses sequence parallelism.
    """

    name: str
    base: str
    produce: Callable[..., Any]
    validate: Callable[[Any], None]
    requires: tuple[str, ...] = ()
    state: Callable[[], Any] = dict
    context_parallel: bool = False

    def request_state(self) -> Any | None:
        """This request's state on this GPU, made on first use; None outside a request."""
        store = _REQUEST.get()
        if store is None:
            return None
        if self.name not in store:
            store[self.name] = self.state()
        return store[self.name]


#: Producers this process's packages registered, by name.
_PRODUCERS: dict[str, AttentionProducer] = {}


def register_attention(producer: AttentionProducer) -> AttentionProducer:
    """Make `producer.name` selectable in this process (`Model.choose_attention`, a request's
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
