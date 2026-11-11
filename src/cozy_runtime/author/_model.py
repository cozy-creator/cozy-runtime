"""The `Model` boundary: its serving facts (`Serving`), component hints, the construction
transaction, and construction-identity coalescing.

A bare typed handler parameter (`model: Fl2VAModel`) is the whole slot declaration, and its
derived path (`generate.models.model`) is the binding point. A class states its serving facts
in one `serving = Serving(...)` attribute.

Construction identity is (package release, model class, exact artifact snapshot + immutable
config, construction/provider variant). Equal identities coalesce ONE single-flight model
generation; unequal ones stay distinct instances with independent lifecycle and admission.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import math
import sys
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Concatenate,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
    final,
)

import cozy_runtime.author._request_path as request_path
from cozy_runtime.author._context import (
    CURRENT,
    AdapterRef,
    Context,
    Device,
    RequestView,
    derive_seed,
    device,
)
from cozy_runtime.author._errors import (
    CapabilityError,
    ConformanceError,
    InvalidRequest,
    OutputError,
)
from cozy_runtime.author._handler import Handler
from cozy_runtime.author._loader import (
    Artifact,
    Backend,
    Config,
    ConstructionRecord,
    DerivedCache,
    Loader,
    Mode,
    ModelFitRefused,
    Scheduler,
    loading,
)
from cozy_runtime.author._stage_memo import StageMemo, is_memoized, memoized

if TYPE_CHECKING:
    import torch
    from torch.nn import Module

    from cozy_runtime.accel.kernels import Plan
    from cozy_runtime.graphs.domain import Domain, Pool
    from cozy_runtime.internal.residency import Binding, Region

_LOG = logging.getLogger(__name__)

PipelineT = TypeVar("PipelineT", covariant=True)
P = ParamSpec("P")
R = TypeVar("R")
ModelT = TypeVar("ModelT", bound="Model[object]")


@dataclass(frozen=True, slots=True)
class AdapterCompatibility:
    """Closed compatibility declaration for one generic model-slot adapter kind."""

    kind: str
    family: str
    components: tuple[str, ...]
    weight_min: float = -1.0
    weight_max: float = 1.0

    def accepts(self, overlay: AdapterRef) -> bool:
        return (
            overlay.kind == self.kind
            and (not self.family or overlay.family == self.family)
            and overlay.component in self.components
            and self.weight_min <= overlay.scale <= self.weight_max
        )


#: THE ENCODED-LEAF CONSENT, and the whole of the author's say in the delivery plane.
#:
#: Every other rung the runtime can choose is invisible to author code by construction: the
#: weights arrive as float tensors in the destinations the census named, and which stored
#: bytes they came from is not a question the package can ask. The `encoded_gemm` route is
#: the one that cannot be invisible. Its leaf has no `.weight` — the payload and its scales
#: ARE the weight, the module that multiplies by them is REPLACED, and a package that
#: reaches into `layer.weight`, re-dtypes a submodule, or rebuilds a `nn.Linear` from one
#: finds something that is not there.
#:
#: So it is DECLARED, once, on the class, and the default is REFUSE. This is not a knob and
#: it is not a request field: it is the author stating whether their code survives having a
#: linear op swapped underneath it, which is a fact about the code and nobody else can know
#: it. `EXCLUDED_KEYWORDS` keeps `quantize` out of every method signature for the opposite
#: and complementary reason — WHICH encoding serves is derived, never authored — and the two
#: rules meet exactly here: the author never picks the encoding, and the runtime never
#: replaces a leaf in a package that did not say it could take one.
#:
#: A string rather than a bool, and a closed vocabulary rather than an open one, because
#: "refuse" then has a spelling — an author can state the conservative answer instead of
#: only implying it.
ENCODED_LEAVES: Mapping[str, str] = {
    "refuse": "the runtime fills float destinations and never replaces a module; an "
    "artifact whose only qualified route on this device is native refuses typed",
    "accept": "this package's code touches its components through their FORWARD only — "
    "no `.weight` read, no per-module dtype walk, no submodule rebuilt from one — so a "
    "linear op may be replaced by a module whose weight is the stored role set",
}

#: `min_bits=`: the lowest weight precision the machine's lane rules may pick by themselves.
MIN_BITS: Mapping[int, str] = {
    16: "16-bit unless the request asks for fewer bits",
    8: "8-bit by default (the default)",
    4: "4-bit for a decode slot, or when the 8-bit weights exceed the GPU budget",
}

#: `phase=`: what bounds the slot's time, so the lane rules know when bytes matter most.
PHASES: Mapping[str, str] = {
    "compute": "diffusion or prefill, bound by arithmetic (the default)",
    "decode": "an LM's token loop, bound by weight bytes",
}


def closed(vocabulary: Collection[object], value: object) -> bool:
    """`value` is one of `vocabulary`'s keys, of the key's own type (no `8.0`, no `True`)."""
    return any(type(value) is type(key) and value == key for key in vocabulary)


def attention_bits(owner: str, value: object) -> int | dict[str, int]:
    """`attention_bits=`: the lowest attention precision the model's sites tolerate (16|8|4),
    whole or per component-qualified module prefix (`"fl2va_dit"`,
    `"transformer.single_transformer_blocks."`; the longest wins, unlisted sites 16)."""
    if closed(MIN_BITS, value):
        return cast(int, value)
    if (
        isinstance(value, Mapping)
        and value
        and all(type(k) is str and k and closed(MIN_BITS, v) for k, v in value.items())
    ):
        return dict(value)
    raise ConformanceError(
        f"{owner}(attention_bits={value!r}): 16, 8 or 4, or a mapping of "
        "component-qualified module prefixes to them",
        code="attention_bits_value",
    )


def coverage_patterns(owner: str, keyword: str, value: object) -> tuple[str, ...]:
    """`unloaded=` / `derived=`: component-qualified key patterns (`fnmatch`) a load does not
    warn about: stored tensors the model deliberately leaves unloaded (a text encoder's LM
    head), persistent buffers it deliberately derives instead of reading from the checkpoint."""
    if isinstance(value, (tuple, list)) and value and all(type(p) is str and p for p in value):
        return tuple(value)
    raise ConformanceError(
        f"{owner}({keyword}={value!r}): a tuple of component-qualified key patterns "
        '(`"text_encoder.lm_head.weight"`, `"model.mtp.*"`)',
        code=f"{keyword}_value",
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Serving:
    """A Model class's serving facts, as its `serving` attribute. Every field has a default, so
    a class states only what differs; a subclass's `serving` replaces its base's.

    - `encoded_leaves`: the Runtime may replace a linear with one whose weight is its stored
      encoding (the code reaches its components through their forward only). False refuses.
    - `min_bits`: the lowest weight precision the lane rules pick by themselves (`MIN_BITS`).
    - `attention_bits`: the lowest attention precision its sites tolerate, whole or per
      component-qualified module prefix.
    - `phase`: "decode" for an LM's token loop, else "compute" (`PHASES`).
    - `unloaded` / `derived`: component-qualified key patterns a load does not warn about.
    - `loras`: the components that take LoRAs.
    """

    encoded_leaves: bool = True
    min_bits: Literal[16, 8, 4] = 8
    attention_bits: Literal[16, 8, 4] | Mapping[str, Literal[16, 8, 4]] = 16
    phase: Literal["compute", "decode"] = "compute"
    unloaded: tuple[str, ...] = ()
    derived: tuple[str, ...] = ()
    loras: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.encoded_leaves) is not bool:
            raise ConformanceError(
                f"Serving(encoded_leaves={self.encoded_leaves!r}): True or False",
                code="encoded_leaves_value",
            )
        for keyword, vocabulary in (("min_bits", MIN_BITS), ("phase", PHASES)):
            value = getattr(self, keyword)
            if not closed(vocabulary, value):
                raise ConformanceError(
                    f"Serving({keyword}={value!r}): one of "
                    + " | ".join(f"{k!r}: {why}" for k, why in vocabulary.items()),
                    code=f"{keyword}_value",
                )
        attention_bits("Serving", self.attention_bits)
        for keyword in ("unloaded", "derived"):
            if getattr(self, keyword):
                coverage_patterns("Serving", keyword, getattr(self, keyword))
        names = self.loras
        if not isinstance(names, tuple) or any(type(n) is not str or not n for n in names):
            raise ConformanceError(
                f"Serving(loras={names!r}): a tuple of component names", code="loras_value"
            )


def uncovered(
    cls: type, ignored: Sequence[str], unstored: Sequence[str]
) -> tuple[list[str], list[str]]:
    """What a load of `cls` warns about: the stored tensors it did not load and the persistent
    buffers no checkpoint supplied (`component/key`), less what the class declares."""

    def declared(names: Sequence[str], patterns: Sequence[str]) -> list[str]:
        return [
            n for n in names if not any(fnmatchcase(n.replace("/", ".", 1), p) for p in patterns)
        ]

    return (
        declared(ignored, getattr(cls, "__unloaded__", ())),
        declared(unstored, getattr(cls, "__derived__", ())),
    )


#: Keyword names an author-facing model method may not declare. Every one of them names a
#: residency, placement or conversion decision the runtime owns (§1.1/§7).
EXCLUDED_KEYWORDS: Mapping[str, str] = {
    "device": "placement is the runtime's; the method wrapper exposes no device",
    "dtype": "compute dtype is a plan decision, not a request knob",
    "torch_dtype": "compute dtype is a plan decision, not a request knob",
    "path": "no filesystem path reaches package code",
    "checkpoint_path": "no filesystem path reaches package code",
    "weights_path": "no filesystem path reaches package code",
    "offload": "there is no manual load/offload/staging API in request code",
    "pin": "pinning is the residency backend's",
    "pinned": "pinning is the residency backend's",
    "evict": "eviction policy is the runtime's",
    "eviction": "eviction policy is the runtime's",
    "compile": "no compile surface is nameable from author code",
    "convert": "source conversion is TensorFS ingest, never a serve-time hook",
    "converter": "source conversion is TensorFS ingest, never a serve-time hook",
    "quantize": "encoded delivery is derived, never authored per call",
    "low_cpu_mem_usage": "residency is the runtime's, not a construction flag",
}

_RUNTIME_MEMBERS = frozenset(
    {
        "load",
        "unload",
        "for_request",
        "checkpoint_ref",
        "serving",
    }
)


# --------------------------------------------------------------------------- scopes


class _ScopedModel(Protocol):
    """What `@uses_components` needs from the instance it wraps — so the decorator is
    ParamSpec-preserving and fully typed without an `Any` seam (§1.7)."""

    def _cozy_scope(self, method: str, components: tuple[str, ...]) -> _Scope: ...

    @property
    def _cozy_memo(self) -> StageMemo | None: ...


M = TypeVar("M", bound=_ScopedModel)


class _Declared(Protocol):
    """A declared method carries its component set as interface content (§1.1)."""

    __uses_components__: tuple[str, ...]
    __deterministic__: bool


class _Scope(Protocol):
    def __enter__(self) -> None: ...

    def __exit__(self, *exc: object) -> None: ...


LoRAServing = tuple[Literal["bake", "per_step"], Literal["eager", "fused"] | None]


class Residency(Protocol):
    """What a Runtime mechanism handed a placed component reaches, installed on a constructed
    generation. Runtime's local residency scopes bind weights around each forward.

    Declared here because the contract has one home; implemented nowhere in this package,
    because it needs a device and this package is torch-free by construction. cozy-runtime
    supplies the body in `internal/weights.py`'s `Placement`.
    """

    def note(self, key: str, value: str) -> None:
        """A fact about how this attempt executed, reported with its memory facts."""
        ...

    def open_attempt(self) -> None:
        """An attempt begins on this placement."""
        ...

    def retain(self, component: str, paths: tuple[str, ...]) -> Binding:
        """Retain ordinary weight storage after prepare(), until the binding closes."""
        ...

    def represent(
        self, component: str, dtype: torch.dtype, *, resident_only: bool = False
    ) -> Callable[[], None]:
        """Validate a final representation and return its pre-capture metadata transition."""
        ...

    def graph_accounted(self) -> bool:
        """Whether per-process driver/context bytes can be included in graph admission."""
        ...

    def graph_domain(self, component: str) -> Domain:
        """Shared capture stream and library workspace, serialized across component graphs."""
        ...

    def region(
        self, owner: str, kind: str, sets: Module | Sequence[Module], *,
        release: Callable[[], None] | None = None, shrink: Callable[[], bool] | None = None,
    ) -> Region:
        """An explicitly owned cache, released only while it has no active user."""
        ...

    def graph_pool(self, component: str) -> Pool:
        """`component`'s ordinary Torch graph pool, owned with its cached recordings."""
        ...

    def graph_pass(self, component: str, entry: str) -> AbstractContextManager[object]:
        """Open the ordinary model pass before capture; nested regions share it."""
        ...

    def graph_idle(self) -> AbstractContextManager[object]:
        """An idle capture's memory request: free bytes only, never an eviction."""
        ...

    def graph_held(self, component: str) -> torch.cuda.Stream | None:
        """The stream `component`'s blocks were held bound on for its plan step, or None."""
        ...

    def using(self, method: str, components: tuple[str, ...]) -> AbstractContextManager[None]:
        """A local scope of `components` (`method`'s call): their memory passes and holds."""
        ...

    def graph_resident(self, binding: Binding) -> bool:
        """Every page a retained region covers is on the device now."""
        ...

    def lora_choices(self, components: tuple[str, ...]) -> dict[str, LoRAServing]:
        """The attempt's selected serving recipe, carried with each mirrored scope."""
        ...

    def mirrored_lora(self, choices: Mapping[str, LoRAServing]) -> AbstractContextManager[None]:
        """Apply the leader's recipes in the follower's call context."""
        ...

    #: How many attempts (requests) it has opened.
    attempts: int


#: A component's root module -> the residency that places it and the component's name, so a
#: Runtime mechanism handed the module (`cozy_runtime.lm`) reaches its placement.
PLACED: weakref.WeakKeyDictionary[object, tuple[Residency, str]] = weakref.WeakKeyDictionary()
#: Kernel builds this process loaded after it started (a boot build landing): calls may take a
#: faster path from each on, so graphs record them again. None loads while a graph records.
LANDED: list[int] = [0]
#: The attribute Runtime sets on a forward wrapper that calls the class's forward unchanged.
TRANSPARENT = "_cozy_transparent"


def hold_dtype(module: torch.nn.Module, dtype: torch.dtype) -> None:
    """Serve a component root in floating `dtype` from now on. A placed one keeps its stored
    bytes; the Runtime casts them when it next fills the weights, so nothing loads here."""
    if next(module.parameters()).dtype == dtype:
        return
    placed = PLACED.get(module)
    if placed is None:
        module.to(dtype=dtype)
    else:
        placed[0].represent(placed[1], dtype)()


def plain_forward(module: torch.nn.Module) -> bool:
    """Whether calling `module` runs its class's forward: it has no forward of its own but the
    Runtime's transparent wrappers, which call the class's forward at each call."""
    own = vars(module).get("forward")
    return own is None or bool(getattr(own, TRANSPARENT, False))


class Building(Exception):
    """Raised in a CUDA graph capture by a call whose kernel is still building and whose
    stand-in cannot record: a recording now would only bake a slower path, so the graphs module
    records once a kernel lands (`LANDED`)."""


class Yielded(Exception):
    """An idle job gives the device back at its next op: a command arrived. An Exception, so a
    module's `always_call` forward hooks run as it unwinds (residency's block exits: as a
    BaseException it left a block claimed and the next call refused "concurrent model use");
    a kernel's fallback passes it on rather than serve a slower path or a failed verdict."""


type WarningRow = tuple[str, dict[str, str | bool | int | float | None]]


class _Warnings:
    """Warnings Runtime's speed mechanisms raise (a graph that would not record or replayed
    wrong), as (message, fields). Each lands in the attempt whose context raised it; one raised
    outside every attempt lands in the next attempt to end."""

    def __init__(self) -> None:
        self._process: list[WarningRow] = []
        self._lock = threading.Lock()
        self._attempt: contextvars.ContextVar[list[WarningRow] | None] = contextvars.ContextVar(
            "cozy_warnings", default=None
        )

    def _rows(self) -> list[WarningRow]:
        rows = self._attempt.get()
        return self._process if rows is None else rows

    def _taken(self) -> list[WarningRow]:
        with self._lock:
            rows, self._process = self._process, []
        return rows

    def append(self, row: WarningRow) -> None:
        self._rows().append(row)

    def clear(self) -> None:
        self._rows().clear()

    def __contains__(self, row: object) -> bool:
        return row in self._rows()

    def __iter__(self) -> Iterator[WarningRow]:
        return iter(list(self._rows()))

    def __len__(self) -> int:
        return len(self._rows())

    @contextmanager
    def attempt(self) -> Iterator[list[WarningRow]]:
        """This context's warnings until it ends, with those raised outside every attempt."""
        rows = self._taken()
        token = self._attempt.set(rows)
        try:
            yield rows
        finally:
            self._attempt.reset(token)
            rows += self._taken()


WARNINGS = _Warnings()


class _Scopes:
    """Calls take component scopes one at a time, in the order they ask: a call entering a scope
    while another call holds one waits for it, told which run (or the Runtime's kernel swap)
    holds it. Within one call, on any of its threads, scopes nest as they always have; a thread
    outside every call (`CURRENT` unset) never waits.

    Known limit: a call's threads are those that carry its context (SDK pools, its event loop's
    executor). A plain `threading.Thread` an author starts carries none, so it enters scopes
    as no call, even inside another call's turn. Harmless with one call at a time; batching
    calls in one executor (cr-202 U4) must close it before authors' plain threads matter; one
    entering during another call's turn warns (`busy`)."""

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._holder: object = None
        self._depth = 0
        self._queue: deque[object] = deque()

    def enter(self, call: object) -> None:
        with self._changed:
            if self._holder is not call:
                ticket = object()
                self._queue.append(ticket)
                request_path.wait(
                    self._changed,
                    lambda: (
                        self._holder is call or (self._holder is None and self._queue[0] is ticket)
                    ),
                    self._held_by,
                )
                self._queue.remove(ticket)
                self._holder = call
                self._changed.notify_all()  # the call's other threads waiting now pass
            self._depth += 1

    def _held_by(self) -> str:
        holder = self._holder
        if holder is None:
            return ""
        return f"run {holder.request_id}" if isinstance(holder, Context) else "a kernel swap"

    def leave(self) -> None:
        with self._changed:
            self._depth -= 1
            if not self._depth:
                self._holder = None
                self._changed.notify_all()

    def busy(self) -> bool:
        """Whether some call holds the turn."""
        return self._holder is not None

    @contextmanager
    def held(self, call: object) -> Iterator[None]:
        """`call`'s turn: while it lasts, no other call is in a scope."""
        self.enter(call)
        try:
            yield
        finally:
            self.leave()


SCOPES = _Scopes()


def graph_keys(**components: tuple[str, ...]) -> Callable[[type[ModelT]], type[ModelT]]:
    """Declare, per component, the attributes of its root module whose values steer its forward
    (`unet=("hidiffusion_active",)`; a dotted path reaches a submodule's): Runtime keys the
    component's CUDA graphs by their values at each call. An attribute that changes between
    calls and is not declared is reported (`task conform` fails on it)."""
    for name, paths in components.items():
        if not isinstance(paths, tuple) or not all(
            isinstance(p, str) and all(part.isidentifier() for part in p.split(".")) for p in paths
        ):
            raise ConformanceError(
                f"@graph_keys({name}=...): a tuple of attribute paths (`a` or `a.b`)",
                code="graph_keys",
                fields=[name],
            )

    def decorate(cls: type[ModelT]) -> type[ModelT]:
        if not (inspect.isclass(cls) and issubclass(cls, Model)):
            raise ConformanceError("@graph_keys decorates a Model class", code="graph_keys")
        cls.__graph_keys__ = dict(components)
        return cls

    return decorate


def uncaptured(*names: str) -> Callable[[type[ModelT]], type[ModelT]]:
    """Declare components whose calls Runtime never records as CUDA graphs: a component whose
    calls depend on state outside their arguments (a cache it mutates, a host-side branch on
    tensor values) runs as written."""
    if not names or any(
        not isinstance(name, str) or not name.replace("_", "").isalnum() for name in names
    ):
        raise ConformanceError(
            f"@uncaptured{names!r}: name one or more components by plain identifier",
            code="uncaptured_components",
        )

    def decorate(cls: type[ModelT]) -> type[ModelT]:
        if not (inspect.isclass(cls) and issubclass(cls, Model)):
            raise ConformanceError(
                "@uncaptured decorates a Model class", code="uncaptured_components"
            )
        cls.__uncaptured__ = tuple(dict.fromkeys(names))
        return cls

    return decorate


class Placement(Protocol):
    """Where a generation's components live across the GPUs of its group, installed by the
    runtime on every GPU of a group that hosts a `placeable` component on one of them.

    A scope whose declared components ALL live on other GPUs holds nothing on this device:
    its module calls run on their home GPUs, and their results cross here. Such a REMOTE scope
    is admitted there, not here.
    """

    def remote(self, components: tuple[str, ...]) -> bool: ...


#: The remote scopes open in this context, innermost last: (model, method, components).
#: Per context rather than per model, so a remote scope on a helper thread is visible to that
#: thread's hosted calls and to nothing else.
#: Open scopes, innermost last: (model, method, components).
type _OpenScopes = tuple[tuple[Model[object], str, tuple[str, ...]], ...]
_REMOTE_SCOPES: contextvars.ContextVar[_OpenScopes] = contextvars.ContextVar(
    "cozy_remote_scopes", default=()
)
#: The local scopes open in this context, innermost last: (model, method, components).
_LOCAL_SCOPES: contextvars.ContextVar[_OpenScopes] = contextvars.ContextVar(
    "cozy_local_scopes", default=()
)
#: Whether a `deterministic=True` scope is open in this context.
_DETERMINISTIC_SCOPE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "cozy_deterministic_scope", default=False
)
#: How a local `deterministic=True` scope sets deterministic cuDNN on this process's device:
#: the Runtime's numerics policy (`internal/numerical_environment.apply`) installs it at
#: executor start. Unset (no executor, no Torch), a scope changes nothing.
DETERMINISTIC: list[Callable[[], AbstractContextManager[None]]] = [nullcontext]


#: What a seal's `enter` hands its `leave`: the refusal count, finer splits and shared-scope count
#: at entry, or None when nothing reads refusals (`internal/contract._Scopes`).
type Sealing = tuple[int, int, int] | None


class Sealed(Protocol):
    def enter(self) -> Sealing: ...
    def leave(self, entered: Sealing, method: str, ok: bool) -> None: ...


#: The package contract's checks at each local scope (`internal/contract._Scopes`: the flags a
#: package may not move, a refusal it returned past), installed when the executor seals torch.
CONTRACT: list[Sealed | None] = [None]


def deterministic(model: object, method: str) -> bool:
    """Whether `model.method` runs under deterministic cuDNN: it declares `deterministic=True`,
    or it is called inside a scope that does. Stage memos key on it."""
    return _DETERMINISTIC_SCOPE.get() or _declares_deterministic(model, method)


def _declares_deterministic(model: object, method: str) -> bool:
    return getattr(getattr(type(model), method, None), "__deterministic__", False) is True


def remote_scope(model: object) -> tuple[str, tuple[str, ...]] | None:
    """The remote scope of `model` open in this context, if any: (method, components)."""
    for owner, method, components in reversed(_REMOTE_SCOPES.get()):
        if owner is model:
            return method, components
    return None


def uses_components(
    *names: str, deterministic: bool = False
) -> Callable[[Callable[Concatenate[M, P], R]], Callable[Concatenate[M, P], R]]:
    """Optional: name the components ONE public `Model` method may touch, so the Runtime fetches
    and holds them for the call. Without it the method just runs, and each component is staged
    at its first forward. Where every public method of a class has one, a model argument of that
    class fetches only the components they name.

    `deterministic=True`: the method, and what it calls on this GPU, runs with cuDNN
    deterministic and without benchmarking, as do its component calls on a group's other
    GPUs; the setting is restored when the method returns, and `@memoize` keys on it. Every
    other global numeric setting is the Runtime's policy; a package never sets
    `torch.backends.*`.
    """
    components = component_set(names)
    if not isinstance(deterministic, bool):
        raise ConformanceError(
            f"@uses_components(deterministic={deterministic!r}): True or False",
            code="deterministic_value",
        )

    def decorate(fn: Callable[Concatenate[M, P], R]) -> Callable[Concatenate[M, P], R]:
        memo = is_memoized(fn)
        if memo:  # `@uses_components` above `@memoize`: the lookup still comes first
            fn = cast("Callable[Concatenate[M, P], R]", inspect.unwrap(fn))
        component_method(fn.__qualname__, inspect.signature(fn).parameters)
        wrapper = _async_wrapper(fn, components) if _is_async(fn) else _sync_wrapper(fn, components)
        if memo:
            wrapper = memoized(fn, wrapper, components, is_async=_is_async(fn))
        declared: _Declared = cast("_Declared", wrapper)
        declared.__uses_components__ = components
        declared.__deterministic__ = deterministic
        return wrapper

    return decorate


def memoize[M: _ScopedModel, **P, R](
    fn: Callable[Concatenate[M, P], R],
) -> Callable[Concatenate[M, P], R]:
    """Mark ONE `Model` method's result a pure function of its code, the weights of the
    components it uses (those `@uses_components` names, else all of the model's), its arguments
    and the numerics it runs under. The Runtime looks for a stored result before anything is
    staged, so a hit stages nothing. Use it alone or with `@uses_components`, in either order."""
    components: tuple[str, ...] = getattr(fn, "__uses_components__", ())
    target = cast("Callable[Concatenate[M, P], R]", inspect.unwrap(fn))
    wrapper = memoized(target, fn, components, is_async=_is_async(target))
    if components:
        declared: _Declared = cast("_Declared", wrapper)
        declared.__uses_components__ = components
        declared.__deterministic__ = bool(getattr(fn, "__deterministic__", False))
    return wrapper


def sequence_parallel(*, degrees: Sequence[int]) -> Callable[[type[ModelT]], type[ModelT]]:
    """Declare that this Model class's attention is head-shardable at `degrees` (cr-068).

    The author's statement, and only the author can make it: at each degree K the class's
    attention heads divide by K, its quantization is shard-invariant (rowwise activation
    scales), and the whole pipeline is built identically on every GPU. The runtime reads
    it into the interface slot (`sequence_parallel.degrees`). Every Model slot of one
    callable declares a common degree, or the package is refused. A call runs on the largest
    declared degree the machine's GPUs can form (one GPU only on a machine narrower than
    every degree). "Qualified" is the measured half, recorded per device class.
    """
    if not degrees:
        raise ConformanceError(
            "@sequence_parallel(degrees=()) declares nothing: omit the decorator instead",
            code="sequence_parallel_degrees",
        )
    values: list[int] = []
    for degree in degrees:
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 2:
            raise ConformanceError(
                f"@sequence_parallel(degrees={list(degrees)!r}): a degree is an int >= 2",
                code="sequence_parallel_degrees",
            )
        values.append(degree)
    if values != sorted(set(values)):
        raise ConformanceError(
            f"@sequence_parallel(degrees={list(degrees)!r}): degrees are sorted and unique",
            code="sequence_parallel_degrees",
        )

    def decorate(cls: type[ModelT]) -> type[ModelT]:
        if not (inspect.isclass(cls) and issubclass(cls, Model)):
            raise ConformanceError(
                "@sequence_parallel decorates a Model class",
                code="sequence_parallel_placement",
            )
        cls.__sequence_parallel__ = tuple(values)
        return cls

    return decorate


def placeable(*names: str) -> Callable[[type[ModelT]], type[ModelT]]:
    """Declare components the runtime may host on another GPU of this Model's group.

    The author's statement: every use of each named component is a module call (its root or
    a submodule) whose arguments and result are tensors, JSON values or model outputs, with
    no effect beside that result. When the group's first GPU cannot hold every component, the
    runtime may then keep one resident on another GPU instead of re-staging it for every
    request.
    """
    if not names or any(
        not isinstance(name, str) or not name.replace("_", "").isalnum() for name in names
    ):
        raise ConformanceError(
            f"@placeable{names!r}: name one or more components by plain identifier",
            code="placeable_components",
        )

    def decorate(cls: type[ModelT]) -> type[ModelT]:
        if not (inspect.isclass(cls) and issubclass(cls, Model)):
            raise ConformanceError(
                "@placeable decorates a Model class", code="placeable_components"
            )
        cls.__placeable__ = tuple(dict.fromkeys(names))
        return cls

    return decorate


def _is_async(fn: Handler) -> bool:
    return inspect.iscoroutinefunction(inspect.unwrap(fn))


def _sync_wrapper[M: _ScopedModel, **P, R](
    fn: Callable[Concatenate[M, P], R], components: tuple[str, ...]
) -> Callable[Concatenate[M, P], R]:
    @functools.wraps(fn)
    def wrapper(self: M, /, *args: P.args, **kwargs: P.kwargs) -> R:
        with self._cozy_scope(fn.__name__, components):
            steps = _entered(fn.__name__)
            result = fn(self, *args, **kwargs)
            _refuse_lazy(fn.__name__, result)
            _refuse_nan(fn.__name__, result, steps)
            return result

    return wrapper


def _async_wrapper[M: _ScopedModel, **P, R](
    fn: Callable[Concatenate[M, P], R], components: tuple[str, ...]
) -> Callable[Concatenate[M, P], R]:
    """The scope stays active THROUGH `await` — the lease is the method's, not the step's."""

    @functools.wraps(fn)
    async def wrapper(self: M, /, *args: P.args, **kwargs: P.kwargs) -> object:
        with self._cozy_scope(fn.__name__, components):
            steps = _entered(fn.__name__)
            result = await fn(self, *args, **kwargs)  # type: ignore[misc]
            _refuse_lazy(fn.__name__, result)
            _refuse_nan(fn.__name__, result, steps)
            return result

    return cast("Callable[Concatenate[M, P], R]", wrapper)


def component_set(names: Sequence[object]) -> tuple[str, ...]:
    """One `@uses_components` hint's names."""
    if not names:
        raise ConformanceError(
            "@uses_components() names no component: leave the decorator off",
            code="empty_component_set",
        )
    seen: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
            raise ConformanceError(
                f"@uses_components({name!r}): component names are plain identifiers",
                code="component_name",
            )
        if name in seen:
            raise ConformanceError(
                f"@uses_components: {name!r} is declared twice", code="duplicate_component"
            )
        seen.append(name)
    return tuple(seen)


def component_method(qualname: str, parameters: Iterable[str]) -> None:
    """Where `@uses_components` may stand: a public, author-owned method that takes no
    residency keyword. Both readers apply it: import at decoration, source per method."""
    name = qualname.rpartition(".")[2]
    if name in _RUNTIME_MEMBERS:
        raise ConformanceError(
            f"@uses_components on {name}: lifecycle and runtime-owned members do "
            "not open component scopes — decorate the public method that forms the "
            "runtime boundary",
            code="component_placement",
        )
    if name.startswith("_"):
        raise ConformanceError(
            f"@uses_components on {name}: private helpers inherit their caller's "
            "active scope and cannot be called as handler operations",
            code="component_placement",
        )
    for parameter in parameters:
        if (why := EXCLUDED_KEYWORDS.get(parameter)) is not None:
            raise ConformanceError(
                f"{qualname}({parameter}=): {why} — the method wrapper exposes no device, "
                "movement, allocation, offload, pinning, eviction or residency backend",
                code="excluded_keyword",
                fields=[parameter],
            )


def _refuse_lazy(method: str, result: object) -> None:
    if inspect.isgenerator(result) or inspect.iscoroutine(result) or inspect.isasyncgen(result):
        raise CapabilityError(
            f"{method} returned a {type(result).__name__}: a lazy object that touches model "
            "components after the method completes escapes its component-use lease",
            code="lazy_escape",
        )


def _steps() -> int:
    context = CURRENT.get()
    return 0 if context is None else context._steps


def _entered(method: str) -> int:
    """A model method call begins: the run shows it as activity (`Context._called`)."""
    context = CURRENT.get()
    if context is None:
        return 0
    if context._called is not None:
        context._called(method)
    return context._steps


def _refuse_nan(method: str, result: object, before: int) -> None:
    """A method that ran steps hands back no NaN. A package's own forward reaches no root's
    per-step check (`graphs.Finite`): its latents pass this one, as its loop ends."""
    if _steps() == before:
        return
    for value in result if isinstance(result, tuple | list) else (result,):
        floating = getattr(value, "is_floating_point", None)
        if floating is not None and floating() and math.isnan(float(value.float().sum())):
            raise OutputError(f"the latents went non-finite in {method}", code="non_finite_latents")


# --------------------------------------------------------------------------- model


class Model[PipelineT]:
    """Base of a package's reusable model class — the ONE long-lived model boundary.

    The author owns WHAT the model is: the lifecycle methods `load` (construct under fake
    tensors, then fill) and `unload` (release when the construction leaves the worker), the declared
    persistent state, and the component-touching methods handlers call. The runtime owns
    WHEN they run and where every byte lives (model-lifecycle.md). A bare typed parameter
    is the whole declaration, and its presence in a signature is what derives `gpu`.
    """

    __sequence_parallel__: tuple[int, ...] = ()
    """`@sequence_parallel(degrees=...)`: the degrees this class shards at. Empty is the
    honest default - a class that said nothing is never sharded."""

    __placeable__: tuple[str, ...] = ()
    """`@placeable(...)`: components the runtime may host on another GPU of the group."""
    __uncaptured__: tuple[str, ...] = ()
    """`@uncaptured(...)`: components whose calls are never recorded as CUDA graphs."""
    __graph_keys__: Mapping[str, tuple[str, ...]] = MappingProxyType({})
    """`@graph_keys(...)`: per component, the root attributes that key its CUDA graphs."""

    serving: Serving = Serving()
    """The class's serving facts. The dunders below are the Runtime's reading of it."""
    __encoded_leaves__: str = "accept"
    __min_bits__: int = 8
    __phase__: str = "compute"
    __attention_bits__: int | dict[str, int] = 16
    __unloaded__: tuple[str, ...] = ()
    __derived__: tuple[str, ...] = ()
    __adapter_compatibility__: tuple[AdapterCompatibility, ...] = ()

    checkpoint_ref: str = ""
    _cozy_selection_ref: str = ""
    """The pinned-ref REPRODUCTION FACT for result structs — an observation, never catalog
    authority. Model-scoped because it is construction identity."""

    _cozy_ready = False
    _cozy_loading = False
    _cozy_record: ConstructionRecord | None = None
    _cozy_identity: ConstructionIdentity | None = None
    _cozy_schedulers: Mapping[str, Scheduler] = {}
    _cozy_caches: tuple[DerivedCache, ...] = ()
    _cozy_adapters: tuple[AdapterRef, ...] = ()
    _cozy_residency: Residency | None = None
    """Runtime-installed; absent when package code runs outside an executor."""
    _cozy_placement: Placement | None = None
    """Runtime-installed on the first GPU of a group that hosts a component on another."""
    _cozy_memo: StageMemo | None = None
    """Runtime-installed: where `memoize=True` methods look before their scope opens."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        serving = cls.__dict__.get("serving")
        if serving is None:
            return
        if not isinstance(serving, Serving):
            raise ConformanceError(
                f"class {cls.__name__}: serving is a Serving(...)", code="serving_value"
            )
        bits = serving.attention_bits
        cls.__encoded_leaves__ = "accept" if serving.encoded_leaves else "refuse"
        cls.__min_bits__ = serving.min_bits
        cls.__phase__ = serving.phase
        cls.__attention_bits__ = bits if isinstance(bits, int) else dict(bits)
        cls.__unloaded__ = serving.unloaded
        cls.__derived__ = serving.derived
        cls.__adapter_compatibility__ = (
            (AdapterCompatibility("lora", "", serving.loras, -math.inf, math.inf),)
            if serving.loras
            else ()
        )

    # ----------------------------------------------------------------- lifecycle

    def load(self, loader: Loader) -> None:
        """Construct this model from the bound artifact. The runtime owns WHEN it runs."""
        raise ConformanceError(
            f"{type(self).__name__} defines no load(loader): a Model constructs through "
            "loader.construct(factory) — the one construction primitive",
            code="no_load",
        )

    def execution_plan(self, plan: Plan) -> str:
        """Install this model's execution plan on its filled components: its own fused
        kernels, glue and hooks. Runtime calls it once per construction on every GPU, after
        the fill and before attention selection and sequence parallelism, outside the
        construction guards: it may load and launch the package's kernels (`plan.load`).
        Returns the installed plan's identity ("" for none), which enters the constructed
        generation's identity. A raise refuses the construction, typed by its `code`."""
        return ""

    def unload(self, loader: Loader) -> None:
        """Release declared state. The runtime drains before calling this."""
        return None

    @property
    def device(self) -> Device:
        """Where this model's calls take and return tensors: the device the runtime runs it on
        (`author.device()`: its GPU, or the CPU where there is none). A package never picks
        one; placement is the runtime's, and reading it is not a door."""
        return device()

    @property
    def _cozy_active(self) -> tuple[str, tuple[str, ...]] | None:
        """This context's innermost open local scope of this instance: (method, components)."""
        for owner, method, components in reversed(_LOCAL_SCOPES.get()):
            if owner is self:
                return method, components
        return None

    def prepare_adapters(self, overlays: tuple[AdapterRef, ...]) -> None:
        """Apply an admitted ordered overlay once during model preparation."""
        if overlays:
            raise CapabilityError(
                f"{type(self).__name__} does not declare a generic adapter composer",
                code="adapter_composition_unsupported",
            )

    @classmethod
    def validate_adapter_stack(cls, overlays: tuple[AdapterRef, ...]) -> None:
        """Validate ordered overlays against the model's closed compatibility declarations."""
        for index, overlay in enumerate(overlays):
            if not any(
                declaration.accepts(overlay) for declaration in cls.__adapter_compatibility__
            ):
                raise CapabilityError(
                    f"adapter {index} ({overlay.kind}:{overlay.component}) is incompatible "
                    f"with {cls.__name__}",
                    code="adapter_incompatible",
                )

    # ----------------------------------------------------------------- request

    def for_request(
        self, ctx: Context, *, seed: int | None = None, sampler: str | None = None
    ) -> RequestView:
        """The request-local view: fresh schedulers, the seeded generator, the applied stack.

        Request-local state lives where its lifetime is true — never as mutable fields on
        the long-lived model.
        """
        self._cozy_require_ready("for_request")
        if sampler is not None:
            compatible = {name for s in self._cozy_schedulers.values() for name in s.compatibles}
            if sampler not in compatible:
                raise InvalidRequest(
                    f"sampler {sampler!r} is not in this family's compatibility table "
                    f"({', '.join(sorted(compatible)) or 'none'})",
                    code="incompatible_sampler",
                    fields=["sampler"],
                )
        effective = derive_seed(ctx.request_id) if seed is None else seed
        return RequestView._make(effective, self._cozy_adapters, self._cozy_schedulers, sampler)

    # ----------------------------------------------------------------- scopes

    def _cozy_scope(self, method: str, components: tuple[str, ...]) -> _Scope:
        return _ComponentScope(self, method, components)

    def _cozy_require_scope(self, what: str) -> None:
        """A derived-cache entry is legal only inside an active component-use scope."""
        self._cozy_require_ready(what)
        if self._cozy_active is None and remote_scope(self) is None:
            raise CapabilityError(
                f"{what} outside an active component-use scope: the runtime joins the entry's "
                "lease to the scope, and there is nowhere for it to live otherwise",
                code="no_active_scope",
            )

    def _cozy_require_ready(self, what: str) -> None:
        if not self._cozy_ready:
            raise CapabilityError(
                f"{type(self).__name__}.{what}: no generation is Ready — construction is "
                "transactional and nothing is observable before it completes",
                code="not_ready",
            )

    # ----------------------------------------------------------------- state fence

    def __setattr__(self, name: str, value: object) -> None:
        if self._cozy_ready and not self._cozy_loading:
            if isinstance(value, RequestView) or _carries_view(value):
                raise CapabilityError(
                    f"{type(self).__name__}.{name} = {type(value).__name__}: request-local "
                    "state cannot escape onto the long-lived model — it lives on the "
                    "RequestView or a package-specific attempt object",
                    code="request_state_escape",
                    fields=[name],
                )
            raise CapabilityError(
                f"{type(self).__name__}.{name} = …: the model generation is immutable after "
                "construction — a model instance never hides unaccounted state; declare a "
                "loader.cache(...) handle for reusable derived entries",
                code="persistent_allocation",
                fields=[name],
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if self._cozy_ready and not self._cozy_loading:
            raise CapabilityError(
                f"del {type(self).__name__}.{name}: declared state is released in "
                "unload(loader), which the runtime calls after draining",
                code="persistent_allocation",
                fields=[name],
            )
        object.__delattr__(self, name)


def _carries_view(value: object) -> bool:
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is None:
        return False
    return any(isinstance(getattr(value, name, None), RequestView) for name in fields)


def _declared_state(cls: type) -> tuple[str, ...]:
    names: list[str] = []
    for base in reversed(cls.__mro__):
        if base is object or base is Model:
            continue
        for name in getattr(base, "__annotations__", {}):
            if not name.startswith("_") and name not in names:
                names.append(name)
    return tuple(names)


@final
class _ComponentScope:
    """A method's declared components, nested or on any of the call's threads (cr-202), each
    marking its own context (`Model._cozy_active`). Residency hooks see each forward, and the
    scope's components are held bound for its plan step (`Residency.using`, which covers a
    component with no blocks that another's forward reads, cr-207). Calls still take scopes one
    at a time (`SCOPES`): the sites they read change between turns (cr-202 U4 lifts it).

    A REMOTE scope, whose every declared component the runtime keeps on another GPU of the
    group (`Placement.remote`), uses nothing on this device."""

    __slots__ = (
        "_caches",
        "_call",
        "_components",
        "_contract",
        "_local",
        "_method",
        "_model",
        "_numerics",
        "_remote",
        "_using",
    )

    def __init__(self, model: Model[object], method: str, components: tuple[str, ...]) -> None:
        self._model = model
        self._method = method
        self._components = components
        self._remote: contextvars.Token[_OpenScopes] | None = None
        self._local: contextvars.Token[_OpenScopes] | None = None
        self._call: Context | None = None
        self._using: AbstractContextManager[None] | None = None
        self._numerics: AbstractContextManager[None] | None = None
        self._caches: list[DerivedCache] = []
        self._contract: tuple[Sealed, Sealing] | None = None

    def __enter__(self) -> None:
        model = self._model
        model._cozy_require_ready(self._method)
        record = model._cozy_record
        unbound = [c for c in self._components if record is not None and c in record.skipped]
        if unbound:
            raise CapabilityError(
                f"{type(model).__name__}.{self._method} uses {', '.join(unbound)}, which this "
                "construction's binding does not select: those components are on meta",
                code="component_unbound",
                fields=unbound,
            )
        placement = model._cozy_placement
        if placement is not None and placement.remote(self._components):
            if remote_scope(model) is not None:
                raise CapabilityError(
                    f"{type(model).__name__}.{self._method} entered inside another remote "
                    "scope of the same instance in this context",
                    code="concurrent_scope",
                )
            self._remote = _REMOTE_SCOPES.set(
                (*_REMOTE_SCOPES.get(), (model, self._method, self._components))
            )
            self._enter_numerics(device=False)
            return
        self._call = CURRENT.get()
        if self._call is not None:
            SCOPES.enter(self._call)
        elif SCOPES.busy():
            row: WarningRow = (
                "a thread outside every call entered a component scope during another call's turn",
                {"method": f"{type(model).__name__}.{self._method}"},
            )
            if row not in WARNINGS:
                WARNINGS.append(row)
        self._local = _LOCAL_SCOPES.set(
            (*_LOCAL_SCOPES.get(), (model, self._method, self._components))
        )
        # A scope that fails to open gets no `__exit__` from its `with`: it gives back the turn
        # and context it took here, or the next call on this executor waits on it forever.
        try:
            if (sealed := CONTRACT[0]) is not None:
                self._contract = (sealed, sealed.enter())
            self._enter_numerics(device=True)
            if model._cozy_residency is not None:
                using = model._cozy_residency.using(self._method, self._components)
                using.__enter__()
                self._using = using
                for cache in model._cozy_caches:
                    cache.enter()
                    self._caches.append(cache)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *exc: object) -> None:
        model = self._model
        if self._remote is not None:
            numerics, self._numerics = self._numerics, None
            if numerics is not None:
                numerics.__exit__(None, None, None)
            _REMOTE_SCOPES.reset(self._remote)
            self._remote = None
            return
        # Only what this scope took, innermost first; each release runs even when one before it
        # raises, so a failed release never keeps the call's turn.
        failed = sys.exc_info()  # a raise in the scope ends its passes as failed
        caches, self._caches = self._caches, []
        using, self._using = self._using, None
        numerics, self._numerics = self._numerics, None
        local, self._local = self._local, None
        call, self._call = self._call, None
        sealed, self._contract = self._contract, None
        with ExitStack() as releases:
            if sealed is not None:  # last: once its numerics and passes are put back
                method = f"{type(model).__name__}.{self._method}"
                releases.callback(sealed[0].leave, sealed[1], method, failed[0] is None)
            if call is not None:
                releases.callback(SCOPES.leave)
            if local is not None:
                releases.callback(_LOCAL_SCOPES.reset, local)
            if numerics is not None:
                releases.callback(numerics.__exit__, None, None, None)
            if using is not None:
                releases.callback(using.__exit__, *failed)
            for cache in caches:
                releases.callback(cache.leave)

    def _enter_numerics(self, *, device: bool) -> None:
        """A `deterministic=True` scope's numerics: marked in this context, and set on this
        process's device when the scope runs here. A remote scope's calls set it on their home
        GPU, whose executor enters this method's scope for them."""
        if _declares_deterministic(self._model, self._method):
            numerics = _deterministic_numerics(device=device)
            numerics.__enter__()
            self._numerics = numerics


@contextmanager
def _deterministic_numerics(*, device: bool) -> Iterator[None]:
    token = _DETERMINISTIC_SCOPE.set(True)
    try:
        with DETERMINISTIC[0]() if device else nullcontext():
            yield
    finally:
        _DETERMINISTIC_SCOPE.reset(token)


# --------------------------------------------------------------------------- identity


@dataclass(frozen=True, slots=True)
class ConstructionIdentity:
    """Equal identities coalesce ONE model generation; unequal ones stay distinct (§1.1).

    No per-use-site minimum and no derived union: the exact artifact and its immutable
    config construct ONE exact graph through the one factory.
    """

    release: str
    model_class: str
    artifact: str
    variant: str

    def __str__(self) -> str:
        return f"{self.model_class}@{self.release} ← {self.artifact} [{self.variant}]"


@dataclass(slots=True)
class Generation:
    """One constructed, Ready model generation and the binding paths sharing it."""

    identity: ConstructionIdentity
    model: Model[object]
    record: ConstructionRecord
    paths: list[str] = field(default_factory=list)


def component_use(cls: type) -> Mapping[str, tuple[str, ...]]:
    """The class's ComponentUseContract: method -> possible components. Descriptor content
    (cr-003), validated against the census at exact binding."""
    contract: dict[str, tuple[str, ...]] = {}
    for name, member in _public_methods(cls):
        declared = getattr(member, "__uses_components__", None)
        if isinstance(declared, tuple):
            contract[name] = declared
    return contract


def _public_methods(cls: type) -> Iterator[tuple[str, Handler]]:
    """Author-defined public methods only — runtime-owned base members do not open scopes."""
    for name in dir(cls):
        if name.startswith("_") or name in _RUNTIME_MEMBERS or hasattr(Model, name):
            continue
        member = inspect.getattr_static(cls, name, None)
        if inspect.isfunction(member):
            yield name, member


def undeclared_methods(cls: type) -> tuple[str, ...]:
    """Public component-touching methods that declare nothing — omitted means ALL."""
    return tuple(
        name
        for name, member in _public_methods(cls)
        if getattr(member, "__uses_components__", None) is None
    )


@final
class ModelRegistry:
    """Single-flight construction and construction-identity coalescing.

    Binding paths keep independent admission and defaults; they converge on one generation
    only when their complete construction identities match.
    """

    def __init__(
        self,
        *,
        release: str,
        backend: Backend | None = None,
        substrate: Callable[[], AbstractContextManager[object]] | None = None,
    ) -> None:
        self._release = release
        self._backend = backend
        self._substrate = substrate
        self._generations: dict[ConstructionIdentity, Generation] = {}
        self._by_path: dict[str, Generation] = {}
        self._lock = threading.RLock()

    def acquire(
        self,
        binding_path: str,
        cls: type[ModelT],
        artifact: Artifact,
        *,
        mode: Mode = "serve",
        adapters: Sequence[AdapterRef] = (),
    ) -> ModelT:
        """Bind one path. Equal identity -> the SAME generation, constructed once."""
        adapter_rows = artifact._adapters or tuple(adapters)
        cls.validate_adapter_stack(adapter_rows)
        variant = artifact.variant
        if adapter_rows:
            variant += "|adapters:" + json.dumps(
                [
                    {
                        "ref": row.ref,
                        "scale": row.scale,
                        "kind": row.kind,
                        "component": row.component,
                        "source_component": row.source_component,
                        "family": row.family,
                    }
                    for row in adapter_rows
                ],
                separators=(",", ":"),
                sort_keys=True,
            )
        identity = ConstructionIdentity(
            release=self._release,
            model_class=f"{cls.__module__}:{cls.__qualname__}",
            artifact=artifact.snapshot,
            variant=variant,
        )
        with self._lock:
            generation = self._generations.get(identity)
            if generation is None:
                generation = self._construct(identity, cls, artifact, mode, adapter_rows)
                self._generations[identity] = generation
            generation.paths.append(binding_path)
            self._by_path[binding_path] = generation
            return cast("ModelT", generation.model)

    def models(self, callable_name: str) -> Mapping[str, Model[object]]:
        """What `Invocation.models` needs for ONE callable, keyed by parameter name.

        Resolution is per binding PATH, never per parameter name: two entrypoints both
        spelling their slot `model` are two independent binding points (§1.1).
        """
        prefix = f"{callable_name}.models."
        return {
            path[len(prefix) :]: g.model
            for path, g in self._by_path.items()
            if path.startswith(prefix)
        }

    def generations(self) -> tuple[Generation, ...]:
        return tuple(self._generations.values())

    def unload_all(self) -> None:
        with self._lock:
            for generation in self._generations.values():
                model = generation.model
                loader = self._loader(model, _EMPTY_ARTIFACT, "serve")
                object.__setattr__(model, "_cozy_loading", True)
                try:
                    with loading(loader):
                        model.unload(loader)
                finally:
                    object.__setattr__(model, "_cozy_ready", False)
                    object.__setattr__(model, "_cozy_loading", False)
                for cache in model._cozy_caches:
                    cache._close()
            self._generations.clear()
            self._by_path.clear()

    # ----------------------------------------------------------------- internals

    def _construct(
        self,
        identity: ConstructionIdentity,
        cls: type[Model[object]],
        artifact: Artifact,
        mode: Mode,
        adapters: Sequence[AdapterRef] = (),
    ) -> Generation:
        started = time.perf_counter()
        model = cls()
        loader = self._loader(model, artifact, mode)
        made = time.perf_counter()
        object.__setattr__(model, "_cozy_loading", True)
        try:
            with ExitStack() as stack:
                # SERVE CONSTRUCTS WEIGHTLESS TOO. `__init__` allocates shapes and dtypes,
                # never values (§1.1), so a serving construction has no more reason to
                # allocate 5 GB of initialized garbage than a derivation does — the fill
                # plane reserves the real destinations (cr-005) and the two graphs stay
                # byte-identical because there is one substrate and one census. Absent a
                # substrate this is exactly the old behaviour: an ordinary construction.
                if self._substrate is not None:
                    stack.enter_context(self._substrate())
                began = time.perf_counter()
                with loading(loader):
                    model.load(loader)
                loaded = time.perf_counter()
            exited = time.perf_counter()
            records = loader.records()
            if not records:
                raise ConformanceError(
                    f"{cls.__name__}.load did not construct anything: the one construction "
                    "primitive is loader.construct(factory)",
                    code="no_construction",
                )
            record = records[0] if len(records) == 1 else _merge(records)
            # The package's whole load() beside its constructions' legs: what its own code spent
            # after them (tokenizers, schedulers, its assets).
            legs = {
                "model": made - started,  # the package's model and its loader
                "substrate_enter": began - made,
                "package_load": loaded - began,
                "substrate_exit": exited - loaded,
            }
            record = replace(
                record,
                stage_ms=(*record.stage_ms, *((k, round(v * 1000, 1)) for k, v in legs.items())),
            )
            if record.skipped and not record.components:
                raise ModelFitRefused(
                    {
                        "ok": False,
                        "code": "destination_absent",
                        "detail": "the binding selects none of the components "
                        f"{cls.__name__} builds ({', '.join(record.skipped)})",
                    }
                )
            _prove_components(cls, record)
            _coverage(cls, record)
            constructed = loader.constructed()
            if adapters and artifact._prepare is None:
                model.prepare_adapters(tuple(adapters))
            object.__setattr__(model, "_cozy_adapters", tuple(adapters))
            # NOTHING IS WRAPPED HERE, AND THAT IS THE FIX (#573).
            #
            # A censused component used to be replaced by a proxy whose `__getattr__`
            # checked the active component-use scope. `Scheduler` is a @runtime_checkable
            # Protocol, so the RUNTIME'S OWN `isinstance(member, Scheduler)` probe below
            # tripped the RUNTIME'S OWN proxy and poisoned the generation with
            # `undeclared_component` naming a component no package had touched — an
            # investigation that first ended in "cause not located". Ordering the scan
            # before the wrap only moved the trap: a proxy is invisible to `isinstance`,
            # un-censusable by `census`, and poisons a live generation permanently on any
            # `hasattr` — which is every duck-type check in the ecosystem.
            #
            # The LEASE never lived in the proxy. `_ComponentScope.__enter__` admits the
            # declared set through `Residency.admit` before the method body runs, so a
            # method whose components cannot be materialized never executes. The proxy only
            # added the ability to BLAME an access after the fact, and it could not even do
            # that — the offender is reached through `__getattr__`, so it is by definition
            # not in the message — while costing a broken `isinstance` on every component.
            object.__setattr__(model, "_cozy_schedulers", _schedulers(constructed, {}))
            object.__setattr__(model, "_cozy_caches", loader.caches())
            object.__setattr__(model, "checkpoint_ref", artifact.snapshot)
            object.__setattr__(
                model, "_cozy_selection_ref", artifact._selection_snapshot or artifact.snapshot
            )
            object.__setattr__(model, "_cozy_record", record)
            object.__setattr__(model, "_cozy_identity", identity)
            object.__setattr__(model, "_cozy_ready", True)
            # The whole construction; beside construct_and_register it leaves only the registry.
            done = round((time.perf_counter() - started) * 1000, 1)
            record = replace(record, stage_ms=(*record.stage_ms, ("construct", done)))
            object.__setattr__(model, "_cozy_record", record)
        except Exception:
            # Transactional: a failure rolls fills, caches and registrations back together
            # and no generation is ever observable half-built (§1.1).
            for cache in loader.caches():
                cache.drop()
            object.__setattr__(model, "_cozy_ready", False)
            raise
        finally:
            object.__setattr__(model, "_cozy_loading", False)
        return Generation(identity, model, record)

    def _loader(self, model: Model[object], artifact: Artifact, mode: Mode) -> Loader:
        return Loader(artifact, owner=model, backend=self._backend, mode=mode)


_EMPTY_ARTIFACT = Artifact(snapshot="unload://drain", tensor_schema={}, config=Config({}))


def _coverage(cls: type, record: ConstructionRecord) -> None:
    """A load whose construction and checkpoint disagree says so once, in the next attempt's
    warnings, naming the tensors, and serves: persistent buffers the checkpoint does not store
    keep their constructed values. Stored tensors no destination reads were not loaded: every
    tensor the model builds was filled, so an unused head changes nothing and is a log line.
    A quarter or more of what it did load left unread is another matter: the code and the
    checkpoint may not be one model, and that warns. What the class declares (`unloaded=`,
    `derived=`) is not warned about."""
    ignored, unstored = uncovered(cls, record.ignored_extras, record.unstored)
    unread = cast("int", (record.fit or {}).get("ignored_bytes", 0))
    if ignored and unread * 4 < record.filled_bytes:
        _LOG.info(
            "%s: %d checkpoint tensors not loaded, %d bytes of %d: %s",
            cls.__name__, len(ignored), unread, record.filled_bytes, ", ".join(ignored[:8]),
        )  # fmt: skip
        ignored = []
    for message, kind, names in (
        ("checkpoint tensors were not loaded: the model builds no destination for them",
         "tensors", ignored),
        ("persistent buffers are not in the checkpoint: they keep their constructed values",
         "buffers", unstored),
    ):  # fmt: skip
        if names:
            more = f", and {len(names) - 8} more" if len(names) > 8 else ""
            row: WarningRow = (
                message,
                {"model": cls.__name__, "count": len(names), kind: ", ".join(names[:8]) + more},
            )
            if row not in WARNINGS:
                WARNINGS.append(row)


def _merge(records: Sequence[ConstructionRecord]) -> ConstructionRecord:
    """Several `construct` calls in one `load()` are one construction contract."""
    return ConstructionRecord(
        destinations=tuple(d for r in records for d in r.destinations),
        components=tuple(c for r in records for c in r.components),
        schedulers=tuple(s for r in records for s in r.schedulers),
        filled=sum(r.filled for r in records),
        filled_bytes=sum(r.filled_bytes for r in records),
        ignored_extras=tuple(e for r in records for e in r.ignored_extras),
        ignored_warnings=tuple(w for r in records for w in r.ignored_warnings),
        unstored=tuple(u for r in records for u in r.unstored),
        skipped=tuple(c for r in records for c in r.skipped),
        fit=next((r.fit for r in records if r.fit is not None), None),
    )


def _prove_components(cls: type, record: ConstructionRecord) -> None:
    """The EXISTENCE clock: every declared name matched against the censused construction,
    and — when the class declares everywhere — no censused component left undeclared."""
    censused = set(record.components) | set(record.skipped)
    contract = component_use(cls)
    unknown = sorted({c for names in contract.values() for c in names} - censused)
    if unknown:
        raise ConformanceError(
            f"{cls.__name__}: @uses_components names {', '.join(unknown)}, which this "
            f"construction does not build (censused: {', '.join(sorted(censused))})",
            code="unknown_component",
            fields=unknown,
        )
    if undeclared_methods(cls):
        return  # an omitted declaration means ALL: conservatively safe, nothing to prove
    declared = {c for names in contract.values() for c in names}
    surplus = sorted(censused - declared)
    if surplus:
        raise ConformanceError(
            f"{cls.__name__} constructs {', '.join(surplus)}, which no method declares — a "
            "class declares only component names EVERY admissible artifact for its slots "
            "constructs, so a construction the declarations do not cover could never pass "
            "exact binding against a narrow artifact",
            code="undeclared_construction",
            fields=surplus,
        )


def _schedulers(objects: Sequence[object], named: Mapping[str, object]) -> Mapping[str, Scheduler]:
    """Scheduler prototypes are CONSTRUCTION facts: whatever the constructed object (or a
    test double standing in for it) carries by name. `make_scheduler` clones them fresh."""
    found: dict[str, Scheduler] = {
        name: member for name, member in named.items() if isinstance(member, Scheduler)
    }
    for obj in objects:
        table = getattr(obj, "components", None)
        if isinstance(table, Mapping):
            for name, member in table.items():
                if isinstance(member, Scheduler):
                    found[str(name)] = member
    return found
