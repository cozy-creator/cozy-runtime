"""The `Model` boundary: role classes, class-keyword stamps, component-use scopes, the
construction transaction, and construction-identity coalescing.

THE SIGNATURE IS THE SLOT (§1.1): a bare typed parameter (`model: Fl2VAModel`) is the whole
declaration and its derived canonical path (`generate.models.model`) is the binding point.
There is no slot object, no `artifacts=` dict, and the `Annotated` override-marker plane is
EMPTY at launch. The CLASS carries the admissibility stamps — one keyword per REQUIRED
stamp from the family's OPEN uploader-minted vocabulary, stated once per role class.

Construction identity is (endpoint release, model class, exact artifact snapshot + immutable
config, construction/provider variant). Equal identities coalesce ONE single-flight model
generation; unequal ones stay distinct instances with independent lifecycle and admission.

The engine-hosted tier (`self_loading=`, `Loader.engine`) rides cr-018 and lands with
se-007's milestone: the keyword refuses here rather than silently becoming a stamp.
"""

from __future__ import annotations

import functools
import inspect
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, field
from typing import (
    Any,
    Concatenate,
    Generic,
    NoReturn,
    ParamSpec,
    Protocol,
    Self,
    TypeVar,
    cast,
    final,
)

from cozy_runtime.author._context import Context, Device, RequestView, derive_seed
from cozy_runtime.author._errors import (
    CapabilityError,
    ConformanceError,
    InvalidRequest,
)
from cozy_runtime.author._loader import (
    Artifact,
    Backend,
    Config,
    ConstructionRecord,
    DerivedCache,
    Loader,
    Mode,
    Scheduler,
    loading,
)

PipelineT = TypeVar("PipelineT", covariant=True)
P = ParamSpec("P")
R = TypeVar("R")
ModelT = TypeVar("ModelT", bound="Model[Any]")

#: Class keywords that are NOT stamps. `self_loading=` is the engine-hosted tier's pinned
#: receiver (§1.5) and its implementation rides cr-018/se-007 — it refuses rather than
#: silently entering the open stamp vocabulary.
RESERVED_KEYWORDS: Mapping[str, str] = {
    "self_loading": "the engine-hosted tier lands with se-007's milestone (cr-018's door)",
    "structure": "structure admissibility is DOCUMENT MATCHING, derived from the code "
    "topology — a structure split enters only on measured need (§1.1)",
}

#: THE ENCODED-LEAF CONSENT, and the whole of the author's say in the delivery plane.
#:
#: Every other rung the runtime can choose is invisible to author code by construction: the
#: weights arrive as float tensors in the destinations the census named, and which stored
#: bytes they came from is not a question the endpoint can ask. The `encoded_gemm` route is
#: the one that cannot be invisible. Its leaf has no `.weight` — the payload and its scales
#: ARE the weight, the module that multiplies by them is REPLACED, and an endpoint that
#: reaches into `layer.weight`, re-dtypes a submodule, or rebuilds a `nn.Linear` from one
#: finds something that is not there.
#:
#: So it is DECLARED, once, on the class, and the default is REFUSE. This is not a knob and
#: it is not a request field: it is the author stating whether their code survives having a
#: linear op swapped underneath it, which is a fact about the code and nobody else can know
#: it. `EXCLUDED_KEYWORDS` keeps `quantize` out of every method signature for the opposite
#: and complementary reason — WHICH encoding serves is derived, never authored — and the two
#: rules meet exactly here: the author never picks the encoding, and the runtime never
#: replaces a leaf in an endpoint that did not say it could take one.
#:
#: A string rather than a bool, and a closed vocabulary rather than an open one, because the
#: class-keyword plane is uniformly string-valued (the stamps are) and because "refuse" then
#: has a spelling — an author can state the conservative answer instead of only implying it.
ENCODED_LEAVES: Mapping[str, str] = {
    "refuse": "the runtime fills float destinations and never replaces a module; an "
    "artifact whose only qualified route on this device is native refuses typed",
    "accept": "this endpoint's code touches its components through their FORWARD only — "
    "no `.weight` read, no per-module dtype walk, no submodule rebuilt from one — so a "
    "linear op may be replaced by a module whose weight is the stored role set",
}

#: Class keywords that are neither stamps nor refusals: the runtime reads them. One entry,
#: deliberately — the plane exists because `encoded_leaves` needed it, and it is a LIST so
#: the next one is a reviewed addition rather than an accident of `__init_subclass__`.
RUNTIME_KEYWORDS: Mapping[str, Mapping[str, str]] = {"encoded_leaves": ENCODED_LEAVES}

#: Names a `Model` subclass may not declare. `model.adapters` is the one people reach for:
#: the applied stack is REQUEST-local state and lives on `RequestView` (§1.4).
DELETED_MODEL_MEMBERS: Mapping[str, str] = {
    "adapters": "the applied stack is request-local: view.adapters, never model.adapters (§1.4)",
    "artifacts": "no artifacts= dict exists — selection is a binding, never code (§1.1)",
    "current_transformer": "residency state is never author-visible (§1.1.2)",
    "adaln_mode": "the modulation source is a CONSTRUCTION fact, not a request mode (§1.1.2)",
    "generator": "the seeded generator is view.generator (§1.3)",
}

#: Keyword names an author-facing model method may not declare. Every one of them names a
#: residency, placement or conversion decision the runtime owns (§1.1/§7).
EXCLUDED_KEYWORDS: Mapping[str, str] = {
    "device": "placement is the runtime's; the method wrapper exposes no device",
    "dtype": "compute dtype is a plan decision, not a request knob",
    "torch_dtype": "compute dtype is a plan decision, not a request knob",
    "path": "no filesystem path reaches endpoint code",
    "checkpoint_path": "no filesystem path reaches endpoint code",
    "weights_path": "no filesystem path reaches endpoint code",
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
    {"load", "unload", "for_request", "for_test", "checkpoint_ref", "harness"}
)


# --------------------------------------------------------------------------- scopes


@dataclass(frozen=True, slots=True)
class ScopeCall:
    """One component-use scope the harness observed. Records, never invents (§1.6)."""

    method: str
    components: tuple[str, ...]


@final
class ModelTestHarness:
    """Installs immutable construction facts and records component-scope calls.

    It never invents outputs for model-specific methods: an endpoint test supplies model
    behavior explicitly as author-built doubles (`for_test(pipe=FakeH3Pipeline())`).
    """

    __slots__ = ("calls",)

    def __init__(self) -> None:
        self.calls: list[ScopeCall] = []

    def __repr__(self) -> str:
        return f"ModelTestHarness({len(self.calls)} scope calls)"

    def components(self) -> tuple[str, ...]:
        seen: list[str] = []
        for call in self.calls:
            seen += [c for c in call.components if c not in seen]
        return tuple(seen)

    def assert_scopes(self, *expected: str) -> None:
        got = tuple(c.method for c in self.calls)
        if got != expected:
            raise AssertionError(f"component scopes were {got}, expected {expected}")


class _ScopedModel(Protocol):
    """What `@uses_components` needs from the instance it wraps — so the decorator is
    ParamSpec-preserving and fully typed without an `Any` seam (§1.7)."""

    def _cozy_scope(self, method: str, components: tuple[str, ...]) -> _Scope: ...


M = TypeVar("M", bound=_ScopedModel)


class _Declared(Protocol):
    """A declared method carries its component set as descriptor content (§1.1)."""

    __uses_components__: tuple[str, ...]


class _Scope(Protocol):
    def __enter__(self) -> None: ...

    def __exit__(self, *exc: object) -> None: ...


class Residency(Protocol):
    """The runtime's component-use ADMISSION, installed on a constructed generation.

    Declared here because the contract has one home; implemented nowhere in this package,
    because admission, materialization, event fencing and eviction all need a device and
    this package is torch-free by construction. cozy-runtime supplies the body
    (`internal/staging.py`); a promoted Varena adapter would supply a different one and the
    endpoint above would not be able to tell.

    `admit` runs BEFORE method entry and either makes the declared set materialized or
    refuses typed. `release` runs after the body returns, records the event fence, and makes
    the leases EVICTABLE — it never forces eviction and never imposes a global scope order.
    """

    def admit(self, method: str, components: tuple[str, ...]) -> None: ...

    def release(self, method: str, components: tuple[str, ...]) -> None: ...


def uses_components(
    *names: str,
) -> Callable[[Callable[Concatenate[M, P], R]], Callable[Concatenate[M, P], R]]:
    """Declare the possible heavyweight component set of ONE public `Model` method.

    A component-use contract, not a phase declaration: while this method executes, any
    branch may touch these components. Omitted means ALL (conservatively safe); explicitly
    empty REFUSES — a method touching no components is module code, not a Model method.
    """
    if not names:
        raise ConformanceError(
            "@uses_components() is empty: a method that touches no components is pure "
            "orchestration and belongs in module code, never on a Model (§1.1)",
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
    components = tuple(names)

    def decorate(fn: Callable[Concatenate[M, P], R]) -> Callable[Concatenate[M, P], R]:
        if fn.__name__ in _RUNTIME_MEMBERS:
            raise ConformanceError(
                f"@uses_components on {fn.__name__}: lifecycle and runtime-owned members do "
                "not open component scopes — decorate the public method that forms the "
                "runtime boundary (§1.1)",
                code="component_placement",
            )
        if fn.__name__.startswith("_"):
            raise ConformanceError(
                f"@uses_components on {fn.__name__}: private helpers inherit their caller's "
                "active scope and cannot be called as handler operations (§1.1)",
                code="component_placement",
            )
        _check_signature(fn)
        wrapper = _async_wrapper(fn, components) if _is_async(fn) else _sync_wrapper(fn, components)
        declared: _Declared = cast("_Declared", wrapper)
        declared.__uses_components__ = components
        return wrapper

    return decorate


def _is_async(fn: Callable[..., object]) -> bool:
    return inspect.iscoroutinefunction(inspect.unwrap(fn))


def _sync_wrapper(
    fn: Callable[Concatenate[M, P], R], components: tuple[str, ...]
) -> Callable[Concatenate[M, P], R]:
    @functools.wraps(fn)
    def wrapper(self: M, /, *args: P.args, **kwargs: P.kwargs) -> R:
        with self._cozy_scope(fn.__name__, components):
            result = fn(self, *args, **kwargs)
            _refuse_lazy(fn.__name__, result)
            return result

    return wrapper


def _async_wrapper(
    fn: Callable[Concatenate[M, P], R], components: tuple[str, ...]
) -> Callable[Concatenate[M, P], R]:
    """The scope stays active THROUGH `await` — the lease is the method's, not the step's."""

    @functools.wraps(fn)
    async def wrapper(self: M, /, *args: P.args, **kwargs: P.kwargs) -> object:
        with self._cozy_scope(fn.__name__, components):
            result = await fn(self, *args, **kwargs)  # type: ignore[misc]
            _refuse_lazy(fn.__name__, result)
            return result

    return cast("Callable[Concatenate[M, P], R]", wrapper)


def _check_signature(fn: Callable[..., object]) -> None:
    for name in inspect.signature(fn).parameters:
        if (why := EXCLUDED_KEYWORDS.get(name)) is not None:
            raise ConformanceError(
                f"{fn.__qualname__}({name}=): {why} — the method wrapper exposes no device, "
                "movement, allocation, offload, pinning, eviction or residency backend (§1.1)",
                code="excluded_keyword",
                fields=[name],
            )


def _refuse_lazy(method: str, result: object) -> None:
    if inspect.isgenerator(result) or inspect.iscoroutine(result) or inspect.isasyncgen(result):
        raise CapabilityError(
            f"{method} returned a {type(result).__name__}: a lazy object that touches model "
            "components after the method completes escapes its component-use lease (§1.1)",
            code="lazy_escape",
        )


# --------------------------------------------------------------------------- model


class Model(Generic[PipelineT]):
    """Base of an endpoint's reusable model class — the ONE long-lived model boundary.

    The author owns WHAT the model is (`load`/`unload`, declared persistent state, the
    component-touching methods handlers call); the runtime owns WHEN they run and where
    every byte lives. A bare typed parameter is the whole declaration, and its presence in
    a signature is what derives `gpu`.
    """

    __stamps__: Mapping[str, str] = {}

    __encoded_leaves__: str = "refuse"
    """The class's `encoded_leaves=` declaration, defaulted to the conservative answer.

    REFUSE is the default because the absence of a statement is not consent. An endpoint
    written before the native rung existed cannot have thought about whether its code
    survives leaf replacement, and reading its silence as "yes" would be the runtime
    deciding a question only the author can answer."""

    checkpoint_ref: str = ""
    """The pinned-ref REPRODUCTION FACT for result structs — an observation, never catalog
    authority. Model-scoped because it is construction identity."""

    _cozy_ready = False
    _cozy_loading = False
    _cozy_poison: str | None = None
    _cozy_active: tuple[str, tuple[str, ...]] | None = None
    _cozy_harness: ModelTestHarness | None = None
    _cozy_record: ConstructionRecord | None = None
    _cozy_identity: ConstructionIdentity | None = None
    _cozy_schedulers: Mapping[str, Scheduler] = {}
    _cozy_caches: tuple[DerivedCache, ...] = ()
    _cozy_residency: Residency | None = None
    """Runtime-installed. Absent under `for_test` and in derive mode, which is the whole
    reason a scope works without one: a test double has no bytes to stage."""

    def __init_subclass__(cls, /, **stamps: str) -> None:
        """The class-keyword plane: one keyword per REQUIRED stamp, open vocabulary.

        Plus the `RUNTIME_KEYWORDS`, which are read by the runtime rather than matched
        against a checkpoint. They are taken FIRST and out of the stamp dict, so a runtime
        keyword can never be mistaken for a stamp the platform would then look for in a
        checkpoint's uploader-minted map and refuse for missing.
        """
        super().__init_subclass__()
        for key, vocabulary in RUNTIME_KEYWORDS.items():
            if key not in stamps:
                continue
            value = stamps.pop(key)
            if value not in vocabulary:
                raise ConformanceError(
                    f"class {cls.__name__}({key}={value!r}): the vocabulary is CLOSED — "
                    + " | ".join(f"{k!r}: {why}" for k, why in vocabulary.items()),
                    code=f"{key}_value",
                )
            setattr(cls, f"__{key}__", value)
        inherited = dict(getattr(cls, "__stamps__", {}))
        for key, value in stamps.items():
            if (why := RESERVED_KEYWORDS.get(key)) is not None:
                raise ConformanceError(
                    f"class {cls.__name__}({key}=…): {why}", code="reserved_keyword"
                )
            if not isinstance(value, str) or not value.strip():
                raise ConformanceError(
                    f"class {cls.__name__}({key}=…): a stamp is a non-empty string naming one "
                    "value in the checkpoint's uploader-minted, family-scoped map — the "
                    "platform blesses no keys and interprets none (§1.1)",
                    code="stamp_value",
                )
            if key in inherited:
                raise ConformanceError(
                    f"class {cls.__name__}({key}={value!r}): {key!r} is already stamped "
                    f"{inherited[key]!r} by a base — a stamp is stated ONCE per role class",
                    code="restamped",
                )
            inherited[key] = value
        cls.__stamps__ = inherited
        for name in (*vars(cls), *getattr(cls, "__annotations__", {})):
            if (why := DELETED_MODEL_MEMBERS.get(name)) is not None:
                raise ConformanceError(
                    f"class {cls.__name__}.{name}: {why}",
                    code="deleted_model_member",
                    fields=[name],
                )

    # ----------------------------------------------------------------- lifecycle

    def load(self, loader: Loader) -> None:
        """Construct this model from the bound artifact. The runtime owns WHEN it runs."""
        raise ConformanceError(
            f"{type(self).__name__} defines no load(loader): a Model constructs through "
            "loader.construct(T, factory=…) — the one construction primitive (§1.1)",
            code="no_load",
        )

    def unload(self, loader: Loader) -> None:
        """Release declared state. The runtime drains before calling this."""
        return None

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
        return RequestView._make(effective, ctx._adapters, self._cozy_schedulers, sampler)

    # ----------------------------------------------------------------- testing

    @classmethod
    def for_test(cls, *, checkpoint_ref: str = "test://model", **members: object) -> Self:
        """Author-built doubles standing in for construction — no hub, no GPU, no weights.

        The SDK cannot generically fake an arbitrary pipeline's behavior, so the test
        provides it. `@uses_components` becomes a no-op lease that RECORDS (§1.6).
        """
        if not checkpoint_ref.startswith("test://"):
            raise ConformanceError(
                f"for_test(checkpoint_ref={checkpoint_ref!r}): a test double names a "
                "test:// ref — author code never names an artifact identifier (§1.1)",
                code="identifier_in_code",
                fields=["checkpoint_ref"],
            )
        declared = _declared_state(cls)
        unknown = [name for name in members if name not in declared]
        if unknown:
            raise ConformanceError(
                f"{cls.__name__}.for_test({', '.join(unknown)}=…): the class declares no such "
                f"state (declared: {', '.join(declared) or 'none'})",
                code="unknown_member",
                fields=unknown,
            )
        model = cls()
        for name, value in members.items():
            object.__setattr__(model, name, value)
        harness = ModelTestHarness()
        object.__setattr__(model, "checkpoint_ref", checkpoint_ref)
        object.__setattr__(model, "_cozy_harness", harness)
        object.__setattr__(model, "_cozy_schedulers", _schedulers(list(members.values()), members))
        object.__setattr__(model, "_cozy_ready", True)
        return model

    @property
    def harness(self) -> ModelTestHarness:
        """The test harness installed by `for_test`. Absent in production, by construction."""
        if self._cozy_harness is None:
            raise CapabilityError(
                f"{type(self).__name__} was constructed by the runtime, not by for_test: "
                "there is no harness and no parallel fake runtime (§1.6)",
                code="no_harness",
            )
        return self._cozy_harness

    # ----------------------------------------------------------------- scopes

    def _cozy_scope(self, method: str, components: tuple[str, ...]) -> _Scope:
        return _ComponentScope(self, method, components)

    def _cozy_require_scope(self, what: str) -> None:
        """A derived-cache entry is legal only inside an active component-use scope."""
        self._cozy_require_ready(what)
        if self._cozy_active is None:
            raise CapabilityError(
                f"{what} outside an active component-use scope: the runtime joins the entry's "
                "lease to the scope, and there is nowhere for it to live otherwise (§1.3)",
                code="no_active_scope",
            )

    def _cozy_check_access(self, component: str) -> None:
        """The guarded component root: every access checks the ACTIVE scope, resident or not."""
        if self._cozy_active is None:
            self._cozy_kill(f"component {component!r} was touched outside any component-use scope")
        method, declared = self._cozy_active
        if component not in declared:
            self._cozy_kill(
                f"{method} touched component {component!r}, which it does not declare "
                f"({', '.join(declared)}) — an undeclared access poisons the executor and "
                "never reactively loads (§1.1)"
            )

    def _cozy_kill(self, why: str) -> NoReturn:
        object.__setattr__(self, "_cozy_poison", why)
        raise CapabilityError(f"{type(self).__name__}: {why}", code="undeclared_component")

    def _cozy_require_ready(self, what: str) -> None:
        if self._cozy_poison is not None:
            raise CapabilityError(
                f"{type(self).__name__}.{what}: this model generation is POISONED "
                f"({self._cozy_poison})",
                code="poisoned_generation",
            )
        if not self._cozy_ready:
            raise CapabilityError(
                f"{type(self).__name__}.{what}: no generation is Ready — construction is "
                "transactional and nothing is observable before it completes (§1.1)",
                code="not_ready",
            )

    # ----------------------------------------------------------------- state fence

    def __setattr__(self, name: str, value: object) -> None:
        if self._cozy_ready and not self._cozy_loading:
            if isinstance(value, RequestView) or _carries_view(value):
                raise CapabilityError(
                    f"{type(self).__name__}.{name} = {type(value).__name__}: request-local "
                    "state cannot escape onto the long-lived model — it lives on the "
                    "RequestView or an endpoint-specific attempt object (§1.1.2)",
                    code="request_state_escape",
                    fields=[name],
                )
            raise CapabilityError(
                f"{type(self).__name__}.{name} = …: the model generation is immutable after "
                "construction — a model instance never hides unaccounted state; declare a "
                "loader.cache(...) handle for reusable derived entries (§1.3)",
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
    """ONE public scope is active at a time per instance (§1.1). A composite operation
    declares its components together on one method; concurrent scopes are a deferred door."""

    __slots__ = ("_components", "_method", "_model")

    def __init__(self, model: Model[Any], method: str, components: tuple[str, ...]) -> None:
        self._model = model
        self._method = method
        self._components = components

    def __enter__(self) -> None:
        model = self._model
        model._cozy_require_ready(self._method)
        if model._cozy_active is not None:
            raise CapabilityError(
                f"{type(model).__name__}.{self._method} entered while "
                f"{model._cozy_active[0]} holds the active component-use scope: one public "
                "scope is active at a time per instance — an operation needing several "
                "components declares them together on one method (§1.1)",
                code="concurrent_scope",
            )
        # ADMISSION HAPPENS BEFORE ENTRY, and before the scope is marked active: a method
        # whose declared set cannot be materialized never runs, so there is no state in
        # which a body is executing against a component the runtime did not admit.
        residency = model._cozy_residency
        if residency is not None:
            residency.admit(self._method, self._components)
        object.__setattr__(model, "_cozy_active", (self._method, self._components))
        if model._cozy_harness is not None:
            model._cozy_harness.calls.append(ScopeCall(self._method, self._components))

    def __exit__(self, *exc: object) -> None:
        model = self._model
        object.__setattr__(model, "_cozy_active", None)
        # RELEASE MAKES BYTES EVICTABLE; it does not evict, and it imposes no global scope
        # sequence (§3.2). The lease ends at the completion event, not at this statement.
        residency = model._cozy_residency
        if residency is not None:
            residency.release(self._method, self._components)


# --------------------------------------------------------------------------- guards


@final
class GuardedComponent:
    """A censused component root. Every registered parameter and buffer is reachable only
    through it, and any access checks the ACTIVE scope — regardless of whether the bytes
    happen to be resident. cr-008b tightens this to per-parameter guards under staging."""

    __slots__ = ("_cozy_model", "_cozy_name", "_cozy_target")

    _cozy_name: str
    _cozy_target: object
    _cozy_model: Model[Any]

    def __init__(self, name: str, target: object, model: Model[Any]) -> None:
        object.__setattr__(self, "_cozy_name", name)
        object.__setattr__(self, "_cozy_target", target)
        object.__setattr__(self, "_cozy_model", model)

    def __repr__(self) -> str:
        return f"<guarded {self._cozy_name}>"

    def __getattr__(self, name: str) -> object:
        self._cozy_model._cozy_check_access(self._cozy_name)
        return getattr(self._cozy_target, name)

    def __call__(self, *args: object, **kwargs: object) -> object:
        self._cozy_model._cozy_check_access(self._cozy_name)
        target = self._cozy_target
        if not callable(target):
            raise TypeError(f"component {self._cozy_name!r} is not callable")
        return target(*args, **kwargs)


def install_guards(obj: object, model: Model[Any], components: Sequence[str]) -> int:
    """Replace each censused component with its guarded root. Author code reaches
    components by name through the constructed object, so the guard is on the only path."""
    table = getattr(obj, "components", None)
    if not isinstance(table, dict):
        return 0
    guarded = [name for name in components if name in table]
    for name in guarded:
        table[name] = GuardedComponent(name, table[name], model)
    return len(guarded)


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
    config_digest: str
    variant: str

    def __str__(self) -> str:
        return (
            f"{self.model_class}@{self.release} ← {self.artifact} "
            f"[{self.config_digest[:16]}… {self.variant}]"
        )


@dataclass(slots=True)
class Generation:
    """One constructed, Ready model generation and the binding paths sharing it."""

    identity: ConstructionIdentity
    model: Model[Any]
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


def _public_methods(cls: type) -> Iterator[tuple[str, Callable[..., object]]]:
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
        device: Device | None = None,
        substrate: Callable[[], AbstractContextManager[object]] | None = None,
    ) -> None:
        self._release = release
        self._backend = backend
        self._device = device or Device()
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
    ) -> ModelT:
        """Bind one path. Equal identity -> the SAME generation, constructed once."""
        identity = ConstructionIdentity(
            release=self._release,
            model_class=f"{cls.__module__}:{cls.__qualname__}",
            artifact=artifact.snapshot,
            config_digest=artifact.config.digest(),
            variant=artifact.variant,
        )
        with self._lock:
            generation = self._generations.get(identity)
            if generation is None:
                generation = self._construct(identity, cls, artifact, mode)
                self._generations[identity] = generation
            generation.paths.append(binding_path)
            self._by_path[binding_path] = generation
            return cast("ModelT", generation.model)

    def models(self, callable_name: str) -> Mapping[str, Model[Any]]:
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
                    cache.drop()
            self._generations.clear()
            self._by_path.clear()

    # ----------------------------------------------------------------- internals

    def _construct(
        self, identity: ConstructionIdentity, cls: type[Model[Any]], artifact: Artifact, mode: Mode
    ) -> Generation:
        _admit(cls, artifact)
        model = cls()
        loader = self._loader(model, artifact, mode)
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
                with loading(loader):
                    model.load(loader)
            records = loader.records()
            if not records:
                raise ConformanceError(
                    f"{cls.__name__}.load did not construct anything: the one construction "
                    "primitive is loader.construct(T, factory=…) (§1.1)",
                    code="no_construction",
                )
            record = records[0] if len(records) == 1 else _merge(records)
            _prove_components(cls, record)
            constructed = loader.constructed()
            for obj in constructed:
                install_guards(obj, model, record.components)
            object.__setattr__(model, "_cozy_schedulers", _schedulers(constructed, {}))
            object.__setattr__(model, "_cozy_caches", loader.caches())
            object.__setattr__(model, "checkpoint_ref", artifact.snapshot)
            object.__setattr__(model, "_cozy_record", record)
            object.__setattr__(model, "_cozy_identity", identity)
            object.__setattr__(model, "_cozy_ready", True)
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

    def _loader(self, model: Model[Any], artifact: Artifact, mode: Mode) -> Loader:
        return Loader(artifact, owner=model, backend=self._backend, mode=mode, device=self._device)


_EMPTY_ARTIFACT = Artifact(snapshot="unload://drain", topology={}, config=Config({}))


def _merge(records: Sequence[ConstructionRecord]) -> ConstructionRecord:
    """Several `construct` calls in one `load()` are one construction contract."""
    return ConstructionRecord(
        factory_identity=" + ".join(r.factory_identity for r in records),
        config_digest=records[0].config_digest,
        destinations=tuple(d for r in records for d in r.destinations),
        components=tuple(c for r in records for c in r.components),
        schedulers=tuple(s for r in records for s in r.schedulers),
        filled=sum(r.filled for r in records),
        filled_bytes=sum(r.filled_bytes for r in records),
        ignored_extras=tuple(e for r in records for e in r.ignored_extras),
        transcribed_markers=tuple(m for r in records for m in r.transcribed_markers),
    )


def _admit(cls: type[Model[Any]], artifact: Artifact) -> None:
    """The class's stamps JOIN THE CHECKPOINT'S VERIFIED STAMP SET BY MEMBERSHIP, fail-closed.

    A stamp key carries a SET of values, not a scalar, and that is not a loosening — it is
    the ratified law (tensorhub.md §lane: "an endpoint `task=` requirement joins by set
    membership"), and the N-ARY PACKAGING RULE cannot be served without it. Stamps are a
    CHECKPOINT fact: the dual H3 artifact carries one `transformer` stamped `task=fl2va` and
    one `transformer_ref` stamped `task=ref2va`, so its lane contract is
    `task=fl2va+ref2va`. Under scalar equality that artifact — the canonical production
    packaging for every multi-task family — admits NEITHER role class, which is the shape
    the serving path actually refused on real H3 weights (#567d).

    Absence still fails closed, and a present-but-wrong value still refuses: unstamped means
    "nobody looked", never "false". What the membership reading does NOT check is that the
    matching value was evidenced on the component THIS class binds — the hub's stamp rows
    carry that component and the lane contract drops it. Binding the wrong transformer is
    already refused one clock earlier by exact-binding topology validation (the class's
    factory constructs only the components it declares), so the per-component tightening is
    a recorded door, not a hole: it needs the evidence rows carried down beside the contract.
    """
    for key, value in cls.__stamps__.items():
        actual = artifact.stamp_set(key)
        if value not in actual:
            carried = "+".join(actual) if actual else "nothing"
            raise ConformanceError(
                f"{cls.__name__} requires stamp {key}={value!r}; {artifact.snapshot} "
                f"carries {key}={carried} — a class keyword joins the checkpoint's stamp "
                "set by membership and fails closed on absence (§1.1)",
                code="stamp_mismatch",
                fields=[key],
            )


def _prove_components(cls: type, record: ConstructionRecord) -> None:
    """The EXISTENCE clock: every declared name matched against the censused construction,
    and — when the class declares everywhere — no censused component left undeclared."""
    censused = set(record.components)
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
            "exact binding against a narrow artifact (§1.1)",
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
