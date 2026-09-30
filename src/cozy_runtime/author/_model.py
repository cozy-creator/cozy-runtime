"""The `Model` boundary: role classes, the one class keyword, component-use scopes, the
construction transaction, and construction-identity coalescing.

THE SIGNATURE IS THE SLOT (§1.1): a bare typed parameter (`model: Fl2VAModel`) is the whole
declaration and its derived canonical path (`generate.models.model`) is the binding point.
There is no slot object, no `artifacts=` dict, and the `Annotated` override-marker plane is
EMPTY at launch. The CLASS carries exactly one keyword, `encoded_leaves`; there are no
stamps, no `task=`/`objective=`/`structure=`, and no model or release name anywhere in code
(model-code-fit §1, D1). A semantic twin is a component name or a config fact.

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
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Concatenate,
    ParamSpec,
    Protocol,
    Self,
    TypeVar,
    cast,
    final,
)

from cozy_runtime.author._attention import AttentionContext
from cozy_runtime.author._context import AdapterRef, Context, Device, RequestView, derive_seed
from cozy_runtime.author._errors import (
    CapabilityError,
    ConformanceError,
    InvalidRequest,
    is_device_oom,
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

_LOG = logging.getLogger(__name__)

PipelineT = TypeVar("PipelineT", covariant=True)
P = ParamSpec("P")
R = TypeVar("R")
ModelT = TypeVar("ModelT", bound="Model[Any]")


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

#: The second class keyword, same shape and same reason (h3a-015): the runtime's fused glue
#: plan replaces a known module's FORWARD with a fused kernel of the same arithmetic. It
#: moves no tensor, but a package that reads a module's class name or an intermediate the
#: fused chain no longer materializes would notice, and only the author knows it does not.
FUSION: Mapping[str, str] = {
    "refuse": "the runtime runs every module's own forward; no fused plan is installed",
    "accept": "this package's code depends on no module's class identity and on no "
    "intermediate between a norm and the GEMM it feeds, so the runtime MAY substitute its "
    "fused glue kernels for the modules it has exact-source matches for — a device without "
    "the built kernels serves the eager forward and the prepare records why",
    "require": "the same consent, and this package will not serve without it: a device "
    "whose fused kernels are absent or stale refuses the placement typed",
}

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
    {"load", "warm", "unload", "for_request", "for_test", "checkpoint_ref", "harness"}
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

    It never invents outputs for model-specific methods: a package test supplies model
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
    """A declared method carries its component set as interface content (§1.1)."""

    __uses_components__: tuple[str, ...]


class _Scope(Protocol):
    def __enter__(self) -> None: ...

    def __exit__(self, *exc: object) -> None: ...


class Residency(Protocol):
    """The runtime's component-use ADMISSION, installed on a constructed generation.

    Declared here because the contract has one home; implemented nowhere in this package,
    because admission, materialization, event fencing and eviction all need a device and
    this package is torch-free by construction. cozy-runtime supplies the body
    (`internal/residency.py`); a promoted Varena adapter would supply a different one and the
    package above would not be able to tell.

    `admit` runs BEFORE method entry and either makes the declared set materialized or
    refuses typed. `release` runs after the body returns, records the event fence, and makes
    the leases EVICTABLE — it never forces eviction and never imposes a global scope order.
    """

    def admit(self, method: str, components: tuple[str, ...]) -> None: ...

    def release(self, method: str, components: tuple[str, ...]) -> None: ...


class Placement(Protocol):
    """Where a generation's components live across the GPUs of its group, installed by the
    runtime on the group's first GPU when it hosts a `placeable` component elsewhere.

    A scope whose declared components ALL live on other GPUs holds nothing on this device:
    its module calls run on their home GPUs. Such a REMOTE scope is admitted there, not
    here, and may be open beside this GPU's one serial scope (`author.concurrently`).
    """

    def remote(self, components: tuple[str, ...]) -> bool: ...

    def enter(self, components: tuple[str, ...]) -> None: ...

    def leave(self, components: tuple[str, ...]) -> None: ...

    def carry[T](self, call: Callable[[], T], components: tuple[str, ...]) -> Callable[[], T]:
        """`call`, to run on another thread under this thread's device and autograd modes.
        Its components' home GPUs are reserved from now until it returns."""
        ...


#: The remote scopes open in this context, innermost last: (model, method, components).
#: Per context rather than per model, so a remote scope on a helper thread is visible to that
#: thread's hosted calls and to nothing else.
_REMOTE_SCOPES: contextvars.ContextVar[tuple[tuple[Model[Any], str, tuple[str, ...]], ...]] = (
    contextvars.ContextVar("cozy_remote_scopes", default=())
)


def remote_scope(model: object) -> tuple[str, tuple[str, ...]] | None:
    """The remote scope of `model` open in this context, if any: (method, components)."""
    for owner, method, components in reversed(_REMOTE_SCOPES.get()):
        if owner is model:
            return method, components
    return None


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


# STOPGAP until memory v3, which deletes it: a device OOM in a scope holding only diffusers'
# image `AutoencoderKL` (SDXL's decode: one call, one returned frame) frees what the scope
# does not hold and runs the same call once more tiled. Nothing else is retried: a video
# VAE's decode streams chunks to its caller, and a second run would deliver them twice.
def _tiles_after_oom(model: Any, components: tuple[str, ...], exc: Exception) -> list[Any]:
    plane: Any = model._cozy_residency
    held = plane.backend.components if plane is not None and is_device_oom(exc) else {}
    modules = [held.get(name) for name in components]
    image_vaes = all(
        any(k.__name__ == "AutoencoderKL" and k.__module__.startswith("diffusers.") for k in
            type(m).__mro__)
        for m in modules
    )
    return modules if modules and image_vaes else []


@contextmanager
def _tiled_retry(model: Any, method: str, modules: list[Any]) -> Iterator[None]:
    plane: Any = model._cozy_residency
    plane.shed()
    if plane.torch is not None and plane.kind == "cuda":
        plane.torch.cuda.empty_cache()
    plane.backend.stage_log.append({"method": method, "action": "decode retried tiled after OOM"})
    _LOG.warning("%s: decode retried tiled after OOM", method)
    untiled = [m for m in modules if not getattr(m, "use_tiling", False)]
    for module in untiled:
        module.enable_tiling()
    try:
        yield
    finally:
        for module in untiled:
            module.disable_tiling()


def _is_async(fn: Callable[..., object]) -> bool:
    return inspect.iscoroutinefunction(inspect.unwrap(fn))


def _sync_wrapper[M: _ScopedModel, **P, R](
    fn: Callable[Concatenate[M, P], R], components: tuple[str, ...]
) -> Callable[Concatenate[M, P], R]:
    @functools.wraps(fn)
    def wrapper(self: M, /, *args: P.args, **kwargs: P.kwargs) -> R:
        with self._cozy_scope(fn.__name__, components):
            tiles: list[Any] = []
            try:
                result = fn(self, *args, **kwargs)
            except Exception as exc:
                if not (tiles := _tiles_after_oom(self, components, exc)):
                    raise
            if tiles:  # outside `except`, so the failed call's frames and tensors are gone
                with _tiled_retry(self, fn.__name__, tiles):
                    result = fn(self, *args, **kwargs)
            _refuse_lazy(fn.__name__, result)
            return result

    return wrapper


def _async_wrapper[M: _ScopedModel, **P, R](
    fn: Callable[Concatenate[M, P], R], components: tuple[str, ...]
) -> Callable[Concatenate[M, P], R]:
    """The scope stays active THROUGH `await` — the lease is the method's, not the step's."""

    @functools.wraps(fn)
    async def wrapper(self: M, /, *args: P.args, **kwargs: P.kwargs) -> object:
        with self._cozy_scope(fn.__name__, components):
            tiles: list[Any] = []
            try:
                result = await fn(self, *args, **kwargs)  # type: ignore[misc]
            except Exception as exc:
                if not (tiles := _tiles_after_oom(self, components, exc)):
                    raise
            if tiles:
                with _tiled_retry(self, fn.__name__, tiles):
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


class Model[PipelineT]:
    """Base of a package's reusable model class — the ONE long-lived model boundary.

    The author owns WHAT the model is: the three lifecycle methods `load` (construct under
    fake tensors, then fill), optional `warm` (post-fill, pre-serving work with real
    tensors) and `unload` (release when the construction leaves the worker), the declared
    persistent state, and the component-touching methods handlers call. The runtime owns
    WHEN they run and where every byte lives (model-lifecycle.md). A bare typed parameter
    is the whole declaration, and its presence in a signature is what derives `gpu`.
    """

    __sequence_parallel__: tuple[int, ...] = ()
    """`@sequence_parallel(degrees=...)`: the degrees this class shards at. Empty is the
    honest default - a class that said nothing is never sharded."""

    __placeable__: tuple[str, ...] = ()
    """`@placeable(...)`: components the runtime may host on another GPU of the group."""

    __encoded_leaves__: str = "refuse"
    """The class's `encoded_leaves=` declaration, defaulted to the conservative answer.

    REFUSE is the default because the absence of a statement is not consent. A package
    written before the native rung existed cannot have thought about whether its code
    survives leaf replacement, and reading its silence as "yes" would be the runtime
    deciding a question only the author can answer."""

    __fusion__: str = "refuse"
    """The class's `fusion=` declaration (h3a-015), defaulted the same way for the same
    reason: silence is not consent to a substituted forward. Nothing defaults to
    `require` — a requirement is only ever spelled out."""

    __adapter_compatibility__: tuple[AdapterCompatibility, ...] = ()
    """Closed generic adapter declarations; empty means overlays are refused."""

    checkpoint_ref: str = ""
    _cozy_selection_ref: str = ""
    """The pinned-ref REPRODUCTION FACT for result structs — an observation, never catalog
    authority. Model-scoped because it is construction identity."""

    _cozy_ready = False
    _cozy_loading = False
    _cozy_active: tuple[str, tuple[str, ...]] | None = None
    _cozy_harness: ModelTestHarness | None = None
    _cozy_record: ConstructionRecord | None = None
    _cozy_identity: ConstructionIdentity | None = None
    _cozy_schedulers: Mapping[str, Scheduler] = {}
    _cozy_caches: tuple[DerivedCache, ...] = ()
    _cozy_adapters: tuple[AdapterRef, ...] = ()
    _cozy_residency: Residency | None = None
    """Runtime-installed. Absent under `for_test` and in derive mode, which is the whole
    reason a scope works without one: a test double has no bytes to stage."""
    _cozy_placement: Placement | None = None
    """Runtime-installed on the first GPU of a group that hosts a component on another."""

    def __init_subclass__(cls, /, **keywords: str) -> None:
        """The class-keyword plane is exactly `encoded_leaves` and `fusion` (model-code-fit
        §1, h3a-015): two closed consents about what the runtime may substitute.

        Anything else refuses typed: the open stamp vocabulary (`task=`, `objective=`),
        `structure=` and `self_loading=` are deleted (D1), and a keyword the runtime does not
        read is not a fact anyone else would look for.
        """
        super().__init_subclass__()
        for keyword, vocabulary, attribute in (
            ("encoded_leaves", ENCODED_LEAVES, "__encoded_leaves__"),
            ("fusion", FUSION, "__fusion__"),
        ):
            if keyword in keywords:
                value = keywords.pop(keyword)
                if value not in vocabulary:
                    raise ConformanceError(
                        f"class {cls.__name__}({keyword}={value!r}): the vocabulary is CLOSED — "
                        + " | ".join(f"{k!r}: {why}" for k, why in vocabulary.items()),
                        code=f"{keyword}_value",
                    )
                setattr(cls, attribute, value)
        if keywords:
            names = ", ".join(f"{key}=…" for key in keywords)
            raise ConformanceError(
                f"class {cls.__name__}({names}): a Model class carries exactly two keywords, "
                "encoded_leaves and fusion — stamps, task=, objective= and structure= are "
                "deleted (model-code-fit D1); a semantic twin is a component name or a config "
                "fact",
                code="class_keyword",
                fields=sorted(keywords),
            )

    # ----------------------------------------------------------------- lifecycle

    def load(self, loader: Loader) -> None:
        """Construct this model from the bound artifact. The runtime owns WHEN it runs."""
        raise ConformanceError(
            f"{type(self).__name__} defines no load(loader): a Model constructs through "
            "loader.construct(T, factory=…) — the one construction primitive (§1.1)",
            code="no_load",
        )

    def choose_attention(self, context: AttentionContext) -> str | Sequence[str] | None:
        """Select an eligible backend before warm-up, or retain Runtime's recommendation.

        This hook chooses a name, not a kernel implementation or device operation.
        A name is an explicit choice: validated and never silently replaced. A sequence
        of names is a preference: Runtime takes the first one this site, device,
        environment and numerical probe admit on each GPU, records why each earlier
        one was skipped, and falls back to its recommendation when none serves.
        Approximate attention remains the model author's quality decision, independent
        of FP8 checkpoint storage. Developer request overrides take precedence.
        """
        return None

    def warm(self, ctx: Context) -> None:
        """Optional post-fill, pre-serving work — the ONE place for it (cr-110, #708).

        The runtime calls it once per construction fill: after the fill and its
        verification, before the placement reports DISPATCHABLE, on every worker, with the
        device lease held and real tensors in place. Never again on a component stage or
        evict — module objects survive residency, only bytes move — and again only when the
        construction is rebuilt. What goes here is the author's choice: an in-place
        `module.compile()` or `compile_repeated_blocks`, Triton/kernel loading, autotune,
        a dry step at the canonical shape, or nothing. `torch.compile(module)` is refused
        here as under derive: it wraps the module and renames every state_dict key, so the
        object the code then calls is not the one the fill and residency planes hold.
        `ctx` carries the device, the fill's deadline and cancellation; it has no request
        and can reserve no package call. A raise is a construction failure naming the
        exception, and the generation never serves.
        """
        return None

    def unload(self, loader: Loader) -> None:
        """Release declared state. The runtime drains before calling this."""
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
        return RequestView._make(
            effective, self._cozy_adapters or ctx._adapters, self._cozy_schedulers, sampler
        )

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
        if self._cozy_active is None and remote_scope(self) is None:
            raise CapabilityError(
                f"{what} outside an active component-use scope: the runtime joins the entry's "
                "lease to the scope, and there is nowhere for it to live otherwise (§1.3)",
                code="no_active_scope",
            )

    def _cozy_require_ready(self, what: str) -> None:
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
                    "RequestView or a package-specific attempt object (§1.1.2)",
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


def warm_context(where: Device, *, deadline: float, cancel: Callable[[], bool]) -> Context:
    """The Context `Model.warm` receives: where it runs, how long the fill may take, and
    whether the generation is still wanted. No request id and no adapters, because there
    is no attempt; the package-call broker only `invoke` binds is absent, so an invocable
    called here refuses `child_broker_absent` by construction (cr-110)."""
    return Context("", deadline, where, cancel)


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


def _derive_model[M: Model[Any]](cls: type[M], checkpoint_ref: str) -> M:
    """A PLAIN RECORD carrying one exact Manifest identity, typed as the author's class.

    No construction runs, so the instance carries `checkpoint_ref` and nothing else: a
    component the load never built is simply absent, and reading one raises the ordinary
    `AttributeError` that names it. Nothing is intercepted to say so — a `__getattribute__`
    that tested a derive-only flag on EVERY attribute read of EVERY model, serving ones
    included, bought a nicer message for the one case at the cost of a branch on the other
    million, and `object.__setattr__` walked around it anyway. `load`/`for_request` refuse
    through the ordinary `not_ready` gate, because no generation is Ready here.
    """
    if not checkpoint_ref.startswith("sha256:") or len(checkpoint_ref) != 71:
        raise ConformanceError(
            "derive-only Model requires one lowercase sha256 Manifest identity",
            code="derive_model_manifest_invalid",
        )
    try:
        raw = bytes.fromhex(checkpoint_ref[7:])
    except ValueError as exc:
        raise ConformanceError(
            "derive-only Model Manifest identity is malformed",
            code="derive_model_manifest_invalid",
        ) from exc
    if len(raw) != 32 or checkpoint_ref != "sha256:" + raw.hex():
        raise ConformanceError(
            "derive-only Model Manifest identity is not canonical",
            code="derive_model_manifest_invalid",
        )
    model = object.__new__(cls)
    object.__setattr__(model, "checkpoint_ref", checkpoint_ref)
    return model


@final
class _ComponentScope:
    """ONE public scope is active at a time per instance (§1.1). A composite operation
    declares its components together on one method.

    The one exception holds nothing here: a REMOTE scope, whose every declared component the
    runtime keeps on another GPU of the group (`Placement.remote`). It admits nothing on
    this device and marks this context, not the instance, so it may be open beside the one
    local scope (`author.concurrently` overlaps the two)."""

    __slots__ = ("_components", "_method", "_model", "_remote")

    def __init__(self, model: Model[Any], method: str, components: tuple[str, ...]) -> None:
        self._model = model
        self._method = method
        self._components = components
        self._remote: contextvars.Token[Any] | None = None

    def __enter__(self) -> None:
        model = self._model
        model._cozy_require_ready(self._method)
        placement = model._cozy_placement
        if placement is not None and placement.remote(self._components):
            if remote_scope(model) is not None:
                raise CapabilityError(
                    f"{type(model).__name__}.{self._method} entered inside another remote "
                    "scope of the same instance in this context (§1.1)",
                    code="concurrent_scope",
                )
            placement.enter(self._components)
            self._remote = _REMOTE_SCOPES.set(
                (*_REMOTE_SCOPES.get(), (model, self._method, self._components))
            )
            if model._cozy_harness is not None:
                model._cozy_harness.calls.append(ScopeCall(self._method, self._components))
            return
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
        if self._remote is not None:
            _REMOTE_SCOPES.reset(self._remote)
            self._remote = None
            placement = model._cozy_placement
            if placement is not None:
                placement.leave(self._components)
            return
        object.__setattr__(model, "_cozy_active", None)
        # RELEASE MAKES BYTES EVICTABLE; it does not evict, and it imposes no global scope
        # sequence (§3.2). The lease ends at the completion event, not at this statement.
        residency = model._cozy_residency
        if residency is not None:
            residency.release(self._method, self._components)


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
        self,
        identity: ConstructionIdentity,
        cls: type[Model[Any]],
        artifact: Artifact,
        mode: Mode,
        adapters: Sequence[AdapterRef] = (),
    ) -> Generation:
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


_EMPTY_ARTIFACT = Artifact(snapshot="unload://drain", tensor_schema={}, config=Config({}))


def _merge(records: Sequence[ConstructionRecord]) -> ConstructionRecord:
    """Several `construct` calls in one `load()` are one construction contract."""
    return ConstructionRecord(
        destinations=tuple(d for r in records for d in r.destinations),
        components=tuple(c for r in records for c in r.components),
        schedulers=tuple(s for r in records for s in r.schedulers),
        filled=sum(r.filled for r in records),
        filled_bytes=sum(r.filled_bytes for r in records),
        ignored_extras=tuple(e for r in records for e in r.ignored_extras),
        fit=next((r.fit for r in records if r.fit is not None), None),
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
