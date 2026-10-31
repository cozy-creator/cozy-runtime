"""The `App` registry — the one object a package module exports as `app = App()`.

The decorators are sugar for explicit registration calls (`app.entrypoint(...)(fn)`):
sibling modules export plain functions and the entry module registers them, so merely
IMPORTING a module never mutates any registry (§1.0). Both spellings register the same
way — `def` is the compute default, `async def` is for genuinely async-shaped handlers
(§1.2) — and both are ParamSpec-preserving, so a decorated handler keeps its exact
signature and callers stay fully typed under `mypy --strict` (§1.7).

Registration is CHEAP: nothing here resolves annotations, so a forward reference to a class
defined later in the module is fine. The surface is derived on first inspection (describe),
and after describe the registry is frozen.

Deliberately absent, and NOT to be added here: any import of cozy_runtime.internal, any
environment or config read, any device/allocator surface. Those are fenced (§7).
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypedDict, TypeVar, Unpack, overload

from cozy_runtime.author._calls import _export
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._markers import Bound
from cozy_runtime.author._memo import MemoDependency, Memoize, memo_policy, one_memo
from cozy_runtime.author._memo import declare as declare_memo
from cozy_runtime.author._model import Model
from cozy_runtime.author._model_defaults import (
    model_components,
    model_defaults,
    model_on_demand,
    model_phases,
    model_prepared_use,
)
from cozy_runtime.author._signature import Surface, analyze
from cozy_runtime.author._video import Implements
from cozy_runtime.author._weights import WeightsOutput

P = ParamSpec("P")
R = TypeVar("R")


class RegistrationOptions(TypedDict, total=False):
    """The registration options `entrypoint` and `child_task` share."""

    name: str | None
    preflight: Callable[..., object] | None
    phases: Sequence[str | Collection[str]]
    prepared_use: Mapping[str, str] | None
    demand: Bound | None
    defaults: Mapping[str, str] | None
    components: Mapping[str, Sequence[str]] | None
    on_demand: Sequence[str]
    memoize: bool | Memoize
    memo_version: str | None
    memo_dependencies: tuple[MemoDependency, ...]
    publishes: bool
    emits_media: bool
    weights: tuple[WeightsOutput, ...]
    accelerator: bool | None


class RegistrationError(ConformanceError):
    """A duplicate or otherwise invalid registration. Refuses at import, not at serve."""

    default_code = "registration"


@dataclass(slots=True)
class Registration:
    """One registered callable plus the doors it opened. cr-003 reads `surface`."""

    name: str
    fn: Callable[..., object]
    invocable: bool = False
    memoize: Memoize | None = None
    model_defaults: Mapping[str, str] = field(default_factory=dict)
    model_components: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    model_on_demand: tuple[str, ...] = ()
    model_prepared_use: Mapping[str, str] = field(default_factory=dict)
    #: A model argument's class where the registration names it (`cozy_runtime.derive.recipes`).
    model_classes: Mapping[str, type[Model[Any]]] = field(default_factory=dict)
    publishes: bool = False
    emits_media: bool = False
    weights_outputs: tuple[WeightsOutput, ...] = ()
    demand_bound: Bound | None = None
    preflight: Callable[..., object] | None = None
    phases: tuple[tuple[str, ...], ...] = ()
    child_task: bool = False
    """Called by this installed package revision's own code only; never a root request."""
    accelerator: bool = True
    """False: CPU only. True (the default): the machine's GPU when it has one."""
    implements: Implements | None = None
    """The interface it implements, as declared."""
    hidden: bool = False
    """DECLARED but NOT PUBLISHED (#572d): absent from the package interface, so no binding
    is staged and no request reaches it. An author fact (an unbuilt seam), not a deployment one."""
    checked: bool = field(default=False, repr=False)
    """Set once the describe-time conformance layer has passed this surface."""
    _surface: Surface | None = field(default=None, repr=False)

    @property
    def surface(self) -> Surface:
        """The derived surface — schemas, capabilities, models, and preflight identity."""
        if self._surface is None:
            surface = analyze(
                self.fn,
                name=self.name,
                publishes=self.publishes,
                emits_media=self.emits_media,
                weights_outputs=self.weights_outputs,
                demand_bound=self.demand_bound,
                preflight=self.preflight,
                hidden=self.hidden,
                child_task=self.child_task,
                accelerator=self.accelerator,
                invocable=self.invocable,
                memoize=self.memoize,
                model_defaults=self.model_defaults,
                model_components=self.model_components,
                model_on_demand=self.model_on_demand,
                model_prepared_use=self.model_prepared_use,
                model_classes=self.model_classes,
                implements=self.implements,
            )
            self._surface = surface
        return self._surface


@dataclass(slots=True)
class App:
    """The one registry a package module exports as `app = App()`."""

    _registry: dict[str, Registration] = field(default_factory=dict)
    _frozen: bool = False

    @overload
    def entrypoint(
        self,
        fn: Callable[P, R],
        /,
        *,
        hidden: bool = False,
        implements: Implements | None = None,
        model_classes: Mapping[str, type[Model[Any]]] | None = None,
        **options: Unpack[RegistrationOptions],
    ) -> Callable[P, R]: ...

    @overload
    def entrypoint(
        self,
        /,
        *,
        hidden: bool = False,
        implements: Implements | None = None,
        model_classes: Mapping[str, type[Model[Any]]] | None = None,
        **options: Unpack[RegistrationOptions],
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def entrypoint(
        self,
        fn: Callable[P, R] | None = None,
        /,
        *,
        hidden: bool = False,
        implements: Implements | None = None,
        model_classes: Mapping[str, type[Model[Any]]] | None = None,
        **options: Unpack[RegistrationOptions],
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Register a public function, bare (`app.entrypoint(fn)`) or as a decorator (§1.2).

        `weights=` names the TensorFS model outputs it writes; `publishes=`/`emits_media=`
        declare its `Outputs`. `accelerator=False` runs it on CPU only; otherwise (`True`,
        or absent) it gets the machine's GPU when there is one. `hidden=True` keeps it out of
        the published interface.

        `memoize=True` admits deterministic results to the operation cache (never for a
        writer); `memo_dependencies`/`memo_version` declare what else decides the result, and
        `Memoize(requires=("seed",))` memoizes only requests that name each listed option.
        `components={"model": ("transformer", "vae")}` names the components each model argument
        uses, so a machine fetches those first; absent means all. `on_demand=("upsampler",)`
        names the model arguments a request may leave unused: each is built, and its weights
        fetched, when the function first reads it. `model_classes` names a model argument's class
        in place of its annotation (the Runtime's own generic functions). `phases=` declares
        immutable component groups in execution order, available to queued preparation before
        handler entry; it has the same residency meaning as the handler's `ctx.plan` hints.
        `implements=Implements("video/shot", max_piece_s=15)` names the interface it
        implements (`ImageRequest`, `ShotRequest`, `UpscaleRequest`) and the facts a caller
        plans by."""
        return self._register(
            fn, False, hidden=hidden, implements=implements, model_classes=model_classes, **options
        )

    @overload
    def child_task(
        self, fn: Callable[P, R], /, **options: Unpack[RegistrationOptions]
    ) -> Callable[P, R]: ...

    @overload
    def child_task(
        self, /, **options: Unpack[RegistrationOptions]
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def child_task(
        self, fn: Callable[P, R] | None = None, /, **options: Unpack[RegistrationOptions]
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Register a function only this package's own calls run: never a root request or
        another package's call. It takes `entrypoint`'s options except `hidden`, `implements` and
        `model_classes`. `prepared_use={"model": "use"}` (a child task's only) names the typed
        field that describes its fixed repeated weight use before construction."""
        return self._register(fn, True, **options)

    def _register(
        self,
        fn: Callable[P, R] | None,
        child_task: bool,
        /,
        *,
        hidden: bool = False,
        implements: Implements | None = None,
        model_classes: Mapping[str, type[Model[Any]]] | None = None,
        name: str | None = None,
        preflight: Callable[..., object] | None = None,
        phases: Sequence[str | Collection[str]] = (),
        prepared_use: Mapping[str, str] | None = None,
        demand: Bound | None = None,
        defaults: Mapping[str, str] | None = None,
        components: Mapping[str, Sequence[str]] | None = None,
        on_demand: Sequence[str] = (),
        memoize: bool | Memoize = False,
        memo_version: str | None = None,
        memo_dependencies: tuple[MemoDependency, ...] = (),
        publishes: bool = False,
        emits_media: bool = False,
        weights: tuple[WeightsOutput, ...] = (),
        accelerator: bool | None = None,
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        # Immutable CPU metadata; the source reader uses the same closed validator.
        declared_phases = model_phases(phases)
        policy = memo_policy(memoize)
        if child_task and (hidden or implements is not None or model_classes):
            raise RegistrationError("child_task takes no hidden=, implements= or model_classes=")
        if accelerator is not None and type(accelerator) is not bool:
            raise RegistrationError(
                "accelerator is True or False: a function gets the machine's GPU unless it "
                "declares accelerator=False"
            )
        if implements is not None and not isinstance(implements, Implements):
            raise RegistrationError("implements is an Implements(interface, ...)")
        declared_defaults = model_defaults(defaults)
        declared_components = model_components(components)
        declared_on_demand = model_on_demand(on_demand)
        declared_use = model_prepared_use(prepared_use)

        def decorate(target: Callable[P, R]) -> Callable[P, R]:
            if self._frozen:
                raise RegistrationError(
                    f"{target.__name__}: the registry is frozen — after describe, "
                    "registration is immutable (§1.0)"
                )
            registered = name or target.__name__
            if registered in self._registry:
                raise RegistrationError(
                    f"duplicate registration {registered!r}: names are unique per app"
                )
            export = _export(target)
            if declared_defaults and export is not None and export.defaults:
                raise ConformanceError(
                    "declare defaults once, on invocable or App registration", code="model_defaults"
                )
            if declared_components and export is not None and export.components:
                raise ConformanceError(
                    "declare components once, on invocable or App registration",
                    code="model_components",
                )
            memo = one_memo(export.memoize, policy) if export is not None else policy
            declare_memo(
                export.implementation if export is not None else target,
                version=memo_version,
                dependencies=memo_dependencies,
            )
            self._registry[registered] = Registration(
                name=registered,
                fn=export.implementation if export is not None else target,
                invocable=export is not None,
                memoize=memo,
                model_defaults=declared_defaults or (export.defaults if export is not None else {}),
                model_components=declared_components
                or (export.components if export is not None else {}),
                model_on_demand=declared_on_demand,
                model_prepared_use=declared_use,
                model_classes=dict(model_classes or {}),
                publishes=publishes,
                emits_media=emits_media,
                weights_outputs=weights,
                demand_bound=demand,
                preflight=preflight,
                phases=declared_phases,
                hidden=hidden,
                child_task=child_task,
                accelerator=accelerator is not False,
                implements=implements,
            )
            return target

        if fn is None:
            return decorate
        return decorate(fn)

    def registrations(self) -> tuple[Registration, ...]:
        """Registered callables in declaration order."""
        return tuple(self._registry.values())

    def get(self, name: str) -> Registration:
        try:
            return self._registry[name]
        except KeyError:
            raise RegistrationError(
                f"no registered callable named {name!r}: have {', '.join(self._registry) or 'none'}"
            ) from None

    def freeze(self) -> None:
        """After describe, registration is immutable (§1.0)."""
        self._frozen = True
