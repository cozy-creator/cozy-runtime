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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import ParamSpec, TypeVar, overload

from cozy_runtime.author._calls import _export
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._markers import Bound
from cozy_runtime.author._memo import MemoDependency
from cozy_runtime.author._memo import declare as declare_memo
from cozy_runtime.author._model_defaults import DefaultLadder, model_defaults
from cozy_runtime.author._signature import Kind, Surface, analyze
from cozy_runtime.author._weights import WeightsOutput

P = ParamSpec("P")
R = TypeVar("R")


class RegistrationError(ConformanceError):
    """A duplicate or otherwise invalid registration. Refuses at import, not at serve."""

    default_code = "registration"


@dataclass(slots=True)
class Registration:
    """One registered callable plus the doors it opened. cr-003 reads `surface`."""

    name: str
    kind: Kind
    fn: Callable[..., object]
    invocable: bool = False
    memoize: bool = False
    model_defaults: Mapping[str, DefaultLadder] = field(default_factory=dict)
    publishes: bool = False
    emits_media: bool = False
    weights_outputs: tuple[WeightsOutput, ...] = ()
    demand_bound: Bound | None = None
    preflight: Callable[..., object] | None = None
    internal: bool = False
    """Managed children from this installed package revision only; never a root request."""
    accelerator: bool | None = None
    """A job's own execution device: True computes on an accelerator, False on CPU. None
    leaves the machine class to the host's reading of the dependency closure."""
    hidden: bool = False
    """This surface is DECLARED but NOT PUBLISHED (#572d). A hidden entrypoint stays in the
    local registry but is absent from the package interface, so no deployment stages a binding and
    no request reaches it.

    It is an AUTHOR fact, not a deployment one: what makes H3's `reference_to_video` hidden
    is that its vision-conditioning seam is unbuilt, which only the author knows. A
    deployment choosing not to offer a working surface is a different thing and belongs in
    a binding. Hiding was a docs fact until a hidden entrypoint's binding was staged, failed
    to prepare, and denied its working sibling the card."""
    checked: bool = field(default=False, repr=False)
    """Set once the describe-time conformance layer has passed this surface."""
    _surface: Surface | None = field(default=None, repr=False)

    @property
    def surface(self) -> Surface:
        """The derived surface — schemas, capabilities, models, and preflight identity."""
        if self._surface is None:
            self._surface = analyze(
                self.fn,
                name=self.name,
                kind=self.kind,
                publishes=self.publishes,
                emits_media=self.emits_media,
                weights_outputs=self.weights_outputs,
                demand_bound=self.demand_bound,
                preflight=self.preflight,
                hidden=self.hidden,
                internal=self.internal,
                accelerator=self.accelerator,
                invocable=self.invocable,
                memoize=self.memoize,
                model_defaults=self.model_defaults,
            )
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
        defaults: Mapping[str, Sequence[Mapping[str, str | int]]] | None = None,
        memoize: bool = False,
        memo_version: str | None = None,
        memo_dependencies: tuple[MemoDependency, ...] = (),
        internal: bool = False,
    ) -> Callable[P, R]: ...

    @overload
    def entrypoint(
        self,
        /,
        *,
        name: str | None = None,
        preflight: Callable[..., object] | None = None,
        demand: Bound | None = None,
        defaults: Mapping[str, Sequence[Mapping[str, str | int]]] | None = None,
        hidden: bool = False,
        internal: bool = False,
        memoize: bool = False,
        memo_version: str | None = None,
        memo_dependencies: tuple[MemoDependency, ...] = (),
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def entrypoint(
        self,
        fn: Callable[P, R] | None = None,
        /,
        *,
        name: str | None = None,
        preflight: Callable[..., object] | None = None,
        demand: Bound | None = None,
        defaults: Mapping[str, Sequence[Mapping[str, str | int]]] | None = None,
        hidden: bool = False,
        internal: bool = False,
        memoize: bool = False,
        memo_version: str | None = None,
        memo_dependencies: tuple[MemoDependency, ...] = (),
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Register a request-serving callable. BARE is the common case (§1.2).

        `hidden=True` declares a surface that is real but NOT READY: it is omitted from the
        published interface and gets no binding staged and no traffic (§1.2, #572d).

        `internal=True` permits managed calls from this installed package revision only.
        It retains bindings and metadata, but cannot be invoked as an external request.

        `memoize=True` explicitly admits deterministic measurement results to the managed
        operation cache. Optional `memo_dependencies` and `memo_version` declare
        result-affecting helpers/resources/native builds just as for `invocable`.
        Ordinary generation remains nonmemoized by default."""
        if type(memoize) is not bool:
            raise RegistrationError("entrypoint memoize must be a boolean")
        return self._register(
            "entrypoint",
            fn,
            name,
            publishes=False,
            weights_outputs=(),
            preflight=preflight,
            demand=demand,
            hidden=hidden,
            internal=internal,
            memoize=memoize,
            memo_version=memo_version,
            memo_dependencies=memo_dependencies,
            defaults=defaults,
        )

    @overload
    def job(
        self,
        fn: Callable[P, R],
        /,
        *,
        name: str | None = None,
        publishes: bool = False,
        emits_media: bool = False,
        weights: tuple[WeightsOutput, ...] = (),
        defaults: Mapping[str, Sequence[Mapping[str, str | int]]] | None = None,
        demand: Bound | None = None,
        internal: bool = False,
        accelerator: bool | None = None,
    ) -> Callable[P, R]: ...

    @overload
    def job(
        self,
        /,
        *,
        name: str | None = None,
        publishes: bool = False,
        emits_media: bool = False,
        weights: tuple[WeightsOutput, ...] = (),
        defaults: Mapping[str, Sequence[Mapping[str, str | int]]] | None = None,
        demand: Bound | None = None,
        internal: bool = False,
        accelerator: bool | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def job(
        self,
        fn: Callable[P, R] | None = None,
        /,
        *,
        name: str | None = None,
        publishes: bool = False,
        emits_media: bool = False,
        weights: tuple[WeightsOutput, ...] = (),
        defaults: Mapping[str, Sequence[Mapping[str, str | int]]] | None = None,
        demand: Bound | None = None,
        internal: bool = False,
        accelerator: bool | None = None,
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Register a RUN-TO-COMPLETION callable (§2).

        Structurally separate from serving, and separate in exactly one place: `kind`. What
        a body may WRITE is its own declaration (`publishes=`, `emits_media=`) — grants mint
        off the declaration, never off the kind — and everything else about the job is
        derived from the signature exactly as it is for an entrypoint. There is no
        JobContext, no second kernel and no serving loop.
        `internal=True` retains this job for same-package managed child calls only.
        `accelerator=` is the job's own execution device: False runs it on a CPU machine
        even when its dependencies include torch, True requires an accelerator.
        """
        return self._register(
            "job",
            fn,
            name,
            publishes=publishes,
            emits_media=emits_media,
            weights_outputs=weights,
            preflight=None,
            demand=demand,
            internal=internal,
            accelerator=accelerator,
            defaults=defaults,
        )

    def _register(
        self,
        kind: Kind,
        fn: Callable[P, R] | None,
        name: str | None,
        *,
        publishes: bool,
        weights_outputs: tuple[WeightsOutput, ...],
        preflight: Callable[..., object] | None,
        demand: Bound | None,
        emits_media: bool = False,
        hidden: bool = False,
        internal: bool = False,
        memoize: bool = False,
        memo_version: str | None = None,
        memo_dependencies: tuple[MemoDependency, ...] = (),
        defaults: Mapping[str, Sequence[Mapping[str, str | int]]] | None = None,
        accelerator: bool | None = None,
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        if type(internal) is not bool:
            raise RegistrationError("internal must be a boolean")
        if accelerator is not None and type(accelerator) is not bool:
            raise RegistrationError("accelerator must be a boolean")
        if hidden and internal:
            raise RegistrationError("an internal callable cannot also be hidden")
        declared_defaults = model_defaults(defaults)

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
            declare_memo(
                export.implementation if export is not None else target,
                memoize=export.memoize if export is not None else memoize,
                version=memo_version,
                dependencies=memo_dependencies,
            )
            self._registry[registered] = Registration(
                name=registered,
                kind=kind,
                fn=export.implementation if export is not None else target,
                invocable=export is not None,
                memoize=export.memoize if export is not None else memoize,
                model_defaults=declared_defaults or (export.defaults if export is not None else {}),
                publishes=publishes,
                emits_media=emits_media,
                weights_outputs=weights_outputs,
                demand_bound=demand,
                preflight=preflight,
                hidden=hidden,
                internal=internal,
                accelerator=accelerator,
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
