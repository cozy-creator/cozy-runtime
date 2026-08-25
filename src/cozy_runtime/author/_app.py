"""The `App` registry — the one object an endpoint module exports as `app = App()`.

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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ParamSpec, TypeVar, overload

from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._markers import Bound, Resources
from cozy_runtime.author._signature import Kind, Surface, analyze

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
    publishes: bool = False
    resources: Resources | None = None
    demand_bound: Bound | None = None
    preflight: Callable[..., object] | None = None
    checked: bool = field(default=False, repr=False)
    """Set once the describe-time conformance layer has passed this surface."""
    _surface: Surface | None = field(default=None, repr=False)

    @property
    def surface(self) -> Surface:
        """The derived surface — schemas, capabilities, `gpu`, preflight identity."""
        if self._surface is None:
            self._surface = analyze(
                self.fn,
                name=self.name,
                kind=self.kind,
                publishes=self.publishes,
                resources=self.resources,
                demand_bound=self.demand_bound,
                preflight=self.preflight,
            )
        return self._surface


@dataclass(slots=True)
class App:
    """The one registry an endpoint module exports as `app = App()`."""

    _registry: dict[str, Registration] = field(default_factory=dict)
    _frozen: bool = False

    @overload
    def entrypoint(self, fn: Callable[P, R], /) -> Callable[P, R]: ...

    @overload
    def entrypoint(
        self,
        /,
        *,
        name: str | None = None,
        preflight: Callable[..., object] | None = None,
        resources: Resources | None = None,
        demand: Bound | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def entrypoint(
        self,
        fn: Callable[P, R] | None = None,
        /,
        *,
        name: str | None = None,
        preflight: Callable[..., object] | None = None,
        resources: Resources | None = None,
        demand: Bound | None = None,
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Register a request-serving callable. BARE is the common case (§1.2)."""
        return self._register(
            "entrypoint",
            fn,
            name,
            publishes=False,
            preflight=preflight,
            resources=resources,
            demand=demand,
        )

    @overload
    def job(self, fn: Callable[P, R], /) -> Callable[P, R]: ...

    @overload
    def job(
        self,
        /,
        *,
        name: str | None = None,
        publishes: bool = False,
        resources: Resources | None = None,
        demand: Bound | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def job(
        self,
        fn: Callable[P, R] | None = None,
        /,
        *,
        name: str | None = None,
        publishes: bool = False,
        resources: Resources | None = None,
        demand: Bound | None = None,
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Register a run-to-completion callable (§2)."""
        return self._register(
            "job", fn, name, publishes=publishes, preflight=None, resources=resources, demand=demand
        )

    def _register(
        self,
        kind: Kind,
        fn: Callable[P, R] | None,
        name: str | None,
        *,
        publishes: bool,
        preflight: Callable[..., object] | None,
        resources: Resources | None,
        demand: Bound | None,
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
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
            self._registry[registered] = Registration(
                registered, kind, target, publishes, resources, demand, preflight
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
