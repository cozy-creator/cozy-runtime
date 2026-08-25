"""The `App` registry — the minimum `cozy-runtime new`'s skeleton needs to import cleanly.

SEAM FOR cr-002a (author core), which owns the real surface: typed request/result structs
and schema derivation, `Model` parameters and `loader.construct`, the Context services
(Telemetry/Outputs/Settings/Secrets), `@uses_components`, the override doors
(`resources=`/`demand=`/`structure=`/`encodings=`/`preflight=`), and the freeze that makes
registration immutable. What is real here: one registry per app, both decorator spellings,
ParamSpec preservation (a decorated handler keeps its exact signature under mypy --strict),
duplicate-name refusal, and the rule that importing a module never mutates a registry —
registration happens only when the entry module executes the decorator.

Deliberately absent, and NOT to be added here: any import of cozy_runtime.internal, any
environment or config read, any device/allocator surface. Those are fenced (§7).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, ParamSpec, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")

Kind = Literal["entrypoint", "job"]


class RegistrationError(Exception):
    """A duplicate or otherwise invalid registration. Refuses at import, not at serve."""


@dataclass(frozen=True, slots=True)
class Registration:
    """One registered callable. cr-002a widens this with the derived schema + capabilities."""

    name: str
    kind: Kind
    fn: Callable[..., object]
    publishes: bool = False


@dataclass(slots=True)
class App:
    """The one registry an endpoint module exports as `app = App()`."""

    _registry: dict[str, Registration] = field(default_factory=dict)

    @overload
    def entrypoint(self, fn: Callable[P, R], /) -> Callable[P, R]: ...

    @overload
    def entrypoint(
        self, /, *, name: str | None = None
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def entrypoint(
        self, fn: Callable[P, R] | None = None, /, *, name: str | None = None
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Register a request-serving callable. Bare is the common case."""
        return self._register("entrypoint", fn, name, publishes=False)

    @overload
    def job(self, fn: Callable[P, R], /) -> Callable[P, R]: ...

    @overload
    def job(
        self, /, *, name: str | None = None, publishes: bool = False
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def job(
        self,
        fn: Callable[P, R] | None = None,
        /,
        *,
        name: str | None = None,
        publishes: bool = False,
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Register a run-to-completion callable (what pods execute)."""
        return self._register("job", fn, name, publishes=publishes)

    def _register(
        self, kind: Kind, fn: Callable[P, R] | None, name: str | None, *, publishes: bool
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        def decorate(target: Callable[P, R]) -> Callable[P, R]:
            registered = name or target.__name__
            if registered in self._registry:
                raise RegistrationError(
                    f"duplicate registration {registered!r}: names are unique per app"
                )
            self._registry[registered] = Registration(registered, kind, target, publishes)
            return target

        if fn is None:
            return decorate
        return decorate(fn)

    def registrations(self) -> tuple[Registration, ...]:
        """Registered callables in declaration order."""
        return tuple(self._registry.values())
