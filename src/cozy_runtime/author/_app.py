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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import ParamSpec, TypeVar, overload

from cozy_runtime.author._artifact import ArtifactOutput
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._markers import Resources
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
    emits_media: bool = False
    artifact_outputs: tuple[ArtifactOutput, ...] = ()
    source_profiles: Mapping[str, str] = field(default_factory=dict)
    resources: Resources | None = None
    preflight: Callable[..., object] | None = None
    hidden: bool = False
    """This surface is DECLARED but NOT PUBLISHED (#572d). A hidden entrypoint stays in the
    local registry but is absent from the descriptor, so no deployment stages a binding and
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
        """The derived surface — schemas, capabilities, `gpu`, preflight identity."""
        if self._surface is None:
            self._surface = analyze(
                self.fn,
                name=self.name,
                kind=self.kind,
                publishes=self.publishes,
                emits_media=self.emits_media,
                artifact_outputs=self.artifact_outputs,
                source_profiles=self.source_profiles,
                resources=self.resources,
                preflight=self.preflight,
                hidden=self.hidden,
            )
        return self._surface


@dataclass(slots=True)
class App:
    """The one registry a package module exports as `app = App()`."""

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
        hidden: bool = False,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def entrypoint(
        self,
        fn: Callable[P, R] | None = None,
        /,
        *,
        name: str | None = None,
        preflight: Callable[..., object] | None = None,
        resources: Resources | None = None,
        hidden: bool = False,
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Register a request-serving callable. BARE is the common case (§1.2).

        `hidden=True` declares a surface that is real but NOT READY: it is omitted from the
        published descriptor and gets no binding staged and no traffic (§1.2, #572d)."""
        return self._register(
            "entrypoint",
            fn,
            name,
            publishes=False,
            artifact_outputs=(),
            source_profiles={},
            preflight=preflight,
            resources=resources,
            hidden=hidden,
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
        emits_media: bool = False,
        artifacts: tuple[ArtifactOutput, ...] = (),
        source_profiles: Mapping[str, str] | None = None,
        resources: Resources | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def job(
        self,
        fn: Callable[P, R] | None = None,
        /,
        *,
        name: str | None = None,
        publishes: bool = False,
        emits_media: bool = False,
        artifacts: tuple[ArtifactOutput, ...] = (),
        source_profiles: Mapping[str, str] | None = None,
        resources: Resources | None = None,
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Register a RUN-TO-COMPLETION callable (§2).

        Structurally separate from serving, and separate in exactly one place: `kind`. What
        a body may WRITE is its own declaration (`publishes=`, `emits_media=`) — grants mint
        off the declaration, never off the kind — and everything else about the job is
        derived from the signature exactly as it is for an entrypoint. There is no
        JobContext, no second kernel and no serving loop.

        ``source_profiles`` is the narrow source-first door: when present it maps every
        inferred Model parameter to one reviewed TensorFS profile. It selects no provider,
        source revision, checkpoint, destination, or placement.
        """
        return self._register(
            "job",
            fn,
            name,
            publishes=publishes,
            emits_media=emits_media,
            artifact_outputs=artifacts,
            source_profiles=dict(source_profiles or {}),
            preflight=None,
            resources=resources,
        )

    def _register(
        self,
        kind: Kind,
        fn: Callable[P, R] | None,
        name: str | None,
        *,
        publishes: bool,
        artifact_outputs: tuple[ArtifactOutput, ...],
        source_profiles: Mapping[str, str],
        preflight: Callable[..., object] | None,
        resources: Resources | None,
        emits_media: bool = False,
        hidden: bool = False,
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
                name=registered,
                kind=kind,
                fn=target,
                publishes=publishes,
                emits_media=emits_media,
                artifact_outputs=artifact_outputs,
                source_profiles=dict(source_profiles),
                resources=resources,
                preflight=preflight,
                hidden=hidden,
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
