"""The `App` registry — the one object a package module exports as `app = App()`.

The decorators are sugar for explicit registration calls (`app.entrypoint(fn)`): sibling
modules export plain functions and the entry module registers them, so merely IMPORTING a
module never mutates any registry. `def` and `async def` register the same way, and a
decorated handler keeps its exact signature.

Registration is CHEAP: nothing here resolves annotations, so a forward reference to a class
defined later in the module is fine. The surface is derived on first inspection (describe),
and after describe the registry is frozen.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypedDict, TypeVar, Unpack, overload

from cozy_runtime.author._calls import _export
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._memo import Memoize, memo_policy, one_memo
from cozy_runtime.author._model import Model
from cozy_runtime.author._model_defaults import (
    model_components,
    model_defaults,
    model_on_demand,
)
from cozy_runtime.author._signature import Surface, analyze
from cozy_runtime.author._tool import Tool
from cozy_runtime.author._weights import WeightsOutput

P = ParamSpec("P")
R = TypeVar("R")


class RegistrationOptions(TypedDict, total=False):
    """The options `entrypoint` and `tool` share."""

    name: str | None
    defaults: Mapping[str, str] | None
    components: Mapping[str, Sequence[str]] | None
    memoize: bool | Memoize
    on_demand: Sequence[str]
    accelerator: bool | None
    weights: tuple[WeightsOutput, ...]
    private: bool
    preflight: Callable[..., object] | None


class RegistrationError(ConformanceError):
    """A duplicate or otherwise invalid registration. Refuses at import, not at serve."""

    default_code = "registration"


@dataclass(slots=True)
class Registration:
    """One registered callable plus the doors it opened."""

    name: str
    fn: Callable[..., object]
    invocable: bool = False
    memoize: Memoize | None = None
    model_defaults: Mapping[str, str] = field(default_factory=dict)
    model_components: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    model_on_demand: tuple[str, ...] = ()
    #: A model argument's class where the Runtime names it (`cozy_runtime.derive.recipes`).
    model_classes: Mapping[str, type[Model[Any]]] = field(default_factory=dict)
    weights_outputs: tuple[WeightsOutput, ...] = ()
    preflight: Callable[..., object] | None = None
    child_task: bool = False
    """`private=True`: called by this package's own code only; never a root request."""
    accelerator: bool = True
    """False: CPU only. True (the default): the machine's GPU when it has one."""
    tool: Tool | None = None
    """`app.tool`'s own options: this function is a box."""
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
                weights_outputs=self.weights_outputs,
                preflight=self.preflight,
                child_task=self.child_task,
                accelerator=self.accelerator,
                tool=self.tool,
                invocable=self.invocable,
                memoize=self.memoize,
                model_defaults=self.model_defaults,
                model_components=self.model_components,
                model_on_demand=self.model_on_demand,
                model_classes=self.model_classes,
            )
        return self._surface


@dataclass(slots=True)
class App:
    """The one registry a package module exports as `app = App()`."""

    _registry: dict[str, Registration] = field(default_factory=dict)
    _frozen: bool = False

    @overload
    def entrypoint(
        self, fn: Callable[P, R], /, **options: Unpack[RegistrationOptions]
    ) -> Callable[P, R]: ...

    @overload
    def entrypoint(
        self, /, **options: Unpack[RegistrationOptions]
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def entrypoint(
        self, fn: Callable[P, R] | None = None, /, **options: Unpack[RegistrationOptions]
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Register a function, bare (`app.entrypoint(fn)`) or as a decorator.

        - `name`: its public name; the function's own by default.
        - `defaults={"model": "org/name@release"}`: each model argument's default release.
        - `components={"model": ("transformer", "vae")}`: optional; the components of a model
          argument this function fetches. Without it: those its class's `@uses_components`
          hints name when every public method has one, else all of them.
        - `memoize=True`: its results are deterministic, so a repeat is answered from the
          result cache (never for a function that writes weights);
          `Memoize(requires=("seed",))` memoizes only requests that name each listed option.
        - `on_demand=("upsampler",)`: model arguments a request may leave unused; each is
          built, and its weights fetched, when the function first reads it.
        - `accelerator=False`: CPU only; otherwise it gets the machine's GPU when there is one.
        - `weights=`: the TensorFS model outputs it writes.
        - `private=True`: only this package's own calls run it, never a root request.
        - `preflight=`: the function that resolves a request's `Preflight[T]` before it runs.
        """
        return self._register(fn, **options)

    @overload
    def tool(
        self,
        fn: Callable[P, R],
        /,
        *,
        label: str = "",
        named: str = "",
        reserved: Sequence[str] = (),
        ui: str = "",
        **options: Unpack[RegistrationOptions],
    ) -> Callable[P, R]: ...

    @overload
    def tool(
        self,
        /,
        *,
        label: str = "",
        named: str = "",
        reserved: Sequence[str] = (),
        ui: str = "",
        **options: Unpack[RegistrationOptions],
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def tool(
        self,
        fn: Callable[P, R] | None = None,
        /,
        *,
        label: str = "",
        named: str = "",
        reserved: Sequence[str] = (),
        ui: str = "",
        **options: Unpack[RegistrationOptions],
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """A public function that is a box: its request struct is the box's fields (tools.md).
        It takes `entrypoint`'s options, and is called like one. `label`: its name in the tool
        list. `named`: a new box's name, numbered (`shot` makes `shot_1`). `reserved`: the model's
        own tags in its text, never mentions. `ui`: a component in this package that draws it.

        Placed on a model's own function, it is the shortcut: its arguments are already the box."""
        box = Tool(label=label, named=named, reserved=reserved, ui=ui)
        return self._register(fn, tool=box, **options)

    def _register(
        self,
        fn: Callable[P, R] | None,
        /,
        *,
        name: str | None = None,
        defaults: Mapping[str, str] | None = None,
        components: Mapping[str, Sequence[str]] | None = None,
        memoize: bool | Memoize = False,
        on_demand: Sequence[str] = (),
        accelerator: bool | None = None,
        weights: tuple[WeightsOutput, ...] = (),
        private: bool = False,
        preflight: Callable[..., object] | None = None,
        tool: Tool | None = None,
        model_classes: Mapping[str, type[Model[Any]]] | None = None,
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        policy = memo_policy(memoize)
        for option, value in (("accelerator", accelerator), ("private", private)):
            if value is not None and type(value) is not bool:
                raise RegistrationError(f"{option} is True or False")
        declared_defaults = model_defaults(defaults)
        declared_components = model_components(components)
        declared_on_demand = model_on_demand(on_demand)

        def decorate(target: Callable[P, R]) -> Callable[P, R]:
            if self._frozen:
                raise RegistrationError(
                    f"{target.__name__}: the registry is frozen — once served, "
                    "registration is immutable"
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
            self._registry[registered] = Registration(
                name=registered,
                fn=export.implementation if export is not None else target,
                invocable=export is not None,
                memoize=one_memo(export.memoize, policy) if export is not None else policy,
                model_defaults=declared_defaults or (export.defaults if export is not None else {}),
                model_components=declared_components
                or (export.components if export is not None else {}),
                model_on_demand=declared_on_demand,
                model_classes=dict(model_classes or {}),
                weights_outputs=weights,
                preflight=preflight,
                child_task=private,
                accelerator=accelerator is not False,
                tool=tool,
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
        """Once served, registration is immutable (§1.0)."""
        self._frozen = True
