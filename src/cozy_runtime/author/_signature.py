"""Signature analysis: the ONE source with two readers (mypy statically, describe at build).

The signature is the capability declaration. Everything the descriptor needs — the payload
and result schemas, the model binding paths, `gpu`, and the AuthorCapabilitySet — is a PURE
FUNCTION of the parameter types, so there is no second document to drift (§1.0/§1.3/§1.7).
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin

import msgspec

from cozy_runtime.author._context import Context
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._markers import Bound, Resources, _PreflightMarker
from cozy_runtime.author._model import Model, component_use
from cozy_runtime.author._services import SERVICES, Secrets, Settings
from cozy_runtime.author._walker import is_struct, strip

Role = Literal["context", "payload", "preflight", "model", "service"]
Kind = Literal["entrypoint", "job"]

_PARAMETERIZED = {"settings": Settings, "secrets": Secrets}


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    role: Role
    annotation: object
    capability: str | None = None
    arg: object | None = None
    """The schema a parameterized service carries: `Settings[T]` -> `T`."""


@dataclass(frozen=True, slots=True)
class PreflightSpec:
    """Hook identity + result schema — the descriptor's "schedule-builder identity"."""

    fn: Callable[..., object]
    identity: str
    payload_type: object
    settings_type: object | None
    result_type: object


@dataclass(frozen=True, slots=True)
class ModelBinding:
    """One model slot as the descriptor sees it. The derived canonical PATH is the binding
    point; the CLASS carries the admissibility stamps and the component-use contract."""

    path: str
    """`<callable>.models.<param>` — renaming a parameter is a breaking descriptor change."""
    param: str
    model_class: type[Model[Any]]
    stamps: Mapping[str, str]
    components: Mapping[str, tuple[str, ...]]

    @property
    def class_key(self) -> str:
        """The by-class binding key: `[bindings."SdxlModel"]` covers every parameter of it."""
        return self.model_class.__name__


@dataclass(frozen=True, slots=True)
class Surface:
    """One registered callable's derived surface. cr-003's descriptor consumes this."""

    name: str
    kind: Kind
    fn: Callable[..., object]
    is_async: bool
    params: tuple[Param, ...]
    payload_type: object
    result_type: object
    capabilities: frozenset[str]
    gpu: bool
    publishes: bool = False
    emits_media: bool = False
    """What a JOB body may write, as its own declaration. Grants mint off the declaration,
    never off the kind (§2) — a job that declares neither cannot be granted a destination."""
    resources: Resources | None = None
    demand_bound: Bound | None = None
    preflight: PreflightSpec | None = None

    @property
    def model_params(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params if p.role == "model")

    @property
    def model_bindings(self) -> tuple[ModelBinding, ...]:
        """The slots this callable declares, each with its own independent admission."""
        bindings = []
        for p in self.params:
            annotation = p.annotation
            if p.role != "model" or not isinstance(annotation, type):
                continue
            if not issubclass(annotation, Model):
                continue
            bindings.append(
                ModelBinding(
                    path=f"{self.name}.models.{p.name}",
                    param=p.name,
                    model_class=annotation,
                    stamps=annotation.__stamps__,
                    components=component_use(annotation),
                )
            )
        return tuple(bindings)


def _hints(fn: Callable[..., object]) -> Mapping[str, object]:
    try:
        return typing.get_type_hints(fn, include_extras=True)
    except Exception as exc:  # a forward reference that never resolves is a BUILD failure
        raise ConformanceError(
            f"{fn.__qualname__}: annotations do not resolve ({exc})", code="unresolved_annotation"
        ) from exc


def _classify(name: str, annotation: object, where: str) -> Param:
    base, markers = strip(annotation)
    if any(isinstance(m, _PreflightMarker) for m in markers):
        if not is_struct(base):
            raise ConformanceError(
                f"{where}: Preflight[...] must carry a msgspec.Struct of frozen facts",
                fields=[name],
            )
        return Param(name, "preflight", base)
    if base is Context:
        return Param(name, "context", base)
    origin = get_origin(base)
    for capability, service in _PARAMETERIZED.items():
        if base is service or origin is service:
            args = get_args(base)
            if not args:
                raise ConformanceError(
                    f"{where}: {service.__name__} must name its strict schema "
                    f"({service.__name__}[T]) — a bare unbounded bag is invalid",
                    fields=[name],
                )
            return Param(name, "service", base, capability, args[0])
    if isinstance(base, type):
        for cap, plain in SERVICES.items():
            if cap in _PARAMETERIZED:
                continue
            if issubclass(base, plain):
                return Param(name, "service", base, cap)
        if issubclass(base, Model):
            if markers:
                raise ConformanceError(
                    f"{where}: model parameter {name} carries {markers[0]!r} — the Annotated "
                    "override-marker plane is EMPTY at launch: a bare typed parameter is "
                    "unconditionally the whole declaration, and the class carries the "
                    "admissibility stamps (§1.1). Future markers enter through cr-018's door",
                    code="empty_marker_plane",
                    fields=[name],
                )
            return Param(name, "model", base)
        if is_struct(base):
            return Param(name, "payload", base)
    raise ConformanceError(
        f"{where}: parameter {name}: {base!r} is not an author surface type — "
        "expected Context, a msgspec.Struct payload, a Model, a service, or Preflight[T]",
        fields=[name],
    )


def analyze(
    fn: Callable[..., object],
    *,
    name: str,
    kind: Kind,
    publishes: bool = False,
    emits_media: bool = False,
    resources: Resources | None = None,
    demand_bound: Bound | None = None,
    preflight: Callable[..., object] | None = None,
) -> Surface:
    """Derive the whole surface from the signature. Every refusal here is a BUILD refusal."""
    where = f"{fn.__module__}:{fn.__qualname__}"
    signature = inspect.signature(fn)
    hints = _hints(fn)
    params: list[Param] = []
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise ConformanceError(
                f"{where}: *{parameter.name} — no **kwargs grab-bags exist on the author "
                "surface; everything injected is named and typed",
                fields=[parameter.name],
            )
        if parameter.default is not parameter.empty:
            raise ConformanceError(
                f"{where}: parameter {parameter.name} carries a default "
                f"({parameter.default!r}) — every parameter here is INJECTED by the runtime, "
                "so a default is a second source of the same fact. A model slot in "
                "particular is declared by its bare type alone: there is no slot object and "
                "no artifacts= dict (§1.1)",
                code="injected_default",
                fields=[parameter.name],
            )
        if parameter.name not in hints:
            raise ConformanceError(
                f"{where}: parameter {parameter.name} is unannotated — the signature IS "
                "the contract",
                fields=[parameter.name],
            )
        params.append(_classify(parameter.name, hints[parameter.name], where))

    payloads = [p for p in params if p.role == "payload"]
    if len(payloads) != 1:
        raise ConformanceError(
            f"{where}: exactly one typed request struct is required, found {len(payloads)}",
            code="payload_arity",
        )
    contexts = [p for p in params if p.role == "context"]
    if len(contexts) > 1:
        raise ConformanceError(f"{where}: Context is injected at most once", code="context_arity")
    facts = [p for p in params if p.role == "preflight"]
    if len(facts) > 1:
        raise ConformanceError(f"{where}: at most one Preflight[T] parameter", code="facts_arity")
    for capability in (p.capability for p in params if p.role == "service"):
        if [p.capability for p in params if p.role == "service"].count(capability) > 1:
            raise ConformanceError(
                f"{where}: service {capability!r} requested twice", code="service_arity"
            )

    result = hints.get("return")
    if result is None or not is_struct(strip(result)[0]):
        raise ConformanceError(
            f"{where}: the return annotation must be a msgspec.Struct result schema "
            f"(got {result!r}) — the result transaction validates against it exactly",
            code="result_schema",
        )

    spec: PreflightSpec | None = None
    if preflight is not None:
        spec = analyze_preflight(preflight, payloads[0].annotation)

    return Surface(
        name=name,
        kind=kind,
        fn=fn,
        is_async=inspect.iscoroutinefunction(fn),
        params=tuple(params),
        payload_type=payloads[0].annotation,
        result_type=strip(result)[0],
        capabilities=frozenset(p.capability for p in params if p.capability),
        gpu=any(p.role == "model" for p in params),
        publishes=publishes,
        emits_media=emits_media,
        resources=resources,
        demand_bound=demand_bound,
        preflight=spec,
    )


#: Attribute names that read asset BYTES. A preflight that names one is refused statically —
#: the runtime refusal in `Asset.read_bytes` is the structural backstop, not the only gate.
BYTE_ACCESS = frozenset({"read_bytes", "open", "tobytes", "read"})


def analyze_preflight(fn: Callable[..., object], payload_type: object) -> PreflightSpec:
    """The ONE pre-admission author hook: pure `(payload[, settings]) -> frozen facts`.

    Narrow by construction — no Context, Model, network, device, output, secret or service
    access, and metadata only, never bytes (§1.2).
    """
    where = f"{fn.__module__}:{fn.__qualname__}"
    hints = _hints(fn)
    positional = [p for p in inspect.signature(fn).parameters]
    if not 1 <= len(positional) <= 2:
        raise ConformanceError(
            f"preflight {where}: takes (payload) or (payload, settings), "
            f"got {len(positional)} parameters",
            code="preflight_signature",
        )
    for pname in positional:
        base, _ = strip(hints.get(pname, object))
        if base is Context or (isinstance(base, type) and issubclass(base, (Model,))):
            raise ConformanceError(
                f"preflight {where}: parameter {pname} reaches {base!r} — a preflight has "
                "no Context, Model, device, network, output or secret access",
                code="preflight_capability",
                fields=[pname],
            )
        if isinstance(base, type) and any(issubclass(base, s) for s in SERVICES.values()):
            raise ConformanceError(
                f"preflight {where}: parameter {pname} requests a service — a preflight is "
                "pure metadata, computed before hydration and before any GPU work",
                code="preflight_capability",
                fields=[pname],
            )
    declared, _ = strip(hints.get(positional[0], object))
    if declared is not payload_type:
        raise ConformanceError(
            f"preflight {where}: first parameter is {declared!r}, but the entrypoint's "
            f"payload is {payload_type!r}",
            code="preflight_payload",
        )
    code = getattr(fn, "__code__", None)
    touched = BYTE_ACCESS & set(code.co_names if code is not None else ())
    if touched:
        raise ConformanceError(
            f"preflight {where}: reads asset bytes via {sorted(touched)} — preflight inspects "
            "asset identity and trusted metadata only, so a refused request downloads nothing",
            code="preflight_bytes",
        )
    result, _ = strip(hints.get("return", None))
    if not is_struct(result):
        raise ConformanceError(
            f"preflight {where}: must return one frozen, canonically serializable "
            f"msgspec.Struct record (got {result!r})",
            code="preflight_result",
        )
    facts = typing.cast(type[msgspec.Struct], result)
    if not facts.__struct_config__.frozen:
        raise ConformanceError(
            f"preflight {where}: {getattr(result, '__name__', result)} must be frozen=True — "
            "the facts record is hashed into admission and handed to the handler unchanged",
            code="preflight_result",
        )
    settings_type = strip(hints.get(positional[1], object))[0] if len(positional) == 2 else None
    return PreflightSpec(fn, where, payload_type, settings_type, result)
