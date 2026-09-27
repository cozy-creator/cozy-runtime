"""Signature analysis: the ONE source with two readers (mypy statically, describe at build).

The signature is the capability declaration. Everything the package interface needs — the payload
and result schemas, model binding paths, and AuthorCapabilitySet — is a PURE FUNCTION of the
parameter types, so there is no second document to drift (§1.0/§1.3/§1.7).
"""

from __future__ import annotations

import enum
import inspect
import math
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, get_args, get_origin

import msgspec

from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._assets import (
    Asset,
    Assets,
    AudioAsset,
    Fidelity,
    FileAsset,
    ImageAsset,
    Tree,
    VideoAsset,
)
from cozy_runtime.author._context import Context
from cozy_runtime.author._decode import DECODED_ASSET_TYPES, DEFAULT_DECODE_LIMITS
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._markers import (
    AssetBound,
    AssetLimits,
    Bound,
    ImagePreparation,
    _PreflightMarker,
)
from cozy_runtime.author._model import Model, component_use
from cozy_runtime.author._model_defaults import DefaultLadder, check_model_arguments
from cozy_runtime.author._services import SERVICES, Secrets, Settings
from cozy_runtime.author._walker import is_struct, strip
from cozy_runtime.author._weights import WeightsOutput

Role = Literal["context", "payload", "wire", "assets", "preflight", "model", "service"]
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
    """Hook identity + result schema — the package interface's "schedule-builder identity"."""

    fn: Callable[..., object]
    identity: str
    payload_type: object
    settings_type: object | None
    result_type: object
    assets_parameter: str | None = None
    argument_order: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AssetsBinding:
    parameter: str
    annotation: object
    kinds: tuple[tuple[type[Asset], AssetBound | None], ...]
    decoded: bool = False
    counts: tuple[tuple[str, int], ...] = ()
    max_length: int | None = None
    image_preparation: ImagePreparation | None = None

    @property
    def wire_type(self) -> object:
        _, declared = strip(self.annotation)
        markers = tuple(
            marker for marker in declared if not isinstance(marker, (AssetLimits, ImagePreparation))
        )
        if self.max_length is not None and not any(
            isinstance(marker, msgspec.Meta) and marker.max_length is not None for marker in markers
        ):
            markers += (msgspec.Meta(max_length=self.max_length),)
        return (
            typing.Annotated[(list[AssetInputValue], *markers)]
            if markers
            else list[AssetInputValue]
        )


class AssetInputValue(msgspec.Struct, forbid_unknown_fields=True):
    """Ordinary payload representation of one explicitly forwarded occurrence."""

    asset: FileAsset
    label: str = ""
    fidelity: Fidelity = "auto"


def assets_binding(name: str, annotation: object) -> AssetsBinding:
    base, metadata = strip(annotation)
    args = get_args(base)
    item = args[0] if args else ImageAsset | VideoAsset | AudioAsset | FileAsset
    inner, _ = strip(item)
    options = get_args(inner) if get_origin(inner) in (typing.Union, types.UnionType) else (item,)
    kinds: dict[str, tuple[type[Asset], AssetBound | None]] = {}
    decoded: bool | None = None
    for option in options:
        cls, markers = strip(option)
        for marker in markers:
            if isinstance(marker, (AssetLimits, ImagePreparation)):
                raise ConformanceError(
                    f"{type(marker).__name__} annotates the Assets collection",
                    code="asset_limits" if isinstance(marker, AssetLimits) else "image_preparation",
                )
        value_type = cls in DECODED_ASSET_TYPES
        if decoded is not None and value_type != decoded:
            raise ConformanceError(
                "Assets cannot mix decoded values and raw handles in one view",
                code="assets_element",
            )
        decoded = value_type
        if value_type:
            cls = DECODED_ASSET_TYPES[typing.cast(type, cls)]
        if cls not in (ImageAsset, VideoAsset, AudioAsset, FileAsset):
            raise ConformanceError(
                "Assets elements must be media values or typed assets", code="assets_element"
            )
        cls = typing.cast(type[Asset], cls)
        if cls.kind in kinds:
            raise ConformanceError("Assets repeats a media kind", code="assets_element")
        bound = next((marker for marker in markers if isinstance(marker, AssetBound)), None)
        if value_type:
            if bound is None:
                bound = AssetBound(max_decoded_bytes=DEFAULT_DECODE_LIMITS.max_decoded_bytes)
            elif bound.max_decoded_bytes is None:
                bound = replace(bound, max_decoded_bytes=DEFAULT_DECODE_LIMITS.max_decoded_bytes)
        if bound is not None:
            bound.check_kind(cls.kind)
        kinds[cls.kind] = cls, bound
    limits = [marker for marker in metadata if isinstance(marker, AssetLimits)]
    if limits and any(marker != limits[0] for marker in limits[1:]):
        raise ConformanceError("conflicting AssetLimits markers", code="asset_limits")
    counts = limits[0].counts if limits else {}
    if set(counts) - set(kinds):
        raise ConformanceError("AssetLimits names an undeclared media kind", code="asset_limits")
    totals = {
        marker.max_length
        for marker in metadata
        if isinstance(marker, msgspec.Meta) and marker.max_length is not None
    }
    if limits and limits[0].total is not None:
        totals.add(limits[0].total)
    if len(totals) > 1:
        raise ConformanceError("conflicting Assets total/max_length", code="asset_limits")
    max_length = next(iter(totals), None)
    if max_length is None and counts and set(counts) == set(kinds):
        max_length = sum(counts.values())
    if max_length is not None and any(
        isinstance(marker, msgspec.Meta)
        and marker.min_length is not None
        and marker.min_length > max_length
        for marker in metadata
    ):
        raise ConformanceError("Assets minimum exceeds its total limit", code="asset_limits")
    preparations = [marker for marker in metadata if isinstance(marker, ImagePreparation)]
    if preparations and (not decoded or "image" not in kinds):
        raise ConformanceError(
            "ImagePreparation requires decoded image Assets", code="image_preparation"
        )
    if preparations and any(marker != preparations[0] for marker in preparations[1:]):
        raise ConformanceError("conflicting ImagePreparation markers", code="image_preparation")
    return AssetsBinding(
        name,
        annotation,
        tuple(kinds[key] for key in sorted(kinds)),
        bool(decoded),
        tuple(sorted(counts.items())),
        max_length,
        preparations[0] if preparations else None,
    )


@dataclass(frozen=True, slots=True)
class ModelBinding:
    """One model slot as the package interface sees it. The derived canonical PATH is the binding
    point; the CLASS carries `encoded_leaves` and the component-use contract."""

    path: str
    """`<callable>.models.<param>` — renaming a parameter is a breaking package interface change."""
    param: str
    model_class: type[Model[Any]]
    components: Mapping[str, tuple[str, ...]]
    sequence_parallel: tuple[int, ...] = ()
    """`@sequence_parallel(degrees=...)` on the class: the degrees it shards at (cr-068)."""

    @property
    def class_key(self) -> str:
        """The class name, for messages and for the publish memo's per-class seed. It is
        NOT a binding key: `package.toml` binds slot paths only (model-code-fit §1)."""
        return self.model_class.__name__

    @property
    def encoded_leaves(self) -> str:
        return str(getattr(self.model_class, "__encoded_leaves__", "refuse"))

    @property
    def fusion(self) -> str:
        return str(getattr(self.model_class, "__fusion__", "refuse"))


@dataclass(frozen=True, slots=True)
class Surface:
    """One registered callable's derived surface. cr-003's package interface consumes this."""

    name: str
    kind: Kind
    fn: Callable[..., object]
    is_async: bool
    params: tuple[Param, ...]
    payload_type: object
    result_type: object
    capabilities: frozenset[str]
    publishes: bool = False
    emits_media: bool = False
    weights_outputs: tuple[WeightsOutput, ...] = ()
    """What a JOB body may write, as its own declaration. Grants mint off the declaration,
    never off the kind (§2) — a job that declares neither cannot be granted a destination."""
    demand_bound: Bound | None = None
    preflight: PreflightSpec | None = None
    internal: bool = False
    accelerator: bool | None = None
    hidden: bool = False
    """DECLARED but NOT PUBLISHED (#572d). Hidden callables stay in the local registry but
    are absent from the package interface, so no binding is staged and no request reaches them."""
    invocable: bool = False
    memoize: bool = False
    model_defaults: Mapping[str, DefaultLadder] = field(default_factory=dict)

    @property
    def assets_binding(self) -> AssetsBinding | None:
        return next(
            (assets_binding(p.name, p.annotation) for p in self.params if p.role == "assets"), None
        )

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
                    components=component_use(annotation),
                    sequence_parallel=tuple(annotation.__sequence_parallel__),
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
    if base is Assets or origin is Assets:
        binding = assets_binding(name, annotation)
        return Param(name, "assets", annotation, "media_decode" if binding.decoded else None)
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
                    "unconditionally the whole declaration (§1.1). Future markers enter "
                    "through cr-018's door",
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
    weights_outputs: tuple[WeightsOutput, ...] = (),
    demand_bound: Bound | None = None,
    preflight: Callable[..., object] | None = None,
    hidden: bool = False,
    internal: bool = False,
    accelerator: bool | None = None,
    invocable: bool = False,
    memoize: bool = False,
    model_defaults: Mapping[str, DefaultLadder] | None = None,
) -> Surface:
    """Derive the whole surface from the signature. Every refusal here is a BUILD refusal."""
    where = f"{fn.__module__}:{fn.__qualname__}"
    signature = inspect.signature(fn)
    hints = _hints(fn)
    params: list[Param] = []
    wire_fields: list[Any] = []
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise ConformanceError(
                f"{where}: *{parameter.name} — no **kwargs grab-bags exist on the author "
                "surface; everything injected is named and typed",
                fields=[parameter.name],
            )
        if parameter.default is not parameter.empty and not invocable:
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
        annotation = hints[parameter.name]
        base, markers = strip(annotation)
        if not (base is Assets or get_origin(base) is Assets):
            for marker in markers:
                if isinstance(marker, (AssetLimits, ImagePreparation)):
                    raise ConformanceError(
                        f"{type(marker).__name__} annotates an Assets input",
                        code="asset_limits"
                        if isinstance(marker, AssetLimits)
                        else "image_preparation",
                    )
        if base is Assets or get_origin(base) is Assets:
            if parameter.default is not parameter.empty:
                raise ConformanceError("Assets inputs cannot have defaults", code="assets_default")
            if invocable and parameter.kind is not parameter.KEYWORD_ONLY:
                raise ConformanceError(
                    "invocable Assets must be keyword-only", code="invocable_signature"
                )
            binding = assets_binding(parameter.name, annotation)
            params.append(
                Param(
                    parameter.name,
                    "assets",
                    annotation,
                    "media_decode" if binding.decoded else None,
                )
            )
            if invocable:
                wire_fields.append((parameter.name, binding.wire_type))
            continue
        if invocable:
            base, _ = strip(annotation)
            injected = (
                base is Context
                or (
                    isinstance(base, type)
                    and (
                        issubclass(base, Model)
                        or any(issubclass(base, service) for service in SERVICES.values())
                    )
                )
                or get_origin(base) in (Settings, Secrets)
            )
            if not injected:
                if parameter.kind is not parameter.KEYWORD_ONLY:
                    raise ConformanceError(
                        "invocable arguments must be keyword-only", code="invocable_signature"
                    )
                _wire_type(annotation)
                params.append(Param(parameter.name, "wire", annotation))
                if parameter.default is parameter.empty:
                    wire_fields.append((parameter.name, annotation))
                else:
                    _wire_default(parameter.default)
                    try:
                        msgspec.convert(parameter.default, type=annotation, strict=True)
                    except (TypeError, ValueError, msgspec.ValidationError) as exc:
                        raise ConformanceError(
                            "invocable default differs from its type", code="invocable_default"
                        ) from exc
                    wire_fields.append((parameter.name, annotation, parameter.default))
                continue
            if parameter.default is not parameter.empty:
                raise ConformanceError(
                    "injected invocable parameters cannot have defaults", code="injected_default"
                )
            if isinstance(base, type) and issubclass(base, Model):
                # Ordinary top-level invocation may use the owner's frozen model
                # binding. A managed caller may supply an exact produced artifact
                # override, which the owner also binds through that same model lane.
                wire_fields.append((parameter.name, ModelArtifact | None, None))
        params.append(_classify(parameter.name, annotation, where))

    payloads = [p for p in params if p.role == "payload"]
    assets_params = [p for p in params if p.role == "assets"]
    if len(assets_params) > 1:
        raise ConformanceError(
            "one Assets input slot is supported per callable", code="assets_arity"
        )
    if invocable:
        declared = list(signature.parameters.values())
        if (
            not inspect.iscoroutinefunction(fn)
            or not declared
            or params[0].role != "context"
            or declared[0].kind
            not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ):
            raise ConformanceError(
                "an invocable must be async with Context first", code="invocable_signature"
            )
        payload_type: object = msgspec.defstruct(
            f"{fn.__name__}Request",
            wire_fields,
            module=fn.__module__,
            kw_only=True,
            frozen=True,
            forbid_unknown_fields=True,
        )
    elif len(payloads) != 1:
        raise ConformanceError(
            f"{where}: exactly one typed request struct is required, found {len(payloads)}",
            code="payload_arity",
        )
    else:
        payload_type = payloads[0].annotation
        if assets_params:
            binding = assets_binding(assets_params[0].name, assets_params[0].annotation)
            payload_type = assets_payload(
                typing.cast(type[msgspec.Struct], payload_type), binding, fn.__name__, fn.__module__
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
    if invocable:
        _wire_type(result)

    spec: PreflightSpec | None = None
    if preflight is not None:
        spec = analyze_preflight(
            preflight, payloads[0].annotation, assets_params[0] if assets_params else None
        )

    check_model_arguments(model_defaults or {}, [p.name for p in params if p.role == "model"])
    return Surface(
        name=name,
        kind=kind,
        fn=fn,
        is_async=inspect.iscoroutinefunction(fn),
        params=tuple(params),
        payload_type=payload_type,
        result_type=strip(result)[0],
        capabilities=frozenset(p.capability for p in params if p.capability)
        | ({"weights"} if weights_outputs else set()),
        publishes=publishes,
        emits_media=emits_media,
        weights_outputs=weights_outputs,
        demand_bound=demand_bound,
        preflight=spec,
        hidden=hidden,
        internal=internal,
        accelerator=accelerator,
        invocable=invocable,
        memoize=memoize,
        model_defaults=model_defaults or {},
    )


def assets_payload(
    payload_type: type[msgspec.Struct], binding: AssetsBinding, fn_name: str, module: str
) -> type[msgspec.Struct]:
    """The request struct with the Assets occurrence field appended — ONE construction, read
    by the import path and the static reader alike."""
    if any(
        f.encode_name == binding.parameter or f.name == binding.parameter
        for f in msgspec.structs.fields(payload_type)
    ):
        raise ConformanceError(
            "Assets parameter collides with a payload field", code="assets_field_collision"
        )
    return msgspec.defstruct(
        f"{fn_name}Request",
        [(binding.parameter, typing.cast(Any, binding.wire_type))],
        bases=(payload_type,),
        module=module,
        kw_only=True,
    )


def _wire_default(value: object) -> None:
    """Only immutable canonical values can change omission into an exact intent."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    raise ConformanceError(
        "invocable defaults must be immutable canonical scalars", code="invocable_default"
    )


def _wire_type(annotation: object, seen: frozenset[type] = frozenset()) -> None:
    """Refuse interpreter-local values before an interface is generated."""
    base, _ = strip(annotation)
    if base in (str, int, float, bool, type(None)):
        return
    origin = get_origin(base)
    if origin is Literal:
        for value in get_args(base):
            _wire_default(value)
        return
    if origin is tuple:
        args = get_args(base)
        if len(args) == 2 and args[1] is Ellipsis:
            _wire_type(args[0], seen)
            return
    if origin in (typing.Union, types.UnionType, list):
        args = get_args(base)
        if not args:
            raise ConformanceError(
                "invocable collections must have element types", code="invocable_wire_type"
            )
        for arg in args:
            _wire_type(arg, seen)
        return
    if isinstance(base, type):
        if issubclass(base, (Asset, Tree)):
            return
        if issubclass(base, enum.Enum):
            for member in base:
                _wire_default(member.value)
            return
        if is_struct(base) and base not in seen:
            for field in msgspec.structs.fields(typing.cast(type[msgspec.Struct], base)):
                if field.default_factory is not msgspec.NODEFAULT:
                    raise ConformanceError(
                        "invocable types cannot have default factories", code="invocable_default"
                    )
                _wire_type(field.type, seen | {base})
            return
    raise ConformanceError(
        "invocable type is recursive or has no portable wire representation",
        code="invocable_wire_type",
    )


#: Attribute names that read asset BYTES. A preflight that names one is refused statically —
#: the runtime refusal in `Asset.read_bytes` is the structural backstop, not the only gate.
BYTE_ACCESS = frozenset({"read_bytes", "open", "tobytes", "read"})


def analyze_preflight(
    fn: Callable[..., object], payload_type: object, assets: Param | None = None
) -> PreflightSpec:
    """The ONE pre-admission author hook: pure `(payload[, settings]) -> frozen facts`.

    Narrow by construction — no Context, Model, network, device, output, secret or service
    access, and metadata only, never bytes (§1.2).
    """
    where = f"{fn.__module__}:{fn.__qualname__}"
    hints = _hints(fn)
    positional = [p for p in inspect.signature(fn).parameters]
    if not 1 <= len(positional) <= (3 if assets else 2):
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
    settings_type = None
    assets_parameter = None
    argument_order = []
    for name in positional[1:]:
        annotation = hints.get(name, object)
        base, _ = strip(annotation)
        if base is Assets or get_origin(base) is Assets:
            if assets is None or annotation != assets.annotation or assets_parameter is not None:
                raise ConformanceError(
                    "preflight Assets differs from its input slot", code="preflight_assets"
                )
            assets_parameter = assets.name
            argument_order.append("assets")
        elif settings_type is None:
            settings_type = base
            argument_order.append("settings")
        else:
            raise ConformanceError("preflight settings repeated", code="preflight_signature")
    return PreflightSpec(
        fn, where, payload_type, settings_type, result, assets_parameter, tuple(argument_order)
    )
