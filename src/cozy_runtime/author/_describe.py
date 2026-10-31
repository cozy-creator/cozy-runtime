"""The describe-time conformance layer — static analysis as a CONTRACT, not a suggestion.

`describe` is the second reader of the one signature (mypy is the first). A package whose
surface is invalid refuses at BUILD with the same verdict class as a schema violation, so a
mis-typed handler cannot reach a device (§1.7).

`assert_declaration_shape` walks a declaration for structural problems a type checker
cannot see. It checks SHAPE, never NAMES: a field called `batch_size`, `dtype` or
`generator` is an ordinary declaration, and a name blocklist over the author's own
vocabulary refused legitimate packages (a training job's batch size, a quantization job's
dtype, `class MyGan(Model): generator: nn.Module`) while preventing nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from cozy_runtime.author._decode import require_decoder_dependency
from cozy_runtime.author._demand import FeatureSpec, feature_specs
from cozy_runtime.author._errors import CapabilityError, ConformanceError
from cozy_runtime.author._files import asset_kinds, field_preparation
from cozy_runtime.author._markers import FileBound
from cozy_runtime.author._model_defaults import check_model_arguments
from cozy_runtime.author._prepared_use import PreparedUse
from cozy_runtime.author._services import UNGRANTABLE
from cozy_runtime.author._signature import Declaration, Surface, media_kinds
from cozy_runtime.author._walker import MISSING, is_struct, walk
from cozy_runtime.author._weights import MAX_WEIGHTS_OUTPUTS

if TYPE_CHECKING:
    from cozy_runtime.author._app import App


def assert_declaration_shape(declaration: object, *, result: bool = False) -> None:
    """Refuse a declaration whose SHAPE cannot be honoured, anywhere in its tree.

    A preparation is input policy: on a `result` declaration nothing would apply it.
    """
    for node in walk(declaration):
        owner = getattr(node.owner, "__name__", node.owner)
        bound = node.marker(FileBound)
        if isinstance(bound, FileBound):
            # cr-012: a bound NARROWS a kind. Checking it at DESCRIBE rather than at
            # hydration is the difference between a package that cannot be built and one
            # that refuses every request that reaches it — a field declaring a media type
            # its kind does not admit promises a decoder this runtime does not have.
            kinds = asset_kinds(node.annotation)
            if not kinds:
                raise ConformanceError(
                    f"{owner}.{node.path} puts FileBound on a field with no direct file; "
                    "put the bound on the concrete file field instead",
                    code="asset_bound_target",
                    fields=[node.path],
                )
            for kind in kinds:
                bound.check_kind(kind)
        preparation = field_preparation(node.annotation, node.markers)
        if result and preparation is not None:
            raise ConformanceError(
                f"{owner}.{node.path} declares {type(preparation).__name__} on a result: "
                "preparations annotate inputs, and a returned file is never prepared",
                code="preparation_target",
                fields=[node.path],
            )


def validate(declared: Declaration) -> None:
    """Every build refusal over one callable, whichever reader filled its record (§1.7).

    The package interface is assembled through it (`package_interface.assemble`), so publish,
    install and the machine's import refuse the same package for the same reason."""
    name = declared.name
    assert_declaration_shape(declared.payload_type)
    assert_declaration_shape(declared.result_type, result=True)
    feature_specs(declared.payload_type)
    if len(declared.contexts) > 1:
        raise ConformanceError(f"{name}: Context is injected at most once", code="context_arity")
    if declared.facts > 1:
        raise ConformanceError(f"{name}: at most one Preflight[T] parameter", code="facts_arity")
    services = declared.services
    for capability in sorted({c for c in services if services.count(c) > 1}):
        raise ConformanceError(
            f"{name}: service {capability!r} requested twice", code="service_arity"
        )
    if declared.memoize and {"egress", "secrets"} & set(services):
        raise ConformanceError(
            "a memoized invocable cannot depend on external egress or unrecorded secrets",
            code="invocable_memoize_effect",
        )
    # A capability nothing can grant refuses at BUILD, not at the first request: a package
    # author can do nothing about it on that clock, and a caller pays for it.
    for capability in sorted(set(services) & set(UNGRANTABLE)):
        raise ConformanceError(
            f"{name} declares the {capability!r} capability: {UNGRANTABLE[capability]}",
            code="ungrantable_capability",
        )
    _writes(declared)
    _models(declared)


def _writes(declared: Declaration) -> None:
    """What a function writes is its own declaration, and grants mint off it (§2)."""
    name = declared.name
    names = [output.name for output in declared.weights_outputs]
    publishes = declared.publishes or declared.emits_media
    if (names or publishes) and declared.memoize:
        raise ConformanceError(
            f"{name}: memoize on a function that declares weights=, publishes= or "
            "emits_media= — a writer's repeat is answered by TensorFS derivation replay and "
            "its destination, never by the memo",
            code="memoized_writer",
        )
    if publishes and "save" not in declared.services:
        raise ConformanceError(
            f"{name}: declares publishes/emits_media but its signature names no "
            "Outputs — a declaration the body cannot act on would grant a destination "
            "nothing can write",
            code="publication_declaration",
        )
    if len(names) > MAX_WEIGHTS_OUTPUTS:
        raise ConformanceError(
            f"{name}: declares {len(names)} weights outputs; the bound is {MAX_WEIGHTS_OUTPUTS}",
            code="weights_output_bound",
        )
    if len(set(names)) != len(names):
        raise ConformanceError(
            f"{name}: weights output names are not unique", code="weights_output_duplicate"
        )
    if names and not declared.contexts:
        raise ConformanceError(
            f"{name}: declares weights outputs but its signature has no Context",
            code="weights_sink_declaration",
        )
    if names and ("save" in declared.services or declared.publishes):
        raise ConformanceError(
            f"{name}: a model-producing function cannot mix TensorFS outputs with "
            "Outputs/publishes",
            code="mixed_weights_outputs",
        )


def _models(declared: Declaration) -> None:
    """The SOURCE clock over the model slots: arguments, classes and shard degrees. It cannot
    prove a component exists in an artifact not yet selected; exact binding does (§1.1)."""
    slots = [slot.parameter for slot in declared.slots]
    constructed = [slot.parameter for slot in declared.slots if not slot.weights]
    check_model_arguments(declared.model_defaults, slots)
    check_model_arguments(declared.model_components, slots, what="components")
    check_model_arguments(declared.model_prepared_use, constructed, what="prepared_use")
    if declared.model_prepared_use:
        if not declared.internal or not declared.accelerator:
            raise ConformanceError(
                "prepared_use belongs to an internal GPU program, not public request knobs",
                code="model_prepared_use",
            )
        nodes = {node.path: node for node in walk(declared.payload_type)}
        for name in declared.model_prepared_use.values():
            node = nodes.get(name)
            if (
                node is None
                or node.optional
                or node.default is not MISSING
                or not is_struct(node.annotation)
                or set(getattr(node.annotation, "__struct_fields__", ()))
                != set(PreparedUse.__struct_fields__)
            ):
                raise ConformanceError(
                    f"prepared_use field {name!r} must be the closed typed fixed-use record",
                    code="model_prepared_use",
                )
    check_model_arguments(dict.fromkeys(declared.model_on_demand), constructed, what="on_demand")
    degrees = [set(slot.spec.sequence_parallel) for slot in declared.slots]
    if any(degrees) and not set.intersection(*degrees):
        raise ConformanceError(
            f"{declared.name}: its Model slots declare @sequence_parallel degrees {degrees} "
            "with none in common; a group shards every slot, so each must share a degree",
            code="sequence_parallel_declaration_conflict",
        )
    for slot in declared.slots:
        if slot.weights:
            continue
        path, spec = f"{declared.name}.models.{slot.parameter}", slot.spec
        uses = declared.model_components.get(slot.parameter, ())
        unknown = sorted(set(uses) - {c for names in spec.components.values() for c in names})
        if unknown:
            raise ConformanceError(
                f"{path}: components names {', '.join(unknown)}, which no "
                f"@uses_components method of {spec.name} declares",
                code="model_components",
                fields=unknown,
            )
        if not spec.loads:
            raise ConformanceError(
                f"{path}: {spec.name} defines no load(loader) — a Model constructs "
                "through loader.construct(T, factory=…), the one construction primitive (§1.1)",
                code="no_load",
            )
        if spec.arguments:
            raise ConformanceError(
                f"{path}: {spec.name}() takes constructor arguments — the runtime "
                "owns WHEN construction runs, so a Model is built empty and filled by load()",
                code="model_init",
            )


def check(surface: Surface) -> Surface:
    """The imported callable's build refusals: `validate`, plus the two only an import can
    state — its release's media decoder, and its preflight hook, whose byte reads and
    demand tables are code the source reader never evaluates."""
    validate(surface)
    _media_decode_bounds(surface)
    if surface.preflight is not None:
        settings = [p for p in surface.params if p.capability == "settings"]
        declared = surface.preflight.settings_type
        if declared is not None:
            if not settings:
                raise ConformanceError(
                    f"{surface.name}: preflight takes settings, but the entrypoint declares "
                    "no Settings[T] parameter",
                    code="preflight_settings",
                )
            if settings[0].arg is not declared:
                raise ConformanceError(
                    f"{surface.name}: preflight settings {declared!r} is not the entrypoint's "
                    f"Settings[{settings[0].arg!r}]",
                    code="preflight_settings",
                )
        facts = [p for p in surface.params if p.role == "preflight"]
        if facts and facts[0].annotation is not surface.preflight.result_type:
            raise ConformanceError(
                f"{surface.name}: the handler's Preflight[{facts[0].annotation!r}] is not the "
                f"hook's result {surface.preflight.result_type!r} — the handler receives the "
                "SAME value admission hashed",
                code="preflight_result",
            )
        _two_producers(surface, feature_specs(surface.payload_type))
    return surface


def _media_decode_bounds(surface: Surface) -> None:
    """A video or audio input needs PyAV in the release's closure. A media input needs no
    declared decoded bound: Runtime decodes what the call reads, within what the machine holds."""
    if "media_decode" not in surface.capabilities:
        return
    if not media_kinds(surface.payload_type, surface.assets) & {"video", "audio"}:
        return
    try:
        require_decoder_dependency()
    except CapabilityError as exc:
        raise ConformanceError(
            f"{surface.name} names a video or audio input but its release cannot import "
            "PyAV; install cozy-runtime[media] in the package closure",
            code="media_decoder_dependency",
        ) from exc


def _two_producers(surface: Surface, specs: Sequence[FeatureSpec]) -> None:
    """Preflight facts COMPOSE with derived RequestFeatures, never restate them (§1.2)."""
    assert surface.preflight is not None
    derivable = {s.axis for s in specs} | {s.field for s in specs}
    duplicated = [n.name for n in walk(surface.preflight.result_type) if n.name in derivable]
    if duplicated:
        raise ConformanceError(
            f"preflight {surface.preflight.identity}: fact(s) {', '.join(sorted(duplicated))} "
            "duplicate a derivable RequestFeature — two producers of one fact is exactly the "
            "drift this fence exists for",
            code="two_producers",
            fields=sorted(duplicated),
        )


def describe(app: App) -> tuple[Surface, ...]:
    """The whole app's surface, or the first refusal. Freezes registration (§1.0)."""
    surfaces = []
    for registration in app.registrations():
        surfaces.append(check(registration.surface))
        registration.checked = True
    app.freeze()
    return tuple(surfaces)
