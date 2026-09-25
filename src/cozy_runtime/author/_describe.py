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

import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING

from cozy_runtime.author._assets import asset_kinds
from cozy_runtime.author._decode import DEFAULT_DECODE_LIMITS, require_decoder_dependency
from cozy_runtime.author._demand import FeatureSpec, feature_specs
from cozy_runtime.author._errors import CapabilityError, ConformanceError
from cozy_runtime.author._markers import AssetBound
from cozy_runtime.author._model import EXCLUDED_KEYWORDS, Model
from cozy_runtime.author._services import JOB_ONLY, SERVING_ONLY, UNGRANTABLE
from cozy_runtime.author._signature import ModelBinding, Surface
from cozy_runtime.author._walker import walk
from cozy_runtime.author._weights import MAX_WEIGHTS_OUTPUTS

if TYPE_CHECKING:
    from cozy_runtime.author._app import App


def assert_declaration_shape(declaration: object) -> None:
    """Refuse a declaration whose SHAPE cannot be honoured, anywhere in its tree."""
    for node in walk(declaration):
        owner = getattr(node.owner, "__name__", node.owner)
        bound = node.marker(AssetBound)
        if isinstance(bound, AssetBound):
            # cr-012: a bound NARROWS a kind. Checking it at DESCRIBE rather than at
            # hydration is the difference between a package that cannot be built and one
            # that refuses every request that reaches it — a field declaring a media type
            # its kind does not admit promises a decoder this runtime does not have.
            kinds = asset_kinds(node.annotation)
            if not kinds:
                raise ConformanceError(
                    f"{owner}.{node.path} puts AssetBound on a field with no direct asset; "
                    "put the bound on the concrete asset field instead",
                    code="asset_bound_target",
                    fields=[node.path],
                )
            for kind in kinds:
                bound.check_kind(kind)


def _grantable(surface: Surface) -> None:
    """A capability nothing can grant refuses at BUILD, not at the first request.

    `net: Egress` passed describe, passed deploy, and then raised
    `egress_broker_unavailable` the first time a real request reached the handler — the one
    clock on which a package author can do nothing about it and a caller pays for it. A
    declaration the platform cannot honour is a conformance fact about the package, so it
    belongs in the same verdict class as a schema violation (§1.7), and the refusal names
    what has to land rather than only that something is missing.
    """
    for capability in sorted(surface.capabilities & set(UNGRANTABLE)):
        raise ConformanceError(
            f"{surface.name} declares the {capability!r} capability: {UNGRANTABLE[capability]}",
            code="ungrantable_capability",
        )


def _separation(surface: Surface) -> None:
    """The STRUCTURAL separation between a job and a serving entrypoint (§2, cr-009).

    A job runs to completion: no streams, no slots, no warm state, and no caller sitting on
    the other end of a request. The separation is not a review rule — the serving surface is
    UNSPELLABLE from a job and the job surface is unspellable from an entrypoint, so a
    mis-declared callable refuses at BUILD with the same verdict class as a schema violation.
    """
    job = surface.kind == "job"
    forbidden = SERVING_ONLY if job else JOB_ONLY
    reached = sorted(surface.capabilities & forbidden)
    if reached:
        raise ConformanceError(
            f"{surface.name}: a {surface.kind} requests {', '.join(reached)} — "
            + (
                "that service writes the caller-visible adjustments envelope, and a "
                "run-to-completion job has no caller to confess to (§2)"
                if job
                else "Scratch, Checkpoints and Budget are JOB services: durable working "
                "state and a spend envelope belong to a bounded attempt, never to a "
                "request handler on a warm serving loop (§2)"
            ),
            code="job_serving_separation",
        )
    if job and surface.preflight is not None:
        raise ConformanceError(
            f"{surface.name}: a job declares a preflight hook — preflight is the SERVING "
            "pre-admission hook (it builds a schedule for a request that has not been "
            "admitted); a job is submitted, accepted once, and runs to completion (§2)",
            code="job_serving_separation",
        )
    if job and surface.demand_bound is not None:
        raise ConformanceError(
            f"{surface.name}: a job declares demand=Bound(…) — a demand envelope is a SERVING "
            "admission fact used to route requests to warm capacity; job placement is derived "
            "from its selected model lane and measured execution facts (§1.2/§2)",
            code="job_serving_separation",
        )
    if not job and (surface.publishes or surface.emits_media):
        raise ConformanceError(
            f"{surface.name}: publishes=/emits_media= on an entrypoint — a publication "
            "declaration is a JOB fact; an entrypoint's outputs ride its result (§2)",
            code="job_serving_separation",
        )
    if (surface.publishes or surface.emits_media) and "save" not in surface.capabilities:
        raise ConformanceError(
            f"{surface.name}: declares publishes/emits_media but its signature names no "
            "Outputs — grants mint off the DECLARATION, so a declaration the body cannot "
            "act on would grant a destination nothing can write (§2)",
            code="publication_declaration",
        )
    if "weights_read" in surface.capabilities and not surface.model_bindings:
        raise ConformanceError(
            f"{surface.name}: WeightsReader requires exact typed model inputs",
            code="weights_source_absent",
        )
    weights_names = [output.name for output in surface.weights_outputs]
    if len(weights_names) > MAX_WEIGHTS_OUTPUTS:
        raise ConformanceError(
            f"{surface.name}: declares {len(weights_names)} weights outputs; "
            f"the bound is {MAX_WEIGHTS_OUTPUTS}",
            code="weights_output_bound",
        )
    if len(set(weights_names)) != len(weights_names):
        raise ConformanceError(
            f"{surface.name}: weights output names are not unique",
            code="weights_output_duplicate",
        )
    if surface.weights_outputs and not any(param.role == "context" for param in surface.params):
        raise ConformanceError(
            f"{surface.name}: declares weights outputs but its signature has no Context",
            code="weights_sink_declaration",
        )
    if surface.weights_outputs and ("save" in surface.capabilities or surface.publishes):
        raise ConformanceError(
            f"{surface.name}: a model-producing job cannot mix TensorFS outputs with the retired "
            "Outputs/publishes path",
            code="mixed_weights_outputs",
        )


def check(surface: Surface) -> Surface:
    """Every build-time refusal that needs the WHOLE surface, not just one signature."""
    assert_declaration_shape(surface.payload_type)
    assert_declaration_shape(surface.result_type)
    if surface.memoize and surface.capabilities & {"egress", "secrets"}:
        raise ConformanceError(
            "a memoized invocable cannot depend on external egress or unrecorded secrets",
            code="invocable_memoize_effect",
        )
    specs = feature_specs(surface.payload_type)
    _grantable(surface)
    _separation(surface)
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
        _two_producers(surface, specs)
    for binding in surface.model_bindings:
        _check_model(binding)
    return surface


def _media_decode_bounds(surface: Surface) -> None:
    """A decoder capability is useful only over explicitly expansion-bounded media.

    Compressed bytes are already deployment- and field-bounded. Decode has a different
    amplification factor, so requesting the service without binding every decodable media
    field would make the most dangerous input the one field a package forgot to annotate.
    Refuse that package at describe instead of offering a request-time unbounded overload.
    """
    if "media_decode" not in surface.capabilities:
        return
    reached = 0
    fields = (
        [
            (node.path, asset_kinds(node.annotation), node.marker(AssetBound))
            for node in walk(surface.payload_type)
        ]
        if any(p.role == "service" and p.capability == "media_decode" for p in surface.params)
        else []
    )
    if binding := surface.assets_binding:
        fields.extend(
            (binding.parameter, frozenset({cls.kind}), bound) for cls, bound in binding.kinds
        )
    for path, kinds, bound in fields:
        if "tree" in kinds:
            # Tree.member supplies an explicit per-member decoded-byte bound.
            reached += 1
        kinds = kinds & {"image", "video", "audio"}
        if not kinds:
            continue
        reached += 1
        if not isinstance(bound, AssetBound) or bound.max_decoded_bytes is None:
            raise ConformanceError(
                f"{surface.name}.{path} is decoded by MediaDecoder but declares no "
                "AssetBound(max_decoded_bytes=...): compressed and decoded size are separate "
                "limits, and the decoder has no unbounded overload",
                code="asset_decode_unbounded",
                fields=[path],
            )
        if bound.max_decoded_bytes > DEFAULT_DECODE_LIMITS.max_decoded_bytes:
            raise ConformanceError(
                f"{surface.name}.{path} declares {bound.max_decoded_bytes} decoded bytes, "
                f"above Runtime's {DEFAULT_DECODE_LIMITS.max_decoded_bytes}-byte hard ceiling",
                code="asset_decode_limit",
                fields=[path],
            )
    if not reached and surface.kind != "job":
        # Jobs can decode bounded outputs produced by their managed children. Those
        # capabilities are granted on child completion, not declared as parent inputs.
        raise ConformanceError(
            f"{surface.name} requests MediaDecoder but its payload contains no image, video "
            "or audio asset field",
            code="unused_capability",
        )
    try:
        require_decoder_dependency()
    except CapabilityError as exc:
        raise ConformanceError(
            f"{surface.name} declares MediaDecoder but its release cannot import PyAV; "
            "install cozy-runtime[media] in the package closure",
            code="media_decoder_dependency",
        ) from exc


def _check_model(binding: ModelBinding) -> None:
    """The SOURCE clock over one model slot: syntax, placement and excluded surfaces only.

    It cannot prove a component name exists in an artifact that has not been selected —
    that existence proof runs at exact binding against the censused construction (§1.1).
    """
    cls = binding.model_class
    if cls.load is Model.load:
        raise ConformanceError(
            f"{binding.path}: {cls.__name__} defines no load(loader) — a Model constructs "
            "through loader.construct(T, factory=…), the one construction primitive (§1.1)",
            code="no_load",
        )
    try:
        inspect.signature(cls).bind()
    except TypeError:
        raise ConformanceError(
            f"{binding.path}: {cls.__name__}() takes constructor arguments — the runtime "
            "owns WHEN construction runs, so a Model is built empty and filled by load()",
            code="model_init",
        ) from None
    for method, names in binding.components.items():
        del names
        member = inspect.getattr_static(cls, method)
        for name in inspect.signature(member).parameters:
            if (why := EXCLUDED_KEYWORDS.get(name)) is not None:
                raise ConformanceError(
                    f"{cls.__name__}.{method}({name}=): {why}",
                    code="excluded_keyword",
                    fields=[name],
                )


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
