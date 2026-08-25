"""The describe-time conformance layer — static analysis as a CONTRACT, not a suggestion.

`describe` is the second reader of the one signature (mypy is the first). An endpoint whose
surface is invalid refuses at BUILD with the same verdict class as a schema violation, so a
mis-typed handler cannot reach a device (§1.7).

This module also owns the ONE SDK-owned deleted-field list. An endpoint asserts what IT
declares via `assert_declaration_shape` and inherits the deletion walk — contracts endpoints
rely on are exercised by conformance, never by convention.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from cozy_runtime.author._demand import FeatureSpec, feature_specs
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._model import EXCLUDED_KEYWORDS, Model
from cozy_runtime.author._signature import ModelBinding, Surface
from cozy_runtime.author._walker import walk

if TYPE_CHECKING:
    from cozy_runtime.author._app import App

#: Names deleted from the author surface, with WHY. One list, one owner: an endpoint that
#: declares any of them refuses, and `scripts/fence.py` proves the SDK never grows one back.
DELETED_SURFACES: Mapping[str, str] = {
    "generator": "the seeded generator is view.generator — there is no ctx.generator (§1.3)",
    "recommended": "'runs better on newer silicon' is a measured ranking fact (§1.2)",
    "vcpus": "vcpus are MEASURED per (lane, plan) fleet-side, never authored (§1.2)",
    "vram": "no VRAM guess is authored in endpoint code (§1.2)",
    "batch_size": "one request produces one output; batch dimensions stay deleted (§1.2)",
    "tuned": "no parallel family-defaults struct or model.tuned exists (§1.2)",
    "callback_on_step_end": "mid-run tensor mutation is an unconfessed quality change (§1.3)",
    "compile": "no compile surface is nameable from author code (§1.3)",
    "retryable": "author code reports neutral FACTS; retryability is a coordinator "
    "projection over them, never an observation the author writes (§2)",
    "retry_after": "no author-raised retry command exists at any tier (§2)",
    "fatal": "FATAL is a coordinator projection, not a worker or author observation (§2)",
}


def assert_declaration_shape(declaration: object) -> None:
    """Refuse a declaration that names a deleted or excluded surface, anywhere in its tree.

    The excluded-keyword list is the same one method wrappers are held to (§1.1): placement,
    dtype, filesystem paths, offload, pinning and eviction are not nameable from author code,
    and a REQUEST FIELD is exactly as much of the author surface as a method parameter is.
    """
    for node in walk(declaration):
        owner = getattr(node.owner, "__name__", node.owner)
        if (why := DELETED_SURFACES.get(node.name)) is not None:
            raise ConformanceError(
                f"{owner}.{node.path} declares the deleted field {node.name!r}: {why}",
                code="deleted_surface",
                fields=[node.path],
            )
        if (why := EXCLUDED_KEYWORDS.get(node.name)) is not None:
            raise ConformanceError(
                f"{owner}.{node.path} declares {node.name!r}: {why} — the schema surface is "
                "held to the same excluded-keyword list as a method wrapper",
                code="excluded_keyword",
                fields=[node.path],
            )


def check(surface: Surface) -> Surface:
    """Every build-time refusal that needs the WHOLE surface, not just one signature."""
    assert_declaration_shape(surface.payload_type)
    assert_declaration_shape(surface.result_type)
    specs = feature_specs(surface.payload_type)

    override = surface.resources
    if override is not None and override.gpu_count is not None and not surface.gpu:
        raise ConformanceError(
            f"{surface.name}: resources=Resources(gpu_count=…) on a signature with no Model "
            "parameter — gpu DERIVES from the signature; the override door is for structural "
            "floors only (§1.2)",
            code="resources_override",
        )

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
