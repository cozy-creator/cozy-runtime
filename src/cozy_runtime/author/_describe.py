"""The describe-time conformance layer — static analysis as a CONTRACT, not a suggestion.

`describe` is the second reader of the one signature (mypy is the first). An endpoint whose
surface is invalid refuses at BUILD with the same verdict class as a schema violation, so a
mis-typed handler cannot reach a device (§1.7).

This module also owns the ONE SDK-owned deleted-field list. An endpoint asserts what IT
declares via `assert_declaration_shape` and inherits the deletion walk — contracts endpoints
rely on are exercised by conformance, never by convention.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from cozy_runtime.author._demand import FeatureSpec, feature_specs
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._signature import Surface
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
}


def assert_declaration_shape(declaration: object) -> None:
    """Refuse a declaration that names a deleted surface, anywhere in its type tree."""
    for node in walk(declaration):
        if (why := DELETED_SURFACES.get(node.name)) is not None:
            raise ConformanceError(
                f"{getattr(node.owner, '__name__', node.owner)}.{node.path} declares the "
                f"deleted field {node.name!r}: {why}",
                code="deleted_surface",
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
    return surface


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
