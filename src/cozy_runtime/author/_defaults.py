"""`ModelDefault[T]` resolution: ONE ATOMIC OVERLAY, never per-field (§1.2).

    code fallback  <-  checkpoint recipe  <-  ordered adapter recipes
                   <-  explicit request   <-  visible deployment clamp

Applied as one immutable overlay whose digest is recorded with every source, after which
the ENTIRE final request revalidates ONCE — cross-field constraints included — so no
unvalidated steps/guidance/scheduler combination can emerge from independently resolved
correlated fields. A partial or internally inconsistent recipe refuses AS A UNIT.

There is no parallel family-defaults struct, no `model.tuned`, no name mapping, and no
package-side `resolve()` call: the resolved values are written into the payload before
invocation, so the handler sees concrete `T`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import msgspec

from cozy_runtime.author._assets import asset_dec_hook
from cozy_runtime.author._errors import DefaultResolutionError, InvalidRequest
from cozy_runtime.author._markers import _ModelDefaultMarker
from cozy_runtime.author._services import AdjustmentRow
from cozy_runtime.author._walker import MISSING, Node, walk

Layer = Literal["checkpoint", "adapter"]


@dataclass(frozen=True, slots=True)
class Recipe:
    """A catalog or adapter default set. Values must NAME an existing marked field."""

    source: str
    """`checkpoint:<ref>` / `adapter:<ref>` — provenance recorded with every resolved value."""
    layer: Layer
    values: Mapping[str, object]
    binding_path: str | None = None
    """The model parameter this recipe binds to. Required when several models can apply."""


@dataclass(frozen=True, slots=True)
class Clamp:
    """An EXPLICIT deployment policy clamp. Caller-visible by construction."""

    field: str
    reason: str
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True, slots=True)
class Overlay:
    """The immutable result. `payload` is the revalidated request the handler receives."""

    payload: Any
    values: Mapping[str, object]
    sources: Mapping[str, str]
    digest: str
    rows: tuple[AdjustmentRow, ...] = ()


def marked_fields(payload_type: object) -> dict[str, Node]:
    """Top-level `ModelDefault[T]` fields. A marker deeper in the tree is not a request
    field the wire can omit, so resolution deliberately does not reach it."""
    return {
        node.name: node
        for node in walk(payload_type)
        if "." not in node.path and node.marker(_ModelDefaultMarker) is not None
    }


def _convert(value: object, node: Node, where: str) -> object:
    """Validate one recipe value against the marked field's own schema, bounds included."""
    try:
        return msgspec.convert(value, type=node.annotation, dec_hook=asset_dec_hook, strict=False)
    except (msgspec.ValidationError, TypeError, NotImplementedError) as exc:
        raise DefaultResolutionError(
            f"{where}: {node.name}={value!r} does not validate against the request field ({exc})",
            fields=[node.name],
        ) from exc


def _validate_recipe(
    recipe: Recipe, marked: Mapping[str, Node], model_params: Sequence[str]
) -> None:
    if len(model_params) > 1 and recipe.binding_path is None:
        raise DefaultResolutionError(
            f"{recipe.source}: with {len(model_params)} model parameters every default "
            "source must name its binding path; the bare form is legal only when exactly "
            "one can apply",
            code="ambiguous_binding",
        )
    if recipe.binding_path is not None and recipe.binding_path not in model_params:
        raise DefaultResolutionError(
            f"{recipe.source}: binding path {recipe.binding_path!r} names no model "
            f"parameter (have {', '.join(model_params) or 'none'})",
            code="unknown_binding",
        )
    unknown = [k for k in recipe.values if k not in marked]
    if unknown:
        raise DefaultResolutionError(
            f"{recipe.source} names {', '.join(sorted(unknown))}, which are not "
            "ModelDefault request fields — the recipe refuses as a unit",
            code="unknown_default",
            fields=sorted(unknown),
        )
    for key, value in recipe.values.items():
        _convert(value, marked[key], recipe.source)


def resolve(
    payload_type: object,
    wire: Mapping[str, object],
    *,
    recipes: Sequence[Recipe] = (),
    clamps: Sequence[Clamp] = (),
    model_params: Sequence[str] = (),
) -> Overlay:
    """Build the one overlay and the revalidated payload."""
    marked = marked_fields(payload_type)

    # Layer 1 of the two-layer clamp: API bounds REJECT what the caller actually sent,
    # before any policy has a chance to quietly clamp it into range.
    _decode(payload_type, wire, "the request")

    for recipe in recipes:
        _validate_recipe(recipe, marked, model_params)

    omitted = [name for name in marked if name not in wire]
    values: dict[str, object] = {}
    sources: dict[str, str] = {}
    rows: list[AdjustmentRow] = []

    for name in omitted:
        node = marked[name]
        if node.default is not MISSING:
            values[name], sources[name] = node.default, "code"

    for layer in ("checkpoint", "adapter"):
        contributors: dict[str, list[Recipe]] = {}
        for recipe in (r for r in recipes if r.layer == layer):
            for name in (n for n in recipe.values if n in omitted):
                contributors.setdefault(name, []).append(recipe)
        for name, applying in contributors.items():
            if layer == "adapter" and len(applying) > 1:
                raise DefaultResolutionError(
                    f"{name}: {' and '.join(r.source for r in applying)} both supply this "
                    "omitted field and it defines no composition rule — stack order never "
                    "silently becomes configuration precedence",
                    code="recipe_conflict",
                    fields=[name],
                )
            recipe = applying[-1]
            values[name] = _convert(recipe.values[name], marked[name], recipe.source)
            sources[name] = recipe.source

    merged: dict[str, object] = {**values, **wire}
    for name in wire:
        sources[name] = "explicit"

    for clamp in clamps:
        if clamp.field not in merged:
            continue
        current = merged[clamp.field]
        if not isinstance(current, (int, float)) or isinstance(current, bool):
            continue
        applied = current
        if clamp.min is not None:
            applied = max(applied, clamp.min)
        if clamp.max is not None:
            applied = min(applied, clamp.max)
        if applied != current:
            applied = type(current)(applied)
            merged[clamp.field] = applied
            sources[clamp.field] = "policy"
            rows.append(
                AdjustmentRow("clamp", clamp.field, current, applied, clamp.reason, "policy")
            )

    for name in omitted:
        if sources.get(name) not in (None, "code", "explicit"):
            rows.append(
                AdjustmentRow(
                    "adjusted", name, None, merged[name], "resolved default", sources[name]
                )
            )

    # ONE whole-request revalidation, cross-field constraints included.
    payload = _decode(payload_type, merged, "the resolved request")
    return Overlay(
        payload, dict(sorted(merged.items())), sources, _digest(merged, sources), tuple(rows)
    )


def _decode(payload_type: object, data: Mapping[str, object], what: str) -> Any:
    try:
        return msgspec.convert(dict(data), type=payload_type, dec_hook=asset_dec_hook)
    except msgspec.ValidationError as exc:
        raise InvalidRequest(f"{what} does not validate: {exc}") from exc
    except (TypeError, NotImplementedError) as exc:
        raise InvalidRequest(f"{what} does not decode: {exc}") from exc


def _digest(values: Mapping[str, object], sources: Mapping[str, str]) -> str:
    canonical = msgspec.json.encode(
        [[k, _plain(v), sources.get(k, "code")] for k, v in sorted(values.items())]
    )
    return "blake2b:" + hashlib.blake2b(canonical, digest_size=16).hexdigest()


def _plain(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
