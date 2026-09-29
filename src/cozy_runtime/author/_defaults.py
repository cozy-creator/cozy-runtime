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

import functools
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import msgspec
import msgspec.inspect as mi

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
    ignored: tuple[str, ...] = ()
    """Request fields the type does not declare, dropped before decoding (`labels[1].foo`)."""


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
    wire, ignored = undeclared(payload_type, wire)
    marked = marked_fields(payload_type)

    # Layer 1 of the two-layer clamp: API bounds REJECT what the caller actually sent,
    # before any policy has a chance to quietly clamp it into range.
    decode_request(cast("type[Any]", payload_type), wire)

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
    payload = decode_request(cast("type[Any]", payload_type), merged, "the resolved request")
    return Overlay(
        payload,
        dict(sorted(merged.items())),
        sources,
        _digest(merged, sources),
        tuple(rows),
        ignored,
    )


def undeclared(
    payload_type: object, wire: Mapping[str, object]
) -> tuple[dict[str, object], tuple[str, ...]]:
    """The request without the fields its type does not declare, at any depth, and their paths.

    An undeclared field is a warning, never a refusal (owner, 2026-09-28): a newer caller may
    name fields this package does not know, whatever its structs' `forbid_unknown_fields`.
    Types, required fields and bounds stay the decoder's, and stay strict.
    """
    ignored: list[str] = []
    kept = _prune(_type_info(payload_type), dict(wire), "", ignored)
    return cast("dict[str, object]", kept), tuple(sorted(ignored))


@functools.lru_cache(maxsize=256)
def _type_info(payload_type: object) -> mi.Type:
    return mi.type_info(cast("Any", payload_type))


_OBJECTS = (mi.StructType, mi.DataclassType, mi.TypedDictType)


def _prune(info: mi.Type, value: object, path: str, ignored: list[str]) -> object:
    if isinstance(info, mi.Metadata):
        return _prune(info.type, value, path, ignored)
    if isinstance(info, mi.UnionType):
        branch = _branch(info, value)
        return value if branch is None else _prune(branch, value, path, ignored)
    if isinstance(info, _OBJECTS) and isinstance(value, dict):
        fields = {field.encode_name: field.type for field in info.fields}
        tag = info.tag_field if isinstance(info, mi.StructType) else None
        kept: dict[object, object] = {}
        for key, item in value.items():
            where = f"{path}.{key}" if path else str(key)
            if key in fields:
                kept[key] = _prune(fields[key], item, where, ignored)
            elif key == tag:
                kept[key] = item
            else:
                ignored.append(where)
        return kept
    if isinstance(info, (mi.CollectionType, mi.TupleType)) and isinstance(value, list):
        if isinstance(info, mi.TupleType):
            items = info.item_types
        else:
            items = (info.item_type,) * len(value)
        return [
            _prune(items[i], item, f"{path}[{i}]", ignored) if i < len(items) else item
            for i, item in enumerate(value)
        ]
    if isinstance(info, mi.DictType) and isinstance(value, dict):
        return {k: _prune(info.value_type, v, f"{path}.{k}", ignored) for k, v in value.items()}
    return value


def _branch(info: mi.UnionType, value: object) -> mi.Type | None:
    """The one member a value decodes as: msgspec admits at most one object-like and one
    array-like member, besides structs told apart by their tag."""
    if not isinstance(value, (dict, list)):
        return None
    fields = value if isinstance(value, dict) else {}
    arrays = (mi.CollectionType, mi.TupleType)
    kinds = (*_OBJECTS, mi.DictType) if isinstance(value, dict) else arrays
    members = [member.type if isinstance(member, mi.Metadata) else member for member in info.types]
    candidates = [member for member in members if isinstance(member, kinds)]
    tagged = [m for m in candidates if isinstance(m, mi.StructType) and m.tag_field is not None]
    for member in tagged:
        if member.tag_field in fields and fields[member.tag_field] == member.tag:
            return member
    untagged = [member for member in candidates if member not in tagged]
    return untagged[0] if len(untagged) == 1 else None


def decode_request[T](
    payload_type: type[T], data: Mapping[str, object], what: str = "the request"
) -> T:
    """Decode wire data into a request struct the way the runtime decodes it.

    The ONE supported path from wire data to a payload carrying asset fields. `Asset` is
    deliberately not a msgspec-native type, so a caller reaching for `msgspec.convert`
    needs `asset_dec_hook` — the hook that accepts a REF STRING and nothing else — and a
    caller who writes their own hook gets a decoder that can hydrate an asset from request
    data. Refusals arrive as `InvalidRequest`, the typed outcome the executor raises.
    """
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
