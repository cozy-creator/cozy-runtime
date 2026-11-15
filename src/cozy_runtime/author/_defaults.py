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
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast, get_origin

import msgspec
import msgspec.inspect as mi

from cozy_runtime.author._errors import DefaultResolutionError, InvalidRequest
from cozy_runtime.author._files import (
    References,
    asset_dec_hook,
    reference_kinds,
    unmentioned,
    unused_reference,
)
from cozy_runtime.author._image import Control
from cozy_runtime.author._markers import _ModelDefaultMarker, applies, condition
from cozy_runtime.author._services import AdjustmentRow
from cozy_runtime.author._walker import MISSING, Node, strip, walk

Layer = Literal["checkpoint", "adapter"]


@dataclass(frozen=True, slots=True)
class DefaultSet:
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

    payload: object
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
    recipe: DefaultSet, marked: Mapping[str, Node], model_params: Sequence[str]
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


def _usable(
    recipe: DefaultSet, marked: Mapping[str, Node], rows: list[AdjustmentRow]
) -> DefaultSet:
    """The values of a recipe this function can take, held to their bounds. A recipe is a model's
    advice, so one that names a field this function doesn't have, or a value it can't use, costs
    that value and a warning, never the run."""
    kept: dict[str, object] = {}
    for key, value in recipe.values.items():
        node = marked.get(key)
        if node is None:
            rows.append(
                AdjustmentRow(
                    "warn", key, value, None, "this function takes no default for it", recipe.source
                )
            )
            continue
        held = _prune(_type_info(node.annotation), value, key, [], rows)
        try:
            kept[key] = _convert(held, node, recipe.source)
        except DefaultResolutionError as exc:
            rows.append(AdjustmentRow("warn", key, value, None, exc.message, recipe.source))
    return replace(recipe, values=kept)


def resolve(
    payload_type: object,
    wire: Mapping[str, object],
    *,
    default_sets: Sequence[DefaultSet] = (),
    clamps: Sequence[Clamp] = (),
    model_params: Sequence[str] = (),
) -> Overlay:
    """Build the one overlay and the revalidated payload."""
    wire, ignored, fitted = fit(payload_type, wire)
    marked = marked_fields(payload_type)

    # What the caller sent decodes on its own, before any recipe joins it: a wrong type or a
    # missing field refuses here. Its numbers and lengths are already within their limits.
    decode_request(cast("type[object]", payload_type), wire)

    rows = list(fitted)
    for recipe in default_sets:
        _validate_recipe(recipe, marked, model_params)
    default_sets = [_usable(recipe, marked, rows) for recipe in default_sets]

    omitted = [name for name in marked if name not in wire]
    values: dict[str, object] = {}
    sources: dict[str, str] = {}

    for name in omitted:
        node = marked[name]
        if node.default is not MISSING:
            values[name], sources[name] = node.default, "code"

    for layer in ("checkpoint", "adapter"):
        contributors: dict[str, list[DefaultSet]] = {}
        for recipe in (r for r in default_sets if r.layer == layer):
            for name in (n for n in recipe.values if n in omitted):
                contributors.setdefault(name, []).append(recipe)
        for name, applying in contributors.items():
            recipe = applying[-1]
            if layer == "adapter" and len(applying) > 1:
                rows.append(
                    AdjustmentRow(
                        "warn",
                        name,
                        None,
                        recipe.values[name],
                        f"{' and '.join(r.source for r in applying)} each set it: the last wins; "
                        "set it in the request to choose",
                        recipe.source,
                    )
                )
            values[name] = recipe.values[name]
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
    payload = decode_request(cast("type[object]", payload_type), merged, "the resolved request")
    return Overlay(
        payload,
        dict(sorted(merged.items())),
        sources,
        _digest(merged, sources),
        tuple(rows),
        ignored,
    )


def fit(
    payload_type: object, wire: Mapping[str, object]
) -> tuple[dict[str, object], tuple[str, ...], tuple[AdjustmentRow, ...]]:
    """The request as its type can take it, at any depth, what it dropped and what it moved.

    An undeclared field is a warning, never a refusal (owner, 2026-09-28): a newer caller may
    name fields this package does not know, whatever its structs' `forbid_unknown_fields`.
    A value over a limit is clamped or cut with a row, never refused (owner, 2026-11-03): a
    number holds to its bounds and `multiple_of`, or a numeric Literal's nearest member, and text
    and lists to their `max_length`, and a resolution takes its enum's nearest tier. A modality
    the function has no use for is ignored with a row (Paul, 11-12): a prompt's reference its
    `Prompt[T]` takes none of, which decoding leaves out, and a prompt given to a text field, as
    its text. Types and required fields stay the decoder's, and stay strict.
    """
    ignored: list[str] = []
    rows: list[AdjustmentRow] = []
    kept = _prune(_type_info(payload_type), dict(wire), "", ignored, rows)
    return cast("dict[str, object]", kept), tuple(sorted(ignored)), tuple(rows)


@functools.lru_cache(maxsize=256)
def _type_info(payload_type: object) -> mi.Type:
    return mi.type_info(payload_type)


_OBJECTS = (mi.StructType, mi.DataclassType, mi.TypedDictType)
_LIMIT = "the field's limit"
_TIER = re.compile(r"[1-9][0-9]*p")
_ASKED = re.compile(r"([0-9]+)(p|k)?")
_K_TIERS = {"1": 720, "2": 1440, "4": 2160, "8": 4320}


def _prune(
    info: mi.Type, value: object, path: str, ignored: list[str], rows: list[AdjustmentRow]
) -> object:
    if isinstance(info, mi.Metadata):
        return _prune(info.type, value, path, ignored, rows)
    if isinstance(info, mi.UnionType):
        branch = _branch(info, value)
        if branch is None and _text(value) is not None:  # a prompt for `str | None`
            members = [m.type if isinstance(m, mi.Metadata) else m for m in info.types]
            branch = next((m for m in members if isinstance(m, mi.StrType)), None)
        return value if branch is None else _prune(branch, value, path, ignored, rows)
    if isinstance(info, _OBJECTS) and isinstance(value, dict):
        fields = {field.encode_name: field.type for field in info.fields}
        tag = info.tag_field if isinstance(info, mi.StructType) else None
        mentions = _mentions(info.cls) if isinstance(info, mi.StructType) else {}
        kept: dict[object, object] = {}
        for key, item in value.items():
            where = f"{path}.{key}" if path else str(key)
            if key in mentions:
                _unused(mentions[key], item, where, rows)
            if key in fields:
                kept[key] = _prune(fields[key], item, where, ignored, rows)
            elif key == tag:
                kept[key] = item
            else:
                ignored.append(where)
        return _applying(info, kept, path, rows)
    if isinstance(info, (mi.CollectionType, mi.TupleType)) and isinstance(value, list):
        if isinstance(info, mi.TupleType):
            items = info.item_types
        else:
            value = _usable_controls(info.item_type, value, path, rows)
            most = info.max_length
            if most is not None and len(value) > most:
                rows.append(
                    AdjustmentRow(
                        "clamp", path, f"{len(value)} items", f"{most} items", _LIMIT, "policy"
                    )
                )
                value = value[:most]
            items = (info.item_type,) * len(value)
        return [
            _prune(items[i], item, f"{path}[{i}]", ignored, rows) if i < len(items) else item
            for i, item in enumerate(value)
        ]
    if isinstance(info, mi.DictType) and isinstance(value, dict):
        return {
            k: _prune(info.value_type, v, f"{path}.{k}", ignored, rows) for k, v in value.items()
        }
    if isinstance(info, mi.StrType) and (text := _text(value)) is not None:
        references = cast("dict[str, object]", value).get("references")
        value = unmentioned(text, _unused(frozenset(), references, f"{path}.references", rows))
    if isinstance(info, mi.StrType) and isinstance(value, str):
        most = info.max_length
        if most is None or len(value) <= most:
            return value
        rows.append(
            AdjustmentRow(
                "clamp", path, f"{len(value)} characters", f"{most} characters", _LIMIT, "policy"
            )
        )
        return value[:most]
    if isinstance(info, mi.EnumType) and isinstance(value, str):
        return _tier(info, value, path, rows)
    applied = _held(info, value)
    if applied != value:
        rows.append(AdjustmentRow("clamp", path, value, applied, _LIMIT, "policy"))
    return applied


@functools.lru_cache(maxsize=256)
def _mentions(cls: type[msgspec.Struct]) -> dict[str, frozenset[str]]:
    """A struct's `References[T]` fields by wire name, each with the media it takes: what
    `_references` is given for it when it decodes."""
    found: dict[str, frozenset[str]] = {}
    for field in msgspec.structs.fields(cls):
        base, _ = strip(field.type)
        if (get_origin(base) or base) is References:
            found[field.encode_name] = reference_kinds(base)
    return found


def _unused(
    kinds: frozenset[str], references: object, path: str, rows: list[AdjustmentRow]
) -> list[str]:
    """An `ignored` row for each wire reference of a modality `kinds` doesn't take; their keys."""
    keys: list[str] = []
    for at, item in enumerate(references if isinstance(references, list) else ()):
        if (kind := unused_reference(kinds, item)) is not None:
            keys.append(str(cast("dict[str, object]", item).get("key", "")))
            rows.append(AdjustmentRow.ignored(f"{path}.{at}", keys[-1], kind))
    return keys


def _text(value: object) -> str | None:
    """A prompt's text, when a value is a prompt (`{"text", "references"}`)."""
    if not isinstance(value, dict) or "text" not in value or value.keys() - {"text", "references"}:
        return None
    text, references = value["text"], value.get("references", [])
    return text if isinstance(text, str) and isinstance(references, list) else None


def _when(info: mi.Type) -> Mapping[str, object] | None:
    """A field's `When`, on it or on the value of its `| None`."""
    members = info.types if isinstance(info, mi.UnionType) else (info,)
    for member in members:
        extra = member.extra if isinstance(member, mi.Metadata) and member.extra else {}
        when = extra.get("when")
        if isinstance(when, dict):
            return when
    return None


def _toggle(info: mi.Type) -> bool:
    """Whether a field is a bool, or a bool's `| None`."""
    if isinstance(info, mi.UnionType):
        kept = [member for member in info.types if not isinstance(member, mi.NoneType)]
        return len(kept) == 1 and _toggle(kept[0])
    return isinstance(info.type if isinstance(info, mi.Metadata) else info, mi.BoolType)


def _default(field: mi.Field) -> object:
    """A field's default as the wire spells it."""
    if field.default is not msgspec.NODEFAULT:
        value = field.default
    else:
        value = None if field.default_factory is msgspec.NODEFAULT else field.default_factory()
    try:
        return msgspec.to_builtins(value)
    except TypeError:
        return value


def _applying(
    info: mi.StructType | mi.DataclassType | mi.TypedDictType,
    kept: dict[object, object],
    path: str,
    rows: list[AdjustmentRow],
) -> dict[object, object]:
    """Without each field given where its `When` doesn't hold: the function takes its default,
    with a row."""
    for field in info.fields:
        when = _when(field.type)
        peer = next((f for f in info.fields if when and f.name == when.get("field")), None)
        if when is None or peer is None or field.encode_name not in kept:
            continue
        held = kept[peer.encode_name] if peer.encode_name in kept else _default(peer)
        if applies(when, held):
            continue
        where = f"{path}.{field.encode_name}" if path else field.encode_name
        given = kept.pop(field.encode_name)
        said = condition(when, _toggle(peer.type))
        rows.append(AdjustmentRow("adjusted", where, given, _default(field), said, "policy"))
    return kept


def _usable_controls(
    info: mi.Type, value: list[object], path: str, rows: list[AdjustmentRow]
) -> list[object]:
    """A `Control[...]` list without the controls of a kind its function has no adapter for, each
    dropped with a row (Paul, 11-12: an input a model can't use is ignored, never refused)."""
    while isinstance(info, mi.Metadata):
        info = info.type
    if not isinstance(info, mi.StructType):
        return value
    if not issubclass(get_origin(info.cls) or info.cls, Control):
        return value
    kinds: tuple[object, ...] = ()
    for field in info.fields:
        if field.name == "kind" and isinstance(field.type, mi.LiteralType):
            kinds = field.type.values
    kept: list[object] = []
    for i, item in enumerate(value):
        kind = item.get("kind") if isinstance(item, dict) else None
        if isinstance(kind, str) and kind not in kinds:
            reason = f"no {kind} control here: ignored"
            rows.append(AdjustmentRow("warn", f"{path}[{i}]", kind, None, reason, "policy"))
            continue
        kept.append(item)
    return kept


def _tier(info: mi.EnumType, value: str, path: str, rows: list[AdjustmentRow]) -> str:
    """A resolution an enum of `1080p`-style tiers doesn't name, as its nearest (`4k` is 2160p);
    any other enum's value as it is, for the decoder."""
    tiers: dict[int, str] = {
        int(member.value[:-1]): member.value
        for member in info.cls
        if _TIER.fullmatch(str(member.value))
    }
    asked = _ASKED.fullmatch(value.strip().casefold())
    if len(tiers) != len(info.cls) or asked is None or value in tiers.values():
        return value
    height = _K_TIERS.get(asked[1]) if asked[2] == "k" else int(asked[1])
    if height is None:
        return value
    applied = tiers[min(tiers, key=lambda tier: (abs(tier - height), tier))]
    if applied != value:
        rows.append(
            AdjustmentRow("clamp", path, value, applied, "the nearest size served", "policy")
        )
    return applied


def _held(info: mi.Type, value: object) -> object:
    """A number within a field's bounds and on its `multiple_of`, or a numeric Literal's nearest
    member (the lower of two); anything else as it is, for the decoder."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if isinstance(info, mi.LiteralType):
        members = [m for m in info.values if isinstance(m, int | float) and type(m) in (int, float)]
        if not members or value in members:
            return value
        return min(members, key=lambda member: (abs(member - value), member))
    whole = isinstance(info, mi.IntType)
    if not isinstance(info, (mi.IntType, mi.FloatType)) or (whole and isinstance(value, float)):
        return value
    low, high = info.ge, info.le
    if info.gt is not None:
        low = info.gt + 1 if whole else math.nextafter(info.gt, math.inf)
    if info.lt is not None:
        high = info.lt - 1 if whole else math.nextafter(info.lt, -math.inf)
    held = value if low is None or value >= low else low
    held = held if high is None or held <= high else high
    step = info.multiple_of
    if not step:
        return held
    near = round(held / step) * step
    if low is not None and near < low:
        return near + step
    return near - step if high is not None and near > high else near


def _branch(info: mi.UnionType, value: object) -> mi.Type | None:
    """The one member a value decodes as: msgspec admits at most one object-like and one
    array-like member, besides structs told apart by their tag; a scalar's is the one member of
    its kind, if it has just one."""
    arrays = (mi.CollectionType, mi.TupleType)
    kinds: tuple[type[mi.Type], ...]
    if isinstance(value, dict):
        kinds = (*_OBJECTS, mi.DictType)
    elif isinstance(value, list):
        kinds = arrays
    elif isinstance(value, str):
        kinds = (mi.StrType, mi.EnumType)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        kinds = (mi.IntType, mi.FloatType, mi.LiteralType)
    else:
        return None
    fields = value if isinstance(value, dict) else {}
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

    The ONE supported path from wire data to a payload carrying file fields. `File` is
    deliberately not a msgspec-native type, so a caller reaching for `msgspec.convert`
    needs `asset_dec_hook` — the hook that accepts a REF STRING and nothing else — and a
    caller who writes their own hook gets a decoder that can hydrate a file from request
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
