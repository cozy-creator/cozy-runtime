"""Type tree -> canonical schema document. The descriptor's request/result half.

SOURCE-STABLE by construction: everything below is read from the author's own declared
types, so nothing a candidate artifact, a binding, a machine or a clock knows can reach it.
It reuses `author._walker`'s primitives (`strip`, `unwrap_optional`, `is_struct`) rather
than re-deriving annotation semantics — one answer to "what does this type tree declare".

Only the wire grammar Creator consumes is emitted. Python names, default factories and
decoded in-memory ceilings stay with the installed signature that enforces them.
"""

from __future__ import annotations

import enum
import types
import typing
from collections.abc import Sequence
from typing import Literal, get_args, get_origin

import msgspec

from cozy_runtime.author._assets import Asset, AudioAsset, FileAsset, ImageAsset, Tree, VideoAsset
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._markers import AssetBound, _ModelDefaultMarker
from cozy_runtime.author._walker import is_struct, strip, unwrap_optional
from cozy_runtime.internal.canonical import Json, write

_PRIMITIVES: dict[object, str] = {
    bool: "bool",
    int: "int",
    float: "float",
    str: "str",
    type(None): "null",
}

_ASSET_KINDS: tuple[tuple[type[Asset], str], ...] = (
    (ImageAsset, "image"),
    (AudioAsset, "audio"),
    (VideoAsset, "video"),
    (FileAsset, "file"),
)


def render(annotation: object) -> Json:
    """One type as a canonical schema value."""
    return _render(annotation, frozenset())


def _render(annotation: object, seen: frozenset[type], *, tag_required: bool = False) -> Json:
    base, _ = strip(annotation)
    inner, optional = unwrap_optional(base)
    if optional:
        return {"union": sorted_union([_render(inner, seen), "null"])}
    return _render_bare(inner, seen, tag_required=tag_required)


def sorted_union(members: list[Json]) -> list[Json]:
    """Union members in a stable order, so a reordered `A | B` is not a surface change."""
    return sorted(members, key=write)


def _render_bare(tp: object, seen: frozenset[type], *, tag_required: bool = False) -> Json:
    if (name := _PRIMITIVES.get(tp)) is not None:
        return name
    origin = get_origin(tp)
    if origin is Literal:
        return {"literal": sorted_union([_literal(a) for a in get_args(tp)])}
    if origin is typing.Union or origin is types.UnionType:
        return _union(get_args(tp), seen)
    if origin in (list, Sequence):
        args = get_args(tp)
        if not args:
            raise ConformanceError(
                "an untyped list has no interoperable descriptor spelling",
                code="unrenderable_type",
            )
        return {"list": _render(args[0], seen)}
    if isinstance(tp, type):
        for asset, kind in _ASSET_KINDS:
            if issubclass(tp, asset):
                return {"asset": kind}
        if issubclass(tp, Tree):
            # A JOB input: a digest-verified materialized tree. The wire spelling is a REF,
            # exactly like an asset — the local path is never schema content (§2).
            return {"input": "tree"}
        if issubclass(tp, enum.Enum):
            return {"literal": sorted_union([_literal(m.value) for m in tp])}
        if is_struct(tp):
            return _struct(typing.cast(type[msgspec.Struct], tp), seen, tag_required=tag_required)
    raise ConformanceError(
        f"{tp!r} has no descriptor spelling — the request/result surface is primitives, "
        "enums, literals, containers, typed Assets, input Trees and msgspec structs",
        code="unrenderable_type",
    )


def _literal(value: object) -> Json:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ConformanceError(f"literal {value!r} is not JSON-representable", code="unrenderable_type")


def _union(members: tuple[object, ...], seen: frozenset[type]) -> Json:
    structs: list[tuple[type[msgspec.Struct], object, object]] = []
    for member in members:
        base, _ = strip(member)
        if not is_struct(base):
            continue
        struct = typing.cast(type[msgspec.Struct], base)
        config = struct.__struct_config__
        structs.append((struct, config.tag_field, config.tag))
    if len(structs) <= 1:
        return {"union": sorted_union([_render(member, seen) for member in members])}
    if any(tag is None for _, _, tag in structs):
        raise ConformanceError(
            "a union with multiple structs requires every struct to declare a tag",
            code="tagged_union_incoherent",
        )
    fields = {field for _, field, _ in structs}
    if len(fields) != 1 or not isinstance(next(iter(fields)), str) or not next(iter(fields)):
        raise ConformanceError(
            "a tagged union's struct members must share one string tag field",
            code="tagged_union_incoherent",
        )
    tag_types = {type(tag) for _, _, tag in structs}
    if len(tag_types) != 1:
        raise ConformanceError(
            "a tagged union's member tags must all have the same scalar type",
            code="tagged_union_incoherent",
        )
    tags = [_literal(tag) for _, _, tag in structs]
    if len({write(value) for value in tags}) != len(tags):
        raise ConformanceError(
            "a tagged union's member tags must be unique",
            code="tagged_union_incoherent",
        )
    rendered = sorted_union(
        [_render(member, seen, tag_required=is_struct(strip(member)[0])) for member in members]
    )
    return {"union": rendered, "tag_field": typing.cast(str, next(iter(fields)))}


def _struct(tp: type[msgspec.Struct], seen: frozenset[type], *, tag_required: bool = False) -> Json:
    if tp in seen:
        raise ConformanceError(
            "recursive structs have no interoperable descriptor spelling",
            code="unrenderable_type",
        )
    nested = seen | {tp}
    config = tp.__struct_config__
    fields = [_field(f, nested) for f in msgspec.structs.fields(tp)]
    if config.tag is not None:
        if not isinstance(config.tag_field, str) or not config.tag_field:
            raise ConformanceError(
                f"tagged struct {tp.__name__} declares no string tag field",
                code="tagged_union_incoherent",
            )
        tag = _literal(config.tag)
        fields.insert(
            0,
            {
                "name": config.tag_field,
                "type": {"literal": [tag]},
                "wire": "required" if tag_required else "optional",
            },
        )
    doc: dict[str, Json] = {"fields": fields}
    if config.tag is not None:
        doc["tag_field"] = typing.cast(str, config.tag_field)
        doc["tag"] = _literal(config.tag)
    return doc


def _field(field: msgspec.structs.FieldInfo, seen: frozenset[type]) -> Json:
    # Excluded/deleted field names refuse one layer up, in the describe-time conformance
    # layer that owns both lists (`author/_describe.assert_declaration_shape`).
    name = field.encode_name
    base, markers = strip(field.type)
    doc: dict[str, Json] = {"name": name, "type": _render(base, seen)}
    omissible = any(isinstance(m, _ModelDefaultMarker) for m in markers)
    if omissible:
        doc["wire"] = "omissible"
    elif field.required:
        doc["wire"] = "required"
    else:
        doc["wire"] = "optional"
    constraints = _constraints(markers)
    if constraints:
        doc["constraints"] = constraints
    for marker in markers:
        if isinstance(marker, AssetBound):
            bound: dict[str, Json] = {}
            if marker.max_bytes is not None:
                bound["max_bytes"] = marker.max_bytes
            if marker.media_types:
                bound["media_types"] = list(marker.media_types)
            if bound:
                doc["asset_bound"] = bound
    return doc


_CONSTRAINTS = ("gt", "ge", "le", "min_length", "max_length")
_UNSUPPORTED_CONSTRAINTS = ("lt", "multiple_of", "pattern", "tz")


def _constraints(markers: Sequence[object]) -> dict[str, Json]:
    out: dict[str, Json] = {}
    for marker in markers:
        if not isinstance(marker, msgspec.Meta):
            continue
        unsupported = [
            name for name in _UNSUPPORTED_CONSTRAINTS if getattr(marker, name, None) is not None
        ]
        if unsupported:
            raise ConformanceError(
                "descriptor constraint(s) are not consumed by Creator: " + ", ".join(unsupported),
                code="unrenderable_constraint",
            )
        for name in _CONSTRAINTS:
            value = getattr(marker, name, None)
            if value is not None:
                out[name] = value
    return out
