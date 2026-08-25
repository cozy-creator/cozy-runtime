"""Type tree -> canonical schema document. The descriptor's request/result half.

SOURCE-STABLE by construction: everything below is read from the author's own declared
types, so nothing a candidate artifact, a binding, a machine or a clock knows can reach it.
It reuses `author._walker`'s primitives (`strip`, `unwrap_optional`, `is_struct`) rather
than re-deriving annotation semantics — one answer to "what does this type tree declare".

`ModelDefault[T]` keeps its CONCRETE handler type here and records the wire fact separately:
`wire: "omissible"` plus the frozen default-source ladder it is eligible for (§1.2). A
handler still sees `int`, never `int | None`.
"""

from __future__ import annotations

import enum
import types
import typing
from collections.abc import Sequence
from typing import Literal, get_args, get_origin

import msgspec

from cozy_runtime.author._assets import Asset, AudioAsset, FileAsset, ImageAsset, VideoAsset
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._markers import Shape, _ModelDefaultMarker
from cozy_runtime.author._walker import MISSING, is_struct, strip, unwrap_optional
from cozy_runtime.internal.canonical import Json, write

#: The atomic ladder a `ModelDefault` field's value may come from, most authoritative last
#: (§1.2). Frozen here because the descriptor is what a hub validates catalog rows against.
DEFAULT_SOURCES = ("code_fallback", "checkpoint_recipe", "adapter_recipe", "request")

_PRIMITIVES: dict[object, str] = {
    bool: "bool",
    int: "int",
    float: "float",
    str: "str",
    bytes: "bytes",
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


def _render(annotation: object, seen: frozenset[type]) -> Json:
    base, _ = strip(annotation)
    inner, optional = unwrap_optional(base)
    if optional:
        return {"union": sorted_union([_render(inner, seen), "null"])}
    return _render_bare(inner, seen)


def sorted_union(members: list[Json]) -> list[Json]:
    """Union members in a stable order, so a reordered `A | B` is not a surface change."""
    return sorted(members, key=write)


def _render_bare(tp: object, seen: frozenset[type]) -> Json:
    if (name := _PRIMITIVES.get(tp)) is not None:
        return name
    origin = get_origin(tp)
    if origin is Literal:
        return {"literal": sorted_union([_literal(a) for a in get_args(tp)])}
    if origin is typing.Union or origin is types.UnionType:
        return {"union": sorted_union([_render(a, seen) for a in get_args(tp)])}
    if origin in (list, Sequence):
        args = get_args(tp)
        return {"list": _render(args[0], seen) if args else "any"}
    if origin in (set, frozenset):
        args = get_args(tp)
        return {"set": _render(args[0], seen) if args else "any"}
    if origin is tuple:
        return {"tuple": [_render(a, seen) for a in get_args(tp) if a is not Ellipsis]}
    if origin is dict:
        args = get_args(tp)
        return {"map": [_render(args[0], seen), _render(args[1], seen)] if args else ["any", "any"]}
    if isinstance(tp, type):
        for asset, kind in _ASSET_KINDS:
            if issubclass(tp, asset):
                return {"asset": kind}
        if issubclass(tp, enum.Enum):
            return {"enum": tp.__name__, "values": sorted(_literal(m.value) for m in tp)}
        if is_struct(tp):
            return _struct(typing.cast(type[msgspec.Struct], tp), seen)
    raise ConformanceError(
        f"{tp!r} has no descriptor spelling — the request/result surface is primitives, "
        "enums, literals, containers, typed Assets and msgspec structs",
        code="unrenderable_type",
    )


def _literal(value: object) -> Json:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ConformanceError(f"literal {value!r} is not JSON-representable", code="unrenderable_type")


def _struct(tp: type[msgspec.Struct], seen: frozenset[type]) -> Json:
    if tp in seen:
        return {"struct_ref": tp.__name__}
    nested = seen | {tp}
    config = tp.__struct_config__
    doc: dict[str, Json] = {
        "struct": tp.__name__,
        "fields": [_field(f, nested) for f in msgspec.structs.fields(tp)],
    }
    if config.forbid_unknown_fields:
        doc["forbid_unknown_fields"] = True
    if config.frozen:
        doc["frozen"] = True
    return doc


def _field(field: msgspec.structs.FieldInfo, seen: frozenset[type]) -> Json:
    # Excluded/deleted field names refuse one layer up, in the describe-time conformance
    # layer that owns both lists (`author/_describe.assert_declaration_shape`).
    name = field.encode_name
    base, markers = strip(field.type)
    doc: dict[str, Json] = {"name": name, "type": _render(base, seen)}
    if field.name != name:
        doc["attribute"] = field.name
    omissible = any(isinstance(m, _ModelDefaultMarker) for m in markers)
    if omissible:
        doc["wire"] = "omissible"
        doc["default_sources"] = list(DEFAULT_SOURCES)
    elif field.required:
        doc["wire"] = "required"
    else:
        doc["wire"] = "optional"
    if not field.required:
        default = _default(field)
        if default is not MISSING:
            doc["default"] = default
    constraints = _constraints(markers)
    if constraints:
        doc["constraints"] = constraints
    for marker in markers:
        if isinstance(marker, Shape):
            doc["shape_axes"] = list(marker.axes())
    return doc


def _default(field: msgspec.structs.FieldInfo) -> Json:
    if field.default_factory is not msgspec.NODEFAULT:
        return {"factory": True}
    value = field.default
    if value is msgspec.NODEFAULT:
        return MISSING
    if isinstance(value, enum.Enum):
        return _literal(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"factory": True}


_CONSTRAINTS = ("gt", "ge", "lt", "le", "multiple_of", "min_length", "max_length", "pattern", "tz")


def _constraints(markers: Sequence[object]) -> dict[str, Json]:
    out: dict[str, Json] = {}
    for marker in markers:
        if not isinstance(marker, msgspec.Meta):
            continue
        for name in _CONSTRAINTS:
            value = getattr(marker, name, None)
            if value is not None:
                out[name] = value
    return out
