"""Type tree -> canonical schema document. The package interface's request/result half.

SOURCE-STABLE by construction: everything below is read from the author's own declared
types, so nothing a candidate artifact, a binding, a machine or a clock knows can reach it.
It reuses `author._walker`'s primitives (`strip`, `unwrap_optional`, `is_struct`) rather
than re-deriving annotation semantics — one answer to "what does this type tree declare".

Only the wire grammar Creator consumes is emitted. Python names, default factories and
input decoded-memory ceilings stay with the installed signature that enforces them.
Result schemas preserve explicit decoded bounds so generated callers can consume
returned media. Input descriptors keep their existing identity.
"""

from __future__ import annotations

import enum
import types
import typing
from collections.abc import Mapping, Sequence, Set
from typing import Literal, get_args, get_origin

import msgspec

from cozy_runtime.author._artifacts import ModelArtifact, SourceArtifact
from cozy_runtime.author._assets import Asset, AudioAsset, FileAsset, ImageAsset, Tree, VideoAsset
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._markers import AssetBound, _ModelDefaultMarker
from cozy_runtime.author._walker import is_struct, strip, unwrap_optional
from cozy_runtime.internal.canonical import CanonicalError, Json, write

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


def render(annotation: object, *, decoded_bounds: bool = False) -> Json:
    """One type as a canonical schema value."""
    return _render(annotation, frozenset(), decoded_bounds=decoded_bounds)


def request(payload_type: object) -> Json:
    """A callable's request: `render`, with each optional top-level field's author default.

    Only the payload's own fields carry one. Nested types render exactly as `render` does,
    because readers compare those by value (the Assets occurrence record, native operations).
    """
    doc = render(payload_type)
    base, _ = strip(payload_type)
    if isinstance(doc, dict) and isinstance(base, type) and is_struct(base):
        fields = {field.encode_name: field for field in msgspec.structs.fields(base)}
        for row in typing.cast(list[dict[str, Json]], doc["fields"]):
            if row.get("wire") == "optional":
                row.update(_default(fields[typing.cast(str, row["name"])]))
    return doc


def _default(field: msgspec.structs.FieldInfo) -> dict[str, Json]:
    """The author's default for clients to show; none when it has no static JSON spelling."""
    value = field.default
    if value is msgspec.NODEFAULT:
        if field.default_factory not in (list, dict):
            return {}
        value = field.default_factory()
    try:
        return {"default": msgspec.json.decode(write(msgspec.to_builtins(value)))}
    except (TypeError, CanonicalError):
        return {}


def _render(
    annotation: object,
    seen: frozenset[type],
    *,
    union_owns_tag_field: bool = False,
    decoded_bounds: bool = False,
) -> Json:
    base, _ = strip(annotation)
    inner, optional = unwrap_optional(base)
    if optional:
        return {
            "union": sorted_union([_render(inner, seen, decoded_bounds=decoded_bounds), "null"])
        }
    return _render_bare(
        inner, seen, union_owns_tag_field=union_owns_tag_field, decoded_bounds=decoded_bounds
    )


def sorted_union(members: list[Json]) -> list[Json]:
    """Union members in a stable order, so a reordered `A | B` is not a surface change."""
    return sorted(members, key=write)


def _render_bare(
    tp: object,
    seen: frozenset[type],
    *,
    union_owns_tag_field: bool = False,
    decoded_bounds: bool = False,
) -> Json:
    if tp is ModelArtifact:
        return {"input": "model"}
    if tp is SourceArtifact:
        return {"input": "source"}
    if (name := _PRIMITIVES.get(tp)) is not None:
        return name
    origin = get_origin(tp)
    if origin is Literal:
        return _enumeration(tp, get_args(tp))
    if origin is typing.Union or origin is types.UnionType:
        return _union(get_args(tp), seen, decoded_bounds=decoded_bounds)
    if origin in (list, Sequence, set, frozenset, Set):
        args = get_args(tp)
        if not args:
            return {"list": _opaque(tp)}
        return {"list": _render(args[0], seen, decoded_bounds=decoded_bounds)}
    if origin in (dict, Mapping):
        args = get_args(tp)
        if len(args) != 2:
            return _opaque(tp)
        return {
            "map": {
                "key": _render(args[0], seen, decoded_bounds=decoded_bounds),
                "value": _render(args[1], seen, decoded_bounds=decoded_bounds),
            }
        }
    if origin is tuple:
        args = get_args(tp)
        if not args:
            return {"list": _opaque(tp)}
        if len(args) == 2 and args[1] is Ellipsis:
            return {"list": _render(args[0], seen, decoded_bounds=decoded_bounds)}
        return {"tuple": [_render(arg, seen, decoded_bounds=decoded_bounds) for arg in args]}
    if isinstance(tp, type):
        for asset, kind in _ASSET_KINDS:
            if issubclass(tp, asset):
                return {"asset": kind}
        if issubclass(tp, Tree):
            # A JOB input: a digest-verified materialized tree. The wire spelling is a REF,
            # exactly like an asset — the local path is never schema content (§2).
            return {"input": "tree"}
        if issubclass(tp, enum.Enum):
            return _enumeration(tp, [member.value for member in tp])
        if is_struct(tp):
            return _struct(
                typing.cast(type[msgspec.Struct], tp),
                seen,
                union_owns_tag_field=union_owns_tag_field,
                decoded_bounds=decoded_bounds,
            )
    # ANYTHING ELSE IS OPAQUE, NOT A BUILD FAILURE. `bytes`, `datetime`, a third-party
    # value type — msgspec still codes it and JSON still carries it. The package interface says
    # "this field exists and Creator cannot introspect it" instead of refusing the package.
    return _opaque(tp)


def _opaque(tp: object) -> Json:
    """A field whose type has no interoperable spelling, named by its Python spelling.

    A bare class spells as its name; anything else — a subscripted alias, whose `__name__`
    is the bare constructor and drops exactly the part that identified it — spells as its
    repr, so `Literal[b"x"]` does not read as `Literal`.
    """
    name = getattr(tp, "__name__", None) if isinstance(tp, type) else None
    return {"opaque": name if isinstance(name, str) and name else repr(tp)}


class _Unspellable(Exception):
    """One enumeration member the wire grammar cannot name. Never a build refusal."""


def _literal(value: object) -> Json:
    """One admissible value as the wire carries it. An enum member spells as its VALUE,
    which is the thing msgspec actually encodes and Creator actually matches on."""
    if isinstance(value, enum.Enum):
        value = value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise _Unspellable(value)


def _enumeration(tp: object, values: Sequence[object]) -> Json:
    """A closed value set, or `opaque` when one member has no interoperable spelling.

    msgspec codes a tuple-valued enum and a `Literal[b"..."]` perfectly well, so refusing
    the BUILD over one made the package interface narrower than the runtime it describes. The node
    degrades instead: the field exists, and Creator cannot enumerate it.
    """
    try:
        return {"literal": sorted_union([_literal(value) for value in values])}
    except _Unspellable:
        return _opaque(tp)


def _tag(value: object) -> Json:
    """A tagged struct's discriminator. msgspec already refuses anything but `str`/`int`
    at class creation; this is the fail-closed reading of that, not a second policy."""
    if isinstance(value, (str, int)):
        return value
    raise ConformanceError(
        f"tag {value!r} is not a string or integer discriminator",
        code="tagged_union_incoherent",
    )


def _union(members: tuple[object, ...], seen: frozenset[type], *, decoded_bounds: bool) -> Json:
    structs: list[tuple[type[msgspec.Struct], object, object]] = []
    for member in members:
        base, _ = strip(member)
        if not is_struct(base):
            continue
        struct = typing.cast(type[msgspec.Struct], base)
        config = struct.__struct_config__
        structs.append((struct, config.tag_field, config.tag))
    if len(structs) <= 1:
        return {
            "union": sorted_union(
                [_render(member, seen, decoded_bounds=decoded_bounds) for member in members]
            )
        }
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
    tags = [_tag(tag) for _, _, tag in structs]
    if len({write(value) for value in tags}) != len(tags):
        raise ConformanceError(
            "a tagged union's member tags must be unique",
            code="tagged_union_incoherent",
        )
    rendered = sorted_union(
        [
            _render(
                member,
                seen,
                union_owns_tag_field=is_struct(strip(member)[0]),
                decoded_bounds=decoded_bounds,
            )
            for member in members
        ]
    )
    return {"union": rendered, "tag_field": typing.cast(str, next(iter(fields)))}


def _struct(
    tp: type[msgspec.Struct],
    seen: frozenset[type],
    *,
    union_owns_tag_field: bool = False,
    decoded_bounds: bool = False,
) -> Json:
    if tp in seen:
        return _opaque(tp)
    nested = seen | {tp}
    config = tp.__struct_config__
    fields = [_field(f, nested, decoded_bounds=decoded_bounds) for f in msgspec.structs.fields(tp)]
    if config.tag is not None and (not isinstance(config.tag_field, str) or not config.tag_field):
        raise ConformanceError(
            f"tagged struct {tp.__name__} declares no string tag field",
            code="tagged_union_incoherent",
        )
    doc: dict[str, Json] = {"fields": fields}
    if config.tag is not None:
        doc["tag"] = _tag(config.tag)
        if not union_owns_tag_field:
            doc["tag_field"] = typing.cast(str, config.tag_field)
    return doc


def _field(
    field: msgspec.structs.FieldInfo, seen: frozenset[type], *, decoded_bounds: bool
) -> Json:
    # Excluded/deleted field names refuse one layer up, in the describe-time conformance
    # layer that owns both lists (`author/_describe.assert_declaration_shape`).
    name = field.encode_name
    base, markers = strip(field.type)
    doc: dict[str, Json] = {
        "name": name,
        "type": _render(base, seen, decoded_bounds=decoded_bounds),
    }
    omissible = any(isinstance(m, _ModelDefaultMarker) for m in markers)
    if omissible:
        doc["wire"] = "omissible"
    elif not field.required:
        doc["wire"] = "optional"
    constraints = _constraints(markers)
    if constraints:
        doc["constraints"] = constraints
    for marker in markers:
        if isinstance(marker, AssetBound):
            bound: dict[str, Json] = {}
            if marker.max_bytes is not None:
                bound["max_bytes"] = marker.max_bytes
            if decoded_bounds and marker.max_decoded_bytes is not None:
                bound["max_decoded_bytes"] = marker.max_decoded_bytes
            if marker.media_types:
                bound["media_types"] = list(marker.media_types)
            if bound:
                doc["asset_bound"] = bound
    return doc


#: Every `msgspec.Meta` constraint, passed through VERBATIM. A consumer that does not
#: understand one ignores it; refusing the build over a constraint Creator has not learned
#: to read yet made the package interface the narrower of the two surfaces for no reason.
_CONSTRAINTS = (
    "gt",
    "ge",
    "lt",
    "le",
    "multiple_of",
    "pattern",
    "min_length",
    "max_length",
    "tz",
)


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
