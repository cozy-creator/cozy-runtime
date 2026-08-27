"""THE Annotated type-tree walker. Every marker reads through this — never a private copy.

`Shape`, `ModelDefault`, `Preflight`, the asset-field scan, the preflight two-producers
fence and the deleted-surface assertion all consume `walk()`. One walker means one answer
to "what does this type tree declare", and a new marker costs zero walking code (§1.2).
"""

from __future__ import annotations

import enum
import functools
import operator
import types
import typing
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, is_dataclass
from typing import Annotated, Any, Literal, TypeVar, get_args, get_origin

import msgspec


class _Missing:
    """Sentinel: this field declares no code fallback."""

    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()

M = TypeVar("M")


@dataclass(frozen=True, slots=True)
class Node:
    """One reachable field in a type tree."""

    path: str
    name: str
    annotation: object
    """The type with `Annotated` metadata stripped — what the handler actually sees."""
    markers: tuple[object, ...]
    owner: type
    default: object
    optional: bool
    """True when `None` is an admissible value (`T | None`)."""

    def marker(self, kind: type[M]) -> M | None:
        for m in self.markers:
            if isinstance(m, kind):
                return m
        return None


def strip(annotation: object) -> tuple[object, tuple[object, ...]]:
    """`Annotated[T, a, b]` -> `(T, (a, b))`, recursively flattened."""
    markers: list[object] = []
    while get_origin(annotation) is Annotated:
        args = get_args(annotation)
        annotation = args[0]
        markers = list(args[1:]) + markers
    return annotation, tuple(markers)


def unwrap_optional(annotation: object) -> tuple[object, bool]:
    """`T | None` -> `(T, True)`. A wider union is returned unchanged."""
    origin = get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) < len(get_args(annotation)):
            if len(args) == 1:
                return args[0], True
            return functools.reduce(operator.or_, args), True
    return annotation, False


def is_struct(tp: object) -> bool:
    return isinstance(tp, type) and issubclass(tp, msgspec.Struct)


def _annotations(tp: type) -> list[tuple[str, object, object]]:
    """(name, annotation, default) for a struct or dataclass, Annotated preserved."""
    if is_struct(tp):
        return [
            (f.name, f.type, MISSING if f.required else f.default)
            for f in msgspec.structs.fields(typing.cast(type[msgspec.Struct], tp))
        ]
    if is_dataclass(tp):
        hints = typing.get_type_hints(tp, include_extras=True)
        return [(n, hints.get(n, object), getattr(tp, n, MISSING)) for n in hints]
    return []


def walk(tp: object, *, path: str = "", _seen: frozenset[type] = frozenset()) -> Iterator[Node]:
    """Every reachable field of `tp`, descending structs, unions and containers.

    Recursion is cycle-guarded by type identity, so a self-referential schema walks once.
    The result is MEMOIZED: a type tree is fixed once its module is imported, and walking
    it per request was the author surface's single largest per-call cost (measured).
    """
    try:
        return iter(_walk_cached(tp, path, _seen))
    except TypeError:  # an unhashable annotation: correctness over the cache
        return _walk(tp, path, _seen)


@functools.lru_cache(maxsize=1024)
def _walk_cached(tp: object, path: str, seen: frozenset[type]) -> tuple[Node, ...]:
    return tuple(_walk(tp, path, seen))


def _walk(tp: object, path: str, _seen: frozenset[type]) -> Iterator[Node]:
    base, _ = strip(tp)
    base, _ = unwrap_optional(base)
    if not (isinstance(base, type) and (is_struct(base) or is_dataclass(base))):
        # A container/union can itself be the member handed down by its parent (for example
        # ``list[ImageReference | VideoReference]``). Descend it here instead of requiring
        # every caller to rebuild the same type-tree recursion. This keeps markers,
        # describe-time bounds, and hydration's declares-assets gate on one walker.
        for member, suffix in _members(base):
            yield from walk(member, path=path + suffix, _seen=_seen)
        return
    if base in _seen:
        return
    seen = _seen | {base}
    for name, annotation, default in _annotations(base):
        inner, markers = strip(annotation)
        inner, optional = unwrap_optional(inner)
        child = f"{path}.{name}" if path else name
        yield Node(child, name, inner, markers, base, default, optional)
        for member, suffix in _members(inner):
            yield from walk(member, path=child + suffix, _seen=seen)


def _members(tp: object) -> list[tuple[object, str]]:
    """Struct-bearing members of a container/union annotation, with their path suffix."""
    origin = get_origin(tp)
    if origin is Literal or (isinstance(tp, type) and issubclass(tp, enum.Enum)):
        return []
    if origin is typing.Union or origin is types.UnionType:
        return [(a, f"{{{_label(a)}}}") for a in get_args(tp) if a is not type(None)]
    if origin in (list, tuple, set, frozenset, Sequence):
        return [(a, "[]") for a in get_args(tp) if a is not Ellipsis]
    if origin is dict:
        args = get_args(tp)
        return [(args[1], "{}")] if len(args) == 2 else []
    if isinstance(tp, type) and (is_struct(tp) or is_dataclass(tp)):
        return [(tp, "")]
    return []


def _label(tp: object) -> str:
    return getattr(tp, "__name__", str(tp))


def value_domain(tp: object) -> tuple[object, ...] | None:
    """The complete admissible value set of a closed type (Enum / Literal), else None."""
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return tuple(tp)
    if get_origin(tp) is Literal:
        return get_args(tp)
    return None


def field_values(value: object) -> dict[str, Any]:
    """The declared fields of a struct instance as a plain mapping."""
    if isinstance(value, msgspec.Struct):
        return {f: getattr(value, f) for f in type(value).__struct_fields__}
    if is_dataclass(value) and not isinstance(value, type):
        return {n: getattr(value, n) for n in typing.get_type_hints(type(value))}
    return {}
