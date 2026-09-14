"""Invocation choices over immutable model artifacts; these values grant no custody."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from typing import Any

import msgspec

from ._artifacts import ModelArtifact
from ._errors import ConformanceError

_COMPONENT = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,127}\Z")
_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?\Z")
MAX_ADAPTERS = 32


def scale_string(value: float) -> str:
    """Finite shortest decimal, shared with Go's FormatFloat(g, -1, 64)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("adapter strength must be a finite number")
    # Decimal construction/as_tuple do not use the caller's decimal context.
    number = Decimal(repr(float(value)))
    sign, digits, exponent = number.as_tuple()
    assert isinstance(exponent, int)
    text = "".join(str(n) for n in digits)
    if not number:
        return "0"
    power = len(text) + exponent - 1
    text = text.rstrip("0")
    if power < -4 or power >= 6:
        fraction = "." + text[1:] if len(text) > 1 else ""
        return f"{'-' if sign else ''}{text[0]}{fraction}e{power:+03d}"
    fixed = format(number, "f")
    return fixed.rstrip("0").rstrip(".") if "." in fixed else fixed


@dataclass(frozen=True, slots=True)
class LoRA:
    """Apply a retained adapter to one explicit component at the supplied strength."""

    source: ModelArtifact
    component: str
    scale: float = 1.0
    source_component: str = "adapter"

    def __post_init__(self) -> None:
        if not isinstance(self.source, ModelArtifact):
            raise ValueError("LoRA source must be a retained ModelArtifact")
        if any(
            not isinstance(value, str) or not _COMPONENT.fullmatch(value)
            for value in (self.component, self.source_component)
        ):
            raise ValueError("LoRA requires exact source and target component names")
        scale_string(self.scale)


@dataclass(frozen=True, slots=True)
class AdaptedModel:
    """A base artifact and ordered invocation adapters; never a new checkpoint."""

    base: ModelArtifact
    adapters: tuple[LoRA, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base, ModelArtifact) or not isinstance(self.adapters, tuple):
            raise ValueError("model binding requires an artifact and immutable adapter tuple")
        if not 0 < len(self.adapters) <= MAX_ADAPTERS or any(
            not isinstance(row, LoRA) for row in self.adapters
        ):
            raise ValueError(f"model binding needs between 1 and {MAX_ADAPTERS} LoRAs")


def with_adapters(base: ModelArtifact, adapters: Sequence[LoRA]) -> ModelArtifact | AdaptedModel:
    """Choose adapters for a Model argument; an empty stack preserves the base value."""
    rows = tuple(adapters)
    if not isinstance(base, ModelArtifact):
        raise ValueError("base must be a retained ModelArtifact")
    return AdaptedModel(base, rows) if rows else base


class _LoRAWire(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    source: ModelArtifact
    component: str
    source_component: str
    scale: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.scale, str)
            or len(self.scale) > 32
            or not _DECIMAL.fullmatch(self.scale)
            or scale_string(float(self.scale)) != self.scale
        ):
            raise ValueError("adapter strength must use the canonical finite decimal")
        LoRA(self.source, self.component, float(self.scale), self.source_component)


class _BindingWire(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    tag="cozy.model-binding/1",
    tag_field="format",
):
    base: ModelArtifact
    adapters: tuple[_LoRAWire, ...]

    def __post_init__(self) -> None:
        if not 0 < len(self.adapters) <= MAX_ADAPTERS:
            raise ValueError(f"model binding needs between 1 and {MAX_ADAPTERS} LoRAs")


def read_binding(value: object) -> tuple[ModelArtifact, tuple[_LoRAWire, ...]]:
    """Strictly decode only after the caller established an injected Model position."""
    if isinstance(value, dict) and value.get("format") == "cozy.model-binding/1":
        row = msgspec.convert(value, type=_BindingWire, strict=True)
        return row.base, row.adapters
    return msgspec.convert(value, type=ModelArtifact, strict=True), ()


def lower_bindings(
    arguments: Mapping[str, object],
    paths: tuple[tuple[str, ...], ...],
) -> tuple[dict[str, Any], dict[tuple[str, ...], dict[str, Any]]]:
    """Validate the original call type using bases, then restore exact wrapper bytes."""
    selected: dict[tuple[str, ...], dict[str, Any]] = {}

    def walk(value: Any, path: tuple[str, ...] = ()) -> Any:
        if len(path) > 32:
            raise ConformanceError("package call nesting exceeds its bound", code="child_arguments")
        if isinstance(value, AdaptedModel):
            if path not in paths:
                raise ConformanceError(
                    "adapter binding is only valid at a serving Model argument",
                    code="adapter_slot",
                )
            wire = _BindingWire(
                value.base,
                tuple(
                    _LoRAWire(
                        row.source, row.component, row.source_component, scale_string(row.scale)
                    )
                    for row in value.adapters
                ),
            )
            selected[path] = msgspec.to_builtins(wire)
            return value.base
        if isinstance(value, dict):
            return {key: walk(item, (*path, key)) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [walk(item, (*path, str(index))) for index, item in enumerate(value)]
        if isinstance(value, msgspec.Struct):
            for field in msgspec.structs.fields(type(value)):
                walk(getattr(value, field.name), (*path, field.encode_name))
        elif is_dataclass(value) and not isinstance(value, type):
            for item in fields(value):
                walk(getattr(value, item.name), (*path, item.name))
        return value

    return walk(dict(arguments)), selected


def restore_bindings(
    payload: dict[str, Any],
    selected: Mapping[tuple[str, ...], dict[str, Any]],
) -> dict[str, Any]:
    for path, value in selected.items():
        parent = payload
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value
    return payload
