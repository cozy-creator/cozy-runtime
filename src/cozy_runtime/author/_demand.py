"""Launch demand: ONE normalized RequestFeatures record per entrypoint.

Nothing is authored for the common case — the payload's declared shape axes normalize to
named scalar features BEFORE admission, and proven-max tables per (variant, plan, device,
cell) do the rest. Every launch family's demand-bearing axes are FINITE products, so no
fitted model exists at launch (§1.2).

DEFERRED as one package (cr-018): the conservative-upper fitting pipeline, `DemandBasis`,
the `basis(...)` override DSL with its typed-operator algebra and vector corpus, and the
authored TIME-formula AST. They enter together when the door's trigger fires — the first
axis that cannot be bucketed, or a measured envelope breach the tables cannot express.
Nothing below anticipates them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass

import msgspec

from cozy_runtime.author._assets import Asset
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._markers import AXES, Shape
from cozy_runtime.author._signature import AssetInputValue
from cozy_runtime.author._walker import Node, field_values, is_struct, value_domain, walk

#: Axis-named request fields normalize with zero author involvement. A field named for an
#: axis IS the declaration; a `Shape` annotation exists only for author-only knowledge.
AXIS_FIELDS: Mapping[str, str] = {
    "width": "width",
    "height": "height",
    "frames": "frames",
    "num_frames": "frames",
    "steps": "steps",
    "num_inference_steps": "steps",
}


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One derivable axis and the field it derives from. Build-time; no values here."""

    axis: str
    field: str
    source: str
    """`field` (axis-named) or `shape` (an author-only preset table)."""


def feature_specs(payload_type: object) -> tuple[FeatureSpec, ...]:
    """Derive the axes this payload yields. REFUSES an underivable axis at BUILD (§1.2)."""
    specs: list[FeatureSpec] = []
    for parent, node in _declaring(payload_type):
        name = f"{parent}.{node.name}" if parent else node.name
        if (axis := AXIS_FIELDS.get(node.name)) is not None:
            specs.append(FeatureSpec(axis, name, "field"))
        shape = node.marker(Shape)
        if shape is None:
            continue
        _check_coverage(node, shape)
        specs += [FeatureSpec(axis, name, "shape") for axis in shape.axes()]
    for axis in AXES:
        fields = [s.field for s in specs if s.axis == axis]
        if len(fields) > 1:
            raise ConformanceError(
                f"axis {axis!r} is declared by {', '.join(fields)}: a request has one value "
                "per axis, so declare it once",
                code="duplicate_axis",
                fields=fields,
            )
    if any(s.axis in ("width", "height") for s in specs) and not any(
        s.axis == "pixels" for s in specs
    ):
        specs.append(FeatureSpec("pixels", "width*height", "derived"))
    for spec in specs:
        if spec.axis not in AXES:
            raise ConformanceError(
                f"{spec.field} declares axis {spec.axis!r}, which nothing normalizes: "
                f"the vocabulary is {', '.join(AXES)}",
                code="underivable_axis",
                fields=[spec.field],
            )
    return tuple(specs)


def _top_level(payload_type: object) -> list[Node]:
    return [node for node in walk(payload_type) if "." not in node.path]


def _declaring(payload_type: object) -> list[tuple[str, Node]]:
    """The fields an axis may be declared on, each with its parent ("" for the request's own):
    the request's fields and those of each struct-typed field. An invocable's request is its
    keyword parameters, so its `payload: T` holds the shape one level down (run 1560: every
    H3 segment shared one cell and a 345-frame call ran on a 158-frame peak)."""
    own = _top_level(payload_type)
    return [("", node) for node in own] + [
        (node.name, child)
        for node in own
        if isinstance(node.annotation, type)
        and (is_struct(node.annotation) or is_dataclass(node.annotation))
        for child in _top_level(node.annotation)
    ]


#: The widest bounded-int field a preset table may cover; wider is not a preset.
MAX_INT_DOMAIN = 4096


def shape_domain(node: Node) -> tuple[object, ...] | None:
    """A closed Enum or Literal, or an int bounded on both sides by its `msgspec.Meta`
    (H3's `duration_s`, 5..15 whole seconds): the values a preset table must cover."""
    domain = value_domain(node.annotation)
    if domain is not None or node.annotation is not int:
        return domain
    low: int | None = None
    high: int | None = None
    for meta in (m for m in node.markers if isinstance(m, msgspec.Meta)):
        if isinstance(meta.ge, int):
            low = meta.ge
        if isinstance(meta.gt, int):
            low = meta.gt + 1
        if isinstance(meta.le, int):
            high = meta.le
        if isinstance(meta.lt, int):
            high = meta.lt - 1
    if low is None or high is None or not 0 <= high - low < MAX_INT_DOMAIN:
        return None
    return tuple(range(low, high + 1))


def _check_coverage(node: Node, shape: Shape[object]) -> None:
    """A preset table must cover its field's COMPLETE value domain, or the axis is
    underivable for some admissible request — which is a build refusal, not a KeyError
    at admission."""
    domain = shape_domain(node)
    if domain is None:
        raise ConformanceError(
            f"Shape on {node.path}: {node.annotation!r} is an open value domain, so the "
            "preset table cannot cover it — annotate a closed Enum or Literal field, or an "
            "int bounded by msgspec.Meta(ge=..., le=...)",
            code="underivable_axis",
            fields=[node.path],
        )
    missing = [repr(v) for v in domain if v not in shape.table_keys()]
    if missing:
        raise ConformanceError(
            f"Shape on {node.path} has no entry for {', '.join(missing)}: the axis is "
            "underivable for those requests",
            code="underivable_axis",
            fields=[node.path],
        )


@dataclass(frozen=True, slots=True)
class RequestFeatures:
    """The normalized per-request demand record admission keys on."""

    values: Mapping[str, int]
    digest: str


def normalize(payload: object) -> RequestFeatures:
    """Request-time normalization. Same rules as `feature_specs`, applied to VALUES."""
    values: dict[str, int] = {}
    own = field_values(payload)
    for parent, node in _declaring(type(payload)):
        value = (field_values(own.get(parent)) if parent else own).get(node.name)
        if (axis := AXIS_FIELDS.get(node.name)) is not None and isinstance(value, int):
            values[axis] = value
        shape = node.marker(Shape)
        if shape is not None and value is not None:
            values.update(shape.resolve(value))
        if isinstance(value, (list, tuple)) and any(
            isinstance(item, (Asset, AssetInputValue)) for item in value
        ):
            # Each listed input (a reference image, a clip) adds tokens and memory, so
            # their count is a demand axis: 1 and 10 references are different cells. An
            # `Assets` parameter's occurrences arrive as `AssetInputValue` rows.
            values["assets"] = values.get("assets", 0) + len(value)
    if "pixels" not in values and "width" in values and "height" in values:
        values["pixels"] = values["width"] * values["height"]
    canonical = json.dumps(dict(sorted(values.items())), separators=(",", ":"))
    return RequestFeatures(
        dict(sorted(values.items())),
        "blake2b:" + hashlib.blake2b(canonical.encode(), digest_size=16).hexdigest(),
    )
