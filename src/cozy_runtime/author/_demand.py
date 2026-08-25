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
from dataclasses import dataclass

from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._markers import AXES, Shape
from cozy_runtime.author._walker import Node, field_values, value_domain, walk

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
    for node in _top_level(payload_type):
        if (axis := AXIS_FIELDS.get(node.name)) is not None:
            specs.append(FeatureSpec(axis, node.name, "field"))
        shape = node.marker(Shape)
        if shape is None:
            continue
        _check_coverage(node, shape)
        specs += [FeatureSpec(axis, node.name, "shape") for axis in shape.axes()]
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


def _check_coverage(node: Node, shape: Shape[object]) -> None:
    """A preset table must cover its field's COMPLETE value domain, or the axis is
    underivable for some admissible request — which is a build refusal, not a KeyError
    at admission."""
    domain = value_domain(node.annotation)
    if domain is None:
        raise ConformanceError(
            f"Shape on {node.path}: {node.annotation!r} is an open value domain, so the "
            "preset table cannot cover it — annotate a closed Enum or Literal field",
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
    fields = field_values(payload)
    for node in _top_level(type(payload)):
        value = fields.get(node.name)
        if (axis := AXIS_FIELDS.get(node.name)) is not None and isinstance(value, int):
            values[axis] = value
        shape = node.marker(Shape)
        if shape is not None and value is not None:
            values.update(shape.resolve(value))
    if "pixels" not in values and "width" in values and "height" in values:
        values["pixels"] = values["width"] * values["height"]
    canonical = json.dumps(dict(sorted(values.items())), separators=(",", ":"))
    return RequestFeatures(
        dict(sorted(values.items())),
        "blake2b:" + hashlib.blake2b(canonical.encode(), digest_size=16).hexdigest(),
    )
