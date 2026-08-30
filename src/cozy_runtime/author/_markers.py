"""Annotation markers and the override doors: `Shape`, `ModelDefault`, `Preflight`,
`Bound`, `Resources`.

Every one of these is an ORDINARY TYPED VALUE — `ModelDefault[int]` is `int` to mypy,
`Shape(...)` is a frozen record inside `Annotated`. There is no mypy plugin and no DSL
(§1.7). The doors are doors: bare `@app.entrypoint` is the common case, and the shapes
below exist only for facts derivation and measurement cannot know.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, TypeAlias, TypeVar

from cozy_runtime.author._errors import ConformanceError

T = TypeVar("T")
K = TypeVar("K")

#: The complete demand-axis vocabulary. A `Shape` cannot name anything else — the kwarg
#: names ARE the axes, so an underivable axis is unspellable rather than unchecked (§1.2).
AXES: tuple[str, ...] = ("width", "height", "pixels", "frames", "steps")


class _Marker:
    """Base of the zero-payload annotation markers, so the walker matches by type."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


class _ModelDefaultMarker(_Marker):
    __slots__ = ()


class _PreflightMarker(_Marker):
    __slots__ = ()


MODEL_DEFAULT = _ModelDefaultMarker()
PREFLIGHT = _PreflightMarker()

#: A concrete request field whose value may be omitted on the wire but is CONCRETE before
#: the handler runs (§1.2). The field owns name, type, bounds and code fallback; the
#: handler sees `T`, never `Optional[T]`.
ModelDefault: TypeAlias = Annotated[T, MODEL_DEFAULT]

#: The frozen record a typed preflight hook produced for THIS request — the same value
#: admission hashed, never recomputed (§1.2).
Preflight: TypeAlias = Annotated[T, PREFLIGHT]


@dataclass(frozen=True, slots=True)
class Shape[K]:
    """Author-only knowledge: a preset value's mapping to demand axes.

    Only needed when the axis is NOT already an axis-named field. Canonical form is
    axis-named kwargs — `Shape(pixels=_BUCKETS)` — where the kwarg names the derived axis
    and the value is the preset -> value table the handler itself indexes (§1.2).
    """

    pixels: Mapping[K, tuple[int, int]] | None = None
    frames: Mapping[K, int] | None = None

    def __post_init__(self) -> None:
        if self.pixels is None and self.frames is None:
            raise ConformanceError(
                "Shape() declares no axis: name one of "
                + ", ".join(f"{a}=" for a in ("pixels", "frames"))
            )

    def axes(self) -> tuple[str, ...]:
        return (("width", "height", "pixels") if self.pixels is not None else ()) + (
            ("frames",) if self.frames is not None else ()
        )

    def table_keys(self) -> frozenset[K]:
        return frozenset(self.pixels or {}) | frozenset(self.frames or {})

    def resolve(self, key: K) -> dict[str, int]:
        out: dict[str, int] = {}
        if self.pixels is not None:
            width, height = self.pixels[key]
            out.update(width=width, height=height, pixels=width * height)
        if self.frames is not None:
            out["frames"] = self.frames[key]
        return out


@dataclass(frozen=True, slots=True)
class Bound:
    """The ONE authored demand number surviving at launch: a hard safety bound.

    Labeled `bound` forever and NEVER rendered as measured demand (§1.2). Measured rows
    live outside the release keyed by exact (profile, plan, runtime, device class); an
    package is not generally routable until its first measured row exists. `reason` is
    mandatory falsifier-first discipline: state the physical fact measurement cannot
    safely discover.
    """

    reason: str
    vram_bytes: int | None = None
    host_ram_bytes: int | None = None

    provenance = "bound"

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ConformanceError("Bound(reason=...) is mandatory: name the physical fact")
        if self.vram_bytes is None and self.host_ram_bytes is None:
            raise ConformanceError("Bound() declares no number: give vram_bytes or host_ram_bytes")
        for name in ("vram_bytes", "host_ram_bytes"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ConformanceError(f"Bound({name}=) must be positive", fields=[name])


@dataclass(frozen=True, slots=True)
class AssetBound:
    """cr-012 — the TYPED bounds on ONE input-asset field.

    An input asset is a typed request field, so its limits belong on the field and not in
    a deployment's config or a downloader's constant:

    * `max_bytes` — the largest this field admits, enforced BEFORE the entrypoint sees the
      value and (as a deployment-wide cap) before acceptance. A field with no bound takes
      the deployment's cap, so "unbounded" is not spellable.
    * `media_types` — a NARROWING of the kind's own set (`_media.KIND_MEDIA`). A package
      that decodes only PNG says so; one that lists a type its kind does not admit is a
      conformance error, because widening a kind would promise a decoder nothing has.
    * `max_decoded_bytes` — the largest immutable RGB8 + float32 PCM value this field may
      expand to. It is bound into the hydrated asset by the runtime and can only narrow the
      runtime's hard frame/pixel/channel/sample ceilings. It accounts for the value retained
      and returned to the handler, not codec working memory or peak process RSS. Request data
      cannot set it.

    COUNT bounds are not here on purpose: `Annotated[list[ImageAsset], Meta(max_length=9)]`
    is msgspec's own, it is already in the descriptor's constraint vocabulary, and a second
    spelling of a limit that already validates would be a second contract.
    """

    max_bytes: int | None = None
    media_types: tuple[str, ...] = ()
    max_decoded_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.max_bytes is None and not self.media_types and self.max_decoded_bytes is None:
            raise ConformanceError(
                "AssetBound() declares nothing: give max_bytes, media_types or "
                "max_decoded_bytes, or drop the annotation and take the deployment's cap"
            )
        for name in ("max_bytes", "max_decoded_bytes"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ConformanceError(f"AssetBound({name}=) must be positive", fields=[name])
        for media_type in self.media_types:
            if media_type != media_type.strip().lower() or "/" not in media_type:
                raise ConformanceError(
                    f"AssetBound(media_types=) takes lowercase `type/subtype`, got {media_type!r}",
                    fields=["media_types"],
                )

    def check_kind(self, kind: str) -> None:
        """A declared type outside the KIND's own set is a conformance error at describe."""
        from cozy_runtime.author._media import KIND_MEDIA

        if self.max_decoded_bytes is not None and kind not in {"image", "video", "audio"}:
            raise ConformanceError(
                f"AssetBound(max_decoded_bytes=) cannot annotate a {kind!r} asset: only "
                "ImageAsset, VideoAsset and AudioAsset have a MediaDecoder",
                fields=["max_decoded_bytes"],
            )
        allowed = KIND_MEDIA.get(kind, ())
        outside = [m for m in self.media_types if allowed and m not in allowed]
        if outside:
            raise ConformanceError(
                f"AssetBound(media_types={outside!r}) is outside what a {kind!r} asset "
                f"admits ({list(allowed)}): a bound NARROWS a kind and never widens it, "
                "because a type the runtime cannot decode is not a capability",
                fields=["media_types"],
            )


_REQUIRES_TOKEN = re.compile(r"^(sm\d{2,3}\+?|vram\d+g|ram\d+g|cuda\d+\.\d+\+?)$")


@dataclass(frozen=True, slots=True)
class Resources:
    """The override door for STRUCTURAL FLOORS ONLY (§1.2).

    `gpu` is derived from the signature and is not spellable here. `vcpus` and host RAM
    are MEASURED per (lane, plan) fleet-side. There is deliberately no `recommended`
    form: "runs better on newer silicon" is a measured ranking fact, not a declaration —
    and because this is a slotted frozen dataclass, `Resources(recommended=...)` is a
    TypeError, not a review comment.
    """

    gpu_count: int | None = None
    placement: str | None = None
    requires: str | None = None
    """Compact incapability grammar the probe vocabulary cannot express: `sm89+, vram24g`."""

    def __post_init__(self) -> None:
        if self.gpu_count is not None and self.gpu_count < 1:
            raise ConformanceError("Resources(gpu_count=) must be >= 1", fields=["gpu_count"])
        if self.placement is not None and self.placement not in ("single_node", "any"):
            raise ConformanceError(
                f"unknown placement class {self.placement!r}: single_node | any",
                fields=["placement"],
            )
        for token in (t.strip() for t in (self.requires or "").split(",") if t.strip()):
            if not _REQUIRES_TOKEN.match(token):
                raise ConformanceError(
                    f"requires={token!r} is outside the grammar: sm89+, vram24g, ram64g, cuda12.8+",
                    fields=["requires"],
                )
