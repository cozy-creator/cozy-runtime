"""Annotation markers and the override doors: `Shape`, `ModelDefault`, `Preflight`,
and `Bound`.

Every one of these is an ORDINARY TYPED VALUE — `ModelDefault[int]` is `int` to mypy,
`Shape(...)` is a frozen record inside `Annotated`. There is no mypy plugin and no DSL
(§1.7). The doors are doors: bare `@app.entrypoint` is the common case, and the shapes
below exist only for facts derivation and measurement cannot know.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, TypeAlias, TypeVar

from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._media import KIND_MEDIA

T = TypeVar("T")
K = TypeVar("K")

#: The complete demand-axis vocabulary. A `Shape` cannot name anything else — the kwarg
#: names ARE the axes, so an underivable axis is unspellable rather than unchecked (§1.2).
AXES: tuple[str, ...] = ("width", "height", "pixels", "frames", "steps", "assets")


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
class AssetLimits:
    """Upper counts for one Assets input; compiled into its normal request schema."""

    images: int | None = None
    videos: int | None = None
    audio: int | None = None
    total: int | None = None

    def __post_init__(self) -> None:
        values = (self.images, self.videos, self.audio, self.total)
        if all(value is None for value in values):
            raise ConformanceError("AssetLimits must declare a count", code="asset_limits")
        for value in values:
            if value is not None and (type(value) is not int or value < 0):
                raise ConformanceError(
                    "asset counts must be nonnegative integers", code="asset_limits"
                )

    @property
    def counts(self) -> dict[str, int]:
        return {
            kind: count
            for kind, count in (
                ("image", self.images),
                ("video", self.videos),
                ("audio", self.audio),
            )
            if count is not None
        }


IMAGE_PREPARATION_PROFILE = "image-fit/1"


@dataclass(frozen=True, slots=True)
class ImagePreparation:
    """Opt-in useful image geometry for an Assets argument, preserving aspect ratio.

    These caps request preparation, not rejection of a larger source image. Source
    resource ceilings remain separate. Fidelity is interpreted by the package.
    """

    max_edge: int | None = None
    max_pixels: int | None = None

    def __post_init__(self) -> None:
        if self.max_edge is None and self.max_pixels is None:
            raise ConformanceError("ImagePreparation needs a cap", code="image_preparation")
        for value in (self.max_edge, self.max_pixels):
            if value is not None and (type(value) is not int or value <= 0):
                raise ConformanceError(
                    "image preparation caps must be positive integers", code="image_preparation"
                )

    def descriptor(self) -> dict[str, str | int]:
        return {
            "profile": IMAGE_PREPARATION_PROFILE,
            **{
                key: value
                for key, value in (("max_edge", self.max_edge), ("max_pixels", self.max_pixels))
                if value is not None
            },
        }


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

    Counts belong to the collection: Meta(max_length=...) or the AssetLimits argument
    annotation. Both compile centrally into the same request and kind constraints.
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
        if self.max_decoded_bytes is not None and kind not in {"image", "video", "audio"}:
            raise ConformanceError(
                f"AssetBound(max_decoded_bytes=) cannot annotate a {kind!r} asset: only "
                "ImageAsset, VideoAsset and AudioAsset have a MediaDecoder",
                fields=["max_decoded_bytes"],
            )
        allowed = KIND_MEDIA.get(kind, ())
        if kind == "tree" and self.media_types:
            raise ConformanceError(
                "Tree bounds govern content bytes; declare media types on Tree.member instead",
                fields=["media_types"],
            )
        outside = [m for m in self.media_types if allowed and m not in allowed]
        if outside:
            raise ConformanceError(
                f"AssetBound(media_types={outside!r}) is outside what a {kind!r} asset "
                f"admits ({list(allowed)}): a bound NARROWS a kind and never widens it, "
                "because a type the runtime cannot decode is not a capability",
                fields=["media_types"],
            )
