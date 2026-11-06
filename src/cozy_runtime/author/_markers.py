"""Annotation markers and the override doors: `Shape`, `ModelDefault`, `Preflight`,
and `Bound`.

Every one of these is an ORDINARY TYPED VALUE — `ModelDefault[int]` is `int` to mypy,
`Shape(...)` is a frozen record inside `Annotated`. There is no mypy plugin and no DSL
(§1.7). The doors are doors: bare `@app.entrypoint` is the common case, and the shapes
below exist only for facts derivation and measurement cannot know.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Literal, TypeAlias, TypeVar

import msgspec

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
class FileLimits:
    """Upper counts for one Files input; compiled into its normal request schema."""

    images: int | None = None
    videos: int | None = None
    audio: int | None = None
    total: int | None = None

    def __post_init__(self) -> None:
        values = (self.images, self.videos, self.audio, self.total)
        if all(value is None for value in values):
            raise ConformanceError("FileLimits must declare a count", code="asset_limits")
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
VIDEO_PREPARATION_PROFILE = "video-trim/1"
VIDEO_FIT_PROFILE = "video-fit/1"
AUDIO_PREPARATION_PROFILE = "audio-trim/1"


@dataclass(frozen=True, slots=True)
class ImagePreparation:
    """Opt-in useful image geometry for an image field or Files, preserving aspect ratio.

    These caps request preparation, not rejection of a larger source image. Source
    resource ceilings remain separate. Fidelity is interpreted by the package.
    """

    max_edge: int | None = None
    max_pixels: int | None = None

    kind = "image"

    def __post_init__(self) -> None:
        if self.max_edge is None and self.max_pixels is None:
            raise ConformanceError("ImagePreparation needs a cap", code="image_preparation")
        for value in (self.max_edge, self.max_pixels):
            if value is not None and (type(value) is not int or value <= 0):
                raise ConformanceError(
                    "image preparation caps must be positive integers", code="image_preparation"
                )

    def descriptor(self) -> dict[str, str | int | float]:
        return {
            "profile": IMAGE_PREPARATION_PROFILE,
            **{
                key: value
                for key, value in (("max_edge", self.max_edge), ("max_pixels", self.max_pixels))
                if value is not None
            },
        }


Excess = Literal["refuse", "truncate"]


@dataclass(frozen=True, slots=True)
class _Trim:
    min_seconds: int | float | None = None
    max_seconds: int | float | None = None
    excess: Excess = "truncate"

    kind = ""
    profile = ""

    def __post_init__(self) -> None:
        name = type(self).__name__
        if self.min_seconds is None and self.max_seconds is None and not self.fit():
            raise ConformanceError(f"{name} needs min_seconds or max_seconds", code="preparation")
        for value in (self.min_seconds, self.max_seconds):
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            ):
                raise ConformanceError(
                    f"{name} seconds must be positive finite numbers", code="preparation"
                )
        if (
            self.min_seconds is not None
            and self.max_seconds is not None
            and self.min_seconds > self.max_seconds
        ):
            raise ConformanceError(f"{name} min_seconds exceeds max_seconds", code="preparation")
        if self.excess not in ("refuse", "truncate"):
            raise ConformanceError(f"{name} excess is refuse or truncate", code="preparation")
        if self.excess == "truncate" and self.max_seconds is None:
            raise ConformanceError(f"{name} truncate needs max_seconds", code="preparation")

    def fit(self) -> dict[str, int | float | bool | str]:
        """The client-hint caps it declares besides its duration policy."""
        return {}

    @property
    def minimum(self) -> Fraction | None:
        return None if self.min_seconds is None else seconds(self.min_seconds)

    @property
    def maximum(self) -> Fraction | None:
        return None if self.max_seconds is None else seconds(self.max_seconds)

    def descriptor(self) -> dict[str, str | int | float]:
        return {
            "profile": self.profile,
            "excess": self.excess,
            **{
                key: value
                for key, value in (
                    ("min_seconds", self.min_seconds),
                    ("max_seconds", self.max_seconds),
                )
                if value is not None
            },
        }


@dataclass(frozen=True, slots=True)
class VideoPreparation(_Trim):
    """Declared duration policy and fit for a video field or the videos of an Files input.

    Shorter than `min_seconds` refuses the request before the handler runs. Longer than
    `max_seconds` ends every read and stream of that input there and reports an `adjusted`
    row (`excess="truncate"`, the default), or refuses with `excess="refuse"`.

    The fit is the most of the input the package reads, never a refusal: `max_short_edge`,
    `max_long_edge` and `max_pixels` bound a frame, `max_fps` the frame rate, and `audio` is
    False when the package never reads the input's audio, or the name of a boolean member of
    the same record that says whether it does. A client may reduce a larger video to the fit
    before it uploads it, so the package's own sizing must give the same result for an input
    already within it. Declaring any fit writes `video-fit/1`; a duration alone, `video-trim/1`.
    """

    max_short_edge: int | None = None
    max_long_edge: int | None = None
    max_pixels: int | None = None
    max_fps: int | float | None = None
    audio: bool | str | None = None

    kind = "video"
    profile = VIDEO_PREPARATION_PROFILE

    def __post_init__(self) -> None:
        _Trim.__post_init__(self)
        for key in ("max_short_edge", "max_long_edge", "max_pixels"):
            value = getattr(self, key)
            if value is not None and (type(value) is not int or value <= 0):
                raise ConformanceError(
                    f"VideoPreparation {key} is a positive integer", code="preparation"
                )
        fps = self.max_fps
        if fps is not None and (
            type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
        ):
            raise ConformanceError(
                "VideoPreparation max_fps is a positive number", code="preparation"
            )
        audio = self.audio
        if (
            audio is not None
            and type(audio) is not bool
            and not (isinstance(audio, str) and audio.isidentifier())
        ):
            raise ConformanceError(
                "VideoPreparation audio is a bool or the name of a boolean member",
                code="preparation",
            )

    def fit(self) -> dict[str, int | float | bool | str]:
        caps = (
            ("max_short_edge", self.max_short_edge),
            ("max_long_edge", self.max_long_edge),
            ("max_pixels", self.max_pixels),
            ("max_fps", self.max_fps),
            ("audio", self.audio),
        )
        return {key: value for key, value in caps if value is not None}

    def descriptor(self) -> dict[str, str | int | float]:
        fit = self.fit()
        trim = _Trim.descriptor(self)
        return {**trim, "profile": VIDEO_FIT_PROFILE, **fit} if fit else trim


@dataclass(frozen=True, slots=True)
class AudioPreparation(_Trim):
    """The `VideoPreparation` duration policy for audio inputs."""

    kind = "audio"
    profile = AUDIO_PREPARATION_PROFILE


Preparation = ImagePreparation | VideoPreparation | AudioPreparation
PREPARATIONS = (ImagePreparation, VideoPreparation, AudioPreparation)


def seconds(value: object) -> Fraction:
    """An exact, positive duration from an int, float or Fraction number of seconds."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError("seconds must be an int, float or Fraction")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("seconds must be finite")
    exact = value if isinstance(value, Fraction) else Fraction(str(value))
    if exact <= 0:
        raise ValueError("seconds must be positive")
    return exact


@dataclass(frozen=True, slots=True)
class FileBound:
    """cr-012 — the TYPED bounds on ONE input-asset field.

    An input file is a typed request field, so its limits belong on the field and not in
    a deployment's config or a downloader's constant:

    * `max_bytes` — the largest this field admits, enforced BEFORE the entrypoint sees the
      value and (as a deployment-wide cap) before acceptance. A field with no bound takes
      the deployment's cap, so "unbounded" is not spellable.
    * `media_types` — a NARROWING of the kind's own set (`_media.KIND_MEDIA`). A package
      that decodes only PNG says so; one that lists a type its kind does not admit is a
      conformance error, because widening a kind would promise a decoder nothing has.
    * `max_decoded_bytes` — the largest immutable RGB8 + float32 PCM value this field may
      expand to. It is bound into the hydrated file by the runtime and can only narrow the
      runtime's hard frame/pixel/channel/sample ceilings. It accounts for the value retained
      and returned to the handler, not codec working memory or peak process RSS. Request data
      cannot set it.

    Counts belong to the collection: Meta(max_length=...) or the FileLimits argument
    annotation. Both compile centrally into the same request and kind constraints.
    """

    max_bytes: int | None = None
    media_types: tuple[str, ...] = ()
    max_decoded_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.max_bytes is None and not self.media_types and self.max_decoded_bytes is None:
            raise ConformanceError(
                "FileBound() declares nothing: give max_bytes, media_types or "
                "max_decoded_bytes, or drop the annotation and take the deployment's cap"
            )
        for name in ("max_bytes", "max_decoded_bytes"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ConformanceError(f"FileBound({name}=) must be positive", fields=[name])
        for media_type in self.media_types:
            if media_type != media_type.strip().lower() or "/" not in media_type:
                raise ConformanceError(
                    f"FileBound(media_types=) takes lowercase `type/subtype`, got {media_type!r}",
                    fields=["media_types"],
                )

    def check_kind(self, kind: str) -> None:
        """A declared type outside the KIND's own set is a conformance error at describe."""
        if self.max_decoded_bytes is not None and kind not in {"image", "video", "audio"}:
            raise ConformanceError(
                f"FileBound(max_decoded_bytes=) cannot annotate a {kind!r} asset: only "
                "ImageFile, VideoFile and AudioFile are read as media",
                fields=["max_decoded_bytes"],
            )
        allowed = KIND_MEDIA.get(kind, ())
        if kind == "tree" and self.media_types:
            raise ConformanceError(
                "Tree bounds govern content bytes; a Tree takes no media types",
                fields=["media_types"],
            )
        outside = [m for m in self.media_types if allowed and m not in allowed]
        if outside:
            raise ConformanceError(
                f"FileBound(media_types={outside!r}) is outside what a {kind!r} file "
                f"admits ({list(allowed)}): a bound NARROWS a kind and never widens it, "
                "because a type the runtime cannot decode is not a capability",
                fields=["media_types"],
            )


# ---------------------------------------------------------------- a tool's box
#
# Each returns a `msgspec.Meta`, so they stack in one `Annotated` and the binder and `fit` hold
# a value to the same bounds the box draws. Their facts ride the interface as `extra`.


def _meta(extra: Mapping[str, object] | None = None, **constraints: object) -> msgspec.Meta:
    said = {key: value for key, value in (extra or {}).items() if value not in (None, "")}
    return msgspec.Meta(**constraints, **({"extra": said} if said else {}))  # type: ignore[arg-type]


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConformanceError(f"{what} is a finite number", code="tool_marker")
    return value


def Range(min: float, max: float, *, step: float | None = None, unit: str = "") -> msgspec.Meta:
    """A number's slider: `min` to `max`, on `step`, counting `unit` (`"s"`, `"MP"`)."""
    low, high = _number(min, "Range min"), _number(max, "Range max")
    if low >= high or (step is not None and _number(step, "Range step") <= 0):
        raise ConformanceError(
            "Range(min, max, step=) needs min < max and a positive step", code="tool_marker"
        )
    return _meta({"step": step, "unit": unit}, ge=low, le=high)


def Count(min: int = 0, max: int | None = None) -> msgspec.Meta:
    """How few and how many rows a list group holds."""
    if (
        type(min) is not int
        or min < 0
        or (max is not None and (type(max) is not int or max < min or max < 1))
    ):
        raise ConformanceError(
            "Count(min, max) takes whole numbers, min <= max", code="tool_marker"
        )
    return msgspec.Meta(min_length=min or None, max_length=max)


def Total() -> msgspec.Meta:
    """On a number in a list group: the box shows its sum (H3's beat durations)."""
    return _meta({"total": True})


def Ratios(*allowed: str, min: float | None = None, max: float | None = None) -> msgspec.Meta:
    """The shapes an `AspectRatio` takes: these ratios, or any from `min` to `max` (W/H)."""
    for ratio in allowed:
        sides = re.split(r"\s*[:/xX\u00d7]\s*", ratio) if isinstance(ratio, str) else []
        try:
            valid = len(sides) == 2 and all(Fraction(side) > 0 for side in sides)
        except (ValueError, ZeroDivisionError):
            valid = False
        if not valid:
            raise ConformanceError(f"Ratios takes W:H, not {ratio!r}", code="tool_marker")
    bounds = {
        k: _number(v, f"Ratios {k}") for k, v in (("min", min), ("max", max)) if v is not None
    }
    if not allowed and not bounds:
        raise ConformanceError("Ratios() names ratios, or a min and max", code="tool_marker")
    return _meta({"ratios": {**({"allowed": list(allowed)} if allowed else {}), **bounds}})


def Snap(every: int, offset: int = 0, first: int = 0) -> msgspec.Meta:
    """A pickup's `out_frame`: it lands on `every * k + offset`, from `first` (H3's `17k + 5`)."""
    if not all(type(n) is int and n >= 0 for n in (every, offset, first)) or every < 1:
        raise ConformanceError(
            "Snap(every, offset, first) takes whole frames, every >= 1", code="tool_marker"
        )
    return _meta({"snap": {"every": every, "offset": offset, "first": first}})


def Spans(rows: str) -> msgspec.Meta:
    """A video slot whose length is the timeline of the list group `rows`: the box shows both."""
    if not isinstance(rows, str) or not rows.isidentifier():
        raise ConformanceError("Spans names a list group of the same struct", code="tool_marker")
    return _meta({"spans": rows})


def Label(text: str, options: Mapping[object, str | tuple[str, str]] | None = None) -> msgspec.Meta:
    """A field's words on the box, and a choice's own words for its options: `{"768p": "Off"}`, or
    `{"swap": ("Character swap", "Puts the reference in place of the person")}`, a dropdown that
    shows each description."""
    words: dict[str, object] = {}
    for value, said in (options or {}).items():
        if isinstance(said, tuple) and len(said) == 2 and all(isinstance(w, str) for w in said):
            words[str(value)] = {"words": said[0], "description": said[1]}
        elif isinstance(said, str):
            words[str(value)] = said
        else:
            raise ConformanceError(
                "Label options give words, or (words, description)", code="tool_marker"
            )
    if not isinstance(text, str):
        raise ConformanceError("Label(text, options) is words", code="tool_marker")
    return _meta({**({"label": text} if text else {}), **({"options": words} if words else {})})


def Placeholder(text: str) -> msgspec.Meta:
    """What an empty text reads."""
    if not isinstance(text, str):
        raise ConformanceError("Placeholder(text) is words", code="tool_marker")
    return _meta({"placeholder": text})


def Advanced() -> msgspec.Meta:
    """Under More, in the collapsed Advanced section: expert settings such as steps."""
    return _meta({"advanced": True})
