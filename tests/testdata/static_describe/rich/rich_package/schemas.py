"""Request and result schemas: tagged unions, bounds, enums, ModelDefault, inheritance."""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Annotated, Literal

import msgspec

from cozy_runtime.author import (
    AssetBound,
    AudioAsset,
    FileAsset,
    ImageAsset,
    ModelDefault,
    Shape,
    Tree,
    VideoAsset,
    data_values,
)
from cozy_runtime.derive import QuantizeResult

from . import bounds
from .bounds import MAX_EDGE, Prompt


class AspectRatio(Enum):
    SQUARE = "1:1"
    WIDE = "16:9"


class Megapixels(IntEnum):
    MP1 = 1
    MP2 = 2


_TIERS = {tier: (1024 * tier.value, 1024) for tier in Megapixels}


class ImageReference(msgspec.Struct, tag="image", tag_field="type", forbid_unknown_fields=True):
    image: Annotated[ImageAsset, bounds.IMAGE_BOUND]


class AudioReference(msgspec.Struct, tag="audio", tag_field="type", forbid_unknown_fields=True):
    audio: Annotated[AudioAsset, bounds.AUDIO_BOUND]


Reference = ImageReference | AudioReference
#: The admitted values are the committed plan's, read as data and never computed (cr-115).
STEPS = data_values(__file__, "plan.json", "schedules", "transformer_evaluations")
LANES = data_values(__file__, "plan.json", "lanes")
Steps = Annotated[Literal[STEPS], msgspec.Meta(description="denoise steps")]  # type: ignore[valid-type]


class Common(msgspec.Struct, kw_only=True):
    seed: int | None = None
    mute: bool = False


class RenderRequest(Common, forbid_unknown_fields=True):
    prompt: Prompt
    references: Annotated[list[Reference], msgspec.Meta(min_length=1, max_length=12)]
    first_frame: Annotated[ImageAsset | None, bounds.IMAGE_BOUND] = None
    aspect_ratio: AspectRatio = AspectRatio.SQUARE
    megapixels: Annotated[Megapixels, Shape(pixels=_TIERS)] = Megapixels.MP1
    steps: Annotated[ModelDefault[int], msgspec.Meta(ge=8, le=50)] = 30
    guidance: Annotated[ModelDefault[float], msgspec.Meta(ge=1.0, le=10.0)] = 4.5
    short_edge: Annotated[int, msgspec.Meta(ge=256, le=MAX_EDGE)] = MAX_EDGE
    denoise: Steps = 30
    labels: dict[str, int] = msgspec.field(default_factory=dict)
    kind: str = msgspec.field(name="kind_of", default="plain")


class Facts(msgspec.Struct, frozen=True):
    reference_count: int


class RenderResult(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(media_types=("video/mp4",))]
    frames: tuple[int, ...]
    warnings: list[str] = msgspec.field(default_factory=list)
    replay: str | None = None


class QuantizeRequest(msgspec.Struct, forbid_unknown_fields=True):
    lanes: Annotated[tuple[Literal[LANES], ...], msgspec.Meta(min_length=1)] = (  # type: ignore[valid-type]
        "fp8",
        "mxfp8",
    )


class QuantizedLanes(msgspec.Struct):
    fp8: QuantizeResult | None = None
    evidence: FileAsset | None = None


class GateRequest(msgspec.Struct, forbid_unknown_fields=True):
    media: Tree
    verdicts: list[Facts] = []


class GateResult(msgspec.Struct):
    passed: int
    reports: list[FileAsset]
