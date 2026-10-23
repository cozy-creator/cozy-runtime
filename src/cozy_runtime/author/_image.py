"""cr-219 — one request and one result for every image model.

The same names, types and meaning in every image package, so swapping one model for another never
breaks a request. Every optional knob is None by default: the model's own recommended value, which
its README and `--describe` state. A model clamps what is outside its range (a `clamp` row) and
warns for a knob it has no use for; it never refuses either. A package subclasses `ImageRequest`
only to narrow `reference_images` to its model's count or to add a knob no other model has.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Annotated

import msgspec

from cozy_runtime.author._assets import ImageAsset
from cozy_runtime.author._errors import InvalidRequest
from cozy_runtime.author._markers import AssetBound, ImagePreparation, ModelDefault
from cozy_runtime.author._services import Adjustments

#: Images, in order, that a request edits or composes; none is text to image. Read upright and
#: scaled down to 16 MP at most. A package states its model's count by overriding the field with
#: `Annotated[ReferenceImages, msgspec.Meta(max_length=N)]`; a text-only model says 0.
ReferenceImages = Annotated[
    list[ImageAsset], AssetBound(max_bytes=32 << 20), ImagePreparation(max_pixels=16 << 20)
]
AspectRatio = Annotated[str, msgspec.Meta(pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")]
Seed = Annotated[int, msgspec.Meta(ge=0, le=(1 << 53) - 1)]

#: Pixels in one megapixel: 1024², as the image packages size their tiers.
MEGAPIXEL = 1 << 20

#: `[x0, y0, x1, y1]`: a box's corners, 0-1 of the output canvas, origin top-left, x0 < x1, y0 < y1.
Box = Annotated[
    list[Annotated[float, msgspec.Meta(ge=0.0, le=1.0)]], msgspec.Meta(min_length=4, max_length=4)
]


class Region(msgspec.Struct):
    """An element a model places by box without a reference (MOSS-Image, Ideogram, Klein's table):
    `prompt` says what goes in the box, `text` is a literal string to render there."""

    box: Box
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    text: Annotated[str, msgspec.Meta(min_length=1)] | None = None


@dataclass(frozen=True, slots=True)
class Target:
    """The size a request asks for, before the model snaps it to its grid and range."""

    ratio: tuple[int, int]
    pixels: float


class ImageRequest(msgspec.Struct):
    prompt: str
    reference_images: Annotated[ReferenceImages, msgspec.Meta(max_length=16)] = msgspec.field(
        default_factory=list
    )
    #: Any W:H. None: the first reference's, else the model's default.
    aspect_ratio: AspectRatio | None = None
    #: Any size, in 1024² pixels. None: the first reference's pixel count, else the model's default.
    megapixels: Annotated[float, msgspec.Meta(gt=0)] | None = None
    #: A checkpoint's or adapter's recipe may supply these three (`ModelDefault`); else the model's.
    steps: ModelDefault[Annotated[int, msgspec.Meta(ge=1)] | None] = None
    guidance: ModelDefault[Annotated[float, msgspec.Meta(ge=0)] | None] = None
    negative_prompt: ModelDefault[str | None] = None
    #: Given, the request is reproducible; omitted, the model draws one and returns it.
    seed: Seed | None = None
    prompt_upsampling: bool | None = None
    #: One per reference, in order: where it goes in the output, or null for none. A placed
    #: reference is an element, not the canvas: the size is the model's text-to-image default.
    reference_boxes: list[Box | None] | None = None

    def target(
        self, first: tuple[int, int] | None, *, megapixels: float, ratio: str = "1:1"
    ) -> Target:
        """`aspect_ratio` and `megapixels` as given. A missing one is the first reference's
        (`first`: its upright width and height), so an edit stays as close as possible to the image
        it edits; with no reference, the model's default (`megapixels`, `ratio`)."""
        if self.aspect_ratio is not None:
            width, height = map(int, self.aspect_ratio.split(":"))
        elif first is not None:
            width, height = first
        else:
            width, height = map(int, ratio.split(":"))
        if self.megapixels is not None:
            pixels = self.megapixels * MEGAPIXEL
        elif first is not None:
            pixels = float(first[0] * first[1])
        else:
            pixels = megapixels * MEGAPIXEL
        return Target((width, height), pixels)

    def placements(self) -> list[tuple[float, float, float, float] | None]:
        """`reference_boxes` checked: one entry per reference, each with x0 < x1 and y0 < y1; none
        given is no placement for any reference."""
        boxes, count = self.reference_boxes, len(self.reference_images)
        if boxes is None:
            return [None] * count
        if len(boxes) != count:
            raise InvalidRequest(
                f"reference_boxes has {len(boxes)} entries for {count} reference_images",
                code="reference_boxes_length",
                fields=("reference_boxes",),
            )
        for index, box in enumerate(boxes):
            if box is not None and not (box[0] < box[2] and box[1] < box[3]):
                raise InvalidRequest(
                    f"a box is [x0, y0, x1, y1] with x0 < x1 and y0 < y1, got {box}",
                    code="invalid_box",
                    fields=(f"reference_boxes.{index}",),
                )
        return [None if box is None else (box[0], box[1], box[2], box[3]) for box in boxes]

    def drawn_seed(self) -> int:
        """The seed given, else a fresh one the result reports."""
        return self.seed if self.seed is not None else secrets.randbits(53)

    def ignored(self, adj: Adjustments, *fields: str, reason: str) -> None:
        """A warning for each of `fields` the request set although this model has no use for it;
        None, False and "" ask for nothing and pass silently."""
        for field in fields:
            if getattr(self, field) not in (None, False, "", []):
                adj.warn(field, f"ignored: {reason}")


class ImageResult(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        AssetBound(
            max_bytes=64 << 20, max_decoded_bytes=256 << 20, media_types=("image/png", "image/webp")
        ),
    ]
    width: int
    height: int
    seed: int
    #: The prompt the image was made from: the upsampled one when upsampling ran.
    prompt: str


def held[T: (int, float)](
    adj: Adjustments, field: str, requested: T, low: T, high: T, reason: str = "the model's range"
) -> T:
    """`requested` inside `low`..`high`; a value that had to move is a `clamp` row (a float shown to
    three decimals)."""
    applied = min(max(requested, low), high)
    if applied != requested:
        shown = round(requested, 3) if isinstance(requested, float) else requested
        adj.clamp(field, shown, applied, reason)
    return applied
