"""cr-219 — one request and one result for every image model.

The same names, types and meaning in every image package, so swapping one model for another never
breaks a request. Every optional knob is None by default: the model's own recommended value, which
its README and `--describe` state. A model clamps what is outside its range (a `clamp` row) and
warns for a knob it has no use for; it never refuses either. A package subclasses `ImageRequest`
only to narrow `reference_images` to its model's count or to add a knob no other model has.
"""

from __future__ import annotations

import math
import secrets
from collections.abc import Sequence
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


@dataclass(frozen=True, slots=True)
class Sizes:
    """The sizes a model makes: `megapixels` low..high (1024² pixels; `reason` names that range in a
    clamp row), sides `sides` px on a `grid` px grid, at most `widest`:1 either way."""

    megapixels: tuple[float, float]
    sides: tuple[int, int]
    grid: int
    reason: str
    widest: float = math.inf

    def pixels(self, adj: Adjustments, target: Target) -> float:
        """The target's pixel count inside `megapixels` (a `clamp` row when it moved)."""
        megapixels = target.pixels / MEGAPIXEL
        return held(adj, "megapixels", megapixels, *self.megapixels, self.reason) * MEGAPIXEL

    def fit(self, adj: Adjustments, ratio: tuple[int, int], pixels: float) -> tuple[int, int]:
        """`pixels` at `W:H` on the grid: the ratio held to the sides and `widest`, then at most
        1/16 over `pixels`. A size off the one the ratio itself rounds to is an `aspect_ratio`
        clamp row."""
        (w, h), (small, large), grid = ratio, self.sides, self.grid
        widest = min(self.widest, large**2 / pixels, pixels / small**2)
        width, height = self._snapped(pixels, min(max(w / h, 1 / widest), widest))
        width, height = min(max(width, small), large), min(max(height, small), large)
        while width * height > pixels * 17 / 16:
            width, height = (width - grid, height) if width >= height else (width, height - grid)
        if (width, height) != self._snapped(pixels, w / h):
            moved = f"{width}:{height}"
            adj.clamp("aspect_ratio", f"{w}:{h}", moved, "the nearest size the model admits")
        return width, height

    def _snapped(self, pixels: float, ratio: float) -> tuple[int, int]:
        grid = self.grid
        return (max(grid, round(math.sqrt(pixels * ratio) / grid) * grid),
                max(grid, round(math.sqrt(pixels / ratio) / grid) * grid))


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

    def size(
        self, adj: Adjustments, first: tuple[int, int] | None, sizes: Sizes, *, megapixels: float,
        ratio: str = "1:1",
    ) -> tuple[int, int]:
        """The output's width and height: `target` held to the model's `sizes`."""
        target = self.target(first, megapixels=megapixels, ratio=ratio)
        return sizes.fit(adj, target.ratio, sizes.pixels(adj, target))

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
        return [None if box is None else corners(box, f"reference_boxes.{index}")
                for index, box in enumerate(boxes)]

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


def corners(box: Sequence[float], field: str) -> tuple[float, float, float, float]:
    """A `Box` as its corners, refused unless x0 < x1 and y0 < y1."""
    x0, y0, x1, y1 = box
    if not (x0 < x1 and y0 < y1):
        raise InvalidRequest(
            f"a box is [x0, y0, x1, y1] with x0 < x1 and y0 < y1, got {list(box)}",
            code="invalid_box",
            fields=(field,),
        )
    return x0, y0, x1, y1


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
