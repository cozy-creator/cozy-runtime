"""What image functions share: the few inputs every model honours, and the types of the rest.

A function declares exactly the inputs its model uses, so what it doesn't have is an unknown field
(a warning) and a box never offers it. `ImageRequest` is what they all honour; a model that takes
pictures declares `prompt: Prompt[Image]` and words its references its own way. `Steps`,
`Guidance` and `NegativePrompt` type the knobs a function adds. Every
optional input is None by default: the model's own recommended value, which its README and
`--describe` state. A model holds an aspect ratio to its range, at most 1:16 through 16:1 (a `clamp`
row).
"""

from __future__ import annotations

import math
import random
import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated

import msgspec

from cozy_runtime.author import _group
from cozy_runtime.author._errors import InvalidRequest
from cozy_runtime.author._files import ImageFile
from cozy_runtime.author._markers import FileBound, ModelDefault
from cozy_runtime.author._services import AdjustmentRows, Adjustments
from cozy_runtime.author._tool import AspectRatio, Seed

#: A checkpoint's recipe may supply these three; else the model's.
Steps = ModelDefault[Annotated[int, msgspec.Meta(ge=1)] | None]
Guidance = ModelDefault[Annotated[float, msgspec.Meta(ge=0)] | None]
NegativePrompt = ModelDefault[str | None]

#: Pixels in one megapixel: 1024², as the image packages size their tiers.
MEGAPIXEL = 1 << 20
MAX_IMAGE_ASPECT_RATIO = 16


def ratio_of(value: str) -> tuple[int, int]:
    """`W:H` in lowest whole numbers, however it is written. Text that names no ratio refuses."""
    sides = re.split(r"\s*[:/xX\u00d7]\s*", value.strip())
    try:
        width, height = map(Fraction, sides if len(sides) > 1 else [*sides, "1"])
        ratio = width / height
    except (ValueError, ZeroDivisionError):
        ratio = Fraction(0)
    if ratio <= 0:
        raise InvalidRequest(
            f"aspect_ratio {value!r} names no ratio: write it as 16:9 or 2.39:1",
            code="aspect_ratio",
            fields=("aspect_ratio",),
        )
    return ratio.numerator, ratio.denominator


#: `[x0, y0, x1, y1]`: a box's corners, 0-1 of the output canvas, origin top-left, x0 < x1, y0 < y1.
Box = Annotated[
    list[Annotated[float, msgspec.Meta(ge=0.0, le=1.0)]], msgspec.Meta(min_length=4, max_length=4)
]


class Region(msgspec.Struct):
    """An element a model places by box without a picture (MOSS-Image, Ideogram, Klein's table):
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

    def pixels(self, adj: Adjustments | AdjustmentRows, target: Target) -> float:
        """The target's pixel count inside `megapixels` (a `clamp` row when it moved)."""
        megapixels = target.pixels / MEGAPIXEL
        return held(adj, "megapixels", megapixels, *self.megapixels, self.reason) * MEGAPIXEL

    def fit(
        self, adj: Adjustments | AdjustmentRows, ratio: tuple[int, int], pixels: float
    ) -> tuple[int, int]:
        """`pixels` at `W:H` on the grid: the ratio held to the sides and `widest`, then at most
        1/16 over `pixels`. A size off the one the ratio itself rounds to is an `aspect_ratio`
        clamp row."""
        (w, h), (small, large), grid = ratio, self.sides, self.grid
        widest = min(self.widest, MAX_IMAGE_ASPECT_RATIO, large**2 / pixels, pixels / small**2)
        width, height = self._snapped(pixels, min(max(w / h, 1 / widest), widest))
        width, height = min(max(width, small), large), min(max(height, small), large)
        # Independent side rounding can cross the global limit. Both sides already lie
        # on the same grid, so shortening the longer side keeps a legal grid canvas.
        width = min(width, MAX_IMAGE_ASPECT_RATIO * height)
        height = min(height, MAX_IMAGE_ASPECT_RATIO * width)
        while width * height > pixels * 17 / 16:
            width, height = (width - grid, height) if width >= height else (width, height - grid)
        if (width, height) != self._snapped(pixels, w / h):
            moved = f"{width}:{height}"
            adj.clamp("aspect_ratio", f"{w}:{h}", moved, "the nearest size the model admits")
        return width, height

    def _snapped(self, pixels: float, ratio: float) -> tuple[int, int]:
        grid = self.grid
        return (
            max(grid, round(math.sqrt(pixels * ratio) / grid) * grid),
            max(grid, round(math.sqrt(pixels / ratio) / grid) * grid),
        )


class ImageRequest(msgspec.Struct):
    """What every image model honours but one, HunyuanImage 3.0, which picks its own size."""

    prompt: str
    #: W:H, held to 1:16 through 16:1. None: the first picture's, else the model's default.
    aspect_ratio: AspectRatio | None = None
    #: Any size, in 1024² pixels. None: the first picture's pixel count, else the model's default.
    megapixels: Annotated[float, msgspec.Meta(gt=0)] | None = None
    #: Given, the request is reproducible; omitted, the model draws one and returns it.
    seed: Seed | None = None

    def __post_init__(self) -> None:
        # msgspec calls this while decoding, before admission or any model work.
        if self.aspect_ratio is not None:
            self.aspect_ratio = AspectRatio(":".join(map(str, ratio_of(self.aspect_ratio))))

    def target(
        self, first: tuple[int, int] | None, *, megapixels: float, ratio: str = "1:1"
    ) -> Target:
        """`aspect_ratio` and `megapixels` as given. A missing one is the first picture's
        (`first`: its upright width and height), so an edit stays as close as possible to the image
        it edits; with no picture, the model's default (`megapixels`, `ratio`)."""
        if self.aspect_ratio is not None:
            width, height = ratio_of(self.aspect_ratio)
        elif first is not None:
            width, height = first
        else:
            width, height = ratio_of(ratio)
        if self.megapixels is not None:
            pixels = self.megapixels * MEGAPIXEL
        elif first is not None:
            pixels = float(first[0] * first[1])
        else:
            pixels = megapixels * MEGAPIXEL
        return Target((width, height), pixels)

    def size(
        self,
        adj: Adjustments,
        first: tuple[int, int] | None,
        sizes: Sizes,
        *,
        megapixels: float,
        ratio: str = "1:1",
    ) -> tuple[int, int]:
        """The output's width and height: `target` held to the model's `sizes`."""
        target = self.target(first, megapixels=megapixels, ratio=ratio)
        return sizes.fit(adj, target.ratio, sizes.pixels(adj, target))

    def drawn_seed(self) -> int:
        """The seed given, else a fresh one the result reports."""
        if self.seed is not None:
            return self.seed
        # Every GPU of a group draws the same one: their `random` starts from the attempt's entropy.
        return random.getrandbits(53) if _group.CURRENT[0] is not None else secrets.randbits(53)


class ImageResult(msgspec.Struct):
    image: Annotated[
        ImageFile,
        # TensorD's output allowance reads `max_bytes`: a finished picture must not be lost to it.
        FileBound(
            max_bytes=1 << 40, max_decoded_bytes=256 << 20, media_types=("image/png", "image/webp")
        ),
    ]
    width: int
    height: int
    seed: int
    #: The prompt the image was made from: the upsampled one when upsampling ran.
    prompt: str


def corners(box: Sequence[float], field: str) -> tuple[float, float, float, float]:
    """A `Box` as its corners, in order whichever way it was written. One with no area refuses."""
    (x0, x1), (y0, y1) = sorted((box[0], box[2])), sorted((box[1], box[3]))
    if x0 == x1 or y0 == y1:
        raise InvalidRequest(
            f"a box is its corners as fractions of the canvas, 0 to 1, with some area: this one "
            f"comes to {list(box)}",
            code="invalid_box",
            fields=(field,),
        )
    return x0, y0, x1, y1


def held[T: (int, float)](
    adj: Adjustments | AdjustmentRows,
    field: str,
    requested: T,
    low: T,
    high: T,
    reason: str = "the model's range",
) -> T:
    """`requested` inside `low`..`high`; a value that had to move is a `clamp` row (a float shown to
    three decimals)."""
    applied = min(max(requested, low), high)
    if applied != requested:
        shown = round(requested, 3) if isinstance(requested, float) else requested
        adj.clamp(field, shown, applied, reason)
    return applied
