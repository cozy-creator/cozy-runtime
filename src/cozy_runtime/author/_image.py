"""cr-219 — one request and one result for every image model.

The same names, types and meaning in every image package, so swapping one model for another never
breaks a request. Every optional knob is None by default: the model's own recommended value, which
its README and `--describe` state. A model holds an aspect ratio to its range, at most 1:16 through
16:1 (a `clamp` row), and warns for a knob it has no use for. A package subclasses `ImageRequest`
only to narrow `pictures` to its model's count or to add a knob no other model has.
"""

from __future__ import annotations

import math
import re
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, NoReturn

import msgspec

from cozy_runtime.author._errors import InvalidRequest
from cozy_runtime.author._files import ImageFile
from cozy_runtime.author._markers import FileBound, ImagePreparation, ModelDefault
from cozy_runtime.author._references import Reference, name_label, render
from cozy_runtime.author._services import AdjustmentRows, Adjustments

AspectRatio = Annotated[
    str,
    msgspec.Meta(
        description="Output aspect ratio: 16:9, 2.39:1, 16/9, 16x9 or 1.85. A model holds it to "
        "its range, at most 1:16 through 16:1, and says so.",
    ),
]
Seed = Annotated[int, msgspec.Meta(ge=0, le=(1 << 53) - 1)]

#: Pixels in one megapixel: 1024², as the image packages size their tiers.
MEGAPIXEL = 1 << 20
MAX_IMAGE_ASPECT_RATIO = 16


def _ratio(value: str) -> tuple[int, int]:
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


class Picture(Reference, kw_only=True):
    """A picture the request edits or composes, named for the prompt's `<name>` mentions. Read
    upright and scaled down to 16 MP at most."""

    image: Annotated[
        ImageFile, FileBound(max_bytes=32 << 20), ImagePreparation(max_pixels=16 << 20)
    ]
    #: Where it goes in the output, for a model with layout control; None places it nowhere in
    #: particular. A placed picture is an element, not the canvas: the size is the model's
    #: text-to-image default.
    box: Box | None = None


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
    #: Names pictures as `<name>`; the model reads each mention in its own words (`rendered`).
    prompt: str
    #: Images, in order, that the request edits or composes; none is text to image. A package
    #: states its model's count with `Annotated[list[Picture], msgspec.Meta(max_length=N)]`;
    #: a text-only model says 0.
    pictures: Annotated[list[Picture], msgspec.Meta(max_length=16)] = msgspec.field(
        default_factory=list
    )
    #: W:H, held to 1:16 through 16:1. None: the first picture's, else the model's default.
    aspect_ratio: AspectRatio | None = None
    #: Any size, in 1024² pixels. None: the first picture's pixel count, else the model's default.
    megapixels: Annotated[float, msgspec.Meta(gt=0)] | None = None
    #: A checkpoint's or adapter's recipe may supply these three (`ModelDefault`); else the model's.
    steps: ModelDefault[Annotated[int, msgspec.Meta(ge=1)] | None] = None
    guidance: ModelDefault[Annotated[float, msgspec.Meta(ge=0)] | None] = None
    negative_prompt: ModelDefault[str | None] = None
    #: Given, the request is reproducible; omitted, the model draws one and returns it.
    seed: Seed | None = None
    prompt_upsampling: bool | None = None

    def __post_init__(self) -> None:
        # msgspec calls this while decoding, before admission or any model work.
        if self.aspect_ratio is not None:
            self.aspect_ratio = ":".join(map(str, _ratio(self.aspect_ratio)))

    def target(
        self, first: tuple[int, int] | None, *, megapixels: float, ratio: str = "1:1"
    ) -> Target:
        """`aspect_ratio` and `megapixels` as given. A missing one is the first picture's
        (`first`: its upright width and height), so an edit stays as close as possible to the image
        it edits; with no picture, the model's default (`megapixels`, `ratio`)."""
        if self.aspect_ratio is not None:
            width, height = _ratio(self.aspect_ratio)
        elif first is not None:
            width, height = first
        else:
            width, height = _ratio(ratio)
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

    # Runtime 0.36.0 renamed these: a package built for an older Runtime reads them and is refused
    # with the remedy, never an AttributeError deep in its handler.
    @property
    def reference_images(self) -> NoReturn:
        raise _removed("reference_images")

    @property
    def reference_boxes(self) -> NoReturn:
        raise _removed("reference_boxes")

    def images(self) -> list[ImageFile]:
        """The pictures' images, in order."""
        return [picture.image for picture in self.pictures]

    def rendered(
        self,
        adj: Adjustments | AdjustmentRows,
        label: Callable[[int, Picture], str] = name_label,
    ) -> str:
        """The prompt as the model reads it: each `<name>` mention in the model's own words for
        the Nth picture (`label`); a mention of no picture stays as written, with a warning."""
        return render(self.prompt, self.pictures, label, adj=adj, listed="pictures")

    def placements(self) -> list[tuple[float, float, float, float] | None]:
        """Each picture's `box` checked (x0 < x1 and y0 < y1), or None where it has none."""
        return [
            None if picture.box is None else corners(picture.box, f"pictures.{index}.box")
            for index, picture in enumerate(self.pictures)
        ]

    def unplaced(self, adj: Adjustments, reason: str) -> None:
        """A warning for each picture's `box`, for a model with no layout control."""
        for index, picture in enumerate(self.pictures):
            if picture.box is not None:
                adj.warn(f"pictures.{index}.box", f"ignored: {reason}")

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
        ImageFile,
        FileBound(
            max_bytes=64 << 20, max_decoded_bytes=256 << 20, media_types=("image/png", "image/webp")
        ),
    ]
    width: int
    height: int
    seed: int
    #: The prompt the image was made from: the upsampled one when upsampling ran.
    prompt: str


def _removed(field: str) -> InvalidRequest:
    return InvalidRequest(
        f"{field} is gone in Runtime 0.36.0 (pictures replaces it, each with its own box): this "
        "package was built for Runtime < 0.36; use its newest release",
        code="removed_name",
        fields=(field,),
    )


def corners(box: Sequence[float], field: str) -> tuple[float, float, float, float]:
    """A `Box` as its corners, in order whichever way it was written. One with no area refuses."""
    (x0, x1), (y0, y1) = sorted((box[0], box[2])), sorted((box[1], box[3]))
    if x0 == x1 or y0 == y1:
        raise InvalidRequest(
            f"a box needs some area, got {list(box)}",
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
