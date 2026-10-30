"""cr-219 — one request and one result for every image model.

The same names, types and meaning in every image package, so swapping one model for another never
breaks a request. Every optional knob is None by default: the model's own recommended value, which
its README and `--describe` state. The runtime rejects aspect ratios outside 1:16 through 16:1.
Within that global envelope, a model clamps to its narrower range (a `clamp` row) and warns for
a knob it has no use for. A package subclasses `ImageRequest`
only to narrow `assets` to its model's count or to add a knob no other model has.
"""

from __future__ import annotations

import math
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, NoReturn

import msgspec

from cozy_runtime.author._assets import Asset, name_label, render
from cozy_runtime.author._errors import InvalidRequest
from cozy_runtime.author._files import ImageFile
from cozy_runtime.author._markers import FileBound, ImagePreparation, ModelDefault
from cozy_runtime.author._services import AdjustmentRows, Adjustments

AspectRatio = Annotated[
    str,
    msgspec.Meta(
        pattern=r"^[1-9][0-9]*:[1-9][0-9]*$",
        description="Output aspect ratio W:H, from 1:16 through 16:1 inclusive.",
    ),
]
Seed = Annotated[int, msgspec.Meta(ge=0, le=(1 << 53) - 1)]

#: Pixels in one megapixel: 1024², as the image packages size their tiers.
MEGAPIXEL = 1 << 20
MAX_IMAGE_ASPECT_RATIO = 16


def _check_ratio(width: int, height: int) -> None:
    """Use integer comparisons: float division can hide a ratio just above the limit."""
    if width <= 0 or height <= 0:
        raise InvalidRequest(
            "aspect_ratio must contain two positive integers W:H",
            code="aspect_ratio",
            fields=("aspect_ratio",),
        )
    if width > MAX_IMAGE_ASPECT_RATIO * height or height > MAX_IMAGE_ASPECT_RATIO * width:
        raise InvalidRequest(
            f"image aspect ratio {width}:{height} is outside 1:16 through 16:1",
            code="aspect_ratio",
            fields=("aspect_ratio",),
        )


def _ratio(value: str) -> tuple[int, int]:
    try:
        width, height = map(int, value.split(":"))
    except (ValueError, AttributeError) as error:
        raise InvalidRequest(
            "aspect_ratio must contain two positive integers W:H",
            code="aspect_ratio",
            fields=("aspect_ratio",),
        ) from error
    _check_ratio(width, height)
    return width, height


#: `[x0, y0, x1, y1]`: a box's corners, 0-1 of the output canvas, origin top-left, x0 < x1, y0 < y1.
Box = Annotated[
    list[Annotated[float, msgspec.Meta(ge=0.0, le=1.0)]], msgspec.Meta(min_length=4, max_length=4)
]


class PictureAsset(Asset, kw_only=True):
    """An asset given as a picture, which a request edits or composes, named for the prompt's
    `<name>` mentions. Read upright and scaled down to 16 MP at most."""

    image: Annotated[
        ImageFile, FileBound(max_bytes=32 << 20), ImagePreparation(max_pixels=16 << 20)
    ]
    #: Where it goes in the output, for a model with layout control; None places it nowhere in
    #: particular. A placed asset is an element, not the canvas: the size is the model's
    #: text-to-image default.
    box: Box | None = None


class Region(msgspec.Struct):
    """An element a model places by box without an asset (MOSS-Image, Ideogram, Klein's table):
    `prompt` says what goes in the box, `text` is a literal string to render there."""

    box: Box
    prompt: Annotated[str, msgspec.Meta(min_length=1)]
    text: Annotated[str, msgspec.Meta(min_length=1)] | None = None


@dataclass(frozen=True, slots=True)
class Target:
    """The size a request asks for, before the model snaps it to its grid and range."""

    ratio: tuple[int, int]
    pixels: float

    def __post_init__(self) -> None:
        _check_ratio(*self.ratio)


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
        _check_ratio(w, h)
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
    #: Names assets as `<name>`; the model reads each mention in its own words (`rendered`).
    prompt: str
    #: Images, in order, that the request edits or composes; none is text to image. A package
    #: states its model's count with `Annotated[list[PictureAsset], msgspec.Meta(max_length=N)]`;
    #: a text-only model says 0.
    assets: Annotated[list[PictureAsset], msgspec.Meta(max_length=16)] = msgspec.field(
        default_factory=list
    )
    #: W:H from 1:16 through 16:1. None: the first asset's, else the model's default.
    aspect_ratio: AspectRatio | None = None
    #: Any size, in 1024² pixels. None: the first asset's pixel count, else the model's default.
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
            _ratio(self.aspect_ratio)

    def target(
        self, first: tuple[int, int] | None, *, megapixels: float, ratio: str = "1:1"
    ) -> Target:
        """`aspect_ratio` and `megapixels` as given. A missing one is the first asset's
        (`first`: its upright width and height), so an edit stays as close as possible to the image
        it edits; with no asset, the model's default (`megapixels`, `ratio`)."""
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
        """The assets' pictures, in order."""
        return [asset.image for asset in self.assets]

    def rendered(
        self,
        adj: Adjustments | AdjustmentRows,
        label: Callable[[int, PictureAsset], str] = name_label,
    ) -> str:
        """The prompt as the model reads it: each `<name>` mention in the model's own words for
        the Nth asset (`label`); a mention of no asset stays as written, with a warning."""
        return render(self.prompt, self.assets, label, adj=adj)

    def placements(self) -> list[tuple[float, float, float, float] | None]:
        """Each asset's `box` checked (x0 < x1 and y0 < y1), or None where it has none."""
        return [
            None if asset.box is None else corners(asset.box, f"assets.{index}.box")
            for index, asset in enumerate(self.assets)
        ]

    def unplaced(self, adj: Adjustments, reason: str) -> None:
        """A warning for each asset's `box`, for a model with no layout control."""
        for index, asset in enumerate(self.assets):
            if asset.box is not None:
                adj.warn(f"assets.{index}.box", f"ignored: {reason}")

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
        f"{field} is gone in Runtime 0.36.0 (assets replaces it, each with its own box): this "
        "package was built for Runtime < 0.36; use its newest release",
        code="removed_name",
        fields=(field,),
    )


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
