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

    def drawn_seed(self) -> int:
        """The seed given, else a fresh one the result reports."""
        return self.seed if self.seed is not None else secrets.randbits(53)

    def ignored(self, adj: Adjustments, *fields: str, reason: str) -> None:
        """A warning for each of `fields` the request set although this model has no use for it;
        None, False and "" ask for nothing and pass silently."""
        for field in fields:
            if getattr(self, field) not in (None, False, ""):
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
