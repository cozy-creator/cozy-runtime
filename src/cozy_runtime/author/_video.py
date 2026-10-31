"""xs-136 — the `video/shot` and `video/upscale` interfaces: one request and one result for every
shot renderer and every upscaler, so a project swaps one package for another and keeps its
requests. Opt-in, like the image interface: a package subclasses these structs to add its own
knobs, and declares `implements=Implements(...)` on its entrypoint so a caller can plan by it.
"""

from __future__ import annotations

from typing import Annotated, Literal, get_args

import msgspec

from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._files import ImageFile, VideoFile
from cozy_runtime.author._image import PictureAsset, Seed
from cozy_runtime.author._markers import FileBound

#: The interfaces a caller knows: stills (`ImageRequest`), shots and upscales.
Interface = Literal["image", "video/shot", "video/upscale"]

Frame = Annotated[
    ImageFile,
    FileBound(
        max_bytes=64 << 20,
        max_decoded_bytes=256 << 20,
        media_types=("image/png", "image/jpeg", "image/webp"),
    ),
]
Video = Annotated[VideoFile, FileBound(max_bytes=1 << 30, max_decoded_bytes=2 << 30)]


class Implements(msgspec.Struct, frozen=True, omit_defaults=True):
    """The interface an entrypoint implements, with the facts a caller plans by. `max_piece_s`
    is the longest piece a `video/shot` renderer makes at once: a longer shot needs beats."""

    interface: Interface
    max_piece_s: Annotated[float, msgspec.Meta(gt=0)] | None = None

    def __post_init__(self) -> None:
        # A constructor checks no types: the declaration is checked here, once, at import and by
        # the static reader alike.
        if self.interface not in get_args(Interface):
            known = ", ".join(get_args(Interface))
            raise ConformanceError(f"implements names {self.interface!r}: one of {known}")
        if self.max_piece_s is not None and (
            self.interface != "video/shot" or not self.max_piece_s > 0
        ):
            raise ConformanceError("max_piece_s is a video/shot's positive number of seconds")


class Beat(msgspec.Struct, kw_only=True):
    """From `at` seconds into the shot, `prompt` is what happens, until the next beat."""

    at: Annotated[float, msgspec.Meta(ge=0)]
    prompt: str


class ShotAsset(PictureAsset, kw_only=True):
    """A character, prop or environment in the shot; the environment's picture is its setup
    plate. Every one is used, named in the text or not."""

    description: str = ""


class WorldState(msgspec.Struct, kw_only=True):
    """How the set stands as the shot begins: a picture, and `layout`, who and what is where."""

    name: str
    image: Frame
    layout: str = ""


class ShotRequest(msgspec.Struct, kw_only=True):
    """One continuous shot of any length. `prompt` applies throughout, with its `<Name>`
    mentions; each beat holds until the next. `writes` asks for frames of it: `first`, `last` or
    seconds in."""

    prompt: str
    duration_s: Annotated[float, msgspec.Meta(gt=0)]
    beats: list[Beat] = msgspec.field(default_factory=list)
    seed: Seed | None = None
    style: str = ""
    assets: list[ShotAsset] = msgspec.field(default_factory=list)
    world_state: list[WorldState] = msgspec.field(default_factory=list)
    writes: list[Literal["first", "last"] | Annotated[float, msgspec.Meta(ge=0)]] = msgspec.field(
        default_factory=list
    )


class ShotResult(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=1 << 30, media_types=("video/mp4",))]
    #: One per `writes` entry, in order.
    frames: list[Frame]
    warnings: list[str]


class UpscaleRequest(msgspec.Struct, kw_only=True):
    """A whole shot at a higher resolution; `prompt` is the text it was made from, for an
    upscaler that re-renders."""

    video: Video
    prompt: str = ""
    seed: Seed | None = None


class UpscaleResult(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=1 << 30, media_types=("video/mp4",))]
