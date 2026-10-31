"""xs-139 — the shot and upscale contract: the field names and shapes a project sends to whatever
function renders a shot or upscales one, so another package can replace any of them. Opt-in, like
the image interface: a package subclasses these structs, narrows `references` to the kinds its model
takes and adds its own fields. The project matches fields by name against the function's request
schema; nothing here is a registry, and Runtime core never reads it.
"""

from __future__ import annotations

from typing import Annotated, Literal

import msgspec

from cozy_runtime.author._files import AudioFile, File, ImageFile, VideoFile
from cozy_runtime.author._image import AspectRatio, Seed
from cozy_runtime.author._markers import FileBound
from cozy_runtime.author._references import Reference

Frame = Annotated[
    ImageFile,
    FileBound(
        max_bytes=64 << 20,
        max_decoded_bytes=256 << 20,
        media_types=("image/png", "image/jpeg", "image/webp"),
    ),
]
Video = Annotated[VideoFile, FileBound(max_bytes=1 << 30, max_decoded_bytes=2 << 30)]
Sound = Annotated[AudioFile, FileBound(max_bytes=256 << 20)]
#: A shot's own latents, in its renderer's format and opaque to the caller: a later call of the same
#: package continues from them, and its upscaler starts from them instead of re-encoding the video.
Latents = Annotated[File, FileBound(max_bytes=1 << 30, media_types=("application/octet-stream",))]
#: Speed against fidelity, as a video encoder's presets: `quality` the slowest and closest to the
#: model's reference, `fast` the quickest; each renderer or upscaler maps them to its own recipes.
Preset = Literal["quality", "balanced", "fast"]
#: How strictly an image reference guides: `exact` keeps it as pictured, `loose` keeps its
#: arrangement and positions, not its frame (a world state, a place's layout).
Follow = Literal["exact", "loose"]
#: An output's short edge: `1080p`, `1440p`, `2160p`.
Resolution = Annotated[str, msgspec.Meta(pattern=r"^[1-9][0-9]*p$")]


class Beat(msgspec.Struct, kw_only=True):
    """From `at` seconds into the shot, `prompt` is what happens, until the next beat."""

    at: Annotated[float, msgspec.Meta(ge=0)]
    prompt: str


class ShotReference(Reference, tag_field="kind", kw_only=True):
    """A file guiding the shot, named for the prompt's `<name>`; `kind` says what it is."""


class CharacterReference(ShotReference, tag="character", kw_only=True):
    """A person. `voice` is their voice's timbre."""

    image: Frame
    description: str = ""
    follow: Follow = "exact"
    voice: Sound | None = None


class PropReference(ShotReference, tag="prop", kw_only=True):
    image: Frame
    description: str = ""
    follow: Follow = "exact"


class EnvironmentReference(ShotReference, tag="environment", kw_only=True):
    """A place: an environment's picture from the shot's camera, or a world state (`loose`)."""

    image: Frame
    description: str = ""
    follow: Follow = "exact"


class VideoReference(ShotReference, tag="video", kw_only=True):
    """A motion and camera guide, second for second with the shot; its sound only with
    `use_audio`."""

    video: Video
    use_audio: bool = False


class AudioReference(ShotReference, tag="audio", kw_only=True):
    """Sound the soundtrack reuses."""

    audio: Sound


ShotReferences = (
    CharacterReference | PropReference | EnvironmentReference | VideoReference | AudioReference
)


class Plate(msgspec.Struct, kw_only=True):
    """The video being edited: the shot keeps its frames, framing and timing, and the prompt (or an
    edit LoRA) decides what changes. It sets the frame size and spans the shot's timeline."""

    video: Video
    use_audio: bool = False


class ContinueFrom(msgspec.Struct, kw_only=True):
    """A video the shot begins with, and its `latents` from the same package: the request then
    runs on that video's timeline, resumes at its end and returns the whole video."""

    video: Video
    latents: Latents | None = None


class ShotRequest(msgspec.Struct, kw_only=True):
    """One continuous shot of any length. `prompt` applies throughout, with its `<Name>`
    mentions; each beat holds until the next. `references` are in the order its model numbers
    them."""

    prompt: str
    duration_s: Annotated[float, msgspec.Meta(gt=0)]
    beats: list[Beat] = msgspec.field(default_factory=list)
    style: str = ""
    soundscape: str = ""
    references: list[ShotReferences] = msgspec.field(default_factory=list)
    plate: Plate | None = None
    first_frame: Frame | None = None
    last_frame: Frame | None = None
    continue_from: ContinueFrom | None = None
    aspect_ratio: AspectRatio | None = None
    seed: Seed | None = None
    preset: Preset = "balanced"


class ShotResult(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=1 << 30, media_types=("video/mp4",))]
    warnings: list[str]
    #: The shot's latents, when its renderer keeps them: a project keeps them with the video, for
    #: `continue_from` and the upscale.
    latents: Latents | None = None


class UpscaleRequest(msgspec.Struct, kw_only=True):
    """A whole shot at `resolution` (its short edge), in the shot's own aspect; `prompt` is the
    text it was made from, for an upscaler that re-renders."""

    video: Video
    resolution: Resolution
    prompt: str = ""
    seed: Seed | None = None
    #: The `latents` the shot came with. An upscaler that reads them starts from them; without
    #: them, or one that cannot, it works from `video`.
    latents: Latents | None = None
    preset: Preset = "balanced"


class UpscaleResult(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=1 << 30, media_types=("video/mp4",))]
