"""xs-139 — a video function's shapes: its references, plate, continuation and result, so another
package can replace any of them. Opt-in, like the image interface. Nothing here is a registry, and
Runtime core never reads it.
"""

from __future__ import annotations

from typing import Annotated, Literal

import msgspec

from cozy_runtime.author._files import AudioFile, File, ImageFile, VideoFile
from cozy_runtime.author._image import AspectRatio, Seed
from cozy_runtime.author._markers import FileBound

#: An output's or latents' byte bound: TensorD's output allowance reads it (256 MiB where none is
#: declared), and the disk is the real limit. A long 2160p upscale must not end in a refusal.
ANY_SIZE = 1 << 40
Frame = Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)]
Video = Annotated[VideoFile, FileBound(max_decoded_bytes=2 << 30)]
Sound = AudioFile
#: A shot's own latents, in its renderer's format and opaque to the caller: a later call of the same
#: package continues from them, and its upscaler starts from them instead of re-encoding the video.
Latents = Annotated[File, FileBound(max_bytes=ANY_SIZE, media_types=("application/octet-stream",))]
#: Speed against fidelity, as a video encoder's presets: `quality` the slowest and closest to the
#: model's reference, `fast` the quickest; each renderer or upscaler maps them to its own recipes.
Preset = Literal["quality", "balanced", "fast"]
#: How strictly an image reference guides: `exact` keeps it as pictured, `loose` keeps its
#: arrangement and positions, not its frame (a world state, a place's layout).
Follow = Literal["exact", "loose"]
#: An output's short edge: `1080p`, `1440p`, `2160p`.
Resolution = Annotated[str, msgspec.Meta(pattern=r"^[1-9][0-9]*p$")]


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


class ShotResult(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=ANY_SIZE, media_types=("video/mp4",))]
    warnings: list[str]
    #: The shot's latents, when its renderer keeps them: a project keeps them with the video, for
    #: `continue_from` and the upscale.
    latents: Latents | None = None


class UpscaleResult(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=ANY_SIZE, media_types=("video/mp4",))]
