# Media inputs

`ImageFile`, `VideoFile` and `AudioFile` are the only media types a signature names,
as a request field or inside `Files[...]`. Each is a handle: nothing is decoded until the
function reads it, and every read goes through the attempt's one reader and budget.
Naming a media handle grants `media_decode`; a function naming a video or audio input
includes `cozy-runtime[media]` (PyAV) in its dependencies.

```python
from typing import Annotated

import msgspec

from cozy_runtime.author import (
    FileBound, FileLimits, Files, AudioFile, Context, ImageFile, ImagePreparation,
    VideoFile, VideoPreparation, invocable,
)

References = Annotated[
    Files[ImageFile | VideoFile | AudioFile],
    FileLimits(images=9, videos=3, audio=3, total=12),
    ImagePreparation(max_edge=4096),
    VideoPreparation(min_seconds=2, max_seconds=15, excess="truncate"),
]


class Shot(msgspec.Struct):
    clip: Annotated[
        VideoFile,
        FileBound(max_bytes=1 << 30, max_decoded_bytes=4 << 30),
        VideoPreparation(max_seconds=15, excess="truncate"),
    ]


class Done(msgspec.Struct):
    frames: int


@invocable
async def render(ctx: Context, *, shot: Shot, refs: References) -> Done:
    video = shot.clip.read(fps=24)  # at most 15 s, on a 24 fps clock
    for item in refs:
        item.probe()  # header facts only
    return Done(video.frame_count)
```

| call | returns |
|---|---|
| `image.read()` | RGB `PIL.Image.Image`, EXIF orientation and `ImagePreparation` applied |
| `image.probe()` | `ImageProbe`: `width`/`height` as `read()` returns them, and the source's |
| `video.probe()` / `audio.probe()` | `VideoProbe` / `AudioProbe`: display-oriented size, `duration`, `source_duration`, frame rate, pixel aspect, soundtrack format |
| `video.read(*, end=None, fps=None, audio=True)` | `DecodedVideo`: frames starting before `end`, soundtrack cut at `end` |
| `video.stream(*, end=None, audio=True, pictures=True)` | `DecodedMediaStream` bounded by `end`; `pictures=False` reads only the soundtrack |
| `audio.read(*, end=None)` / `audio.stream(*, end=None)` | native-rate `DecodedAudio` / stream cut at `end` |

- Each frame shows until the next one starts, whatever duration the container records
  (B-frame and variable-rate files record decode-order timing); only the last frame uses
  its own. Time running backwards refuses.
- `end` counts seconds from the first decoded frame (or, for audio, sample), with or
  without pictures. The first frame starting at or after it is never validated or
  converted and the demux stops; the last kept frame's duration is clipped to it and the
  soundtrack ends on the sample at it. A source longer than the 3600-frame ceiling reads
  its first seconds: ceilings count frames a read keeps. A stream retains one event at a
  time and has no frame or sample ceiling.
- `fps` maps frames onto a constant clock starting at 0: each frame shows until the next
  starts, rounded to the nearest slot. A frame filling several slots is the same `bytes`
  object. Video pixels are never resized; sample-rate conversion stays in the package.
- `probe()` reads headers only. A container without a recorded duration (live WebM) is
  measured from its packets without decoding.
- Reads refuse inside an active component-use scope and after the attempt closes. Each
  `read()` decodes again; keep the value. A refused read returns what it charged, so a
  shorter `end` can follow. `read_bytes()` still returns the whole file.

## Preparation

`ImagePreparation(max_edge=, max_pixels=)`, `VideoPreparation(min_seconds=, max_seconds=,
excess="refuse"|"truncate")` and `AudioPreparation(...)` annotate one field or an `Files`
argument, where each applies to its kind.

- `VideoPreparation(max_short_edge=, max_long_edge=, max_pixels=, max_fps=, audio=)` also
  declares the video's fit: the most of it the package reads. `audio` is `False` when the
  package never reads the input's audio, or the name of a boolean member of the same record
  that says whether it does (`"use_audio"`). A fit is a hint, never a refusal: a client may
  reduce a larger source to it before it uploads (the cozy CLI does, with ffmpeg), so the
  package's own sizing must give the same result for an input already within it (resample
  to a fixed frame rate and size; never halve).

- Images keep their aspect ratio and are never upscaled. Pillow decodes, applies EXIF
  orientation (including mirrors) and resamples with Lanczos.
- At hydration the runtime probes each prepared video or audio input. Shorter than
  `min_seconds`, or longer than `max_seconds` with `excess="refuse"`, refuses the request
  with `media_too_short` / `media_too_long` naming the field, before the function runs.
- With `excess="truncate"` every read and stream of a longer input ends at `max_seconds`;
  `probe().duration` reports that and `probe().source_duration` the file's. The attempt
  records an `adjusted` row (`field` = input id, `requested` = source seconds, `applied` =
  `max_seconds`, `source` = `policy`) and a warning log line.

## Budget

Retained media is budgeted by the machine's memory: every `read()` of an attempt shares one
budget of the host's physical memory (never below 2 GiB), so a run that fits the machine
succeeds. `FileBound(max_decoded_bytes=)` narrows one field's reads to the RGB8/float32 value
it retains, charged after `end`, `fps` and preparation: dropped frames, repeated slots and
resized pixels cost nothing extra. Every image, video and audio request field declares it
(describe refuses `asset_decode_unbounded`), and any size describes. An `Files` kind without
one takes the documented 2 GiB default, which the interface publishes; a 15-second 1080p
read at 24 fps retains about 2.2 GB, so declare more on a kind that needs it. Streams are
bounded per event.

## Collections

`assets["first"]` requires an occurrence labelled `first`; `assets.get("last")` returns
`None` when absent. Integer indexing preserves input order. Counts in `FileLimits` are
upper limits and compile into the request's list constraint; conflicting totals refuse at
describe. `assets.info(key)` returns readonly `FileInfo` (`id`, `position`, `label`,
`fidelity`, `kind`, `media_type`, `size_bytes`, `digest`) without reading. Preflight sees
metadata only.

Fidelity (`auto`, `low`, `medium`, `high`) is an occurrence hint the package interprets;
Runtime preserves it in invocation identity and never resizes because of it.

Pass `assets` to another managed function. `assets.select("product", "woman")` or
`assets[:2]` reorders references without reading and keeps custody, labels and fidelity.
Child input IDs are rebound; nonempty labels stay unique.

## Child results

A function returning media declares the consumer's decode bound on its result:

```python
class SegmentResult(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=256 << 20, max_decoded_bytes=32 << 20)]
```

The parent reads the returned handle (`result.video.stream()`) through its own attempt
reader, including through a generated caller. Without a decoded bound the handle stays
forwardable but reads refuse `decoded_bound_missing`.

## Package interface

An `Files` kind row carries `prepare`, and a field's `asset_bound` carries `prepare`,
each with one profile: `image-fit/1` `{max_edge, max_pixels}`, `video-trim/1` and
`audio-trim/1` `{min_seconds, max_seconds, excess}`, and `video-fit/1`, `video-trim/1`'s
members with `{max_short_edge, max_long_edge, max_pixels, max_fps, audio}`, written only when
a video declares a fit. A known profile admits exactly those
members and must prepare a kind its declaration carries; readers ignore a profile they do
not know. Preparations annotate inputs (fields, invocable parameters, `Files`); a result
declaring one refuses at describe.

## Tests

`cozy_runtime.author.fakes` reads real files the way the executor grants them:
`file_input(VideoFile, path, preparation=...)` returns a hydrated handle,
`fake_input(saved_output)` re-grants an output, and `granted_input(path, input_id=...)` is
the verified grant `invoke_with_fakes(assets=...)` and `Invocation(assets=...)` take.
