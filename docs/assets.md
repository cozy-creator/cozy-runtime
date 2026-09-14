# Ordered media inputs

Declare media as an explicit function input. `Image` is the actual
`PIL.Image.Image` class. Indexing or iterating the collection decodes lazily through
Runtime's existing bounded media decoder; the function needs no decoder parameter.
Packages using decoded media include `cozy-runtime[media]` in their dependencies.

```python
from typing import Annotated

import msgspec

from cozy_runtime.author import App, AssetLimits, Assets, Context, Image, invocable


class Sizes(msgspec.Struct):
    values: list[list[int]]


@invocable
async def inspect_images(
    ctx: Context,
    *,
    assets: Annotated[Assets[Image], AssetLimits(images=2)],
) -> Sizes:
    return Sizes([list(image.size) for image in assets])


app = App()
app.job(inspect_images)
```

`assets["first"]` requires an occurrence labelled `first`.
`assets.get("last")` returns `None` when the label is absent. Returned images are
ordinary RGB PIL images that can be passed directly to an image pipeline. Integer
indexing preserves input order. Different occurrences of the same file have distinct
mutable decoded images; repeated lookup of one occurrence reuses its decoded value.

For mixed references, use
`Annotated[Assets[Mixed], AssetLimits(images=9, videos=3, audio=3, total=12)]`.
`Mixed` is `Image | DecodedVideo | DecodedAudio`; video and audio retain the existing
clock, soundtrack and PCM metadata. Model-specific conversions remain in the package.

Counts are upper limits, and zero is allowed. `AssetLimits` belongs on the Assets
argument, never the function decorator. It compiles into the ordinary request list
constraint and media-kind counts. Existing `msgspec.Meta(max_length=...)` remains
supported; conflicting explicit totals refuse at describe. If every allowed kind has
a count and no total is specified, their sum bounds the collection.

`assets.info(index_or_label)` returns readonly `AssetInfo`: `id`, `position`, `label`,
`fidelity`, `kind`, `media_type`, `size_bytes` and `digest`. Metadata access does not
decode. Preflight sees declared input metadata; byte hydration verifies it before
handler entry. Decoded value access refuses during preflight and after the attempt
closes. Already-returned image/audio/video values are ordinary data.

Fidelity is an occurrence hint: `auto` (default), `low`, `medium` or `high`. Runtime
preserves it and includes it in invocation identity; it does not resize inputs or
rewrite prompts. The model package decides how to interpret the hint.

Pass `assets` directly to another managed function. `assets.select("product", "woman")`
or `assets[:2]` selects/reorders references without decoding and preserves their
verified custody, labels and fidelity. Child input IDs are rebound normally. Nonempty
labels must remain unique; duplicate content is allowed.

The advanced `Assets[ImageAsset | VideoAsset | AudioAsset]` view yields the existing
raw handles. `AssetBound` can narrow MIME, encoded bytes or decoded bytes on individual
kinds. Raw and decoded value types cannot be mixed in one view. Both views use the
same occurrence records, input grants, decoder and attempt budget.

For a native `Tree`, project a member through the same typed input interface:

```python
image = prefix.member(
    "shots/000000/continuation.png",
    ImageAsset,
    bound=AssetBound(max_bytes=8 << 20, max_decoded_bytes=64 << 20),
)
decoded = media.decode_image(image)
await next_shot(opening_frame=image)
```

`member` also accepts `VideoAsset`, `AudioAsset` and `FileAsset`. Runtime verifies
the native Tree membership, byte identity, media type and explicit bound. It uses
the already materialized member; selecting it does not copy the Tree again.
Forwarding creates independent child custody over that one file. The handle is
valid only during this attempt. Use `Outputs` to copy it into a declared output
when exporting it. A manually constructed Tree or a directory path cannot grant
this capability.

A function returning media can declare its consumer decode bound on the result:

```python
class SegmentResult(msgspec.Struct):
    video: Annotated[
        VideoAsset,
        AssetBound(max_bytes=256 << 20, max_decoded_bytes=32 << 20),
    ]
```

The parent receives that bound with the verified native video, including through
a generated imported caller, and can use `MediaDecoder.stream_video` directly.
Without an explicit decoded-byte bound the returned handle remains readable and
forwardable, but decoding refuses. Result schema metadata carries this bound;
existing input schema identities remain unchanged.
