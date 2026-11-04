"""cr-012 — MEDIA-TYPE HONESTY: what the bytes actually are, against what was declared.

A declared media type is a CLAIM by whoever minted the grant. A file field is a typed
promise to author code that `payload.photo` is an image. Between those two sits nothing at
all unless something reads the first bytes, so this module reads them.

Three rules:

**The bytes win within a family.** A `.jpg` that is really WebP, or `audio/x-wav` for a WAV,
is the image or audio the field asked for: `reconcile` takes the sniffed type and the field's
own check (`admits`) still decides. Only an image declared and video delivered (another
family) refuses. Bytes no signature names keep their declaration; the decoder is the judge.

**A kind is a set of media types, declared here and nowhere else.** `ImageFile` means
exactly `KIND_MEDIA["image"]`. A package narrows that with `FileBound(media_types=…)`;
it never widens it, because a decoder the runtime does not have is not a capability. A kind
admits what its decoder reads, not only its own family: `AudioFile` reads the sound of a video
file (an `.m4a` or a browser's `.webm` is often sniffed as one), `VideoFile` an animated GIF. In
a slot of several kinds each type still binds to one (`slot_kind`): a video is a video when the
slot takes video, and only otherwise a source of sound.

It lives under `author/` because the KIND table is a fact about the author surface's own
types, and both sides need one copy: the worker sniffs the fetched bytes before
acceptance, the executor checks the sniffed type against the FIELD it is about to fill. Two
tables would be two contracts. Stdlib only, so the package fence is satisfied in the
direction that matters (`internal` may import `author`; never the reverse).
"""

from __future__ import annotations

from collections.abc import Iterable

#: magic-number prefix -> media type. Byte signatures only: no extension, no caller hint,
#: no libmagic. Each entry is (offset, signature, media type) so container formats whose
#: marker is not at zero (RIFF/ISO-BMFF) are one rule and not a special case.
SIGNATURES: tuple[tuple[int, bytes, str], ...] = (
    (0, b"\x89PNG\r\n\x1a\n", "image/png"),
    (0, b"\xff\xd8\xff", "image/jpeg"),
    (0, b"GIF87a", "image/gif"),
    (0, b"GIF89a", "image/gif"),
    (0, b"BM", "image/bmp"),
    (0, b"II*\x00", "image/tiff"),
    (0, b"MM\x00*", "image/tiff"),
    (8, b"WEBP", "image/webp"),
    (4, b"ftypavif", "image/avif"),
    (4, b"ftypheic", "image/heif"),
    (4, b"ftypheix", "image/heif"),
    (4, b"ftyphevc", "image/heif"),
    (4, b"ftypmif1", "image/avif"),
    (4, b"ftypisom", "video/mp4"),
    (4, b"ftypiso5", "video/mp4"),  # StreamingMP4Encoder's fragmented MP4 brand.
    (4, b"ftypmp42", "video/mp4"),
    (4, b"ftypM4V ", "video/mp4"),
    (4, b"ftypqt  ", "video/quicktime"),
    (4, b"ftypM4A ", "audio/m4a"),
    (4, b"ftypM4B ", "audio/m4a"),
    (4, b"ftyp", "video/mp4"),  # every other ISO-BMFF brand: mp41, avc1, iso4, dash, 3gp...
    (0, b"\x1a\x45\xdf\xa3", "video/webm"),
    (8, b"WAVE", "audio/wav"),
    (0, b"OggS", "audio/ogg"),
    (0, b"fLaC", "audio/flac"),
    (0, b"ID3", "audio/mpeg"),
    (0, b"\xff\xfb", "audio/mpeg"),
    (0, b"\xff\xf3", "audio/mpeg"),
    (0, b"\xff\xfa", "audio/mpeg"),
    (0, b"\xff\xf2", "audio/mpeg"),
    (0, b"%PDF-", "application/pdf"),
)

#: How many bytes a sniff needs. The furthest signature starts at 8 and is 8 long.
SNIFF_BYTES = 32

#: media type -> the filename suffix a DIGEST-NAMED output takes (`<sha256>.<ext>`). This
#: is the one table cozy-creator's `resultfiles.Extension` also spells, and the two must
#: stay equal: the worker names the file under a directory grant, Calcifer looks for
#: exactly that name at the terminal. An unlisted type gets no guessed suffix.
EXTENSIONS: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "audio/wav": ".wav",
    "application/json": ".json",
}

#: The media types each ASSET KIND admits. `file` is deliberately open: `File` is the
#: kind that promises nothing about its contents, and pretending otherwise would make the
#: three specific kinds mean less.
_VIDEO: tuple[str, ...] = (
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-matroska",
    "video/x-msvideo",
    "video/x-m4v",
    "video/mpeg",
    "video/x-flv",
    "video/x-ms-wmv",
)
KIND_MEDIA: dict[str, tuple[str, ...]] = {
    "image": (
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/bmp",
        "image/tiff",
        "image/webp",
        "image/avif",
        "image/jp2",
        "image/vnd.adobe.photoshop",
        "image/vnd.microsoft.icon",
    ),
    "video": (*_VIDEO, "image/gif"),
    "audio": (
        "audio/wav",
        "audio/ogg",
        "audio/flac",
        "audio/mpeg",
        "audio/m4a",
        "audio/aac",
        "audio/opus",
        "audio/x-aiff",
        "audio/amr",
        "audio/x-ape",
        *_VIDEO,
    ),
    "file": (),
}


def sniff(head: bytes) -> str:
    """The media type the BYTES are, or `""` when no signature matches."""
    for offset, signature, media_type in SIGNATURES:
        if head[offset : offset + len(signature)] == signature:
            return media_type
    return ""


def admits(kind: str, media_type: str) -> bool:
    """Whether a file KIND admits a media type. An open kind admits everything."""
    allowed = KIND_MEDIA.get(kind)
    if allowed is None:
        return False
    return not allowed or media_type in allowed


def slot_kind(kinds: Iterable[str], media_type: str) -> str | None:
    """The one of a slot's `kinds` a file of `media_type` binds to: its own family's when the slot
    takes that, else the first that reads it. None when only an open kind can hold it."""
    readers = [kind for kind in kinds if media_type in KIND_MEDIA.get(kind, ())]
    family = media_type.partition("/")[0]
    return family if family in readers else next(iter(readers), None)


#: Spellings people and tools use for the types above.
ALIASES: dict[str, str] = {
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "image/x-png": "image/png",
    "image/x-ms-bmp": "image/bmp",
    "video/x-m4v": "video/mp4",
    "audio/mp4": "audio/m4a",
    "audio/x-m4a": "audio/m4a",
    "audio/aiff": "audio/x-aiff",
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "audio/vnd.wave": "audio/wav",
    "audio/mp3": "audio/mpeg",
    "audio/x-mpeg": "audio/mpeg",
    "audio/x-flac": "audio/flac",
}


def normalize(declared: str) -> str:
    """The media type without its parameters, lowercased and unaliased: `image/jpg` and
    `image/jpeg; q=1` are `image/jpeg`."""
    media_type = declared.split(";")[0].strip().lower()
    return ALIASES.get(media_type, media_type)


def reconcile(declared: str, actual: str) -> str | None:
    """The type bytes sniffed `actual` are used as under a `declared` claim: the bytes within
    one family, the claim when no signature matched, None across families."""
    declared = normalize(declared)
    if not actual or not declared or declared == "application/octet-stream":
        return actual or declared
    return actual if actual.partition("/")[0] == declared.partition("/")[0] else None


def extension(media_type: str) -> str:
    """`.png` from `image/png`; `""` for a type the closed table does not name."""
    return EXTENSIONS.get(normalize(media_type), "")
