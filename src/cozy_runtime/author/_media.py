"""cr-012 — MEDIA-TYPE HONESTY: what the bytes actually are, against what was declared.

A declared media type is a CLAIM by whoever minted the grant. An asset field is a typed
promise to author code that `payload.photo` is an image. Between those two sits nothing at
all unless something reads the first bytes, so this module reads them.

Three rules:

**The sniff is a REFUSAL, not a correction.** When the bytes disagree with the declaration
the input refuses typed and pre-entrypoint. Rewriting the declared type to match the bytes
would make a caller's mislabelled upload succeed under a type the package never agreed to,
and the package's typed field is the whole contract.

**A kind is a set of media types, declared here and nowhere else.** `ImageAsset` means
exactly `KIND_MEDIA["image"]`. A package narrows that with `AssetBound(media_types=…)`;
it never widens it, because a decoder the runtime does not have is not a capability.

**An unrecognised container is UNKNOWN, never "probably fine".** `sniff` returns `""` when
no signature matches, and the caller decides — which for a typed image field is a refusal,
and for a `FileAsset` (the kind that promises nothing about its contents) is not.

It lives under `author/` because the KIND table is a fact about the author surface's own
types, and both sides need one copy: the worker sniffs the fetched bytes before
acceptance, the executor checks the sniffed type against the FIELD it is about to fill. Two
tables would be two contracts. Stdlib only, so the package fence is satisfied in the
direction that matters (`internal` may import `author`; never the reverse).
"""

from __future__ import annotations

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
    (4, b"ftypmif1", "image/avif"),
    (4, b"ftypisom", "video/mp4"),
    (4, b"ftypmp42", "video/mp4"),
    (4, b"ftypM4V ", "video/mp4"),
    (4, b"ftypqt  ", "video/quicktime"),
    (0, b"\x1a\x45\xdf\xa3", "video/webm"),
    (8, b"WAVE", "audio/wav"),
    (0, b"OggS", "audio/ogg"),
    (0, b"fLaC", "audio/flac"),
    (0, b"ID3", "audio/mpeg"),
    (0, b"\xff\xfb", "audio/mpeg"),
    (0, b"\xff\xf3", "audio/mpeg"),
    (0, b"%PDF-", "application/pdf"),
)

#: How many bytes a sniff needs. The furthest signature starts at 8 and is 8 long.
SNIFF_BYTES = 32

#: media type -> the filename suffix a DIGEST-NAMED output takes (`<sha256>.<ext>`). This
#: is the one table cozy-creator's `resultfiles.Extension` also spells, and the two must
#: stay equal: the worker names the file under a directory grant, the daemon looks for
#: exactly that name at the terminal. An unlisted type gets no guessed suffix.
EXTENSIONS: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "audio/wav": ".wav",
    "application/json": ".json",
}

#: The media types each ASSET KIND admits. `file` is deliberately open: `FileAsset` is the
#: kind that promises nothing about its contents, and pretending otherwise would make the
#: three specific kinds mean less.
KIND_MEDIA: dict[str, tuple[str, ...]] = {
    "image": (
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/bmp",
        "image/tiff",
        "image/webp",
        "image/avif",
    ),
    "video": ("video/mp4", "video/webm", "video/quicktime"),
    "audio": ("audio/wav", "audio/ogg", "audio/flac", "audio/mpeg"),
    "file": (),
}


def sniff(head: bytes) -> str:
    """The media type the BYTES are, or `""` when no signature matches."""
    for offset, signature, media_type in SIGNATURES:
        if head[offset : offset + len(signature)] == signature:
            return media_type
    return ""


def admits(kind: str, media_type: str) -> bool:
    """Whether an asset KIND admits a media type. An open kind admits everything."""
    allowed = KIND_MEDIA.get(kind)
    if allowed is None:
        return False
    return not allowed or media_type in allowed


def normalize(declared: str) -> str:
    """The media type without its parameters, lowercased. `image/png; charset=x` is a
    media type with a parameter, not a different type, and a comparison that says
    otherwise refuses a well-formed declaration."""
    return declared.split(";")[0].strip().lower()


def extension(media_type: str) -> str:
    """`.png` from `image/png`; `""` for a type the closed table does not name."""
    return EXTENSIONS.get(normalize(media_type), "")
