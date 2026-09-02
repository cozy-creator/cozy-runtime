"""Digest identity: ONE spelling, derived documents, measured files (job-015).

A content digest is spelled `sha256:` + 64 lowercase hex, everywhere. Bare hex is
accepted at a struct boundary only because upstream tools emit it, and it is normalized
there — it never survives past the boundary into a schema, a manifest or a comparison.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from cozy_runtime.author import canonical_json

_HEX = frozenset("0123456789abcdef")


def canonical_sha256(value: str, *, field: str, allow_empty: bool = False) -> str:
    """The one digest spelling, or a refusal. ValueError so msgspec `__post_init__`
    surfaces it as a ValidationError at the struct boundary."""
    if value == "":
        if allow_empty:
            return ""
        raise ValueError(f"{field}: an empty digest identifies nothing")
    bare = value.removeprefix("sha256:").lower()
    if len(bare) != 64 or not set(bare) <= _HEX:
        raise ValueError(f"{field}: {value!r} is not a sha256 digest (`sha256:` + 64 hex)")
    return "sha256:" + bare


def document_digest(document: object, *, domain: str) -> str:
    """The formatting-independent identity of semantic JSON within one use.

    A private digest preimage needs domain separation, not a globally versioned field in
    the stored document. The domain is deliberately outside ``document`` so it cannot
    leak into an output file and masquerade as another file format.
    """
    if not domain.isascii() or not domain or "\x00" in domain:
        raise ValueError("digest domain must be non-empty NUL-free ASCII")
    digest = hashlib.sha256()
    digest.update(b"cozy-jobs\x00")
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00")
    digest.update(canonical_json.encode(document))
    return "sha256:" + digest.hexdigest()


def hash_file(path: Path, window: int = 8 << 20) -> tuple[str, int]:
    """(canonical digest, byte count) of a file, streamed."""
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(window):
            digest.update(chunk)
            total += len(chunk)
    return "sha256:" + digest.hexdigest(), total
