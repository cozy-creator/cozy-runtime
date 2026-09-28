"""Public canonical JSON identity for semantic Cozy documents.

Formatting is never identity: decode refuses duplicate keys, non-finite numbers and values
outside the interoperable profile; encode produces the single RFC 8785 representation.
Blob and artifact integrity must continue to hash their exact bytes instead.
"""

from __future__ import annotations

from typing import Any

import msgspec

from cozy_runtime.internal import canonical


def encode(value: Any) -> bytes:
    """Return the canonical JSON bytes of a JSON value."""
    try:
        return canonical.write(value)
    except canonical.CanonicalError as exc:
        raise ValueError(str(exc)) from exc


def decode(data: bytes) -> Any:
    """Parse bounded UTF-8 JSON, refusing duplicate keys and invalid numbers."""
    try:
        return canonical.parse(data)
    except canonical.CanonicalError as exc:
        raise ValueError(str(exc)) from exc


def decode_as[T](data: bytes, into: type[T]) -> T:
    """`decode`, then convert once into `into`: a wrong shape raises `msgspec.ValidationError`
    naming the JSON path, at this boundary rather than as a KeyError in the reader."""
    return msgspec.convert(decode(data), type=into, strict=True)


def normalize(data: bytes) -> bytes:
    """Return canonical bytes for a possibly formatted JSON document."""
    return encode(decode(data))


def digest(value: Any) -> str:
    """Return ``sha256:<hex>`` over a JSON value's canonical semantic bytes."""
    try:
        return canonical.digest(value)
    except canonical.CanonicalError as exc:
        raise ValueError(str(exc)) from exc


def digest_bytes(data: bytes) -> str:
    """Return the formatting-invariant digest of JSON input bytes."""
    return digest(decode(data))
