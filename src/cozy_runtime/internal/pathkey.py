"""Opaque filesystem names for wire identities.

Request and output ids remain exact on the wire and in journals.  They are never path
components: a path is local authority, while these strings are caller-controlled identity.
"""

from __future__ import annotations

import hashlib


def opaque_key(namespace: str, *parts: str | int) -> str:
    """A framed, domain-separated SHA-256 name for caller-controlled identity fields."""
    digest = hashlib.sha256()
    for part in (namespace, *(str(part) for part in parts)):
        raw = part.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()
