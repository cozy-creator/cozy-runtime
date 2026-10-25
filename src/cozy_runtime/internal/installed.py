"""Installed distributions identified by their files."""

from __future__ import annotations

import hashlib
import importlib.metadata

import msgspec


def record_digest(found: importlib.metadata.Distribution | None) -> str:
    """A distribution's own installed files by their RECORD hashes: the same wherever this build
    is installed. Empty when unrecorded or editable (a path file says nothing of the code)."""
    rows = sorted(
        (str(file), file.hash.value)
        for file in (found.files if found else None) or []
        if file.hash and file.parts[0] != ".." and not file.parts[0].endswith(".dist-info")
    )
    if not rows or any(path.endswith(".pth") for path, _ in rows):
        return ""
    return "sha256:" + hashlib.sha256(msgspec.json.encode(rows)).hexdigest()
