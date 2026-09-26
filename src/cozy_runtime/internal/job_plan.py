"""Look up a callable within one retained installation lifecycle."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_DIGEST = re.compile(r"sha256:([0-9a-f]{64})")


def path(root: Path, installation_id: str, descriptor_id: str) -> Path:
    build = re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", installation_id)
    descriptor = _DIGEST.fullmatch(descriptor_id)
    if build is None or descriptor is None:
        raise ValueError("job plan requires an installation ID and callable descriptor")
    return root / installation_id / (descriptor.group(1) + ".json")


def write(root: Path, record: dict[str, Any]) -> Path:
    """Publish a complete immutable plan; exact replay leaves the old file intact."""
    target = path(root, record["installation_id"], record["job_descriptor_id"])
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    content = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    descriptor, name = tempfile.mkstemp(dir=target.parent, prefix=".stage-")
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staged, target)
        except FileExistsError:
            if target.is_symlink() or target.read_bytes() != content:
                raise ValueError(
                    "job plan conflicts with its immutable build and descriptor"
                ) from None
    finally:
        staged.unlink(missing_ok=True)
    # Close and remove the staging name before marking the shared inode readonly:
    # Windows refuses unlinking an open or readonly staging file.
    target.chmod(0o444)
    if os.name == "posix":
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return target
