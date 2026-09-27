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
    """Publish one installation callable's plan; a re-derivation replaces an older one.

    The key (installation, descriptor) is the identity and every other field is derived
    from it, so newer derived bytes replace older ones. Only a different application or
    job under the same key is a genuine conflict.
    """
    target = path(root, record["installation_id"], record["job_descriptor_id"])
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    content = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    if target.is_file() and not target.is_symlink():
        existing = target.read_bytes()
        if existing == content:
            return target
        try:
            held = json.loads(existing)
        except ValueError:
            held = {}
        if isinstance(held, dict) and any(
            key in held and held[key] != record.get(key) for key in ("application", "job")
        ):
            raise ValueError(
                f"job plan {target.name} already names another callable for this installation"
            )
    descriptor, name = tempfile.mkstemp(dir=target.parent, prefix=".stage-")
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        staged.chmod(0o444)
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            target.chmod(0o644)  # Windows refuses replacing a read-only file
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    if os.name == "posix":
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return target
