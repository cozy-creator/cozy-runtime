"""Exact executable revision plus callable identity for prepared job lookup."""

from __future__ import annotations

import re
from pathlib import Path

_DIGEST = re.compile(r"sha256:([0-9a-f]{64})")


def path(root: Path, build_id: str, descriptor_id: str) -> Path:
    build, descriptor = _DIGEST.fullmatch(build_id), _DIGEST.fullmatch(descriptor_id)
    if build is None or descriptor is None:
        raise ValueError("job plan requires exact build and descriptor SHA-256 identities")
    return root / build.group(1) / (descriptor.group(1) + ".json")
