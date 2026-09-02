"""Assemble the package constructor document from inline CozyTensors configs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from cozy_runtime.internal import canonical


class ModelConfigRefusal(Exception):
    pass


def construction_document(configs: object) -> tuple[str, bytes]:
    """Return one exact config document; multiple component configs become one mapping."""
    if not isinstance(configs, Mapping) or not configs:
        raise ModelConfigRefusal("the CozyTensors header declares no inline construction configs")
    decoded: dict[str, object] = {}
    held: list[bytes] = []
    for name, value in sorted(configs.items(), key=lambda row: str(row[0])):
        if not isinstance(name, str) or not name or not isinstance(value, bytes):
            raise ModelConfigRefusal("the CozyTensors header carries a malformed inline config")
        raw = value
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelConfigRefusal(f"config {name!r} is not JSON") from exc
        if not isinstance(document, dict):
            raise ModelConfigRefusal(f"config {name!r} is not an object")
        try:
            normalized = canonical.write(document)
        except canonical.CanonicalError as exc:
            raise ModelConfigRefusal(f"config {name!r} is not canonical JSON: {exc}") from exc
        if normalized != raw:
            raise ModelConfigRefusal(f"config {name!r} is not canonical JSON")
        decoded[name] = document
        held.append(raw)
    if len(held) == 1:
        raw = held[0]
        return "sha256:" + hashlib.sha256(raw).hexdigest(), raw
    raw = canonical.write(decoded)
    return "sha256:" + hashlib.sha256(raw).hexdigest(), raw
