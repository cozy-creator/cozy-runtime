"""Derive the one constructor document from a Manifest's exact named configs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from cozy_runtime.internal import canonical


class ModelConfigRefusal(Exception):
    pass


def construction_document(store: Any, configs: object) -> tuple[str, bytes]:
    """Return one exact config document; multiple component configs become one mapping."""
    if not isinstance(configs, Mapping) or not configs:
        raise ModelConfigRefusal("the model Manifest declares no construction configs")
    decoded: dict[str, object] = {}
    held: list[bytes] = []
    for name, value in sorted(configs.items(), key=lambda row: str(row[0])):
        if not isinstance(name, str) or not name or not isinstance(value, Mapping):
            raise ModelConfigRefusal("the model Manifest carries a malformed config row")
        digest = "sha256:" + str(value.get("sha256", ""))
        try:
            length = int(value.get("length", 0))
            raw = bytes(store.document(digest))
        except Exception as exc:
            raise ModelConfigRefusal(f"config {name!r} is unavailable: {exc}") from exc
        if (
            length <= 0
            or len(raw) != length
            or "sha256:" + hashlib.sha256(raw).hexdigest() != digest
        ):
            raise ModelConfigRefusal(f"config {name!r} does not match its exact reference")
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelConfigRefusal(f"config {name!r} is not JSON") from exc
        if not isinstance(document, dict):
            raise ModelConfigRefusal(f"config {name!r} is not an object")
        decoded[name] = document
        held.append(canonical.write(document))
    if len(held) == 1:
        raw = held[0]
        return "sha256:" + hashlib.sha256(raw).hexdigest(), raw
    raw = canonical.write(decoded)
    return "sha256:" + hashlib.sha256(raw).hexdigest(), raw
