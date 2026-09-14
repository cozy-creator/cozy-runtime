"""Byte capacity derived from accepted fixed output bindings, never executor claims."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cozy_runtime.author._services import MAX_OUTPUT_BYTES


def admitted(spec: Mapping[str, Any]) -> int:
    """Keep each byte slot bounded while honoring the sum of its admitted slots."""
    outputs = spec.get("outputs", [])
    if not isinstance(outputs, list) or len(outputs) > 32:
        raise ValueError("fixed output inventory exceeds its bound")
    sizes = []
    for row in outputs:
        if row.get("mime_type") == "application/vnd.cozy.model-manifest":
            continue
        size = row.get("max_bytes", 0)
        if type(size) is not int or size < 0:
            raise ValueError("output byte bound must be a nonnegative integer")
        sizes.append(min(size or MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
    return sum(sizes) if sizes else MAX_OUTPUT_BYTES


def effective(spec: Mapping[str, Any], reply: Mapping[str, Any]) -> int:
    """An executor may use a smaller budget but cannot enlarge admitted capacity."""
    limit = admitted(spec)
    value = reply.get("max_output_bytes", limit)
    if type(value) is not int or value < 0 or value > limit:
        raise ValueError("executor output budget exceeds its admitted bound")
    return value
