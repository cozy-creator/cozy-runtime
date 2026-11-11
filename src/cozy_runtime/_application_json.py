"""Bounded typed application JSON, independent of protocol JCS numeric semantics."""

from __future__ import annotations

from cozy_runtime.internal import canonical


def encode(value: object) -> bytes:
    try:
        return canonical.write_application(value)
    except canonical.CanonicalError as exc:
        raise ValueError(str(exc)) from exc


def decode(data: bytes) -> canonical.Json:
    try:
        return canonical.parse_application(data)
    except canonical.CanonicalError as exc:
        raise ValueError(str(exc)) from exc
