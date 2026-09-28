"""Byte capacity derived from accepted fixed output bindings, never executor claims."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

import msgspec

from cozy_runtime.author._services import MAX_OUTPUT_BYTES

_MANIFEST = "application/vnd.cozy.model-manifest"
_Bytes = Annotated[int, msgspec.Meta(ge=0)]


class _Output(msgspec.Struct, frozen=True):
    mime_type: str = ""
    #: 0: unbounded, which is the service cap
    max_bytes: _Bytes = 0


class _Spec(msgspec.Struct, frozen=True):
    """What the budget reads of an InvocationSpec document."""

    outputs: Annotated[list[_Output], msgspec.Meta(max_length=32)] = []


class _Reply(msgspec.Struct, frozen=True):
    """What the budget reads of an executor reply: absent, the admitted bound applies."""

    max_output_bytes: _Bytes | msgspec.UnsetType = msgspec.UNSET


def admitted(spec: Mapping[str, object]) -> int:
    """Keep each byte slot bounded while honoring the sum of its admitted slots.

    A malformed inventory raises `msgspec.ValidationError`, a `ValueError`."""
    sizes = [
        min(row.max_bytes or MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES)
        for row in msgspec.convert(spec, _Spec).outputs
        if row.mime_type != _MANIFEST
    ]
    return sum(sizes) if sizes else MAX_OUTPUT_BYTES


def effective(spec: Mapping[str, object], reply: Mapping[str, object]) -> int:
    """The reported working allowance cannot enlarge exact final-output authority."""
    limit = admitted(spec)
    value = msgspec.convert(reply, _Reply).max_output_bytes
    if value is msgspec.UNSET:
        return limit
    if value > intermediate(spec):
        raise ValueError("executor output budget exceeds its admitted bound")
    return min(value, limit)


def intermediate(spec: Mapping[str, object]) -> int:
    """A small final result does not reduce the existing retained-work allowance."""
    return max(MAX_OUTPUT_BYTES, admitted(spec))
