"""Negotiated machine-owned model byte sources over the existing durable executor seam.

Only preparation requests descriptors. Header/fit/plan/model behavior remain the existing
TensorFS and Runtime implementations; no step or block crosses this broker.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import msgspec

try:
    from tensorfs import plane as native_plane
except ImportError:
    native_plane = None

from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._executor_requests import Exchange, ModelSourceRead, SourceBlob

if TYPE_CHECKING:
    from tensorfs import ReadPlan
    from tensorfs.plane import Plane, Source, SourceDescriptor

CAPABILITY = "model_sources.descriptors/1"


class ModelSourceProvider(Protocol):
    def header(self, manifest: str) -> bytes: ...
    def assets(self, manifest: str, names: Sequence[str]) -> dict[str, bytes]: ...
    def source(self, plane: Plane, manifest: str, plan: ReadPlan) -> Source: ...
    def close(self) -> None: ...


class _ObjectItem(msgspec.Struct, tag="object", tag_field="source", frozen=True):
    object: str
    object_length: int


class _InlineItem(msgspec.Struct, tag="inline", tag_field="source", frozen=True):
    pass


def available() -> bool:
    return native_plane is not None and hasattr(native_plane, "SourceDescriptor") and hasattr(
        native_plane.Plane, "source_from_descriptors"
    )


class BrokeredModelSources:
    def __init__(self, exchange: Exchange):
        if not available():
            raise CapabilityError("this weight plane has no descriptor source operation",
                                  code="model_source_unavailable")
        self.exchange = exchange
        self.headers: dict[str, bytes] = {}
        self.descriptors: dict[tuple[str, str], SourceDescriptor] = {}

    def _open(self, request: ModelSourceRead) -> SourceBlob:
        reply = self.exchange(request, SourceBlob)
        if not reply.ok or reply.descriptor < 0:
            if reply.descriptor >= 0:
                os.close(reply.descriptor)
            raise CapabilityError(reply.detail or "machine declined this source operation",
                                  code=reply.code or "model_source_unavailable")
        return reply

    def _bytes(self, request: ModelSourceRead) -> bytes:
        reply = self._open(request)
        try:
            if os.fstat(reply.descriptor).st_size != reply.length:
                raise CapabilityError("source blob length changed", code="model_source_changed")
            data = bytearray()
            digest = hashlib.sha256()
            while len(data) < reply.length:
                chunk = os.pread(
                    reply.descriptor, min(1 << 20, reply.length - len(data)), len(data)
                )
                if not chunk:
                    raise CapabilityError("source blob ended early", code="model_source_changed")
                data.extend(chunk)
                digest.update(chunk)
            if digest.hexdigest() != reply.sha256:
                raise CapabilityError("source blob digest differs", code="model_source_changed")
            return bytes(data)
        finally:
            os.close(reply.descriptor)

    def header(self, manifest: str) -> bytes:
        if manifest not in self.headers:
            self.headers[manifest] = self._bytes(ModelSourceRead(manifest=manifest, role="header"))
        return self.headers[manifest]

    def assets(self, manifest: str, names: Sequence[str]) -> dict[str, bytes]:
        return {name: self._bytes(ModelSourceRead(manifest=manifest, role="asset", name=name))
                for name in names}

    def source(self, plane: Plane, manifest: str, plan: ReadPlan) -> Source:
        rows = msgspec.convert(plan.items(), list[_ObjectItem | _InlineItem], strict=True)
        wanted: dict[str, int] = {}
        for row in rows:
            if isinstance(row, _ObjectItem):
                if row.object in wanted and wanted[row.object] != row.object_length:
                    raise CapabilityError("read plan conflicts on object length",
                                          code="model_source_changed")
                wanted[row.object] = row.object_length
        descriptors = []
        for object_id, length in wanted.items():
            key = (manifest, object_id)
            if key not in self.descriptors:
                reply = self._open(ModelSourceRead(manifest=manifest, role="object",
                                                   name=object_id, length=length))
                try:
                    if reply.sha256 != object_id.removeprefix("sha256:") or reply.length != length:
                        raise CapabilityError("object grant differs from its read plan",
                                              code="model_source_changed")
                    self.descriptors[key] = native_plane.SourceDescriptor(
                        reply.sha256, reply.length, reply.descriptor
                    )
                finally:
                    os.close(reply.descriptor)
            descriptors.append(self.descriptors[key])
        return plane.source_from_descriptors(descriptors)

    def close(self) -> None:
        self.descriptors.clear()
        self.headers.clear()
