"""Portable exact model references. Identity is data; the owner grants custody."""

from __future__ import annotations

import re

import msgspec

from cozy_runtime.author._errors import ConformanceError

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_REQUEST = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_SLOT = re.compile(r"[a-z][a-z0-9_-]{0,63}")


class ObjectRef(msgspec.Struct, frozen=True):
    digest: str
    length: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.digest, str)
            or _DIGEST.fullmatch(self.digest) is None
            or type(self.length) is not int
            or not 0 < self.length <= (1 << 53) - 1
        ):
            raise ConformanceError(
                "artifact reference requires an exact digest and positive length",
                code="artifact_reference",
            )


class ModelArtifact(msgspec.Struct, frozen=True):
    """A produced model's original receipt provenance, independent of its current holder.

    Reusing it never rewrites the producer or receipt. The RecordOwner verifies
    current access and acquires a new native hold before delegating it to a child.
    Neither this value nor its digest grants access by itself.
    """

    producer_request_id: str
    output_slot: str
    manifest: ObjectRef
    tensorfs_receipt_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.producer_request_id, str)
            or _REQUEST.fullmatch(self.producer_request_id) is None
            or not isinstance(self.output_slot, str)
            or _SLOT.fullmatch(self.output_slot) is None
            or not isinstance(self.manifest, ObjectRef)
            or not isinstance(self.tensorfs_receipt_digest, str)
            or _DIGEST.fullmatch(self.tensorfs_receipt_digest) is None
        ):
            raise ConformanceError(
                "model artifact requires exact producer, output and native receipt provenance",
                code="model_artifact",
            )


class SourceArtifact(msgspec.Struct, frozen=True):
    """Exact retained raw source bytes. Identity never grants access by itself.

    The native source receipt preserves original producer history while each
    request and memo entry acquire independent byte-tree custody.
    """

    producer_request_id: str
    output_slot: str
    manifest: ObjectRef
    tensorfs_receipt_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.producer_request_id, str)
            or _REQUEST.fullmatch(self.producer_request_id) is None
            or self.output_slot != "source"
            or not isinstance(self.manifest, ObjectRef)
            or not isinstance(self.tensorfs_receipt_digest, str)
            or _DIGEST.fullmatch(self.tensorfs_receipt_digest) is None
        ):
            raise ConformanceError(
                "source artifact requires exact producer and native byte-tree receipt",
                code="source_artifact",
            )
