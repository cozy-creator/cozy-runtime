"""Fixed private publication effects sharing ordinary managed call-index admission."""

from __future__ import annotations

import dataclasses

import msgspec

from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._calls import _CallType
from cozy_runtime.author._files import File
from cozy_runtime.author.publication import AssessmentRef, CheckpointRef, ReleaseReceipt

MODULE = "cozy_runtime.author.publication"


class Upload(msgspec.Struct, frozen=True):
    artifact: ModelArtifact
    destination: str


class Publish(msgspec.Struct, frozen=True):
    destination: str
    release: str
    lanes: dict[str, CheckpointRef]
    expected_revision: int | None = None


class AttachAssessment(msgspec.Struct, frozen=True):
    checkpoint: CheckpointRef
    report: File
    workloads: File


TYPES: dict[str, tuple[type[msgspec.Struct], type[msgspec.Struct]]] = {
    "upload_checkpoint": (Upload, CheckpointRef),
    "attach_assessment": (AttachAssessment, AssessmentRef),
    "publish_release": (Publish, ReleaseReceipt),
}


def bindings() -> dict[str, _CallType]:
    return {
        name: _CallType("", MODULE, name, request, result)
        for name, (request, result) in TYPES.items()
    }


def unserved(export: str) -> _CallType:
    """An effect the machine did not offer this executor: a call to it is refused."""
    detail = f'native operation "{export}" is not offered by this machine'
    return dataclasses.replace(bindings()[export], unavailable=detail)
