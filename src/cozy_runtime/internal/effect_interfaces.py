"""Fixed private publication effects sharing ordinary managed call-index admission."""

from __future__ import annotations

import hashlib

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._assets import FileAsset
from cozy_runtime.author._calls import _CallType
from cozy_runtime.author.publication import AssessmentRef, CheckpointRef, ReleaseReceipt
from cozy_runtime.internal import schema

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
    report: FileAsset
    workloads: FileAsset


_TYPES: dict[str, tuple[type[msgspec.Struct], type[msgspec.Struct]]] = {
    "upload_checkpoint": (Upload, CheckpointRef),
    "attach_assessment": (AttachAssessment, AssessmentRef),
    "publish_release": (Publish, ReleaseReceipt),
}
DOCUMENT = canonical_json.encode(
    {
        "format": "cozy.native-effect-interface/1",
        "module": MODULE,
        "operations": {
            name: {"request": schema.render(request), "result": schema.render(result)}
            for name, (request, result) in _TYPES.items()
        },
    }
)
DIGEST = "sha256:" + hashlib.sha256(DOCUMENT).hexdigest()


def bindings() -> dict[tuple[str, str, str], _CallType]:
    return {
        (prefix, MODULE, name): _CallType(DIGEST, MODULE, name, request, result)
        for name, (request, result) in _TYPES.items()
        for prefix in ("", DIGEST)
    }


def rows() -> list[dict[str, object]]:
    return [
        {"interface_digest": DIGEST, "module": MODULE, "export": name, "native_effect": True}
        for name in _TYPES
    ]
