"""Explicit publication effects delegated to the admitted unpublished package owner."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from typing import cast

import msgspec

from cozy_runtime.author._artifacts import ModelArtifact, ObjectRef
from cozy_runtime.author._assets import FileAsset
from cozy_runtime.author._calls import _current
from cozy_runtime.author._errors import CapabilityError


class CheckpointRef(msgspec.Struct, frozen=True):
    destination: str
    checkpoint: str
    manifest: ObjectRef
    publication: str
    observation: str


class AssessmentRef(msgspec.Struct, frozen=True):
    destination: str
    checkpoint: str
    report: ObjectRef
    verdict: str
    observation: str


class ReleaseReceipt(msgspec.Struct, frozen=True):
    destination: str
    release: str
    revision: int
    lanes: dict[str, str]
    observation: str


def _call(operation: str, arguments: dict[str, object]) -> Awaitable[object]:
    broker = _current.get()
    if broker is None:
        raise CapabilityError("publication has no admitted attempt", code="child_broker_absent")
    return broker.reserve(__name__, operation, arguments)


def upload_checkpoint(artifact: ModelArtifact, *, destination: str) -> Awaitable[CheckpointRef]:
    """Explicitly retain this artifact at the destination; no script or memo history uploads."""
    return cast(
        Awaitable[CheckpointRef],
        _call("upload_checkpoint", {"artifact": artifact, "destination": destination}),
    )


def publish_release(
    *,
    destination: str,
    release: str,
    lanes: Mapping[str, CheckpointRef],
    expected_revision: int | None = None,
) -> Awaitable[ReleaseReceipt]:
    """Publish lane changes under one frozen revision, preserving unmentioned lanes.

    Omitted revision observes and freezes the current revision. Zero requires absence.
    A CheckpointRef is inert data and never grants destination write permission.
    """
    return cast(
        Awaitable[ReleaseReceipt],
        _call(
            "publish_release",
            {
                "destination": destination,
                "release": release,
                "lanes": dict(lanes),
                "expected_revision": expected_revision,
            },
        ),
    )


def attach_assessment(
    checkpoint: CheckpointRef, *, report: FileAsset, workloads: FileAsset
) -> Awaitable[AssessmentRef]:
    """Attach a verified assessment and workload sidecar to a published checkpoint.

    Both files must be completed native results, including owned files completed
    with ``await out.commit(...)``. The owner verifies their
    retained render/capture provenance and current destination authority before
    publication; report bytes never travel inside the managed call payload.
    """
    return cast(
        Awaitable[AssessmentRef],
        _call(
            "attach_assessment",
            {"checkpoint": checkpoint, "report": report, "workloads": workloads},
        ),
    )
