"""Fixed effects reserve the same call index namespace and never invoke package code."""

from __future__ import annotations

import time
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.author import (
    App,
    CheckpointRef,
    Context,
    FileAsset,
    Invocation,
    ModelArtifact,
    ObjectRef,
    attach_assessment,
    attempt,
    publish_release,
    upload_checkpoint,
)
from cozy_runtime.author._calls import _Broker
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.internal.effect_interfaces import bindings
from durable_seam import wire


class Empty(msgspec.Struct):
    pass


def test_publication_requires_admitted_call_broker() -> None:
    model = ModelArtifact(
        "producer", "model", ObjectRef("sha256:" + "1" * 64, 100), "sha256:" + "2" * 64
    )
    with pytest.raises(CapabilityError, match="no admitted attempt"):
        upload_checkpoint(model, destination="alice/model")
    with pytest.raises(CapabilityError, match="no admitted attempt"):
        publish_release(destination="alice/model", release="v1", lanes={})


def test_effect_reservations_use_existing_parent_index_and_refuse_unknown_inputs(
    tmp_path: Path,
) -> None:
    app = App()
    model = ModelArtifact(
        "producer", "model", ObjectRef("sha256:" + "1" * 64, 100), "sha256:" + "2" * 64
    )

    @app.job
    async def script(ctx: Context, payload: Empty) -> Empty:
        upload_checkpoint(model, destination="alice/model")
        publish_release(destination="alice/model", release="v1", lanes={})
        return Empty()

    def exchange(*_: object) -> dict[str, object]:
        raise AssertionError("unawaited effects must not be sent")

    broker = _Broker(
        "parent", {(b.module, b.export): b for b in bindings().values()}, wire(exchange)
    )
    _, outcome, _ = attempt(
        app.get("script"), {}, Invocation("parent", tmp_path, time.monotonic() + 10, calls=broker)
    )
    assert outcome.code == "unawaited_child_call"
    assert [call.index for call in broker.calls.values()] == [0, 1]
    assert [call.binding.export for call in broker.calls.values()] == [
        "upload_checkpoint",
        "publish_release",
    ]


def test_assessment_refuses_files_without_received_custody(tmp_path: Path) -> None:
    app = App()
    digest = "sha256:" + "1" * 64
    checkpoint = CheckpointRef(
        "alice/model", digest, ObjectRef(digest, 100), "upload", "acknowledged"
    )

    @app.job
    async def script(ctx: Context, payload: Empty) -> Empty:
        await attach_assessment(checkpoint, report=FileAsset(digest), workloads=FileAsset(digest))
        return Empty()

    def exchange(*_: object) -> dict[str, object]:
        raise AssertionError("unreceived artifact must not reach the effect owner")

    broker = _Broker(
        "parent", {(b.module, b.export): b for b in bindings().values()}, wire(exchange)
    )
    _, outcome, _ = attempt(
        app.get("script"), {}, Invocation("parent", tmp_path, time.monotonic() + 10, calls=broker)
    )
    assert outcome.code == "child.asset_ungranted"
    assert not broker.calls
