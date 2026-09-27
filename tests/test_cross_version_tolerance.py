"""Independently deployed Runtime peers ignore additive and advisory metadata."""

from __future__ import annotations

import hashlib
import socket
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.author import CapabilityError, FileAsset
from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._call_results import decode
from cozy_runtime.author._capture import ExecutionObservation
from cozy_runtime.author.publication import CheckpointRef
from cozy_runtime.internal import census_cache, derive_child, effect_interfaces, source_interfaces
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.seam import Channel

STALE = "sha256:" + "0" * 64


def _broker(tmp_path: Path, rows: list[dict[str, object]]) -> object:
    left, right = socket.socketpair()
    with left, right:
        return Executor(Channel(left), tmp_path)._call_broker(
            {"request_id": "parent", "call_interfaces": rows}
        )


def test_native_operations_bind_when_the_recorded_interface_digest_moved(tmp_path: Path) -> None:
    rows = [
        {**row, "interface_digest": STALE}
        for row in (*source_interfaces.rows(), *effect_interfaces.rows())
    ]
    broker = _broker(tmp_path, rows)
    assert set(broker.bindings) == {(row["module"], row["export"]) for row in rows}  # type: ignore[attr-defined]


def test_an_operation_this_runtime_lacks_still_refuses(tmp_path: Path) -> None:
    row = {**source_interfaces.rows()[0], "export": "not_an_operation"}
    with pytest.raises(CapabilityError, match="no native operation"):
        _broker(tmp_path, [row])


def test_child_observations_and_receipts_accept_future_fields() -> None:
    observation = msgspec.convert(
        {"environment": {"runtime_version": "9.0.0", "future": 1}, "future": {}},
        type=ExecutionObservation,
        strict=True,
    )
    assert observation.environment.runtime_version == "9.0.0"
    digest = "sha256:" + "1" * 64
    reference = {"digest": digest, "length": 1, "future": True}
    checkpoint = msgspec.convert(
        {
            "destination": "alice/model",
            "checkpoint": digest,
            "manifest": reference,
            "publication": "upload",
            "observation": "acknowledged",
            "future": [],
        },
        type=CheckpointRef,
    )
    assert checkpoint.manifest.length == 1
    artifact = msgspec.convert(
        {
            "producer_request_id": "producer",
            "output_slot": "model",
            "manifest": reference,
            "tensorfs_receipt_digest": digest,
            "future": "advisory",
        },
        type=ModelArtifact,
    )
    assert artifact.output_slot == "model"


def test_child_result_asset_records_ignore_advisory_fields(tmp_path: Path) -> None:
    data = b"child bytes"
    path = tmp_path / "report.txt"
    path.write_bytes(data)
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    wire = msgspec.defstruct("Report", [("file", FileAsset)])
    record = {
        "asset_ref": digest,
        "kind": "file",
        "digest": digest,
        "size_bytes": len(data),
        "media_type": "text/plain",
        "label": "added by a newer child",
    }
    grant = {
        "output_id": "file",
        "kind": "file",
        "digest": digest,
        "length": len(data),
        "media_type": "text/plain",
        "local": str(path),
    }
    result, _ = decode(
        {"file": record},
        wire,
        [grant],
        request_id="parent",
        guard=lambda: None,
        observation=None,
    )
    assert result.file.digest == digest
    forged = {**grant, "digest": "sha256:" + "2" * 64}
    with pytest.raises(ValueError, match="verified host grant"):
        decode(
            {"file": record},
            wire,
            [forged],
            request_id="parent",
            guard=lambda: None,
            observation=None,
        )


def test_derive_child_rows_from_another_runtime_ignore_additive_fields() -> None:
    request = derive_child.DeriveRequest(
        (derive_child.SlotRequest("model", b"{}"), derive_child.SlotRequest("source", b"{}")),
        application="package:app",
    )
    rows = derive_child.read_response(
        {
            "requirements": [
                {
                    "slot": "model",
                    "components": ["unet"],
                    "tensors": [
                        {"component": "unet", "key": "w", "dtype": None, "shape": [2], "x": 1}
                    ],
                    "future": True,
                },
                {"slot": "source", "source": True, "future": True},
            ],
            "future": {},
        },
        request,
    )
    assert isinstance(rows[0], derive_child.TensorRequirements)
    assert rows[0].rows == (("unet", "w", None, (2,)),)
    assert rows[1] == derive_child.SourceSlot("source")


def test_a_torn_census_entry_is_a_miss(tmp_path: Path) -> None:
    entry = tmp_path / "census.json"
    entry.write_bytes(b'{"format": ')
    assert census_cache._read(entry) is None
