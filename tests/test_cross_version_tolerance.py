"""Independently deployed Runtime peers ignore additive and advisory metadata."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.author import CapabilityError, ConformanceError, FileAsset
from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._call_results import decode
from cozy_runtime.author._capture import ExecutionObservation
from cozy_runtime.author._executor_requests import (
    Answer,
    Checkpoint,
    CheckpointReceipt,
    ChildCall,
    respond,
)
from cozy_runtime.author._executor_requests import decode as decode_request
from cozy_runtime.author.fakes import fake_context
from cozy_runtime.author.publication import CheckpointRef
from cozy_runtime.internal import (
    census_cache,
    derive_child,
    native_interfaces,
    source_interfaces,
    witness,
)
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import CallInterface
from cozy_runtime.internal.seam import Channel
from cozy_runtime.models.minimax_h3.table_layout import TableLayout


def _broker(tmp_path: Path, rows: list[dict[str, object]]) -> object:
    left, right = socket.socketpair()
    with left, right:
        return Executor(Channel(left), tmp_path)._call_broker(
            "parent", msgspec.convert(rows, tuple[CallInterface, ...])
        )


def test_native_rows_bind_per_operation(tmp_path: Path) -> None:
    rows = native_interfaces.rows({"native_interfaces": native_interfaces.stated()})
    listed = [{**rows[0], "export": "not_an_operation"}, *rows[1:]]
    broker = _broker(tmp_path, listed)
    bindings = broker.bindings  # type: ignore[attr-defined]
    # An operation this Runtime lacks cannot be called from here, so its row is ignored.
    assert (source_interfaces.MODULE, "not_an_operation") not in bindings
    # One the worker did not list is refused alone, naming the rental update.
    assert not any(b.unavailable for key, b in bindings.items() if key[1] != rows[0]["export"])
    broker.bind(fake_context(request_id="parent"))  # type: ignore[attr-defined]
    with pytest.raises(CapabilityError, match="cozy rental update") as refused:
        broker.reserve(rows[0]["module"], rows[0]["export"], {})  # type: ignore[attr-defined]
    assert refused.value.code == "child_capability_unavailable"


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
    newer = {
        "requirements": [
            {
                "slot": "model",
                "components": ["unet"],
                "tensors": [{"component": "unet", "key": "w", "dtype": None, "shape": [2], "x": 1}],
                "future": True,
            },
            {"slot": "source", "source": True, "future": True},
        ],
        "future": {},
    }
    response = msgspec.msgpack.decode(
        msgspec.msgpack.encode(newer), type=derive_child.DeriveResponse
    )
    rows = derive_child.read_response(response, request)
    assert isinstance(rows[0], derive_child.TensorRequirements)
    assert rows[0].rows == (("unet", "w", None, (2,)),)
    assert rows[1] == derive_child.SourceSlot("source")


def test_durable_requests_and_answers_cross_older_and_newer_peers(tmp_path: Path) -> None:
    digest = "sha256:" + "0" * 64
    checkpoint = Checkpoint(operation_key="op", logical_key="k", content_digest=digest, length=3)
    # An older executor's frames: a checkpoint path, no capture or label, a future field.
    older = {**msgspec.to_builtins(checkpoint), "path": "/spool/k", "future": 1}
    assert decode_request(older) == checkpoint
    call = {"kind": "child_call", "call_index": 0, "module": "m", "export": "e", "payload": "{}"}
    assert decode_request(call) == ChildCall(call_index=0, module="m", export="e", payload="{}")
    # A newer executor's kind is refused alone, answered under its own sequence number.
    answer, handoff = respond(
        {"event": "request", "seq": 7, "kind": "future"}, lambda _: Answer(ok=True)
    )
    assert handoff is None
    assert (answer["seq"], answer["ok"], answer["code"]) == (7, False, "unknown_durable_request")
    # The executor's real exchange against an older worker's answer, which has no code.
    left, right = socket.socketpair()
    with left, right:
        worker = Channel(right)
        worker.send({"event": "answer", "seq": 1, "ok": True, "receipt_id": "r", "future": 1})
        received = Executor(Channel(left), tmp_path)._durable(checkpoint, CheckpointReceipt)
        assert received == CheckpointReceipt(ok=True, receipt_id="r")
        # Every key an older worker indexes is written.
        assert worker.recv() == {
            "event": "request",
            "seq": 1,
            "kind": "checkpoint",
            "operation_key": "op",
            "logical_key": "k",
            "content_digest": digest,
            "length": 3,
        }


def test_a_torn_census_entry_is_a_miss(tmp_path: Path) -> None:
    entry = tmp_path / "census.json"
    entry.write_bytes(b'{"format": ')
    assert census_cache._read(entry) is None


def test_witness_logs_from_other_versions_keep_their_known_rows(tmp_path: Path) -> None:
    log = tmp_path / "witness" / "events.jsonl"
    log.parent.mkdir()
    death = witness.name_death(0, 0, 0, child=7, started=0.0)
    rows = [{"event": "future_kind"}, {**msgspec.to_builtins(death), "future": 1}]
    log.write_text("\n".join(json.dumps(row) for row in rows) + '\n{"event": "de')
    assert witness.read_death(tmp_path) == death


def test_h3_table_labels_from_other_writers_ignore_extras_and_default_optionals() -> None:
    t = (1.0).hex()
    layout = TableLayout.parse(
        {
            "final_normalization": [{"timestep": t, "writer": "future"}],
            "block_modulation": [
                {"index": 0, "timestep": t, "modality_tag": 0, "modality": "video"},
                {"timestep": t, "modality": "text"},
                {"timestep": t, "modality_tag": 2, "note": 1},
            ],
            "provenance": {"tool": "future"},
        }
    )
    assert layout.block_keys == ((0, 0), (0, 1), (0, 2))
    with pytest.raises(ConformanceError, match="modalities"):
        TableLayout.parse(
            {
                "final_normalization": [{"timestep": t}],
                "block_modulation": [{"timestep": t, "modality_tag": 0, "modality": "audio"}],
            }
        )
