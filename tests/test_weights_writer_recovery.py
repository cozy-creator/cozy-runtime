"""The native writer channel preserves completed roles across attempt replacement."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorfs
from tensorfs.derived import Derivation, Part, Source, Target, Tensor

from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._executor_requests import Request, WriterOutput, decode, encode
from cozy_runtime.internal.weights_writer import WriterBroker
from cozy_runtime.internal.worker.grants import MODEL_PREFIX
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from weights_channel import open_output


def test_parent_broker_resumes_native_roles_and_rejects_child_fingerprint_or_head(
    tmp_path: Path,
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    spool = tmp_path / "spool"
    spool.mkdir()
    encoding = next(d for a, d in tensorfs.seed_digests() if a == "fp8-rowwise/1")
    order = [("model", "weight")]
    maximum = (4 << 20) + 2048
    fingerprint = "sha256:" + "71" * 32
    observed: list[Mapping[str, object]] = []
    committed: list[Mapping[str, object]] = []
    broker = WriterBroker(
        lambda: store,
        checkpoint=lambda _a, _t, _b, facts: observed.append(facts),
        receipt=lambda _a, _t, _b, facts: committed.append(facts),
    )
    attempt = SimpleNamespace(
        request_id="request",
        attempt=1,
        state="running",
        canceling="",
        spool=spool,
        weights_work_fingerprint=fingerprint,
        spec={"inputs": [], "outputs": [{"output_id": "model", "max_bytes": maximum}]},
    )
    frames: list[Request] = []

    transaction = "sha256:" + "72" * 32
    definition = Derivation(
        {},
        {
            "model": Target(
                add={
                    "weight": Tensor(
                        "bf16",
                        (512, 8192),
                        encoding,
                        {"data": Part("f8_e4m3fn", (512, 8192)), "scale": Part("f32", (512,))},
                    )
                }
            )
        },
        {},
        order,
    )
    with pytest.raises(CapabilityError) as retired:
        decode(
            {
                "kind": "weights_writer",
                "operation": "open",
                "weights_transaction_id": transaction,
                "writer_epoch": 91,
                "work_fingerprint": "sha256:" + "73" * 32,
            }
        )
    assert retired.value.code == "durable_request_malformed"
    assert store.derived_lookup(transaction)["state"] == "absent"
    writer = open_output(broker, attempt, transaction, definition, frames=frames)
    assert writer.completed_parts() == []
    data = b"\x31" * (4 << 20)
    writer.add_part("model", "weight", "data", io.BytesIO(data))
    assert writer.completed_parts() == [("model", "weight", "data")]
    head = writer.checkpoint()
    assert observed == [head]
    with pytest.raises(tensorfs.errors.ArtifactIncomplete):
        writer.commit()  # Native refusal fences only authority, preserving the complete role.
    broker.close_attempt(attempt)
    assert broker.writers == {} and broker.bindings == {}
    tensorfs.gc(store.root)

    attempt.attempt = 2
    restored_head = (head["head"], head["head_length"])
    writer = open_output(
        broker, attempt, transaction, definition, checkpoint=restored_head, frames=frames
    )
    assert writer.completed_parts() == [("model", "weight", "data")]
    assert writer.completed_configs() == []
    assert writer.checkpoint() == head
    writer.add_part("model", "weight", "scale", io.BytesIO(bytes(2048)))
    receipt = writer.commit()
    assert committed == [receipt]
    assert len(receipt["added_objects"]) == 2
    broker.close_attempt(attempt)
    assert broker.writers == {} and broker.bindings == {}
    assert {type(frame) for frame in frames} == {WriterOutput}
    assert not list(spool.glob("*-input")) and not list(spool.glob("*-output"))
    # No payload or model-sized role inventory crosses the durable control request.
    assert all(
        set(encode(frame)) == {"kind", "operation", "output_slot", "length"} for frame in frames
    )


def test_parent_broker_spends_zero_protobuf_output_grant_on_native_graft(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    plain = dict(tensorfs.seed_digests())["plain/1"]
    source_writer = store.begin_derived(
        "sha256:" + "81" * 32,
        1,
        {},
        {
            "model": {
                "drop": [],
                "add": {
                    "weight": {
                        "logical_dtype": "f32",
                        "shape": [2],
                        "encoding": plain,
                        "parts": {"value": {"dtype": "f32", "shape": [2]}},
                    }
                },
            }
        },
        {},
        [("model", "weight")],
        8,
        work_fingerprint="sha256:" + "82" * 32,
    )
    source_writer.add_part("model", "weight", "value", io.BytesIO(bytes(8)))
    manifest = source_writer.commit()["manifest"]
    source = "sha256:" + manifest["sha256"], manifest["length"]
    sources = {"source": source}
    targets = {"model": {"source": "source", "source_component": "model", "drop": [], "add": {}}}
    order = [("model", "weight")]
    fingerprint = "sha256:" + "83" * 32
    declaration = store.derived_declaration(
        sources, targets, {}, order, 0, work_fingerprint=fingerprint
    )
    spec = documents.body(
        pb.InvocationSpec(
            inputs=[
                pb.InputBinding(
                    input_id=MODEL_PREFIX + "source", digest=source[0], length=source[1]
                )
            ],
            outputs=[
                pb.OutputBinding(
                    output_id="model", max_bytes=0, mime_type="application/vnd.cozy.model-manifest"
                )
            ],
        )
    )
    assert "max_bytes" not in spec["outputs"][0]
    spool = tmp_path / "spool"
    spool.mkdir()
    attempt = SimpleNamespace(
        request_id="graft",
        attempt=1,
        state="running",
        canceling="",
        spool=spool,
        weights_work_fingerprint=fingerprint,
        spec=spec,
    )
    recorded: list[Mapping[str, object]] = []
    broker = WriterBroker(
        lambda: store,
        checkpoint=lambda _a, _t, _b, facts: recorded.append(facts),
        receipt=lambda _a, _t, _b, facts: recorded.append(facts),
    )
    transaction = "sha256:" + "84" * 32
    broker.authorize(attempt, transaction, 1, hashlib.sha256(declaration).digest(), "model")
    definition = Derivation(
        {"source": Source(*source)},
        {"model": Target(source="source", source_component="model")},
        {},
        order,
    )
    with pytest.raises(CapabilityError) as excess:
        open_output(broker, attempt, transaction, definition, maximum=1)
    assert excess.value.code == "weights_writer_ungranted"
    writer = open_output(broker, attempt, transaction, definition)
    receipt = writer.commit()
    assert recorded == [receipt]
    assert receipt["added_objects"] == []
    assert receipt["manifest"] == manifest
    broker.close_attempt(attempt)
