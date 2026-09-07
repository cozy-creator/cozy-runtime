"""The real parent broker moves native role state through its existing bounded spool."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tensorfs

from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.internal.weights_writer import StoreProxy, WriterBroker


def test_parent_broker_resumes_native_roles_and_rejects_child_fingerprint_or_head(
    tmp_path: Path,
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    spool = tmp_path / "spool"
    spool.mkdir()
    encoding = next(d for a, d in tensorfs.seed_digests() if a == "fp8-rowwise/1")
    targets = {
        "model": {
            "drop": [],
            "add": {
                "weight": {
                    "logical_dtype": "bf16",
                    "shape": [512, 8192],
                    "encoding": encoding,
                    "parts": {
                        "data": {"dtype": "f8_e4m3fn", "shape": [512, 8192]},
                        "scale": {"dtype": "f32", "shape": [512]},
                    },
                }
            },
        }
    }
    order = [("model", "weight")]
    maximum = (4 << 20) + 2048
    fingerprint = "sha256:" + "71" * 32
    declaration = store.derived_declaration(
        {}, targets, {}, order, maximum, work_fingerprint=fingerprint
    )
    digest = hashlib.sha256(declaration).digest()
    observed: list[dict[str, Any]] = []
    broker = WriterBroker(
        lambda: store, checkpoint=lambda _a, _t, _b, facts: observed.append(facts)
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
    frames: list[dict[str, Any]] = []

    def exchange(kind: str, frame: dict[str, Any]) -> dict[str, Any]:
        assert kind == "weights_writer"
        frames.append(dict(frame))
        return broker.handle(attempt, frame)

    proxy = StoreProxy(store, spool, exchange)
    transaction = "sha256:" + "72" * 32
    broker.authorize(attempt, transaction, 1, digest, "model")
    with pytest.raises(CapabilityError) as changed:
        proxy.begin_derived(
            transaction, 1, {}, targets, {}, order, maximum, work_fingerprint="sha256:" + "73" * 32
        )
    assert changed.value.code == "weights_writer_declaration"
    assert store.derived_lookup(transaction)["state"] == "absent"
    writer = proxy.begin_derived(
        transaction, 1, {}, targets, {}, order, maximum, work_fingerprint=fingerprint
    )
    assert writer.completed_parts() == []
    data = b"\x31" * (4 << 20)
    writer.add_part("model", "weight", "data", io.BytesIO(data))
    assert writer.completed_parts() == [["model", "weight", "data"]]
    head = writer.checkpoint("request", "model")
    assert observed == [head]
    with pytest.raises(CapabilityError):
        writer.commit()  # Native refusal fences only authority, preserving the complete role.
    broker.close_attempt(attempt)
    assert broker.writers == {} and broker.bindings == {}
    tensorfs.gc(store.root)

    attempt.attempt = 2
    restored_head = (head["head"], head["head_length"])
    broker.authorize(attempt, transaction, 2, digest, "model", checkpoint=restored_head)
    with pytest.raises(CapabilityError) as wrong_head:
        proxy.begin_derived(
            transaction, 2, {}, targets, {}, order, maximum, work_fingerprint=fingerprint
        )
    assert wrong_head.value.code == "weights_writer_declaration"
    writer = proxy.begin_derived(
        transaction,
        2,
        {},
        targets,
        {},
        order,
        maximum,
        work_fingerprint=fingerprint,
        checkpoint=restored_head,
    )
    assert writer.completed_parts() == [["model", "weight", "data"]]
    assert writer.completed_configs() == []
    assert writer.checkpoint("request", "model", previous=restored_head) == head
    writer.add_part("model", "weight", "scale", io.BytesIO(bytes(2048)))
    receipt = writer.commit()
    assert len(receipt["added_objects"]) == 2
    broker.close_attempt(attempt)
    assert broker.writers == {} and broker.bindings == {}
    # No payload or model-sized role inventory crosses the durable control request.
    assert all(
        set(frame)
        <= {
            "operation",
            "weights_transaction_id",
            "writer_epoch",
            "length",
            "target_operation",
            "component",
            "key",
            "role",
            "operation_id",
            "slot",
            "previous",
        }
        for frame in frames
    )
