"""Real native custody, rather than surviving receipt metadata, governs replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tensorfs
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.author import WeightsReceipt
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.internal.weights_sink import receipt_from_native, same_receipt
from native_weights import NativeExecution


def _committed(
    tmp_path: Path,
) -> tuple[tensorfs.Store, NativeExecution, Derivation, WeightsReceipt]:
    store = tensorfs.Store.init(tmp_path / "store")
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    request = Derivation(
        {},
        {
            "model": Target(
                add={"weight": Tensor("f32", (512,), plain, {"value": Part("f32", (512,))})}
            )
        },
        {},
        (("model", "weight"),),
    )
    host = NativeExecution(store, tmp_path, "request", {}, {"result": 2048})
    with host.client.open_output("result", request) as writer:
        writer.add_part("model", "weight", "value", b"\x31" * 2048)
        facts = writer.commit()
    receipt = receipt_from_native(
        "result", facts["transaction_id"], facts, replayed=False, request_id="request"
    )
    return store, host, request, receipt


def test_pending_and_adopted_results_replay_but_disposed_metadata_refuses(tmp_path: Path) -> None:
    store, host, request, receipt = _committed(tmp_path)
    for disposition in ("pending", "adopted"):
        if disposition == "adopted":
            store.derived_adopt(receipt.weights_transaction_id, "owned-output")
        replay = host.client.open_output("result", request)
        assert replay.receipt is not None
        assert replay.commit() == json.loads(receipt.tensorfs_receipt)
        observed = store.derived_lookup(receipt.weights_transaction_id)
        assert observed["disposition"]["kind"] == disposition

    store.derived_dispose(receipt.weights_transaction_id)
    retained_metadata = store.derived_lookup(receipt.weights_transaction_id)
    assert retained_metadata["state"] == "committed"
    assert same_receipt(retained_metadata["receipt"], json.loads(receipt.tensorfs_receipt))
    with pytest.raises(CapabilityError) as error:
        host.client.open_output("result", request)
    assert error.value.code == "weights_transaction_closed"
    tensorfs.gc(store.root)
    facts = json.loads(receipt.tensorfs_receipt)
    with pytest.raises(tensorfs.errors.Refusal):
        store.manifest("sha256:" + facts["manifest"]["sha256"])
    with pytest.raises(CapabilityError) as adoption:
        host.client.adopt_model(facts)
    assert adoption.value.code == "weights_transaction_closed"
    host.broker.close()


def test_pending_receipt_without_native_root_is_not_reusable(tmp_path: Path) -> None:
    store, host, request, receipt = _committed(tmp_path)
    # Corrupt only this fixture's private root. The metadata still says Pending and
    # all payload bytes remain resident; neither fact may replace native retention.
    root = (
        Path(store.root)
        / "roots/derived"
        / (receipt.weights_transaction_id.removeprefix("sha256:") + ".json")
    )
    root.unlink()
    assert store.derived_lookup(receipt.weights_transaction_id)["state"] == "committed"
    with pytest.raises(CapabilityError) as error:
        host.client.open_output("result", request)
    assert error.value.code == "weights_receipt_unavailable"
    with pytest.raises(CapabilityError) as adoption:
        host.client.adopt_model(json.loads(receipt.tensorfs_receipt))
    assert adoption.value.code == "weights_receipt_unavailable"
    host.broker.close()
