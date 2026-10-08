"""Native executor writer/adoption and stored receipts without worker protobuf imports."""

from __future__ import annotations

import hashlib
import io
import struct
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorfs
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime import canonical_json
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.internal import weights_receipts, weights_sink
from cozy_runtime.internal.weights_writer import (
    ExecutionStorage,
    WriterAttempt,
    WriterBinding,
    WriterBroker,
)
from durable_seam import seam
from weights_channel import broker_lane


def native_exchange(root: Path) -> None:
    """Exercise the production executor storage client over real descriptor/native channels."""
    store = tensorfs.Store.init(root / "store")
    spool = root / "spool"
    spool.mkdir()
    attempt = SimpleNamespace(
        request_id="native-receipt-proof",
        attempt=1,
        digest=b"\x19" * 32,
        state="running",
        canceling="",
        spool=spool,
        weights_work_fingerprint="sha256:" + "20" * 32,
        spec={"inputs": [], "outputs": [{"output_id": "model", "max_bytes": 8}]},
    )
    recorded: list[Mapping[str, object]] = []
    checkpoints: list[Mapping[str, object]] = []
    broker: WriterBroker

    def bind(current: WriterAttempt, slot: str, declaration: bytes) -> tuple[str, int]:
        transaction = weights_sink.weights_transaction_id(
            "proof-owner", current.request_id, weights_receipts.spell(current.digest), slot
        )
        broker.authorize(current, transaction, 1, hashlib.sha256(declaration).digest(), slot)
        return transaction, 1

    def record(
        current: WriterAttempt,
        transaction: str,
        binding: WriterBinding,
        facts: Mapping[str, object],
    ) -> None:
        native = weights_sink.receipt_from_native(
            binding.slot, transaction, facts, replayed=False, request_id=current.request_id
        )
        assert native.output_slot == binding.slot
        assert native.weights_transaction_id == transaction
        assert native.tensorfs_receipt == canonical_json.encode(facts)
        assert (
            weights_receipts.raw(native.tensorfs_receipt_digest)
            == hashlib.sha256(native.tensorfs_receipt).digest()
        )
        assert store.derived_lookup(transaction)["receipt"] == facts
        recorded.append(dict(facts))

    broker = WriterBroker(
        lambda: store,
        checkpoint=lambda _a, _t, _b, facts: checkpoints.append(facts),
        receipt=record,
        bind_output=bind,
    )
    client = ExecutionStorage(spool, seam(broker_lane(broker, attempt)), {"model": 8})
    plain = dict(tensorfs.seed_digests())["plain/1"]
    definition = Derivation(
        {},
        {"model": Target(add={"weight": Tensor("f32", (2,), plain, {"value": Part("f32", (2,))})})},
        {},
        [("model", "weight")],
    )
    try:
        writer = client.open_output("model", definition)
        writer.add_part("model", "weight", "value", io.BytesIO(struct.pack("<ff", 1.0, 2.0)))
        head = writer.checkpoint()
        assert checkpoints == [head]
        facts = writer.commit()
        artifact = client.adopt_model(facts)
        receipt = weights_sink.native_receipt(facts)
        assert (
            artifact.producer_request_id == attempt.request_id and artifact.output_slot == "model"
        )
        assert artifact.manifest.digest == receipt.manifest.digest
        assert artifact.manifest.length == receipt.manifest.length
        assert recorded and all(reference == recorded[0] for reference in recorded)

        extended = {**facts, "future_observation": {"compiler": "newer"}}
        assert client.adopt_model(extended) == artifact
        assert all(reference == facts for reference in recorded)
        for field, value in (
            ("transaction_id", "sha256:" + "fe" * 32),
            ("manifest", {"sha256": "ff" * 32, "length": artifact.manifest.length}),
        ):
            with pytest.raises(CapabilityError) as changed_identity:
                client.adopt_model({**facts, field: value})
            assert changed_identity.value.code == "weights_receipt_mismatch"
        with pytest.raises(CapabilityError) as malformed:
            client.adopt_model(
                {**facts, "manifest": {"sha256": receipt.manifest.sha256, "length": "8"}}
            )
        assert malformed.value.code == "weights_receipt_mismatch"

        with pytest.raises(CapabilityError) as changed:
            client.adopt_model({**facts, "declaration_digest": "sha256:" + "ff" * 32})
        assert changed.value.code == "weights_receipt_mismatch"
        # Native receipt content remains bounded independently of the stored wrapper.
        with pytest.raises(CapabilityError) as oversized:
            client.adopt_model({**facts, "extra": "x" * weights_receipts.MAX_RECEIPT_BYTES})
        assert oversized.value.code == "weights_writer_length"
        assert store.derived_lookup(receipt.transaction_id)["receipt"] == facts
    finally:
        broker.close()


def test_native_executor_writer_receipt_adoption_without_old_protocol(tmp_path: Path) -> None:
    script = """
import importlib.abc
import pathlib
import runpy
import sys
class NoOldProtocol(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('cozy_runtime.protocol', 'cozy.worker.v1', 'google.protobuf')):
            raise AssertionError('native writer imported old protocol: ' + fullname)
sys.meta_path.insert(0, NoOldProtocol())
sys.path.insert(0, str(pathlib.Path(sys.argv[1]).parent))
proof = runpy.run_path(sys.argv[1])
proof['native_exchange'](pathlib.Path(sys.argv[2]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
