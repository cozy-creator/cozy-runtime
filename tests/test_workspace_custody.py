"""Workspace ownership crosses real processes and protects actual native bytes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.author import WeightsPart, WeightsReceipt, WeightsTarget, WeightsTensor
from cozy_runtime.author._weights import WeightsCommit
from cozy_runtime.internal import canonical
from cozy_runtime.internal.weights_sink import (
    WeightsHostBinding,
    WeightsTransactionHost,
    protocol_receipt,
)
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def produced(
    tmp_path: Path, *, request_id: str = "producer", result_schema: bytes = b""
) -> tuple[tensorfs.Store, Workspace, WeightsReceipt]:
    store = tensorfs.Store.ensure(tmp_path / "store")
    workspace = Workspace(Path(store.root))
    value = json.loads(
        (
            Path(__file__).parent / "testdata/worker-protocol/canonical/invocation_spec_job.json"
        ).read_text()
    )
    for outputs in (value["outputs"], value["job"]["publication_contract"]["outputs"]):
        for output in outputs:
            output["mime_type"] = "application/vnd.cozy.model-manifest"
    invocation = canonical.write(value)
    spec = hashlib.sha256(invocation).digest()
    workspace.accept(
        "owner",
        pb.AttemptOffer(
            request_id=request_id,
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            invocation_spec_canonical_bytes=invocation,
        ),
        result_schema=result_schema,
    )
    declaration_digest = b""
    epoch = 0

    def bind(slot: str, transaction: str, digest: bytes, declaration: bytes) -> WeightsHostBinding:
        nonlocal declaration_digest, epoch
        declaration_digest = digest
        row = workspace.begin_weights(
            "owner",
            pb.WeightsIntentFrame(
                request_id=request_id,
                attempt_ordinal=1,
                invocation_spec_digest=spec,
                output_slot=slot,
                weights_transaction_id=transaction,
                tensorfs_declaration_digest=digest,
                tensorfs_declaration_canonical_bytes=declaration,
            ),
        )
        epoch = int(row["epoch"])
        return WeightsHostBinding(transaction, epoch)

    def record(receipt: WeightsReceipt) -> None:
        reference, _, _ = protocol_receipt(
            receipt,
            owner_scope="owner",
            request_id=request_id,
            invocation_spec_digest=documents.spell(spec),
        )
        manifest = json.loads(receipt.tensorfs_receipt)["manifest"]
        workspace.record_receipt(
            "owner",
            pb.WeightsReceiptFrame(
                request_id=request_id,
                attempt_ordinal=1,
                invocation_spec_digest=spec,
                output_slot="model",
                weights_transaction_id=receipt.weights_transaction_id,
                writer_epoch=epoch,
                tensorfs_declaration_digest=declaration_digest,
                weights_receipt=reference,
                manifest=pb.Ref(
                    digest=bytes.fromhex(manifest["sha256"]), length=manifest["length"]
                ),
            ),
        )

    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    commit = WeightsCommit(
        output_slot="model",
        sources={},
        targets={
            "model": WeightsTarget(
                add={
                    "weight": WeightsTensor(
                        logical_dtype="f32",
                        shape=(512,),
                        encoding=plain,
                        parts={"value": WeightsPart(dtype="f32", shape=(512,))},
                    )
                },
            )
        },
        configs={},
        order=(("model", "weight"),),
        max_new_bytes=2048,
    )
    producer = WeightsTransactionHost(
        store=store,
        owner_scope="owner",
        request_id=request_id,
        invocation_spec_digest=documents.spell(spec),
        work_fingerprint="sha256:" + "42" * 32,
        writer_session_id=1,
        allowed_sources={},
        output_bounds={"model": 2048},
        bind_intent=bind,
        record_receipt=record,
    )
    writer = producer.open(commit)
    writer.add_part("model", "weight", "value", b"\x31" * 2048)
    receipt = writer.commit()
    assert epoch == 1
    return store, workspace, receipt


def retention(receipt: WeightsReceipt, number: int) -> pb.DerivedRetentionRequest:
    return pb.DerivedRetentionRequest(
        weights_transaction_id=receipt.weights_transaction_id,
        tensorfs_receipt_digest=documents.raw(receipt.tensorfs_receipt_digest),
        retention_id="sha256:" + f"{number:064x}",
    )


def test_shared_journal_retains_native_result_after_original_disposal(tmp_path: Path) -> None:
    store, first, receipt = produced(tmp_path)
    held = first.retain("owner", retention(receipt, 1))
    first.release_result(
        "owner",
        pb.DerivedResultReleaseRequest(
            weights_transaction_id=receipt.weights_transaction_id,
            tensorfs_receipt_digest=documents.raw(receipt.tensorfs_receipt_digest),
        ),
    )
    second = Workspace(Path(store.root))
    next_hold = second.retain("owner", retention(receipt, 2))
    assert held.manifest == next_hold.manifest
    first.retain("owner", retention(receipt, 1), release=True)
    tensorfs.gc(str(store.root))
    assert store.manifest(documents.spell(held.manifest.digest))["manifest"]
    with pytest.raises(WorkspaceRefusal, match="cannot be recreated"):
        second.retain("owner", retention(receipt, 1))
    with pytest.raises(WorkspaceRefusal, match="another subject"):
        second.retain("foreign", retention(receipt, 2))


def test_process_restart_replays_same_hold_and_honors_release_tombstone(tmp_path: Path) -> None:
    store, workspace, receipt = produced(tmp_path)
    request = retention(receipt, 3)
    expected = workspace.retain("owner", request)
    command = [
        sys.executable,
        "-c",
        """
import sys
from pathlib import Path
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import worker_pb2 as pb
request = pb.DerivedRetentionRequest.FromString(bytes.fromhex(sys.argv[2]))
result = Workspace(Path(sys.argv[1])).retain('owner', request)
print(result.manifest.digest.hex())
""",
        str(store.root),
        request.SerializeToString().hex(),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected.manifest.digest.hex()
    workspace.retain("owner", request, release=True)
    refused = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert refused.returncode != 0
    assert "released retention cannot be recreated" in refused.stderr


def test_native_serving_lease_rechecks_owner_identity_and_release(tmp_path: Path) -> None:
    store, workspace, receipt = produced(tmp_path)
    source = retention(receipt, 17)
    held = workspace.retain("owner", source)
    with workspace.held_model("owner", source, held.manifest):
        with pytest.raises(tensorfs.errors.StoreBusy):
            tensorfs.gc(str(store.root))
        assert store.manifest(documents.spell(held.manifest.digest))["manifest"]
    with (
        pytest.raises(WorkspaceRefusal, match="exact retained"),
        workspace.held_model("foreign", source, held.manifest),
    ):
        pass
    wrong = pb.Ref(digest=held.manifest.digest, length=held.manifest.length + 1)
    with (
        pytest.raises(WorkspaceRefusal, match="exact retained"),
        workspace.held_model("owner", source, wrong),
    ):
        pass
    with (
        pytest.raises(WorkspaceRefusal, match="exact retained"),
        workspace.held_model("owner", source, held.manifest),
    ):
        workspace.retain("owner", source, release=True)
    with (
        pytest.raises(WorkspaceRefusal, match="exact retained"),
        workspace.held_model("owner", source, held.manifest),
    ):
        pass
