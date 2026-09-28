"""Owner finalization through real TensorFS state, including the absent-writer case."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from threading import Event
from typing import Any

import pytest
import tensorfs

from cozy_runtime.internal import weights_sink
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.weights import WeightsExchange, WeightsExchangeError
from cozy_runtime.internal.worker.weights_finalize import FinalizationRefusal, finalize
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_derived_config_runtime import _released_source


def _request() -> pb.WeightsFinalizeRequest:
    raw, digest = documents.identity(
        pb.InvocationSpec(
            installation_id="local-" + "e" * 32,
            payload_digest="sha256:" + "d" * 64,
            outputs=[
                pb.OutputBinding(
                    output_id="model",
                    mime_type="application/vnd.cozy.model-manifest",
                    max_bytes=1 << 20,
                )
            ],
            job=pb.JobInvocationSpec(
                installation_id="local-" + "b" * 32, job_descriptor_id="c" * 64
            ),
        )
    )
    return pb.WeightsFinalizeRequest(
        request_id="request-finalize",
        invocation_spec_digest=digest,
        invocation_spec_canonical_bytes=raw,
        output_slot="model",
        owner_authority_scope="owner",
        disposition=pb.WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED,
    )


def _attempt(request: pb.WeightsFinalizeRequest, *, spec: bool = True) -> AttemptRecord:
    return AttemptRecord(
        request_id=request.request_id,
        attempt=1,
        digest=request.invocation_spec_digest,
        spec={
            "job": {},
            "outputs": [{"output_id": "model", "mime_type": "application/vnd.cozy.model-manifest"}],
        }
        if spec
        else {},
        state="outcome",
        kind="job",
    )


def _transaction(request: pb.WeightsFinalizeRequest) -> str:
    return weights_sink.weights_transaction_id(
        "owner", request.request_id, "sha256:" + request.invocation_spec_digest.hex(), "model"
    )


def _finish(
    request: pb.WeightsFinalizeRequest, root: Path, attempts: list[AttemptRecord]
) -> pb.WeightsFinalizeResult:
    return finalize(
        request,
        owner_scope="owner",
        attempts=attempts,
        tensorfs_root=root,
        journal_root=root.parent / "worker",
    )


def _writer(store: Any, request: pb.WeightsFinalizeRequest, manifest: str, length: int) -> Any:
    return store.begin_derived(
        _transaction(request),
        1,
        {"source": (manifest, length)},
        {"unet": {"source": "source", "source_component": "unet", "drop": [], "add": {}}},
        {},
        [("unet", "weight")],
        1 << 20,
        work_fingerprint="sha256:" + "54" * 32,
    )


def _committed(tmp_path: Path) -> tuple[pb.WeightsFinalizeRequest, Path, Any]:
    store, manifest, length = _released_source(tmp_path)
    root = tmp_path / "store"
    request = _request()
    _writer(store, request, manifest, length).commit()
    # The worker's own Stores name a receipt's declaration by digest.
    store = tensorfs.Store.open(str(root))
    facts = store.derived_lookup(_transaction(request))["receipt"]
    receipt = weights_sink.receipt_from_native(
        "model", _transaction(request), facts, replayed=True, request_id=request.request_id
    )
    _ref, _bytes, digest = weights_sink.protocol_receipt(
        receipt,
        owner_scope="owner",
        request_id=request.request_id,
        invocation_spec_digest="sha256:" + request.invocation_spec_digest.hex(),
    )
    request.disposition = pb.WEIGHTS_FINALIZE_DISPOSITION_ADOPT
    request.weights_receipt_digest = digest
    request.scratch_root_id = "creator-retention-name"
    return request, root, store


def test_absent_writer_is_tombstoned_before_reply_and_replays_after_history_is_gone(
    tmp_path: Path,
) -> None:
    store, manifest, length = _released_source(tmp_path)
    root = tmp_path / "store"
    request = _request()
    result = _finish(request, root, [_attempt(request, spec=False)])
    assert result.outcome == pb.WEIGHTS_FINALIZE_OUTCOME_ABANDONED
    assert store.derived_lookup(_transaction(request))["state"] == "abandoned"
    assert _finish(request, root, []) == result
    with pytest.raises(tensorfs.errors.Refusal, match="TRANSACTION_CLOSED"):
        _writer(store, request, manifest, length)


def test_undeclared_slot_and_active_attempt_are_refused_without_a_tombstone(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = tensorfs.Store.ensure(str(root))
    request = _request()
    attempt = _attempt(request)
    request.output_slot = "other"
    with pytest.raises(FinalizationRefusal, match="declared"):
        _finish(request, root, [attempt])
    request.output_slot = "model"
    attempt.state = "running"
    with pytest.raises(FinalizationRefusal, match="active"):
        _finish(request, root, [attempt])
    assert store.derived_lookup(_transaction(request))["state"] == "absent"


def test_original_declaration_is_verified_after_history_loss(tmp_path: Path) -> None:
    root = tmp_path / "store"
    request = _request()
    original = request.invocation_spec_canonical_bytes
    request.invocation_spec_canonical_bytes = b"{}"
    with pytest.raises(FinalizationRefusal, match="digest"):
        _finish(request, root, [])
    request.invocation_spec_canonical_bytes = original
    assert _finish(request, root, []).outcome == pb.WEIGHTS_FINALIZE_OUTCOME_ABANDONED


def test_history_loss_does_not_authorize_fencing_an_existing_writer(tmp_path: Path) -> None:
    store, manifest, length = _released_source(tmp_path)
    request = _request()
    writer = _writer(store, request, manifest, length)
    with pytest.raises(FinalizationRefusal, match="existing writer"):
        _finish(request, tmp_path / "store", [])
    assert writer.commit()["transaction_id"] == _transaction(request)


def test_transaction_identity_matches_the_shared_protocol_vector() -> None:
    path = Path(__file__).parent / "testdata/worker-protocol/canonical/weights_receipt.json"
    row = json.loads(path.read_bytes())
    actual = weights_sink.weights_transaction_id(
        row["owner_authority_scope"],
        row["request_id"],
        row["invocation_spec_digest"],
        row["output_slot"],
    )
    assert actual == row["weights_transaction_id"]


def test_named_adoption_replays_and_conflicting_disposition_cannot_release_it(
    tmp_path: Path,
) -> None:
    request, root, store = _committed(tmp_path)
    request.scratch_root_id = "invalid retention name"
    with pytest.raises(FinalizationRefusal, match="root name"):
        _finish(request, root, [_attempt(request)])
    request.scratch_root_id = "creator-retention-name"
    result = _finish(request, root, [_attempt(request)])
    assert result.outcome == pb.WEIGHTS_FINALIZE_OUTCOME_ADOPTED
    assert result.weights_receipt.weights_receipt_digest == request.weights_receipt_digest
    assert _finish(request, root, []) == result
    adopted = store.derived_adopt(_transaction(request), request.scratch_root_id)
    assert adopted["disposition"]["kind"] == "adopted"
    with pytest.raises(tensorfs.errors.Refusal):
        store.derived_adopt(_transaction(request), "another-root")
    request.disposition = pb.WEIGHTS_FINALIZE_DISPOSITION_ABANDON
    request.scratch_root_id = ""
    with pytest.raises(FinalizationRefusal, match="first-wins"):
        _finish(request, root, [])


def test_bad_receipt_is_not_journaled_as_an_accepted_decision(tmp_path: Path) -> None:
    request, root, _store = _committed(tmp_path)
    correct = request.weights_receipt_digest
    request.weights_receipt_digest = b"x" * 32
    with pytest.raises(FinalizationRefusal, match="receipt differs"):
        _finish(request, root, [_attempt(request)])
    request.weights_receipt_digest = correct
    assert (
        _finish(request, root, [_attempt(request)]).outcome == pb.WEIGHTS_FINALIZE_OUTCOME_ADOPTED
    )


def test_commit_first_abandon_uncommitted_returns_the_released_receipt(tmp_path: Path) -> None:
    request, root, store = _committed(tmp_path)
    committed = store.derived_lookup(_transaction(request))["receipt"]
    request.disposition = pb.WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED
    request.weights_receipt_digest = b""
    request.scratch_root_id = ""
    result = _finish(request, root, [_attempt(request)])
    assert result.outcome == pb.WEIGHTS_FINALIZE_OUTCOME_ABANDONED
    # With no recorded receipt to match, the worker returns its own custody's form.
    returned = documents.read(
        result.weights_receipt.weights_receipt_canonical_bytes, pb.WeightsReceipt
    )
    native = json.loads(base64.b64decode(returned["tensorfs_receipt_canonical_bytes"]))
    assert weights_sink.same_receipt(native, committed)
    with pytest.raises(tensorfs.errors.Refusal):
        store.derived_adopt(_transaction(request), "released-root")


def test_abandon_drops_upload_lease_and_late_receipt_cannot_reopen_it(tmp_path: Path) -> None:
    request, root, store = _committed(tmp_path)
    facts = store.derived_lookup(_transaction(request))["receipt"]
    receipt = weights_sink.receipt_from_native(
        "model", _transaction(request), facts, replayed=True, request_id=request.request_id
    )
    reference, _raw, _digest = weights_sink.protocol_receipt(
        receipt,
        owner_scope="owner",
        request_id=request.request_id,
        invocation_spec_digest="sha256:" + request.invocation_spec_digest.hex(),
    )
    exchange = WeightsExchange(store_root=root, stop=Event(), owner_scope=lambda: "owner")
    exchange._hold(_transaction(request), 1, reference)
    request.disposition = pb.WEIGHTS_FINALIZE_DISPOSITION_ABANDON
    request.scratch_root_id = ""
    result = finalize(
        request,
        owner_scope="owner",
        attempts=[_attempt(request)],
        tensorfs_root=root,
        journal_root=tmp_path / "worker",
        on_abandon=exchange.abandon,
    )
    assert result.outcome == pb.WEIGHTS_FINALIZE_OUTCOME_ABANDONED
    assert _transaction(request) not in exchange._held
    with pytest.raises(WeightsExchangeError, match="released"):
        exchange._hold(_transaction(request), 1, reference)
    restarted = WeightsExchange(store_root=root, stop=Event(), owner_scope=lambda: "owner")
    with pytest.raises(WeightsExchangeError, match="released"):
        restarted._hold(_transaction(request), 1, reference)
