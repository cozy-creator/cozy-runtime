"""First-wins owner decisions over real TensorFS transaction roots."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill, weights_sink
from cozy_runtime.protocol import documents, weights_limits
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .attempts import AttemptRecord


class FinalizationRefusal(Exception):
    pass


def _decision(request: pb.WeightsFinalizeRequest, owner_scope: str) -> dict[str, object]:
    if (
        not owner_scope
        or request.owner_authority_scope != owner_scope
        or not request.request_id
        or len(request.invocation_spec_digest) != 32
        or re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", request.output_slot) is None
        or request.ByteSize() > weights_limits.MAX_INLINE_CONTROL_BYTES
    ):
        raise FinalizationRefusal("weights finalization has an invalid owner/request/spec/slot")
    if (
        documents.digest_of(request.invocation_spec_canonical_bytes)
        != request.invocation_spec_digest
    ):
        raise FinalizationRefusal(
            "weights finalization's original invocation digest does not match"
        )
    try:
        spec = documents.parse(request.invocation_spec_canonical_bytes, pb.InvocationSpec)
    except documents.DocumentError as exc:
        raise FinalizationRefusal(
            "weights finalization has no canonical invocation declaration"
        ) from exc
    if not spec.HasField("job") or not any(
        row.output_id == request.output_slot
        and row.mime_type == "application/vnd.cozy.model-manifest"
        for row in spec.outputs
    ):
        raise FinalizationRefusal("weights finalization names no declared model output")
    uncommitted = request.disposition == pb.WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED
    if request.disposition not in {
        pb.WEIGHTS_FINALIZE_DISPOSITION_ADOPT,
        pb.WEIGHTS_FINALIZE_DISPOSITION_ABANDON,
        pb.WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED,
    } or len(request.weights_receipt_digest) != (0 if uncommitted else 32):
        raise FinalizationRefusal("weights finalization has an invalid disposition or receipt")
    if request.disposition == pb.WEIGHTS_FINALIZE_DISPOSITION_ADOPT:
        # Creator assigns the retention name; TensorFS owns its storage and replay.
        # Validate its public ASCII-name contract before journaling the decision.
        if re.fullmatch(r"[A-Za-z0-9_./-]{1,128}", request.scratch_root_id) is None:
            raise FinalizationRefusal("weights adoption has an invalid private scratch root name")
    elif request.scratch_root_id:
        raise FinalizationRefusal("weights abandonment cannot name an adopted scratch root")
    return {
        "owner": owner_scope,
        "request": request.request_id,
        "spec": documents.spell(bytes(request.invocation_spec_digest)),
        "slot": request.output_slot,
        "disposition": request.disposition,
        "receipt": request.weights_receipt_digest.hex(),
        "scratch_root": request.scratch_root_id,
    }


def _read(path: Path) -> bytes | None:
    try:
        with path.open("rb") as stream:
            data = stream.read(weights_limits.MAX_WEIGHTS_RECEIPT_AGGREGATE_BYTES + 1)
    except FileNotFoundError:
        return None
    if len(data) > weights_limits.MAX_WEIGHTS_RECEIPT_AGGREGATE_BYTES:
        raise FinalizationRefusal("weights finalization journal exceeds its byte bound")
    return data


def _record(path: Path, data: bytes) -> None:
    """Publish a complete immutable journal record, then sync its directory entry."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".finalize-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if _read(path) != data:
                    raise FinalizationRefusal(
                        "weights finalization replay changed its first decision"
                    ) from None
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink()


def _receipt(
    request: pb.WeightsFinalizeRequest, transaction_id: str, native: object
) -> pb.WeightsReceiptRef:
    """The recorded protocol receipt."""
    if not isinstance(native, dict) or native.get("transaction_id") != transaction_id:
        raise FinalizationRefusal("TensorFS finalization returned another transaction receipt")
    receipt = weights_sink.receipt_from_native(
        request.output_slot,
        transaction_id,
        native,
        replayed=True,
        request_id=request.request_id,
    )
    ref, _raw, digest = weights_sink.protocol_receipt(
        receipt,
        owner_scope=request.owner_authority_scope,
        request_id=request.request_id,
        invocation_spec_digest=documents.spell(bytes(request.invocation_spec_digest)),
    )
    if request.weights_receipt_digest and request.weights_receipt_digest != digest:
        raise FinalizationRefusal("weights finalization receipt differs from the committed receipt")
    return ref


def _authorize(
    request: pb.WeightsFinalizeRequest, attempts: Sequence[AttemptRecord], writer_active: bool
) -> None:
    matching = [attempt for attempt in attempts if attempt.request_id == request.request_id]
    if any(attempt.state not in {"outcome", "closed"} for attempt in matching):
        raise FinalizationRefusal("weights finalization cannot race an active attempt")
    matching = [attempt for attempt in matching if attempt.digest == request.invocation_spec_digest]
    if writer_active and not matching:
        raise FinalizationRefusal("an existing writer has no matching recorded terminal attempt")


def finalize(
    request: pb.WeightsFinalizeRequest,
    *,
    owner_scope: str,
    attempts: Sequence[AttemptRecord],
    tensorfs_root: Path,
    journal_root: Path,
    on_abandon: Callable[[str], None] | None = None,
    retain_partial: bool = False,
) -> pb.WeightsFinalizeResult:
    """``retain_partial`` fences an open abandoned writer but keeps its checkpointed parts
    for a later identical attempt to adopt; the owner holds nothing either way."""
    if not tensorfs_root.is_absolute() or not journal_root.is_absolute():
        raise FinalizationRefusal("weights finalization requires absolute worker storage roots")
    decision = _decision(request, owner_scope)
    _authorize(request, attempts, False)
    transaction_id = weights_sink.weights_transaction_id(
        owner_scope,
        request.request_id,
        documents.spell(bytes(request.invocation_spec_digest)),
        request.output_slot,
    )
    directory = journal_root / "weights-finalizations"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = transaction_id.removeprefix("sha256:")
    intent, result_path = directory / f"{key}.intent", directory / f"{key}.result"
    encoded = canonical_json.encode(decision)
    previous = _read(intent)
    if previous is not None and previous != encoded:
        raise FinalizationRefusal("weights finalization conflicts with its first-wins decision")
    recorded = _read(result_path)
    if recorded is not None:
        if previous is None:
            raise FinalizationRefusal("weights finalization result has no durable intent")
        result = pb.WeightsFinalizeResult.FromString(recorded)
        if (
            result.request_id != request.request_id
            or result.invocation_spec_digest != request.invocation_spec_digest
            or result.output_slot != request.output_slot
            or result.owner_authority_scope != owner_scope
            or result.outcome
            != (
                pb.WEIGHTS_FINALIZE_OUTCOME_ADOPTED
                if request.disposition == pb.WEIGHTS_FINALIZE_DISPOSITION_ADOPT
                else pb.WEIGHTS_FINALIZE_OUTCOME_ABANDONED
            )
            or (
                request.weights_receipt_digest
                and result.weights_receipt.weights_receipt_digest != request.weights_receipt_digest
            )
        ):
            raise FinalizationRefusal("weights finalization journal has another result identity")
        if result.outcome == pb.WEIGHTS_FINALIZE_OUTCOME_ABANDONED and on_abandon is not None:
            on_abandon(transaction_id)
        return result
    store = fill.ensure_store(tensorfs_root)
    observed = store.derived_lookup(transaction_id)
    committed_receipt = None
    if request.disposition != pb.WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED:
        if observed["state"] != "committed":
            raise FinalizationRefusal("weights finalization requires an existing committed receipt")
        committed_receipt = _receipt(request, transaction_id, observed["receipt"])
    if previous is None:
        _authorize(request, attempts, bool(observed.get("writer_session_id")))
        _record(intent, encoded)
    result = pb.WeightsFinalizeResult(
        request_id=request.request_id,
        invocation_spec_digest=request.invocation_spec_digest,
        output_slot=request.output_slot,
        owner_authority_scope=owner_scope,
    )
    if request.disposition != pb.WEIGHTS_FINALIZE_DISPOSITION_ADOPT and on_abandon is not None:
        on_abandon(transaction_id)
    if request.disposition == pb.WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED:
        if observed["state"] == "open" and observed.get("writer_session_id"):
            store.derived_fence(transaction_id, observed["writer_session_id"])
        if not (retain_partial and observed["state"] == "open"):
            observed = store.derived_abandon(transaction_id)
        if observed["state"] == "committed":
            result.weights_receipt.CopyFrom(_receipt(request, transaction_id, observed["receipt"]))
            store.derived_dispose(transaction_id)
        result.outcome = pb.WEIGHTS_FINALIZE_OUTCOME_ABANDONED
    else:
        assert committed_receipt is not None
        result.weights_receipt.CopyFrom(committed_receipt)
        if request.disposition == pb.WEIGHTS_FINALIZE_DISPOSITION_ADOPT:
            store.derived_adopt(transaction_id, request.scratch_root_id)
            result.outcome = pb.WEIGHTS_FINALIZE_OUTCOME_ADOPTED
        else:
            store.derived_dispose(transaction_id)
            result.outcome = pb.WEIGHTS_FINALIZE_OUTCOME_ABANDONED
    _record(result_path, result.SerializeToString(deterministic=True))
    return result
