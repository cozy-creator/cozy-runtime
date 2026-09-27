"""Resolve exact produced-model provenance for independent machine-side custody."""

from __future__ import annotations

import hashlib

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def model_retention(
    workspace: Workspace,
    owner: str,
    recipient: str,
    output_path: str,
    artifact: ModelArtifact,
) -> pb.DerivedRetentionRequest:
    """Plan a stable hold; ``Workspace.retain`` still authorizes and establishes it.

    Matching bytes alone do not prove the supplied producer/slot/receipt. Both
    ordinary transformations and native conversion services preserve that original
    provenance when a later caller obtains its own hold from a memo result.
    """
    Workspace.owner(owner)
    if not recipient or len(recipient.encode()) > 256 or len(output_path.encode()) > 1024:
        raise WorkspaceRefusal("model hold recipient or output path exceeds its bound")
    digest = documents.raw(artifact.tensorfs_receipt_digest)
    manifest = documents.raw(artifact.manifest.digest)
    with workspace.locked() as db:
        rows = db.execute(
            "SELECT id FROM weights WHERE owner=? AND request=? AND slot=? "
            "AND native_digest=? AND manifest=? AND manifest_length=? "
            "AND state IN ('receipt','released') ORDER BY ordinal DESC",
            (
                owner,
                artifact.producer_request_id,
                artifact.output_slot,
                digest,
                manifest,
                artifact.manifest.length,
            ),
        ).fetchall()
        if len(rows) > 1:
            raise WorkspaceRefusal("model artifact has ambiguous producer provenance")
        transaction = str(rows[0]["id"]) if rows else ""
        if not transaction:
            native = db.execute(
                "SELECT native_owner,native_receipt,result FROM native_calls "
                "WHERE owner=? AND service_id=? AND operation='convert_cozytensors' "
                "AND state IN ('complete','releasing','released')",
                (owner, artifact.producer_request_id),
            ).fetchone()
            if (
                native is None
                or _stored(native["result"]) != artifact
                or hashlib.sha256(native["native_receipt"]).digest() != digest
            ):
                raise WorkspaceRefusal("model artifact has no exact owned producer provenance")
            receipt = canonical_json.decode(native["native_receipt"])
            subject = receipt.get("manifest")
            if (
                receipt.get("transaction_id") != native["native_owner"]
                or not isinstance(subject, dict)
                or subject.get("sha256") != artifact.manifest.digest[7:]
                or subject.get("length") != artifact.manifest.length
            ):
                raise WorkspaceRefusal("native model receipt changed its exact subject")
            transaction = str(native["native_owner"])
    documents.raw(transaction)  # One exact native transaction, never a caller path.
    hold_id = documents.spell(
        hashlib.sha256(
            canonical_json.encode(
                [
                    "cozy.machine-model-hold/1",
                    owner,
                    recipient,
                    output_path,
                    msgspec.to_builtins(artifact),
                ]
            )
        ).digest()
    )
    return pb.DerivedRetentionRequest(
        weights_transaction_id=transaction,
        tensorfs_receipt_digest=digest,
        retention_id=hold_id,
    )


def _stored(result: bytes) -> ModelArtifact | None:
    """The members this reader consumes; another Runtime version's additions are ignored."""
    try:
        return msgspec.convert(canonical_json.decode(result), type=ModelArtifact)
    except (ValueError, msgspec.ValidationError, ConformanceError):
        return None
