"""Execution identity and protocol projections for native TensorFS output receipts.

TensorFS owns source and writer APIs. Runtime binds native facts to the accepted
execution and converts retained native receipts into its existing protocol documents.
The native source API supplies all reader metadata and bytes.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact, ObjectRef
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._weights import (
    WeightsReceipt,
)
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    import tensorfs

_OUTPUT_SLOT = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_EXCHANGE_KINDS = frozenset({"declaration", "receipt", "derivation", "native-receipt"})


class ManifestRef(msgspec.Struct, frozen=True):
    sha256: Annotated[str, msgspec.Meta(pattern="^[0-9a-f]{64}$")]
    length: int

    @property
    def digest(self) -> str:
        return "sha256:" + self.sha256


class NativeReceipt(msgspec.Struct, frozen=True):
    """The TensorFS derived-writer receipt facts Runtime reads; the rest is TensorFS's."""

    transaction_id: str
    declaration_digest: str
    manifest: ManifestRef


def native_receipt(facts: Mapping[str, object]) -> NativeReceipt:
    """Decode a native receipt once; a receipt missing a fact Runtime reads is refused."""
    try:
        return msgspec.convert(facts, NativeReceipt)
    except msgspec.ValidationError as exc:
        raise CapabilityError(
            f"native receipt is malformed: {exc}", code="weights_receipt_mismatch"
        ) from exc


class Disposition(msgspec.Struct, frozen=True):
    kind: str


class DerivedLookup(msgspec.Struct, frozen=True):
    """`Store.derived_lookup`: only a committed transaction carries a receipt and disposition."""

    state: str
    receipt: dict[str, object] = {}
    disposition: Disposition = Disposition("")


def lookup(store: tensorfs.Store, transaction_id: str) -> DerivedLookup:
    return msgspec.convert(store.derived_lookup(transaction_id), DerivedLookup)


def declaration_digest(facts: Mapping[str, object]) -> bytes:
    """The declaration a native receipt names by digest."""
    return documents.raw(native_receipt(facts).declaration_digest)


def same_receipt(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Whether two native receipts state the same facts."""
    return dict(left) == dict(right)


def work_fingerprint(
    spec: Mapping[str, object],
    numerical: bytes,
    *,
    operation_identity: str = "",
) -> str:
    """Bind native partial work to what determines its output: the operation (its
    declared version and source identity, else its job declaration), the actual inputs,
    and the numerical policy.

    Installation ids, attempt/custody clocks and SDK inventories never define it, so
    interrupted work resumes after a Runtime upgrade or a reinstall of the same release.
    """
    job = spec.get("job")
    if not isinstance(job, Mapping) or len(numerical) != 32:
        raise CapabilityError(
            "weights work has no selected job or numerical policy", code="weights_work_identity"
        )
    facts = {
        "operation": operation_identity or {"job_descriptor_id": job["job_descriptor_id"]},
        "payload_digest": spec["payload_digest"],
        "inputs": spec.get("inputs", []),
        "outputs": spec.get("outputs", []),
        "numerical_environment_digest": documents.spell(numerical),
    }
    return (
        "sha256:"
        + hashlib.sha256(
            b"cozy.runtime.derived-work/1\0" + canonical_json.encode(facts)
        ).hexdigest()
    )


def exchange_file(spool: Path, output_slot: str, kind: str) -> Path:
    """One worker-chosen spool file; no child-authored path crosses the seam."""
    if _OUTPUT_SLOT.fullmatch(output_slot) is None or kind not in _EXCHANGE_KINDS:
        raise CapabilityError("weights exchange identity is invalid", code="weights_host_identity")
    return spool / f"weights-{output_slot}-{kind}.canonical"


def write_exchange_file(spool: Path, output_slot: str, kind: str, data: bytes) -> Path:
    """Publish immutable exact bytes into the already-brokered attempt spool."""
    target = exchange_file(spool, output_slot, kind)
    try:
        with target.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        target.chmod(0o444)
    except FileExistsError:
        if target.read_bytes() != data:
            raise CapabilityError(
                f"weights {kind} replay changed exact bytes",
                code="weights_receipt_mismatch" if kind == "receipt" else "weights_intent_conflict",
            ) from None
    return target


def receipt_from_native(
    output_slot: str,
    transaction_id: str,
    facts: Mapping[str, object],
    *,
    replayed: bool,
    request_id: str,
) -> WeightsReceipt:
    manifest = native_receipt(facts).manifest
    raw = canonical_json.encode(facts)
    receipt_digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    return WeightsReceipt(
        output_slot=output_slot,
        weights_transaction_id=transaction_id,
        tensorfs_receipt_digest=receipt_digest,
        tensorfs_receipt=raw,
        replayed=replayed,
        _artifact=ModelArtifact(
            request_id,
            output_slot,
            ObjectRef(manifest.digest, manifest.length),
            receipt_digest,
        ),
    )


def weights_transaction_id(
    owner_scope: str, request_id: str, invocation_spec_digest: str, output_slot: str
) -> str:
    """Runtime alone assigns this ID for creation and finalization; hosts record and echo it."""
    subject = canonical_json.encode(
        {
            "owner_authority_scope": owner_scope,
            "request_id": request_id,
            "invocation_spec_digest": invocation_spec_digest,
            "output_slot": output_slot,
        }
    )
    return "sha256:" + hashlib.sha256(b"cozy.runtime.artifact-transaction\0" + subject).hexdigest()


def protocol_receipt(
    receipt: WeightsReceipt,
    *,
    owner_scope: str,
    request_id: str,
    invocation_spec_digest: str,
) -> tuple[pb.WeightsReceiptRef, bytes, bytes]:
    """Wrap exact TensorFS receipt bytes in Worker Protocol WeightsReceipt/1."""
    body = pb.WeightsReceipt(
        owner_authority_scope=owner_scope,
        request_id=request_id,
        invocation_spec_digest=invocation_spec_digest,
        output_slot=receipt.output_slot,
        weights_transaction_id=receipt.weights_transaction_id,
        tensorfs_receipt_digest=receipt.tensorfs_receipt_digest,
        tensorfs_receipt_canonical_bytes=receipt.tensorfs_receipt,
    )
    canonical_bytes, digest = documents.identity(body)
    return (
        pb.WeightsReceiptRef(
            weights_receipt_digest=digest,
            weights_receipt_canonical_bytes=canonical_bytes,
        ),
        canonical_bytes,
        digest,
    )
