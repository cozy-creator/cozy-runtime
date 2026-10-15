"""Execution identity and stored documents for native TensorFS output receipts.

TensorFS owns source and writer APIs. Runtime binds native facts to the accepted
execution and converts retained native receipts into its stored receipt documents.
The native source API supplies all reader metadata and bytes.
"""

from __future__ import annotations

import hashlib
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


def same_receipt(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Compare adoption identity; record authoritative native facts, not peer observations."""
    return native_receipt(left) == native_receipt(right)


def exchange_file(spool: Path, output_slot: str, kind: str) -> Path:
    """One worker-chosen spool file; no child-authored path crosses the seam."""
    if _OUTPUT_SLOT.fullmatch(output_slot) is None or kind not in _EXCHANGE_KINDS:
        raise CapabilityError("weights exchange identity is invalid", code="weights_host_identity")
    return spool / f"weights-{output_slot}-{kind}.canonical"


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
