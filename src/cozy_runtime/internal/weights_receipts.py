"""Execution receipt documents, independent of the retired worker protobuf types.

The stored format and exact-byte identity stay unchanged. Native TensorFS receipts
remain opaque bytes; Runtime consumes only its execution binding and digest.
"""

from __future__ import annotations

import base64
import binascii
import hashlib

import msgspec

from cozy_runtime.internal import canonical

MAX_RECEIPT_BYTES = 1 << 20
_FORMAT = "cozy.worker.v1.WeightsReceipt/1"


class ReceiptError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def spell(digest: bytes) -> str:
    if len(digest) != 32:
        raise ReceiptError("malformed_digest", f"{len(digest)} B is not a 32-byte SHA-256")
    return "sha256:" + digest.hex()


def raw(digest: str) -> bytes:
    if canonical.DIGEST_RE.fullmatch(digest) is None:
        raise ReceiptError("malformed_digest", f"{digest!r} is not a lowercase sha256: digest")
    return bytes.fromhex(digest[7:])


class Receipt(msgspec.Struct, frozen=True):
    owner_authority_scope: str = ""
    request_id: str = ""
    invocation_spec_digest: str = ""
    output_slot: str = ""
    weights_transaction_id: str = ""
    tensorfs_receipt_digest: str = ""
    tensorfs_receipt_canonical_bytes: bytes = b""


class Reference(msgspec.Struct, frozen=True):
    weights_receipt_digest: bytes
    weights_receipt_canonical_bytes: bytes


def identity(receipt: Receipt) -> tuple[bytes, bytes]:
    """Omit default fields and hash the stored document's exact canonical bytes."""
    fields = msgspec.to_builtins(receipt)
    for name, value in fields.items():
        if not isinstance(value, str) or any(not 0x20 <= ord(c) <= 0x7E for c in value):
            raise ReceiptError("non_ascii_field", f"{name}: fields are printable ASCII")
    document = {name: value for name, value in fields.items() if value}
    document["format"] = _FORMAT
    data = canonical.write(document)
    return data, hashlib.sha256(data).digest()


def read(data: bytes) -> Receipt:
    """Read known receipt fields while keeping all stored bytes as the identity."""
    try:
        value = canonical.parse_canonical(data)
    except canonical.CanonicalError as exc:
        raise ReceiptError(exc.code, exc.detail) from exc
    if not isinstance(value, dict):
        raise ReceiptError("wrong_type", "a receipt is a JSON object")
    if value.get("format") != _FORMAT:
        raise ReceiptError("unknown_format", f"{value.get('format')!r} is not {_FORMAT!r}")
    fields = {name: value[name] for name in Receipt.__struct_fields__ if name in value}
    native = fields.get("tensorfs_receipt_canonical_bytes", "")
    if not isinstance(native, str):
        raise ReceiptError("wrong_type", "native receipt bytes are not a base64 string")
    try:
        fields["tensorfs_receipt_canonical_bytes"] = base64.b64decode(native, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ReceiptError("malformed_bytes", "native receipt bytes are not base64") from exc
    try:
        return msgspec.convert(fields, Receipt)
    except msgspec.ValidationError as exc:
        raise ReceiptError("wrong_type", str(exc)) from exc
