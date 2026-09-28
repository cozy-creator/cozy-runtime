"""The operation memo row body, decoded once where a journal row is read."""

from __future__ import annotations

import hashlib
from typing import Literal

import msgspec
from msgspec.structs import replace

from cozy_runtime import canonical_json
from cozy_runtime.internal import canonical
from cozy_runtime.internal.worker.workspace import Journal, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


class Hold(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """One native root a memo row owns; `retention_id` is empty only while planned."""

    transaction_id: str
    retention_id: str
    native_receipt_digest: str
    manifest_digest: str
    manifest_length: int
    kind: Literal["derived", "tree"] = "derived"
    content_bytes: int = 0
    output_id: str = ""

    def request(self) -> pb.DerivedRetentionRequest:
        return pb.DerivedRetentionRequest(
            weights_transaction_id=self.transaction_id,
            retention_id=self.retention_id,
            tensorfs_receipt_digest=documents.raw(self.native_receipt_digest),
        )


class Source(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """The producing terminal: an attempt outcome, or a native service call."""

    kind: Literal["attempt", "native"] = "attempt"
    request_id: str = ""
    attempt_ordinal: int = 0
    invocation_spec_digest: str = ""
    outcome_id: str = ""
    outcome_digest: str = ""
    outcome_canonical_bytes: bytes = b""
    service_id: str = ""
    operation: str = ""
    parent_request: str = ""
    parent_ordinal: int = 0
    result_digest: str = ""
    receipt_digest: str = ""
    result: bytes = b""
    native_receipt: bytes = b""

    def identity(self) -> Source:
        """Without the delivered copies and a native call's parent attempt."""
        return replace(
            self, outcome_canonical_bytes=b"", result=b"", native_receipt=b"", parent_ordinal=0
        )

    def producer(self) -> tuple[str | int, ...]:
        if self.kind == "native":
            return ("native", self.service_id)
        return ("attempt", self.request_id, self.attempt_ordinal, self.outcome_id)


class Memo(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """An `operation_cache`/`operation_lookups` body; a disabled mapping names its contradiction."""

    source: Source
    holds: tuple[Hold, ...]
    schema: canonical.Json = None
    conflicting_source: Source | None = None

    def encode(self) -> bytes:
        return canonical_json.encode(msgspec.to_builtins(self))

    def require(self, kind: str) -> None:
        if self.source.kind != kind:
            raise WorkspaceRefusal("memo result has another terminal kind")

    def compact(self) -> Memo:
        """Identity and custody only: delivered copies and the result schema are dropped."""
        conflicting = self.conflicting_source
        return Memo(
            source=self.source.identity(),
            holds=self.holds,
            conflicting_source=None if conflicting is None else conflicting.identity(),
        )

    def assigned(self, kind: str, owner: str, recipient: str, key: bytes) -> Memo:
        return replace(
            self,
            holds=tuple(
                replace(
                    hold, retention_id=hold_id(kind, owner, recipient, key, hold.transaction_id)
                )
                for hold in self.holds
            ),
        )


class Lookup(msgspec.Struct, frozen=True):
    key: bytes
    cache_id: str
    state: Literal["retaining", "ready", "missing", "miss", "released", "acknowledged"]
    memo: Memo


class Ref(msgspec.Struct, frozen=True):
    digest: str
    length: int


class Artifact(msgspec.Struct, frozen=True):
    """A native call's result value; members another version added are ignored."""

    producer_request_id: str
    output_slot: str
    manifest: Ref
    tensorfs_receipt_digest: str


class _NativeManifest(msgspec.Struct, frozen=True):
    sha256: str
    length: int


class NativeReceipt(msgspec.Struct, frozen=True):
    """The TensorFS receipt facts a memo compares; the rest is the receipt's own business."""

    manifest: _NativeManifest

    def ref(self) -> Ref:
        return Ref("sha256:" + self.manifest.sha256, self.manifest.length)


def decode[T](data: bytes, into: type[T]) -> T:
    try:
        return canonical_json.decode_as(data, into)
    except ValueError as exc:
        raise WorkspaceRefusal(f"memo {into.__name__} is malformed: {exc}") from exc


def lookup_row(db: Journal, owner: str, consumer: str) -> Lookup | None:
    row = db.execute(
        "SELECT key,cache_id,state,body FROM operation_lookups WHERE owner=? AND consumer=?",
        (owner, consumer),
    ).fetchone()
    if row is None:
        return None
    return Lookup(row["key"], row["cache_id"], row["state"], decode(row["body"], Memo))


def hold_id(kind: str, owner: str, recipient: str, key: bytes, transaction: str) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            canonical_json.encode([kind, owner, recipient, key.hex(), transaction])
        ).hexdigest()
    )
