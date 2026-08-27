"""Runtime ownership for job artifact transactions.

The executor turns the path-free author declaration into plain typed TensorFS arguments.
The worker owns the semantic identity, the monotone writer fence, and durable replay.  This
module contains the common closed declaration and journal projection; neither side learns a
model family or a repository/publication vocabulary.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from cozy_runtime.author._artifact_sink import (
    ArtifactBytes,
    ArtifactComponent,
    ArtifactConfig,
    ArtifactReader,
    ArtifactValue,
    CopyComponent,
    CopyConfig,
    CreateComponent,
    CreateConfig,
    DeriveComponent,
    _ArtifactRequest,
    _ordered_unique,
)
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.internal import canonical
from cozy_runtime.internal.journal import Journal, JournalError

DECLARATION_FORMAT = "cozy.runtime.ArtifactDeclaration/1"
TRANSACTION_FORMAT = "cozy.runtime.ArtifactTransactionIdentity/1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_STAGED_NAME = re.compile(r"^artifact-[0-9a-f]{64}\.(declaration|receipt)$")

StreamKey: TypeAlias = tuple[str, ...]


class ArtifactTransactionError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class PreparedDeclaration:
    canonical_bytes: bytes
    digest: str
    document: dict[str, Any]
    streams: dict[StreamKey, ArtifactReader]


def _reader(data: ArtifactBytes, what: str) -> ArtifactReader:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return io.BytesIO(bytes(data))
    if isinstance(data, ArtifactReader):
        return data
    raise CapabilityError(
        f"{what} is {type(data).__name__}, not bytes or a binary reader; artifact values "
        "are bounded streams, never filesystem paths",
        code="artifact_reader",
    )


def _shape(value: tuple[int, ...], what: str) -> list[int]:
    if any(
        not isinstance(extent, int) or isinstance(extent, bool) or extent < 0
        for extent in value
    ):
        raise CapabilityError(
            f"{what} shape contains a non-integer or negative extent", code="artifact_declaration"
        )
    return list(value)


def _value(
    component: str,
    value: ArtifactValue,
    streams: dict[StreamKey, ArtifactReader],
) -> dict[str, Any]:
    if not value.key or not value.logical_dtype or not value.encoding or not value.parts:
        raise CapabilityError(
            f"target {component!r} has an incomplete added value {value.key!r}",
            code="artifact_declaration",
        )
    if _DIGEST.fullmatch(value.encoding) is None:
        raise CapabilityError(
            f"{component}.{value.key}: encoding is not an exact sha256 identity",
            code="artifact_declaration",
        )
    roles = _ordered_unique(
        tuple(part.role for part in value.parts), f"{component}.{value.key} roles"
    )
    by_role = {part.role: part for part in value.parts}
    parts: list[dict[str, Any]] = []
    for role in roles:
        part = by_role[role]
        if not part.dtype:
            raise CapabilityError(
                f"{component}.{value.key}#{role} has no dtype", code="artifact_declaration"
            )
        streams[("value", component, value.key, role)] = _reader(
            part.data, f"{component}.{value.key}#{role}"
        )
        parts.append({"role": role, "dtype": part.dtype, "shape": _shape(part.shape, role)})
    return {
        "key": value.key,
        "logical_dtype": value.logical_dtype,
        "shape": _shape(value.shape, f"{component}.{value.key}"),
        "encoding": value.encoding,
        "parts": parts,
    }


def _values(
    component: str,
    values: tuple[ArtifactValue, ...],
    streams: dict[StreamKey, ArtifactReader],
) -> list[dict[str, Any]]:
    keys = _ordered_unique(tuple(value.key for value in values), f"{component} added keys")
    by_key = {value.key: value for value in values}
    return [_value(component, by_key[key], streams) for key in keys]


def _target(
    name: str,
    target: ArtifactComponent,
    aliases: frozenset[str],
    streams: dict[StreamKey, ArtifactReader],
) -> dict[str, Any]:
    if isinstance(target, CopyComponent):
        row: dict[str, Any] = {
            "component": name,
            "mode": "copy",
            "source_alias": target.source,
            "source_component": target.component,
        }
    elif isinstance(target, DeriveComponent):
        row = {
            "component": name,
            "mode": "derive",
            "source_alias": target.source,
            "source_component": target.component,
            "drop": list(_ordered_unique(target.drop, f"{name} dropped keys"))
            if target.drop
            else [],
            "add": _values(name, target.add, streams),
        }
    elif isinstance(target, CreateComponent):
        if not target.values:
            raise CapabilityError(
                f"created target {name!r} has no values", code="artifact_declaration"
            )
        row = {"component": name, "mode": "create", "values": _values(name, target.values, streams)}
    else:  # the public union is closed; protect a dynamically forged object too
        raise CapabilityError(
            f"target {name!r} has unsupported type {type(target).__name__}",
            code="artifact_declaration",
        )
    source = row.get("source_alias")
    if source is not None and source not in aliases:
        raise CapabilityError(
            f"target {name!r} selects undeclared source alias {source!r}",
            code="artifact_source_ungranted",
        )
    if not name or not row.get("component") or (
        source is not None and not row.get("source_component")
    ):
        raise CapabilityError(
            f"target {name!r} has an empty component selector", code="artifact_declaration"
        )
    return row


def _config(
    name: str,
    config: ArtifactConfig,
    aliases: frozenset[str],
    streams: dict[StreamKey, ArtifactReader],
) -> dict[str, Any]:
    if isinstance(config, CopyConfig):
        if config.source not in aliases:
            raise CapabilityError(
                f"config {name!r} selects undeclared source alias {config.source!r}",
                code="artifact_source_ungranted",
            )
        if not config.name:
            raise CapabilityError(
                f"config {name!r} has an empty source name", code="artifact_declaration"
            )
        return {
            "name": name,
            "mode": "copy",
            "source_alias": config.source,
            "source_name": config.name,
        }
    if isinstance(config, CreateConfig):
        streams[("config", name)] = _reader(config.data, f"config {name}")
        return {"name": name, "mode": "create"}
    raise CapabilityError(
        f"config {name!r} has unsupported type {type(config).__name__}",
        code="artifact_declaration",
    )


def prepare_declaration(
    request: _ArtifactRequest,
    *,
    granted_sources: dict[str, str],
) -> PreparedDeclaration:
    """Close an author request against the executor's verified source grant.

    `granted_sources` is ref -> exact snapshot digest, projected from the worker-verified
    InvocationSpec.  A mutable ref or a materialized path never substitutes for that digest.
    """
    sources: list[dict[str, str]] = []
    for alias, tree in sorted(request.sources.items()):
        granted = granted_sources.get(tree.ref, "")
        if _DIGEST.fullmatch(tree.digest) is None or tree.digest != granted:
            raise CapabilityError(
                f"source alias {alias!r} names {tree.ref!r} at {tree.digest or '(no digest)'}, "
                f"but the verified InvocationSpec grants {granted or 'no exact snapshot'}",
                code="artifact_source_ungranted",
            )
        sources.append({"alias": alias, "input_ref": tree.ref, "snapshot": tree.digest})
    aliases = frozenset(source["alias"] for source in sources)
    streams: dict[StreamKey, ArtifactReader] = {}
    targets = [
        _target(name, target, aliases, streams) for name, target in sorted(request.targets.items())
    ]
    configs = [
        _config(name, config, aliases, streams) for name, config in sorted(request.configs.items())
    ]
    document = {
        "format": DECLARATION_FORMAT,
        "output_slot": request.output_slot,
        "sources": sources,
        "configs": configs,
        "targets": targets,
    }
    data = canonical.write(document)
    return PreparedDeclaration(
        canonical_bytes=data,
        digest="sha256:" + hashlib.sha256(data).hexdigest(),
        document=document,
        streams=streams,
    )


def transaction_id(
    owner_authority_scope: str,
    request_id: str,
    invocation_spec_digest: str,
    output_slot: str,
) -> str:
    """The retry-stable semantic identity. Attempts/executors are deliberately absent."""
    if not owner_authority_scope or not request_id or not output_slot:
        raise ArtifactTransactionError(
            "artifact_identity", "owner scope, request id, and output slot are required"
        )
    if _DIGEST.fullmatch(invocation_spec_digest) is None:
        raise ArtifactTransactionError(
            "artifact_identity", "InvocationSpec identity is not one lowercase sha256 digest"
        )
    data = canonical.write(
        {
            "format": TRANSACTION_FORMAT,
            "owner_authority_scope": owner_authority_scope,
            "request_id": request_id,
            "invocation_spec_digest": invocation_spec_digest,
            "output_slot": output_slot,
        }
    )
    return "sha256:" + hashlib.sha256(data).hexdigest()


def stage_exact(root: Path, suffix: str, data: bytes) -> tuple[str, str]:
    """Stage Runtime-authored bytes under a digest name; return name and digest."""
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    name = f"artifact-{digest[7:]}.{suffix}"
    target = root / name
    if target.exists():
        if target.read_bytes() != data:
            raise ArtifactTransactionError(
                "artifact_stage_conflict", f"staged {name} exists with different bytes"
            )
        return name, digest
    temporary = root / f".{name}.{os.getpid()}.partial"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return name, digest


def read_staged_exact(
    root: Path,
    name: str,
    *,
    expected_digest: str,
    expected_length: int,
    max_bytes: int,
) -> bytes:
    """Read one Runtime-minted spool member without following paths or symlinks."""
    if _STAGED_NAME.fullmatch(name) is None or _DIGEST.fullmatch(expected_digest) is None:
        raise ArtifactTransactionError(
            "artifact_stage_name", "artifact staging uses one Runtime-minted digest basename"
        )
    if expected_length <= 0 or expected_length > max_bytes:
        raise ArtifactTransactionError(
            "artifact_stage_size",
            f"declared {expected_length} B is outside the 1..{max_bytes} B control-document bound",
        )
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory,
        )
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size != expected_length:
                raise ArtifactTransactionError(
                    "artifact_stage_identity",
                    f"staged {name} is not a regular {expected_length} B file",
                )
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                data = handle.read(max_bytes + 1)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)
    actual = "sha256:" + hashlib.sha256(data).hexdigest()
    if len(data) != expected_length or actual != expected_digest:
        raise ArtifactTransactionError(
            "artifact_stage_identity",
            f"staged {name} read as {len(data)} B at {actual}, not {expected_length} B at "
            f"{expected_digest}",
        )
    return data


@dataclass(frozen=True, slots=True)
class ArtifactIntent:
    transaction_id: str
    declaration_digest: str
    declaration_bytes: bytes
    request_id: str
    invocation_spec_digest: str
    output_slot: str
    writer_session_id: int
    attempt_ordinal: int
    executor_generation: int


@dataclass(frozen=True, slots=True)
class StoredReceipt:
    transaction_id: str
    receipt_digest: str
    receipt_bytes: bytes
    snapshot_digest: str
    header_digest: str


class ArtifactJournal:
    """Projection and append rules for Runtime's three durable artifact facts."""

    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        self.intents: dict[str, list[ArtifactIntent]] = {}
        self.receipts: dict[str, StoredReceipt] = {}
        self.final: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        for row in self.journal.of_kind("artifact_intent"):
            body = row.body
            intent = ArtifactIntent(
                transaction_id=str(body["transaction_id"]),
                declaration_digest=str(body["declaration_digest"]),
                declaration_bytes=base64.b64decode(str(body["declaration_b64"])),
                request_id=str(body["request_id"]),
                invocation_spec_digest=str(body["invocation_spec_digest"]),
                output_slot=str(body["output_slot"]),
                writer_session_id=int(body["writer_session_id"]),
                attempt_ordinal=int(body["attempt_ordinal"]),
                executor_generation=int(body["executor_generation"]),
            )
            self.intents.setdefault(intent.transaction_id, []).append(intent)
        for row in self.journal.of_kind("artifact_receipt"):
            body = row.body
            receipt = StoredReceipt(
                transaction_id=str(body["transaction_id"]),
                receipt_digest=str(body["receipt_digest"]),
                receipt_bytes=base64.b64decode(str(body["receipt_b64"])),
                snapshot_digest=str(body["snapshot_digest"]),
                header_digest=str(body["header_digest"]),
            )
            held = self.receipts.get(receipt.transaction_id)
            if held is not None and held != receipt:
                raise JournalError(
                    "journal_corrupt",
                    f"artifact transaction {receipt.transaction_id} has conflicting receipts",
                )
            self.receipts[receipt.transaction_id] = receipt
        for row in self.journal.of_kind("artifact_final"):
            transaction = str(row.body["transaction_id"])
            final_held = self.final.get(transaction)
            if final_held is not None and final_held != row.body:
                raise JournalError(
                    "journal_corrupt",
                    f"artifact transaction {transaction} has conflicting final dispositions",
                )
            self.final[transaction] = row.body

    def open(
        self,
        *,
        transaction: str,
        declaration_digest: str,
        declaration_bytes: bytes,
        request_id: str,
        invocation_spec_digest: str,
        output_slot: str,
        attempt_ordinal: int,
        executor_generation: int,
    ) -> tuple[ArtifactIntent, bool]:
        intents = self.intents.get(transaction, [])
        if intents:
            first = intents[0]
            if (
                first.declaration_digest != declaration_digest
                or first.declaration_bytes != declaration_bytes
                or first.request_id != request_id
                or first.invocation_spec_digest != invocation_spec_digest
                or first.output_slot != output_slot
            ):
                raise ArtifactTransactionError(
                    "artifact_declaration_conflict",
                    "the semantic transaction is already locked to a different declaration",
                )
            for intent in intents:
                if (
                    intent.attempt_ordinal == attempt_ordinal
                    and intent.executor_generation == executor_generation
                ):
                    return intent, True
        writer_session = max((intent.writer_session_id for intent in intents), default=0) + 1
        intent = ArtifactIntent(
            transaction,
            declaration_digest,
            declaration_bytes,
            request_id,
            invocation_spec_digest,
            output_slot,
            writer_session,
            attempt_ordinal,
            executor_generation,
        )
        body = {
            "transaction_id": transaction,
            "declaration_digest": declaration_digest,
            "declaration_b64": base64.b64encode(declaration_bytes).decode(),
            "request_id": request_id,
            "invocation_spec_digest": invocation_spec_digest,
            "output_slot": output_slot,
            "writer_session_id": writer_session,
            "attempt_ordinal": attempt_ordinal,
            "executor_generation": executor_generation,
        }
        self.journal.append("artifact_intent", body)
        self.intents.setdefault(transaction, []).append(intent)
        return intent, False

    def commit(
        self,
        *,
        transaction: str,
        receipt_bytes: bytes,
        snapshot_digest: str,
        header_digest: str,
    ) -> tuple[StoredReceipt, bool]:
        digest = "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()
        receipt = StoredReceipt(
            transaction, digest, receipt_bytes, snapshot_digest, header_digest
        )
        held = self.receipts.get(transaction)
        if held is not None:
            if held != receipt:
                raise ArtifactTransactionError(
                    "artifact_receipt_conflict",
                    "TensorFS returned different receipt bytes for one committed transaction",
                )
            return held, True
        self.journal.append(
            "artifact_receipt",
            {
                "transaction_id": transaction,
                "receipt_digest": digest,
                "receipt_b64": base64.b64encode(receipt_bytes).decode(),
                "snapshot_digest": snapshot_digest,
                "header_digest": header_digest,
            },
        )
        self.receipts[transaction] = receipt
        return receipt, False

    def finalize(self, transaction: str, result: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        body = {"transaction_id": transaction, **result}
        held = self.final.get(transaction)
        if held is not None:
            if held != body:
                raise ArtifactTransactionError(
                    "artifact_finalize_conflict",
                    "one transaction already has a different durable final disposition",
                )
            return held, True
        self.journal.append("artifact_final", body)
        self.final[transaction] = body
        return body, False
