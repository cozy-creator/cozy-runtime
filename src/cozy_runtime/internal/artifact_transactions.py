"""Runtime ownership for job artifact transactions.

The executor turns the path-free author declaration into plain typed TensorFS arguments.
The worker owns the semantic identity, the monotone writer fence, and durable replay.  This
module contains the common closed declaration and journal projection; neither side learns a
model family or a repository/publication vocabulary.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

from cozy_runtime.author._artifact_sink import (
    ArtifactBytes,
    ArtifactCommit,
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
        if not config.media_type or config.length <= 0:
            raise CapabilityError(
                f"config {name!r} needs an exact media type and positive length",
                code="artifact_declaration",
            )
        streams[("config", name)] = _reader(config.data, f"config {name}")
        return {
            "name": name,
            "mode": "create",
            "media_type": config.media_type,
            "length": config.length,
        }
    raise CapabilityError(
        f"config {name!r} has unsupported type {type(config).__name__}",
        code="artifact_declaration",
    )


def prepare_declaration(
    request: _ArtifactRequest,
    *,
    granted_sources: dict[str, tuple[str, int]],
) -> PreparedDeclaration:
    """Close an author request against the executor's verified source grant.

    `granted_sources` is ref -> exact snapshot digest, projected from the worker-verified
    InvocationSpec.  A mutable ref or a materialized path never substitutes for that digest.
    """
    sources: list[dict[str, Any]] = []
    for alias, tree in sorted(request.sources.items()):
        granted_digest, granted_length = granted_sources.get(tree.ref, ("", 0))
        if (
            _DIGEST.fullmatch(tree.digest) is None
            or tree.digest != granted_digest
            or granted_length <= 0
        ):
            raise CapabilityError(
                f"source alias {alias!r} names {tree.ref!r} at {tree.digest or '(no digest)'}, "
                "but the verified InvocationSpec grants "
                f"{granted_digest or 'no exact snapshot'} at {granted_length} B",
                code="artifact_source_ungranted",
            )
        sources.append(
            {
                "alias": alias,
                "input_ref": tree.ref,
                "snapshot": tree.digest,
                "snapshot_length": granted_length,
            }
        )
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
    if target.exists() or target.is_symlink():
        existing = read_staged_exact(
            root,
            name,
            expected_digest=digest,
            expected_length=len(data),
            max_bytes=canonical.DOC_MAX_BYTES,
        )
        if existing != data:
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
            transaction = str(body["transaction_id"])
            prior = self.intents.get(transaction, [])
            encoded = str(body.get("declaration_b64", ""))
            if encoded:
                declaration_bytes = base64.b64decode(encoded)
            elif prior:
                declaration_bytes = prior[0].declaration_bytes
            else:
                raise JournalError(
                    "journal_corrupt",
                    f"first artifact intent for {transaction} carries no declaration bytes",
                )
            declaration_digest = str(body["declaration_digest"])
            if "sha256:" + hashlib.sha256(declaration_bytes).hexdigest() != declaration_digest:
                raise JournalError(
                    "journal_corrupt",
                    f"artifact declaration bytes do not match {declaration_digest}",
                )
            intent = ArtifactIntent(
                transaction_id=transaction,
                declaration_digest=declaration_digest,
                declaration_bytes=declaration_bytes,
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
            receipt_bytes = base64.b64decode(str(body["receipt_b64"]))
            receipt_digest = str(body["receipt_digest"])
            if "sha256:" + hashlib.sha256(receipt_bytes).hexdigest() != receipt_digest:
                raise JournalError(
                    "journal_corrupt", f"artifact receipt bytes do not match {receipt_digest}"
                )
            receipt = StoredReceipt(
                transaction_id=str(body["transaction_id"]),
                receipt_digest=receipt_digest,
                receipt_bytes=receipt_bytes,
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
            "request_id": request_id,
            "invocation_spec_digest": invocation_spec_digest,
            "output_slot": output_slot,
            "writer_session_id": writer_session,
            "attempt_ordinal": attempt_ordinal,
            "executor_generation": executor_generation,
        }
        if not intents:
            body["declaration_b64"] = base64.b64encode(declaration_bytes).decode()
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


def _closed(row: Any, fields: frozenset[str], what: str) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != fields:
        actual = sorted(row) if isinstance(row, dict) else type(row).__name__
        raise ArtifactTransactionError(
            "artifact_declaration_schema",
            f"{what} has fields {actual}, expected {sorted(fields)}",
        )
    return row


def validate_declaration(data: bytes) -> dict[str, Any]:
    """Read the executor-authored control document under a closed schema.

    TensorFS remains authoritative for dtype, shape, encoding, source component, edit, and
    complete-target validity.  This pass holds Runtime's smaller authority boundary: no
    transaction/session id, path, store, tool, publisher, or undeclared source/slot can hide
    in the document handed back by a disposable executor.
    """
    try:
        value = canonical.parse_canonical(data)
    except canonical.CanonicalError as exc:
        raise ArtifactTransactionError(exc.code, exc.detail) from exc
    document = _closed(
        value,
        frozenset({"format", "output_slot", "sources", "configs", "targets"}),
        "ArtifactDeclaration",
    )
    if document["format"] != DECLARATION_FORMAT or not isinstance(document["output_slot"], str):
        raise ArtifactTransactionError(
            "artifact_declaration_schema", "ArtifactDeclaration has the wrong format or slot"
        )
    sources = document["sources"]
    if not isinstance(sources, list) or not sources:
        raise ArtifactTransactionError(
            "artifact_declaration_schema", "ArtifactDeclaration sources are a non-empty list"
        )
    aliases: list[str] = []
    for position, source in enumerate(sources):
        row = _closed(
            source,
            frozenset({"alias", "input_ref", "snapshot", "snapshot_length"}),
            f"sources[{position}]",
        )
        if not all(
            isinstance(row[key], str) and row[key]
            for key in ("alias", "input_ref", "snapshot")
        ) or not isinstance(row["snapshot_length"], int) or row["snapshot_length"] <= 0:
            raise ArtifactTransactionError(
                "artifact_declaration_schema", f"sources[{position}] fields are non-empty strings"
            )
        if _DIGEST.fullmatch(row["snapshot"]) is None:
            raise ArtifactTransactionError(
                "artifact_declaration_schema", f"sources[{position}] snapshot is not a digest"
            )
        aliases.append(row["alias"])
    if aliases != sorted(set(aliases)):
        raise ArtifactTransactionError(
            "artifact_declaration_schema", "source aliases are unique and canonically ordered"
        )

    configs = document["configs"]
    if not isinstance(configs, list):
        raise ArtifactTransactionError(
            "artifact_declaration_schema", "ArtifactDeclaration configs are a list"
        )
    config_names: list[str] = []
    for position, config in enumerate(configs):
        if not isinstance(config, dict):
            raise ArtifactTransactionError(
                "artifact_declaration_schema", f"configs[{position}] is an object"
            )
        mode = config.get("mode")
        fields = (
            frozenset({"name", "mode", "source_alias", "source_name"})
            if mode == "copy"
            else frozenset({"name", "mode", "media_type", "length"})
            if mode == "create"
            else frozenset()
        )
        row = _closed(config, fields, f"configs[{position}]")
        if not fields or not isinstance(row["name"], str) or not row["name"]:
            raise ArtifactTransactionError(
                "artifact_declaration_schema", f"configs[{position}] has an invalid mode/name"
            )
        if mode == "copy" and (
            row["source_alias"] not in aliases
            or not isinstance(row["source_name"], str)
            or not row["source_name"]
        ):
            raise ArtifactTransactionError(
                "artifact_source_ungranted", f"configs[{position}] selects no granted source"
            )
        if mode == "create" and (
            not isinstance(row["media_type"], str)
            or not row["media_type"]
            or not isinstance(row["length"], int)
            or row["length"] <= 0
        ):
            raise ArtifactTransactionError(
                "artifact_declaration_schema", f"configs[{position}] has no exact byte contract"
            )
        config_names.append(row["name"])
    if config_names != sorted(set(config_names)):
        raise ArtifactTransactionError(
            "artifact_declaration_schema", "config names are unique and canonically ordered"
        )

    targets = document["targets"]
    if not isinstance(targets, list) or not targets:
        raise ArtifactTransactionError(
            "artifact_declaration_schema", "ArtifactDeclaration targets are a non-empty list"
        )
    target_names: list[str] = []
    for position, target in enumerate(targets):
        if not isinstance(target, dict):
            raise ArtifactTransactionError(
                "artifact_declaration_schema", f"targets[{position}] is an object"
            )
        mode = str(target.get("mode", ""))
        fields = {
            "copy": frozenset({"component", "mode", "source_alias", "source_component"}),
            "derive": frozenset(
                {
                    "component",
                    "mode",
                    "source_alias",
                    "source_component",
                    "drop",
                    "add",
                }
            ),
            "create": frozenset({"component", "mode", "values"}),
        }.get(mode, frozenset())
        row = _closed(target, fields, f"targets[{position}]")
        if not fields or not isinstance(row["component"], str) or not row["component"]:
            raise ArtifactTransactionError(
                "artifact_declaration_schema", f"targets[{position}] has an invalid mode/name"
            )
        if mode in ("copy", "derive") and (
            row["source_alias"] not in aliases
            or not isinstance(row["source_component"], str)
            or not row["source_component"]
        ):
            raise ArtifactTransactionError(
                "artifact_source_ungranted", f"targets[{position}] selects no granted source"
            )
        additions = row.get("add", row.get("values", []))
        if not isinstance(additions, list) or (mode == "create" and not additions):
            raise ArtifactTransactionError(
                "artifact_declaration_schema", f"targets[{position}] values are invalid"
            )
        value_keys: list[str] = []
        for value_position, addition in enumerate(additions):
            value = _closed(
                addition,
                frozenset({"key", "logical_dtype", "shape", "encoding", "parts"}),
                f"targets[{position}].values[{value_position}]",
            )
            if (
                not isinstance(value["key"], str)
                or not value["key"]
                or not isinstance(value["logical_dtype"], str)
                or not value["logical_dtype"]
                or _DIGEST.fullmatch(value["encoding"]) is None
                or not isinstance(value["shape"], list)
                or not isinstance(value["parts"], list)
                or not value["parts"]
            ):
                raise ArtifactTransactionError(
                    "artifact_declaration_schema", "added value metadata is incomplete"
                )
            roles: list[str] = []
            for part_position, part in enumerate(value["parts"]):
                part_row = _closed(
                    part,
                    frozenset({"role", "dtype", "shape"}),
                    f"targets[{position}].values[{value_position}].parts[{part_position}]",
                )
                if (
                    not isinstance(part_row["role"], str)
                    or not part_row["role"]
                    or not isinstance(part_row["dtype"], str)
                    or not part_row["dtype"]
                    or not isinstance(part_row["shape"], list)
                ):
                    raise ArtifactTransactionError(
                        "artifact_declaration_schema", "added part metadata is incomplete"
                    )
                roles.append(part_row["role"])
            if roles != sorted(set(roles)):
                raise ArtifactTransactionError(
                    "artifact_declaration_schema", "added part roles are unique and ordered"
                )
            value_keys.append(value["key"])
        if value_keys != sorted(set(value_keys)):
            raise ArtifactTransactionError(
                "artifact_declaration_schema", "added value keys are unique and ordered"
            )
        drops = row.get("drop", [])
        if not isinstance(drops, list) or drops != sorted(set(drops)):
            raise ArtifactTransactionError(
                "artifact_declaration_schema", "dropped value keys are unique and ordered"
            )
        target_names.append(row["component"])
    if target_names != sorted(set(target_names)):
        raise ArtifactTransactionError(
            "artifact_declaration_schema", "target component names are unique and ordered"
        )
    return document


class ArtifactCoordinator:
    """Worker-owned authorization, identity, and journal boundary around TensorFS."""

    def __init__(self, journal: Journal, store_root: Path) -> None:
        self.state = ArtifactJournal(journal)
        self.store_root = store_root

    def open(
        self,
        *,
        owner_authority_scope: str,
        request_id: str,
        invocation_spec_digest: str,
        attempt_ordinal: int,
        executor_generation: int,
        spool: Path,
        staged_name: str,
        staged_digest: str,
        staged_length: int,
        granted_sources: dict[str, tuple[str, int]],
        granted_outputs: dict[str, int],
    ) -> dict[str, Any]:
        data = read_staged_exact(
            spool,
            staged_name,
            expected_digest=staged_digest,
            expected_length=staged_length,
            max_bytes=canonical.DOC_MAX_BYTES,
        )
        document = validate_declaration(data)
        slot = str(document["output_slot"])
        max_new_bytes = granted_outputs.get(slot, 0)
        if max_new_bytes <= 0:
            raise ArtifactTransactionError(
                "artifact_output_ungranted",
                f"output slot {slot!r} is not an exact positive-byte InvocationSpec output",
            )
        for source in document["sources"]:
            granted = granted_sources.get(str(source["input_ref"]), ("", 0))
            if (source["snapshot"], source["snapshot_length"]) != granted:
                raise ArtifactTransactionError(
                    "artifact_source_ungranted",
                    f"source {source['input_ref']!r} at "
                    f"{(source['snapshot'], source['snapshot_length'])} is not granted "
                    f"at that identity ({granted})",
                )
        transaction = transaction_id(
            owner_authority_scope, request_id, invocation_spec_digest, slot
        )
        if transaction in self.state.final:
            raise ArtifactTransactionError(
                "artifact_transaction_final",
                f"artifact transaction {transaction} already has a final disposition",
            )
        receipt = self.state.receipts.get(transaction)
        observed = self._lookup_state(transaction)
        if receipt is None and observed["state"] == "committed":
            actual = observed["receipt"]
            data = canonical.write(actual)
            receipt, _ = self.state.commit(
                transaction=transaction,
                receipt_bytes=data,
                snapshot_digest=_object_digest(actual.get("snapshot"), "receipt.snapshot"),
                header_digest=_object_digest(actual.get("header"), "receipt.header"),
            )
        if receipt is not None:
            return {
                "state": "committed",
                "transaction_id": transaction,
                "receipt_digest": receipt.receipt_digest,
                "snapshot_digest": receipt.snapshot_digest,
                "header_digest": receipt.header_digest,
                "replayed": True,
            }
        intent, replayed = self.state.open(
            transaction=transaction,
            declaration_digest=staged_digest,
            declaration_bytes=data,
            request_id=request_id,
            invocation_spec_digest=invocation_spec_digest,
            output_slot=slot,
            attempt_ordinal=attempt_ordinal,
            executor_generation=executor_generation,
        )
        observed_session = observed.get("writer_session_id")
        if (
            observed["state"] == "open"
            and isinstance(observed_session, int)
            and observed_session != intent.writer_session_id
        ):
            self._fence(transaction, observed_session)
        return {
            "state": "open",
            "transaction_id": transaction,
            "writer_session_id": intent.writer_session_id,
            "max_new_bytes": max_new_bytes,
            "replayed": replayed,
        }

    def commit(
        self,
        *,
        transaction: str,
        writer_session_id: int,
        spool: Path,
        staged_name: str,
        staged_digest: str,
        staged_length: int,
    ) -> dict[str, Any]:
        intents = self.state.intents.get(transaction, [])
        if not any(intent.writer_session_id == writer_session_id for intent in intents):
            raise ArtifactTransactionError(
                "artifact_writer_fenced",
                f"writer session {writer_session_id} was never minted for {transaction}",
            )
        data = read_staged_exact(
            spool,
            staged_name,
            expected_digest=staged_digest,
            expected_length=staged_length,
            max_bytes=canonical.DOC_MAX_BYTES,
        )
        try:
            carried = canonical.parse_canonical(data)
        except canonical.CanonicalError as exc:
            raise ArtifactTransactionError(exc.code, exc.detail) from exc
        if not isinstance(carried, dict) or carried.get("transaction_id") != transaction:
            raise ArtifactTransactionError(
                "artifact_receipt", "the staged TensorFS receipt names another transaction"
            )
        actual = self._lookup(transaction)
        if actual is None or canonical.write(actual) != data:
            raise ArtifactTransactionError(
                "artifact_receipt_unproven",
                "TensorFS lookup does not return the exact staged committed receipt",
            )
        snapshot = _object_digest(carried.get("snapshot"), "receipt.snapshot")
        header = _object_digest(carried.get("header"), "receipt.header")
        receipt, replayed = self.state.commit(
            transaction=transaction,
            receipt_bytes=data,
            snapshot_digest=snapshot,
            header_digest=header,
        )
        return {
            "transaction_id": transaction,
            "receipt_digest": receipt.receipt_digest,
            "snapshot_digest": receipt.snapshot_digest,
            "header_digest": receipt.header_digest,
            "replayed": replayed,
        }

    def settle(self, *, attempt_ordinal: int, executor_generation: int) -> list[StoredReceipt]:
        """Recover commit-before-journal, or fence every still-open writer for an attempt."""
        settled: list[StoredReceipt] = []
        for transaction, intents in sorted(self.state.intents.items()):
            matching = [
                intent
                for intent in intents
                if intent.attempt_ordinal == attempt_ordinal
                and intent.executor_generation == executor_generation
            ]
            if not matching:
                continue
            receipt = self.state.receipts.get(transaction)
            if receipt is None:
                actual = self._lookup(transaction)
                if actual is not None:
                    data = canonical.write(actual)
                    receipt, _ = self.state.commit(
                        transaction=transaction,
                        receipt_bytes=data,
                        snapshot_digest=_object_digest(actual.get("snapshot"), "receipt.snapshot"),
                        header_digest=_object_digest(actual.get("header"), "receipt.header"),
                    )
                else:
                    self._fence(transaction, matching[-1].writer_session_id)
            if receipt is not None:
                settled.append(receipt)
        return settled

    def finalize(
        self,
        *,
        transaction: str,
        decision: str,
        receipt_digest: str = "",
        scratch_root: str = "",
    ) -> tuple[dict[str, Any], bool]:
        """Drive one stored-root disposition after owner authentication at the wire edge.

        This is deliberately transport-neutral. th-049 supplies the authenticated request
        and exact replay id; this method owns only Runtime/TensorFS behavior.
        """
        if decision not in ("adopt", "abandon", "abandon_uncommitted"):
            raise ArtifactTransactionError(
                "artifact_finalize_decision", f"unknown artifact decision {decision!r}"
            )
        if (decision == "adopt") != bool(scratch_root):
            raise ArtifactTransactionError(
                "artifact_finalize_decision",
                "only adopt carries one non-empty owner-scoped scratch root",
            )
        if (decision == "abandon_uncommitted") == bool(receipt_digest):
            raise ArtifactTransactionError(
                "artifact_finalize_decision",
                "committed decisions require a receipt digest; abandon_uncommitted forbids one",
            )
        prior = self.state.final.get(transaction)
        if prior is not None:
            if (
                prior.get("decision") != decision
                or prior.get("requested_receipt_digest", "") != receipt_digest
                or prior.get("scratch_root", "") != scratch_root
            ):
                raise ArtifactTransactionError(
                    "artifact_finalize_conflict", "the transaction has another final decision"
                )
            return prior, True

        store = self._store()
        held = self.state.receipts.get(transaction)
        evidence = held
        if decision in ("adopt", "abandon"):
            if held is None or held.receipt_digest != receipt_digest:
                raise ArtifactTransactionError(
                    "artifact_finalize_receipt",
                    "the committed decision does not name Runtime's exact durable receipt",
                )
            disposed = store.derived_dispose(
                transaction, scratch_root if decision == "adopt" else None
            )
            self._verify_disposition(transaction, held, disposed, decision, scratch_root)
        else:
            observed = self._lookup_state(transaction)
            if observed["state"] == "open" and isinstance(
                observed.get("writer_session_id"), int
            ):
                store.derived_fence(transaction, int(observed["writer_session_id"]))
            abandoned = store.derived_abandon(transaction)
            if not isinstance(abandoned, dict) or abandoned.get("state") not in {
                "abandoned",
                "committed",
            }:
                raise ArtifactTransactionError(
                    "artifact_finalize_result", "TensorFS returned an unknown abandon result"
                )
            if abandoned["state"] == "committed":
                receipt = abandoned.get("receipt")
                if not isinstance(receipt, dict):
                    raise ArtifactTransactionError(
                        "artifact_finalize_result", "commit-first abandon returned no receipt"
                    )
                receipt_bytes = canonical.write(receipt)
                evidence, _ = self.state.commit(
                    transaction=transaction,
                    receipt_bytes=receipt_bytes,
                    snapshot_digest=_object_digest(receipt.get("snapshot"), "receipt.snapshot"),
                    header_digest=_object_digest(receipt.get("header"), "receipt.header"),
                )
                disposed = store.derived_dispose(transaction, None)
                self._verify_disposition(transaction, evidence, disposed, "abandon", "")

        result = {
            "decision": decision,
            "requested_receipt_digest": receipt_digest,
            "scratch_root": scratch_root,
            "status": "adopted" if decision == "adopt" else "abandoned",
            "evidence_receipt_digest": evidence.receipt_digest if evidence else "",
        }
        return self.state.finalize(transaction, result)

    @staticmethod
    def _verify_disposition(
        transaction: str,
        receipt: StoredReceipt,
        result: Any,
        decision: str,
        scratch_root: str,
    ) -> None:
        if not isinstance(result, dict):
            raise ArtifactTransactionError(
                "artifact_finalize_result", "TensorFS disposition result is not an object"
            )
        carried = result.get("receipt")
        disposition = result.get("disposition")
        expected = (
            {"kind": "transferred", "root": scratch_root}
            if decision == "adopt"
            else {"kind": "released"}
        )
        if (
            not isinstance(carried, dict)
            or carried.get("transaction_id") != transaction
            or canonical.write(carried) != receipt.receipt_bytes
            or disposition != expected
        ):
            raise ArtifactTransactionError(
                "artifact_finalize_result",
                "TensorFS disposition does not preserve the immutable receipt and exact decision",
            )

    def _store(self) -> Any:
        try:
            import tensorfs

            return tensorfs.Store.open(str(self.store_root))
        except Exception as exc:
            raise ArtifactTransactionError(
                str(getattr(exc, "code", "artifact_store_unavailable")),
                str(getattr(exc, "detail", exc)),
            ) from exc

    def _lookup(self, transaction: str) -> dict[str, Any] | None:
        value = self._lookup_state(transaction)
        if value["state"] in ("absent", "open"):
            return None
        if value["state"] == "abandoned":
            raise ArtifactTransactionError(
                "artifact_transaction_final", f"artifact transaction {transaction} is abandoned"
            )
        receipt = value.get("receipt")
        if not isinstance(receipt, dict) or receipt.get("transaction_id") != transaction:
            raise ArtifactTransactionError(
                "artifact_receipt", "TensorFS committed state carries no matching receipt"
            )
        return receipt

    def _lookup_state(self, transaction: str) -> dict[str, Any]:
        value = self._store().derived_lookup(transaction)
        if not isinstance(value, dict) or value.get("state") not in {
            "absent",
            "open",
            "committed",
            "abandoned",
        }:
            raise ArtifactTransactionError(
                "artifact_receipt", "TensorFS derived_lookup returned an unknown state"
            )
        return value

    def _fence(self, transaction: str, writer_session_id: int) -> None:
        self._store().derived_fence(transaction, writer_session_id)


def _tensorfs_declaration(
    document: Mapping[str, Any],
) -> tuple[
    dict[str, tuple[str, int]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    sources = {
        str(row["alias"]): (str(row["snapshot"]), int(row["snapshot_length"]))
        for row in document["sources"]
    }
    targets: dict[str, dict[str, Any]] = {}
    for row in document["targets"]:
        additions = row.get("add", row.get("values", []))
        target: dict[str, Any] = {
            "drop": list(row.get("drop", [])),
            "add": {
                value["key"]: {
                    "logical_dtype": value["logical_dtype"],
                    "shape": value["shape"],
                    "encoding": value["encoding"],
                    "parts": {
                        part["role"]: {"dtype": part["dtype"], "shape": part["shape"]}
                        for part in value["parts"]
                    },
                }
                for value in additions
            },
        }
        if row["mode"] in ("copy", "derive"):
            target["source"] = row["source_alias"]
            target["source_component"] = row["source_component"]
        targets[str(row["component"])] = target
    configs: dict[str, dict[str, Any]] = {}
    for row in document["configs"]:
        configs[str(row["name"])] = (
            {"source": row["source_alias"], "source_config": row["source_name"]}
            if row["mode"] == "copy"
            else {"media_type": row["media_type"], "length": row["length"]}
        )
    return sources, targets, configs


def _object_digest(row: Any, what: str) -> str:
    if not isinstance(row, Mapping):
        raise ArtifactTransactionError("artifact_receipt", f"{what} is not an ObjectRef")
    digest = str(row.get("sha256", ""))
    length = row.get("length")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ArtifactTransactionError("artifact_receipt", f"{what} has no exact sha256")
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise ArtifactTransactionError("artifact_receipt", f"{what} has no positive length")
    return "sha256:" + digest


class ExecutorArtifactSink:
    """Executor-side TensorFS writer hidden behind the author-facing ArtifactSink."""

    def __init__(
        self,
        *,
        store_root: Path,
        spool: Path,
        granted_sources: dict[str, tuple[str, int]],
        granted_outputs: dict[str, int],
        durable: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.store_root = store_root
        self.spool = spool
        self.granted_sources = granted_sources
        self.granted_outputs = granted_outputs
        self.durable = durable

    def commit(self, request: _ArtifactRequest) -> ArtifactCommit:
        if request.output_slot not in self.granted_outputs:
            raise CapabilityError(
                f"output slot {request.output_slot!r} is not granted to this attempt",
                code="artifact_output_ungranted",
            )
        prepared = prepare_declaration(request, granted_sources=self.granted_sources)
        staged_name, staged_digest = stage_exact(
            self.spool, "declaration", prepared.canonical_bytes
        )
        opened = self.durable(
            "artifact_open",
            {
                "staged_name": staged_name,
                "staged_digest": staged_digest,
                "staged_length": len(prepared.canonical_bytes),
            },
        )
        if not opened.get("ok"):
            raise CapabilityError(
                str(opened.get("detail", "the worker refused the artifact transaction")),
                code=str(opened.get("code", "artifact_open_refused")),
            )
        if opened.get("state") == "committed":
            return ArtifactCommit(
                request.output_slot,
                str(opened["snapshot_digest"]),
                str(opened["header_digest"]),
                replayed=True,
            )
        if opened.get("state") != "open":
            raise CapabilityError(
                f"the worker returned unknown artifact state {opened.get('state')!r}",
                code="artifact_open_refused",
            )
        transaction = str(opened["transaction_id"])
        writer_session = int(opened["writer_session_id"])
        max_new_bytes = int(opened["max_new_bytes"])
        sources, targets, configs = _tensorfs_declaration(prepared.document)
        writer: Any = None
        try:
            import tensorfs

            store = tensorfs.Store.open(str(self.store_root))
            writer = store.begin_derived(
                transaction,
                writer_session,
                sources,
                cast("Any", targets),
                cast("Any", configs),
                max_new_bytes,
            )
            effective = writer.sources()
            observed_sources = (
                {
                    str(row.get("alias", "")): (
                        str(row.get("snapshot_id", "")),
                        int(row.get("snapshot_length", 0)),
                    )
                    for row in effective
                    if isinstance(row, Mapping)
                }
                if isinstance(effective, list)
                else {}
            )
            if observed_sources != sources:
                raise ArtifactTransactionError(
                    "artifact_source_drift",
                    "TensorFS effective sources differ from the exact granted source set",
                )
            for key, reader in sorted(prepared.streams.items()):
                if key[0] == "config":
                    writer.add_config(key[1], reader)
                else:
                    _, component, value, role = key
                    writer.add_part(component, value, role, reader)
            receipt = writer.commit()
            receipt_bytes = canonical.write(receipt)
            receipt_name, receipt_digest = stage_exact(self.spool, "receipt", receipt_bytes)
            answer = self.durable(
                "artifact_commit",
                {
                    "transaction_id": transaction,
                    "writer_session_id": writer_session,
                    "staged_name": receipt_name,
                    "staged_digest": receipt_digest,
                    "staged_length": len(receipt_bytes),
                },
            )
            if not answer.get("ok"):
                raise ArtifactTransactionError(
                    str(answer.get("code", "artifact_commit_refused")),
                    str(answer.get("detail", "the worker refused the committed receipt")),
                )
            return ArtifactCommit(
                request.output_slot,
                str(answer["snapshot_digest"]),
                str(answer["header_digest"]),
                replayed=bool(answer.get("replayed", False)),
            )
        except CapabilityError:
            raise
        except Exception as exc:
            if writer is not None:
                with contextlib.suppress(Exception):
                    writer.fence()
            code = str(getattr(exc, "code", "artifact_transaction_failed"))
            detail = str(getattr(exc, "detail", exc))
            raise CapabilityError(detail, code=code) from exc
