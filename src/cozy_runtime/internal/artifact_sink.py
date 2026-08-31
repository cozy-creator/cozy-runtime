"""The Runtime/host adapter for TensorFS derived artifact transactions.

The author service emits one declaration plus bounded byte readers. This module is the
privileged half: it validates the immutable invocation grant, opens the real tfs-019
transaction before pulling bytes, streams directly into TensorFS, and returns exact receipt
bytes suitable for Worker Protocol's ``ArtifactReceipt`` wrapper.

The worker supplies two callbacks for the pod lane: bind exact intent before TensorFS opens,
then durably record the exact protocol receipt before commit returns to package code. Local
execution omits them and retains the same TensorFS transaction implementation.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.author._artifact import (
    ArtifactBytes,
    ArtifactCommit,
    ArtifactConfig,
    ArtifactHostTransaction,
    ArtifactReceipt,
    ArtifactTarget,
    Readable,
)
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.protocol import artifact_limits, documents
from cozy_runtime.protocol import worker_pb2 as pb

_OUTPUT_SLOT = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_EXCHANGE_KINDS = frozenset({"declaration", "receipt"})


@dataclass(frozen=True, slots=True)
class ArtifactHostBinding:
    """The supervisor's durable answer to one exact TensorFS declaration."""

    transaction_id: str
    writer_generation: int
    artifact_receipt_digest: bytes = b""


def exchange_file(spool: Path, output_slot: str, kind: str) -> Path:
    """One worker-chosen spool file; no child-authored path crosses the seam."""
    if _OUTPUT_SLOT.fullmatch(output_slot) is None or kind not in _EXCHANGE_KINDS:
        raise CapabilityError(
            "artifact exchange identity is invalid", code="artifact_host_identity"
        )
    return spool / f"artifact-{output_slot}-{kind}.canonical"


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
                f"artifact {kind} replay changed exact bytes",
                code="artifact_receipt_mismatch"
                if kind == "receipt"
                else "artifact_intent_conflict",
            ) from None
    return target


class ArtifactTransactionHost:
    """One immutable invocation's real TensorFS artifact authority."""

    def __init__(
        self,
        *,
        store: Any,
        owner_scope: str,
        request_id: str,
        invocation_spec_digest: str,
        writer_session_id: int,
        allowed_sources: Mapping[str, int],
        output_bounds: Mapping[str, int],
        bind_intent: Callable[[str, bytes, bytes], ArtifactHostBinding] | None = None,
        record_receipt: Callable[[ArtifactReceipt], None] | None = None,
    ) -> None:
        if (
            not owner_scope
            or not request_id
            or not invocation_spec_digest.startswith("sha256:")
            or writer_session_id <= 0
        ):
            raise CapabilityError(
                "artifact host identity is incomplete", code="artifact_host_identity"
            )
        self.store = store
        self.owner_scope = owner_scope
        self.request_id = request_id
        self.invocation_spec_digest = invocation_spec_digest
        self.writer_session_id = writer_session_id
        self.allowed_sources = dict(allowed_sources)
        self.output_bounds = dict(output_bounds)
        self.bind_intent = bind_intent
        self.record_receipt = record_receipt

    def open(self, request: ArtifactCommit) -> ArtifactHostTransaction:
        """Lock one tfs-019 declaration before exposing any source reader or writer."""
        maximum = self.output_bounds.get(request.output_slot)
        if maximum is None or maximum != request.max_new_bytes:
            raise CapabilityError(
                f"artifact output {request.output_slot!r} is not in the immutable grant",
                code="artifact_output_ungranted",
            )
        sources: dict[str, tuple[str, int]] = {}
        for alias, manifest in request.sources.items():
            length = self.allowed_sources.get(manifest)
            if length is None:
                raise CapabilityError(
                    f"artifact source {alias!r} ({manifest}) is not in the immutable grant",
                    code="artifact_source_ungranted",
                )
            sources[alias] = (manifest, length)
        transaction_id = self.transaction_id(request.output_slot)
        targets = {name: self._target(target) for name, target in request.targets.items()}
        configs = {name: self._config(config) for name, config in request.configs.items()}
        expected_declaration = {
            "components": {
                name: self._locked_target(target)
                for name, target in sorted(request.targets.items())
            },
            "configs": configs,
            "max_new_bytes": request.max_new_bytes,
            "order": [list(row) for row in request.order],
            "sources": {
                alias: {"manifest": {"sha256": manifest.removeprefix("sha256:"), "length": length}}
                for alias, (manifest, length) in sources.items()
            },
        }
        declaration = canonical_json.encode(expected_declaration)
        declaration_digest = hashlib.sha256(declaration).digest()
        binding = (
            self.bind_intent(request.output_slot, declaration_digest, declaration)
            if self.bind_intent is not None
            else ArtifactHostBinding(transaction_id, self.writer_session_id)
        )
        if binding.transaction_id != transaction_id or binding.writer_generation <= 0:
            raise CapabilityError(
                "the artifact host bound another transaction or an invalid writer generation",
                code="artifact_host_identity",
            )
        self.writer_session_id = binding.writer_generation
        existing = self.store.derived_lookup(transaction_id)
        if existing.get("state") == "committed":
            facts = self._receipt(existing.get("receipt"), transaction_id, expected_declaration)
            receipt = self._wrap(request.output_slot, transaction_id, facts, replayed=True)
            _, _, protocol_digest = protocol_receipt(
                receipt,
                owner_scope=self.owner_scope,
                request_id=self.request_id,
                invocation_spec_digest=self.invocation_spec_digest,
            )
            if binding.artifact_receipt_digest:
                if binding.artifact_receipt_digest != protocol_digest:
                    raise CapabilityError(
                        "the artifact host replayed another receipt digest",
                        code="artifact_receipt_mismatch",
                    )
            elif self.record_receipt is not None:
                # TensorFS committed before the host receipt ACK. Close that crash gap now.
                self.record_receipt(receipt)
            return OpenedArtifactTransaction(
                host=self,
                request=request,
                transaction_id=transaction_id,
                declaration=expected_declaration,
                writer=None,
                receipt=receipt,
            )
        if binding.artifact_receipt_digest:
            raise CapabilityError(
                "the artifact host replayed a receipt absent from local TensorFS",
                code="artifact_receipt_mismatch",
            )
        if existing.get("state") == "abandoned":
            raise CapabilityError(
                f"artifact transaction {transaction_id} was abandoned",
                code="artifact_transaction_closed",
            )
        try:
            writer = self.store.begin_derived(
                transaction_id,
                binding.writer_generation,
                sources,
                targets,
                configs,
                list(request.order),
                request.max_new_bytes,
            )
        except Exception as exc:
            raise CapabilityError(
                f"TensorFS artifact declaration refused: {type(exc).__name__}: {exc}",
                code="artifact_tensorfs_refused",
            ) from exc
        return OpenedArtifactTransaction(
            host=self,
            request=request,
            transaction_id=transaction_id,
            declaration=expected_declaration,
            writer=writer,
            receipt=None,
        )

    def transaction_id(self, output_slot: str) -> str:
        subject = canonical_json.encode(
            {
                "owner_authority_scope": self.owner_scope,
                "request_id": self.request_id,
                "invocation_spec_digest": self.invocation_spec_digest,
                "output_slot": output_slot,
            }
        )
        return (
            "sha256:" + hashlib.sha256(b"cozy.runtime.artifact-transaction\0" + subject).hexdigest()
        )

    def fence(self, output_slot: str) -> bool:
        return bool(
            self.store.derived_fence(self.transaction_id(output_slot), self.writer_session_id)
        )

    def acknowledge(self, receipt: ArtifactReceipt) -> None:
        """Do not return a TensorFS commit to package code before the host records it."""
        if self.record_receipt is not None:
            self.record_receipt(receipt)

    def abandon(self, output_slot: str) -> ArtifactReceipt | None:
        """Fence then atomically abandon, returning a commit-race winner if one exists."""
        transaction_id = self.transaction_id(output_slot)
        state = self.store.derived_lookup(transaction_id).get("state")
        if state == "open":
            self.store.derived_fence(transaction_id, self.writer_session_id)
        result = self.store.derived_abandon(transaction_id)
        if result.get("state") != "committed":
            return None
        facts = result.get("receipt")
        if not isinstance(facts, dict):
            raise CapabilityError(
                "TensorFS returned a committed abandonment without a receipt",
                code="artifact_receipt_absent",
            )
        return self._wrap(output_slot, transaction_id, facts, replayed=True)

    def release(self, output_slot: str) -> ArtifactReceipt:
        """Release a committed transaction root after publication or rejection.

        TensorFS currently has no scratch-root transfer operation.  This is therefore the
        honest ABANDON path, not an ADOPT implementation.
        """
        transaction_id = self.transaction_id(output_slot)
        result = self.store.derived_dispose(transaction_id)
        facts = result.get("receipt") if isinstance(result, dict) else None
        if not isinstance(facts, dict):
            raise CapabilityError(
                "TensorFS disposition returned no receipt", code="artifact_receipt_absent"
            )
        return self._wrap(output_slot, transaction_id, facts, replayed=True)

    @staticmethod
    def _target(target: ArtifactTarget) -> dict[str, Any]:
        value: dict[str, Any] = {
            "drop": sorted(target.drop),
            "add": {
                key: {
                    "logical_dtype": tensor.logical_dtype,
                    "shape": list(tensor.shape),
                    "encoding": tensor.encoding,
                    "parts": {
                        role: {"dtype": part.dtype, "shape": list(part.shape)}
                        for role, part in sorted(tensor.parts.items())
                    },
                }
                for key, tensor in sorted(target.add.items())
            },
        }
        if target.source:
            value["source"] = target.source
            value["source_component"] = target.source_component
        return value

    @staticmethod
    def _locked_target(target: ArtifactTarget) -> dict[str, Any]:
        """TensorFS receipt spelling after the Python facade has typed the declaration."""
        value: dict[str, Any] = {
            "drop": sorted(target.drop),
            "add": {
                key: {
                    "encoding": tensor.encoding,
                    "logical": {
                        "dtype": tensor.logical_dtype,
                        "shape": list(tensor.shape),
                    },
                    "parts": {
                        role: {"dtype": part.dtype, "shape": list(part.shape)}
                        for role, part in sorted(tensor.parts.items())
                    },
                }
                for key, tensor in sorted(target.add.items())
            },
        }
        if target.source:
            value["source"] = target.source
            value["source_component"] = target.source_component
        return value

    @staticmethod
    def _config(config: ArtifactConfig) -> dict[str, Any]:
        value: dict[str, Any] = {}
        if config.source:
            value.update(source=config.source, source_config=config.source_config)
        if config.data is not None:
            known = (
                len(config.data)
                if isinstance(config.data, (bytes, bytearray, memoryview))
                else config.length
            )
            value.update(media_type=config.media_type, length=known)
        return value

    @staticmethod
    def _reader(source: ArtifactBytes) -> Readable:
        if isinstance(source, (bytes, bytearray, memoryview)):
            return BytesIO(bytes(source))
        if isinstance(source, Readable):
            return source
        raise CapabilityError("artifact part source is not readable", code="artifact_part_source")

    @staticmethod
    def _receipt(value: object, transaction_id: str, declaration: object) -> dict[str, Any]:
        if not isinstance(value, dict) or value.get("transaction_id") != transaction_id:
            raise CapabilityError(
                "TensorFS returned a receipt for another transaction",
                code="artifact_receipt_mismatch",
            )
        if value.get("declaration") != declaration:
            raise CapabilityError(
                "TensorFS receipt declaration differs from the locked artifact request",
                code="artifact_receipt_mismatch",
            )
        evidence = value.get("release_evidence_canonical_bytes")
        if (
            not isinstance(evidence, bytes)
            or not 0 < len(evidence) <= artifact_limits.MAX_RELEASE_EVIDENCE_BYTES
        ):
            raise CapabilityError(
                "TensorFS receipt carries no bounded ReleaseEvidence",
                code="artifact_receipt_mismatch",
            )
        return value

    @staticmethod
    def _wrap(
        output_slot: str,
        transaction_id: str,
        facts: dict[str, Any],
        *,
        replayed: bool,
    ) -> ArtifactReceipt:
        evidence = facts["release_evidence_canonical_bytes"]
        assert isinstance(evidence, bytes)
        serializable = dict(facts)
        try:
            serializable["release_evidence_canonical_bytes"] = evidence.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CapabilityError(
                "TensorFS ReleaseEvidence is not canonical UTF-8",
                code="artifact_receipt_mismatch",
            ) from exc
        raw = canonical_json.encode(serializable)
        return ArtifactReceipt(
            output_slot=output_slot,
            artifact_transaction_id=transaction_id,
            tensorfs_receipt_digest="sha256:" + hashlib.sha256(raw).hexdigest(),
            tensorfs_receipt=raw,
            release_evidence_canonical_bytes=evidence,
            replayed=replayed,
        )


class OpenedArtifactTransaction:
    """One declaration-locked TensorFS writer exposed through the narrow host protocol."""

    def __init__(
        self,
        *,
        host: ArtifactTransactionHost,
        request: ArtifactCommit,
        transaction_id: str,
        declaration: object,
        writer: Any,
        receipt: ArtifactReceipt | None,
    ) -> None:
        self.host = host
        self.request = request
        self.transaction_id = transaction_id
        self.declaration = declaration
        self.writer = writer
        self.receipt = receipt
        self.replayed = receipt is not None
        self.closed = self.replayed

    def source_read_into(
        self,
        source: str,
        component: str,
        key: str,
        role: str,
        offset: int,
        into: memoryview,
    ) -> None:
        self._call(
            "source read",
            lambda writer: writer.source_read_into(source, component, key, role, offset, into),
        )

    def add_part(self, component: str, key: str, role: str, data: ArtifactBytes) -> None:
        self._call(
            "artifact part",
            lambda writer: writer.add_part(component, key, role, self.host._reader(data)),
        )

    def add_config(self, name: str, data: ArtifactBytes) -> None:
        self._call(
            "artifact config", lambda writer: writer.add_config(name, self.host._reader(data))
        )

    def commit(self) -> ArtifactReceipt:
        if self.receipt is not None:
            return self.receipt
        writer = self._writer("commit")
        try:
            facts = self.host._receipt(writer.commit(), self.transaction_id, self.declaration)
        except Exception as exc:
            self._fence_after(exc, "commit")
        self.closed = True
        self.receipt = self.host._wrap(
            self.request.output_slot, self.transaction_id, facts, replayed=False
        )
        self.host.acknowledge(self.receipt)
        return self.receipt

    def fence(self) -> None:
        if self.closed:
            return
        writer = self._writer("fence")
        try:
            writer.fence()
        finally:
            self.closed = True

    def _writer(self, what: str) -> Any:
        if self.closed or self.writer is None:
            raise CapabilityError(
                f"artifact transaction {self.transaction_id} is closed during {what}",
                code="artifact_transaction_closed",
            )
        return self.writer

    def _call(self, what: str, operation: Any) -> None:
        writer = self._writer(what)
        try:
            operation(writer)
        except Exception as exc:
            self._fence_after(exc, what)

    def _fence_after(self, cause: Exception, what: str) -> None:
        fence_error = ""
        try:
            if self.writer is not None:
                self.writer.fence()
        except Exception as fence:
            fence_error = f"; writer fence also failed: {type(fence).__name__}: {fence}"
        self.closed = True
        if isinstance(cause, CapabilityError) and not fence_error:
            raise cause
        raise CapabilityError(
            f"TensorFS {what} refused: {type(cause).__name__}: {cause}{fence_error}",
            code="artifact_tensorfs_refused",
        ) from cause


def protocol_receipt(
    receipt: ArtifactReceipt,
    *,
    owner_scope: str,
    request_id: str,
    invocation_spec_digest: str,
) -> tuple[pb.ArtifactReceiptRef, bytes, bytes]:
    """Wrap exact TensorFS receipt bytes in Worker Protocol ArtifactReceipt/1."""
    body = pb.ArtifactReceipt(
        owner_authority_scope=owner_scope,
        request_id=request_id,
        invocation_spec_digest=invocation_spec_digest,
        output_slot=receipt.output_slot,
        artifact_transaction_id=receipt.artifact_transaction_id,
        tensorfs_receipt_digest=receipt.tensorfs_receipt_digest,
        tensorfs_receipt_canonical_bytes=receipt.tensorfs_receipt,
        release_evidence_canonical_bytes=receipt.release_evidence_canonical_bytes,
    )
    canonical_bytes, digest = documents.identity(body)
    return (
        pb.ArtifactReceiptRef(
            artifact_receipt_digest=digest,
            artifact_receipt_canonical_bytes=canonical_bytes,
        ),
        canonical_bytes,
        digest,
    )
