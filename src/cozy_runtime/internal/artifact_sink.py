"""The Runtime/host adapter for TensorFS derived artifact transactions.

The author service emits one declaration plus bounded byte readers. This module is the
privileged half: it validates the immutable invocation grant, opens the real tfs-019
transaction before pulling bytes, streams directly into TensorFS, and returns exact receipt
bytes suitable for Worker Protocol's ``ArtifactReceipt`` wrapper.

Pod-supervisor does not yet expose this handshake to its stateless Runtime child.  The
adapter therefore has no implicit worker hookup: local Runtime or the future host bridge
must construct it with exact owner/request/spec/source/output facts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from io import BytesIO
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.author._artifact import (
    ArtifactCommit,
    ArtifactConfig,
    ArtifactReceipt,
    ArtifactTarget,
    Readable,
)
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


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

    def commit(self, request: ArtifactCommit) -> ArtifactReceipt:
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
        existing = self.store.derived_lookup(transaction_id)
        if existing.get("state") == "committed":
            facts = self._receipt(existing.get("receipt"), transaction_id, expected_declaration)
            return self._wrap(request.output_slot, transaction_id, facts, replayed=True)
        if existing.get("state") == "abandoned":
            raise CapabilityError(
                f"artifact transaction {transaction_id} was abandoned",
                code="artifact_transaction_closed",
            )
        writer: Any = None
        try:
            # Declaration and aggregate new-byte bound lock BEFORE TensorFS asks any
            # produced stream for one byte. No model-sized output copy exists between the
            # job and the CAS writer.
            writer = self.store.begin_derived(
                transaction_id,
                self.writer_session_id,
                sources,
                targets,
                configs,
                list(request.order),
                request.max_new_bytes,
            )
            for component, target in sorted(request.targets.items()):
                for key, tensor in sorted(target.add.items()):
                    for role, part in sorted(tensor.parts.items()):
                        writer.add_part(component, key, role, self._reader(part.data))
            for name, config in sorted(request.configs.items()):
                if config.data is not None:
                    writer.add_config(name, self._reader(config.data))
            facts = self._receipt(writer.commit(), transaction_id, expected_declaration)
        except CapabilityError:
            if writer is not None:
                writer.fence()
            raise
        except Exception as exc:
            fence_error = ""
            if writer is not None:
                try:
                    writer.fence()
                except Exception as fence:
                    fence_error = f"; writer fence also failed: {type(fence).__name__}: {fence}"
            raise CapabilityError(
                f"TensorFS artifact transaction refused: {type(exc).__name__}: {exc}{fence_error}",
                code="artifact_tensorfs_refused",
            ) from exc
        return self._wrap(request.output_slot, transaction_id, facts, replayed=False)

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
    def _reader(source: object) -> Readable:
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
        return value

    @staticmethod
    def _wrap(
        output_slot: str,
        transaction_id: str,
        facts: dict[str, Any],
        *,
        replayed: bool,
    ) -> ArtifactReceipt:
        raw = canonical_json.encode(facts)
        return ArtifactReceipt(
            output_slot=output_slot,
            artifact_transaction_id=transaction_id,
            tensorfs_receipt_digest="sha256:" + hashlib.sha256(raw).hexdigest(),
            tensorfs_receipt=raw,
            replayed=replayed,
        )


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
