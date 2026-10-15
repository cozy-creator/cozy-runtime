"""The executor's TensorFS source and output handles, granted per attempt by its owner.

Only bounded declarations and receipts cross the executor's control exchange; TensorFS
owns the source and writer channels, which TensorD serves from its Store.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import tensorfs
from tensorfs.derived import DerivedTransaction, SourceCapability

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact, ObjectRef
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._executor_requests import (
    Adopted,
    Exchange,
    Opened,
    WriterAdopt,
    WriterOutput,
    WriterRequest,
    WriterSource,
)
from cozy_runtime.internal import weights_receipts, weights_sink

if TYPE_CHECKING:
    from tensorfs.derived import Derivation


def declaration_bound() -> int:
    """A declaration grows with the tensors it declares; TensorFS bounds its own documents."""
    return int(tensorfs.manifest_max_bytes())


class ExecutionStorage:
    """Execution bindings only: the executor receives TensorFS handles, never a Store."""

    def __init__(self, spool: Path, exchange: Exchange, outputs: Mapping[str, int]) -> None:
        self.spool, self.exchange, self.outputs = spool, exchange, dict(outputs)
        self.opened: dict[str, str] = {}

    def _call[A: Opened | Adopted](self, request: WriterRequest, into: type[A]) -> A:
        answer = self.exchange(request, into)
        if not answer.ok:
            raise CapabilityError(answer.detail, code=answer.code)
        return answer

    def source(self, manifest: str) -> SourceCapability:
        answer = self._call(WriterSource(manifest=manifest), Opened)
        return SourceCapability.from_fd(manifest, answer.length, answer.descriptor)

    def open_output(self, output_slot: str, definition: Derivation) -> DerivedTransaction:
        maximum = self.outputs.get(output_slot)
        if maximum is None:
            raise CapabilityError("output is not declared", code="weights_output_ungranted")
        data = json.dumps(definition.native_arguments(maximum), separators=(",", ":")).encode()
        length = _write(
            weights_sink.exchange_file(self.spool, output_slot, "derivation"),
            data,
            declaration_bound(),
        )
        answer = self._call(WriterOutput(output_slot=output_slot, length=length), Opened)
        writer = DerivedTransaction.accept(answer.descriptor)
        self.opened[answer.transaction] = output_slot
        return writer

    def adopt_model(self, facts: Mapping[str, object]) -> ModelArtifact:
        transaction = weights_sink.native_receipt(facts).transaction_id
        slot = self.opened.get(transaction)
        if slot is None:
            raise CapabilityError("receipt has no opened output", code="weights_receipt_mismatch")
        length = _write(
            weights_sink.exchange_file(self.spool, slot, "native-receipt"),
            canonical_json.encode(dict(facts)),
            weights_receipts.MAX_RECEIPT_BYTES,
        )
        answer = self._call(
            WriterAdopt(output_slot=slot, transaction=transaction, length=length), Adopted
        )
        return ModelArtifact(
            answer.request_id,
            slot,
            ObjectRef(answer.manifest, answer.manifest_length),
            answer.receipt_digest,
        )


def _write(path: Path, data: bytes, maximum: int) -> int:
    if len(data) > maximum:
        raise CapabilityError("writer stream exceeds its bound", code="weights_writer_length")
    # An old name is never followed or opened for overwrite, including a planted symlink.
    path.unlink(missing_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "wb") as target:
        target.write(data)
    return len(data)
