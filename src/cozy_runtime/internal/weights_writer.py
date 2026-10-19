"""The executor's weights sources and outputs, granted per attempt by TensorD (`weights.rs`).

A source is one of the call's `Weights[M]` checkpoints; an output one declared weights output,
written through TensorD's derive channel. Only bounded exchange files cross the control
exchange; TensorD serves the channels from its Store.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

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
from cozy_runtime.tensord_channels import Derivation, Output, Source, ref

#: An exchange file stays this small (TensorD's bound).
EXCHANGE_MAX = 64 << 20
_OUTPUT_SLOT = re.compile(r"[a-z][a-z0-9_-]{0,63}")


def exchange_file(spool: Path, output_slot: str, kind: str) -> Path:
    """One spool file the executor writes for TensorD; no child-authored path crosses."""
    if _OUTPUT_SLOT.fullmatch(output_slot) is None or kind not in ("derivation", "native-receipt"):
        raise CapabilityError("weights exchange identity is invalid", code="weights_host_identity")
    return spool / f"weights-{output_slot}-{kind}.canonical"


class ExecutionStorage:
    """Execution bindings only: the executor receives channels, never a Store."""

    def __init__(self, spool: Path, exchange: Exchange, outputs: Mapping[str, int]) -> None:
        self.spool, self.exchange, self.outputs = spool, exchange, dict(outputs)
        self.opened: dict[str, str] = {}

    def _call[A: Opened | Adopted](self, request: WriterRequest, into: type[A]) -> A:
        answer = self.exchange(request, into)
        if not answer.ok:
            raise CapabilityError(answer.detail, code=answer.code)
        return answer

    def source(self, manifest: str) -> Source:
        answer = self._call(WriterSource(manifest=manifest), Opened)
        return Source(manifest, answer.length, answer.descriptor)

    def open_output(self, output_slot: str, derivation: Derivation) -> Output:
        maximum = self.outputs.get(output_slot)
        if maximum is None:
            raise CapabilityError("output is not declared", code="weights_output_ungranted")
        data = json.dumps(derivation.document(maximum), separators=(",", ":")).encode()
        length = _write(exchange_file(self.spool, output_slot, "derivation"), data)
        answer = self._call(WriterOutput(output_slot=output_slot, length=length), Opened)
        self.opened[answer.transaction] = output_slot
        return Output(answer.descriptor, answer.transaction)

    def adopt_model(self, transaction: str, manifest: tuple[str, int]) -> ModelArtifact:
        """Keep a committed output: TensorD checks the relayed manifest against its own."""
        slot = self.opened.get(transaction)
        if slot is None:
            raise CapabilityError("receipt has no opened output", code="weights_receipt_mismatch")
        receipt = {"transaction_id": transaction, "manifest": ref(*manifest)}
        length = _write(
            exchange_file(self.spool, slot, "native-receipt"),
            canonical_json.encode(receipt),
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


def _write(path: Path, data: bytes) -> int:
    if len(data) > EXCHANGE_MAX:
        raise CapabilityError("writer stream exceeds its bound", code="weights_writer_length")
    # An old name is never followed or opened for overwrite, including a planted symlink.
    path.unlink(missing_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "wb") as target:
        target.write(data)
    return len(data)
