"""Runtime model-output declarations, retained receipt references and reader metadata.

TensorFS supplies derivation declarations and source/output handles through Context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._errors import CapabilityError, ConformanceError

MAX_WEIGHTS_OUTPUTS = 16
MAX_WEIGHTS_SOURCE_READ = 64 << 20
MAX_WEIGHTS_NEW_BYTES = (1 << 53) - 1
_OUTPUT_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class WeightsOutput:
    """One package-interface model output and its new-byte ceiling."""

    name: str
    max_new_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _OUTPUT_NAME.fullmatch(self.name) is None:
            raise ConformanceError(
                f"weights output {self.name!r} is not a portable name",
                code="weights_output_name",
                fields=[self.name],
            )
        if (
            not isinstance(self.max_new_bytes, int)
            or isinstance(self.max_new_bytes, bool)
            or not 0 <= self.max_new_bytes <= MAX_WEIGHTS_NEW_BYTES
        ):
            raise ConformanceError(
                f"weights output {self.name} max_new_bytes must be in 0..={MAX_WEIGHTS_NEW_BYTES}",
                code="weights_output_bound",
                fields=[self.name],
            )


@dataclass(frozen=True, slots=True)
class WeightsReceipt:
    output_slot: str
    weights_transaction_id: str
    tensorfs_receipt_digest: str
    tensorfs_receipt: bytes
    replayed: bool = False
    _artifact: ModelArtifact | None = None

    @property
    def artifact(self) -> ModelArtifact:
        """The verified retained model reference produced by this exact receipt."""
        if self._artifact is None:
            raise CapabilityError(
                "weights receipt has no verified model artifact", code="weights_receipt_artifact"
            )
        return self._artifact
