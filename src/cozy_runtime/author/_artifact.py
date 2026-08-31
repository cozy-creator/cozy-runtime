"""Runtime-owned model artifact transaction on the author surface.

The job supplies a closed structural edit and byte streams.  It never sees a TensorFS
store, transaction id, destination repository, upload grant, or publication credential.
Runtime hands one immutable declaration plus bounded readers to the host transaction
implementation. TensorFS locks geometry/order/sources/output bounds before it pulls a byte;
no complete output is copied through the attempt spool.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from cozy_runtime.author._errors import CapabilityError, ConformanceError
from cozy_runtime.author._model import Model

MAX_ARTIFACT_OUTPUTS = 16
MAX_ARTIFACT_SOURCES = 16
MAX_ARTIFACT_COMPONENTS = 64
MAX_ARTIFACT_ADDITIONS = 65_536
_OUTPUT_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


@runtime_checkable
class Readable(Protocol):
    def read(self, size: int = -1) -> bytes: ...


ArtifactBytes = bytes | bytearray | memoryview | Readable


@dataclass(frozen=True, slots=True)
class ArtifactOutput:
    """One descriptor-declared model output and its new-byte ceiling."""

    name: str
    max_new_bytes: int

    def __post_init__(self) -> None:
        if _OUTPUT_NAME.fullmatch(self.name) is None:
            raise ConformanceError(
                f"artifact output {self.name!r} is not a portable name",
                code="artifact_output_name",
                fields=[self.name],
            )
        if self.max_new_bytes < 0:
            raise ConformanceError(
                f"artifact output {self.name} max_new_bytes must be non-negative",
                code="artifact_output_bound",
                fields=[self.name],
            )


@dataclass(frozen=True, slots=True)
class ArtifactPart:
    dtype: str
    shape: tuple[int, ...]
    data: ArtifactBytes

    def __post_init__(self) -> None:
        if not self.dtype or any(dimension <= 0 for dimension in self.shape):
            raise CapabilityError(
                "an artifact part needs a dtype and positive dimensions",
                code="artifact_part_declaration",
            )
        if not isinstance(self.data, (bytes, bytearray, memoryview, Readable)):
            raise CapabilityError(
                "an artifact part is bytes or a readable stream",
                code="artifact_part_source",
            )


@dataclass(frozen=True, slots=True)
class ArtifactTensor:
    logical_dtype: str
    shape: tuple[int, ...]
    encoding: str
    parts: Mapping[str, ArtifactPart]

    def __post_init__(self) -> None:
        if (
            not self.logical_dtype
            or not self.encoding
            or not self.parts
            or any(dimension <= 0 for dimension in self.shape)
        ):
            raise CapabilityError(
                "an artifact tensor needs logical dtype, shape, encoding and parts",
                code="artifact_tensor_declaration",
            )


@dataclass(frozen=True, slots=True)
class ArtifactTarget:
    """Copy/derive/create one target component through TensorFS tfs-019."""

    source: str = ""
    source_component: str = ""
    drop: tuple[str, ...] = ()
    add: Mapping[str, ArtifactTensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if bool(self.source) != bool(self.source_component):
            raise CapabilityError(
                "artifact target source and source_component are present together",
                code="artifact_target_source",
            )
        if not self.source and not self.add:
            raise CapabilityError(
                "an artifact target either derives from a component or creates values",
                code="artifact_target_empty",
            )
        if len(set(self.drop)) != len(self.drop) or set(self.drop) & set(self.add):
            raise CapabilityError(
                "artifact target drop/add keys must be unique and disjoint",
                code="artifact_target_keys",
            )


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    """Copy, add, or replace one config in the derived Manifest."""

    source: str = ""
    source_config: str = ""
    media_type: str = ""
    data: ArtifactBytes | None = None
    length: int = 0

    def __post_init__(self) -> None:
        copied = bool(self.source or self.source_config)
        added = self.data is not None or bool(self.media_type)
        if bool(self.source) != bool(self.source_config) or not (copied or added):
            raise CapabilityError(
                "an artifact config is an exact copy, addition, or copy-plus-replacement",
                code="artifact_config_declaration",
            )
        if added and (self.data is None or not self.media_type):
            raise CapabilityError(
                "an added artifact config needs media_type and data",
                code="artifact_config_declaration",
            )
        if self.data is not None:
            known = len(self.data) if isinstance(self.data, (bytes, bytearray, memoryview)) else 0
            if (
                self.length < 0
                or (known and self.length not in (0, known))
                or (not known and not self.length)
            ):
                raise CapabilityError(
                    "an artifact config stream needs its exact non-negative length",
                    code="artifact_config_length",
                )


@dataclass(frozen=True, slots=True)
class ArtifactCommit:
    output_slot: str
    sources: Mapping[str, str]
    targets: Mapping[str, ArtifactTarget]
    configs: Mapping[str, ArtifactConfig]
    order: tuple[tuple[str, str], ...]
    max_new_bytes: int


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    output_slot: str
    artifact_transaction_id: str
    tensorfs_receipt_digest: str
    tensorfs_receipt: bytes
    replayed: bool = False


class ArtifactSink:
    """Attempt-bound capability for one or more descriptor-declared model outputs."""

    __slots__ = ("_attempt", "_models", "_outputs", "_sink", "_spent")

    def __init__(
        self,
        attempt: object,
        models: Mapping[str, Model[object]],
        outputs: Mapping[str, int],
        sink: Callable[[ArtifactCommit], ArtifactReceipt] | None,
    ) -> None:
        self._attempt = attempt
        self._models = dict(models)
        self._outputs = dict(outputs)
        self._sink = sink
        self._spent: set[str] = set()

    def derive(
        self,
        output: str,
        *,
        sources: Mapping[str, Model[object]],
        targets: Mapping[str, ArtifactTarget],
        configs: Mapping[str, ArtifactConfig] | None = None,
        order: Sequence[tuple[str, str]],
    ) -> ArtifactReceipt:
        """Commit one exact derived Manifest and return only its durable receipt."""
        from cozy_runtime.author._services import Attempt

        attempt = self._attempt
        assert isinstance(attempt, Attempt)
        attempt.check_open("ArtifactSink.derive")
        if self._sink is None:
            raise CapabilityError(
                "this attempt has no artifact transaction host",
                code="artifact_sink_unavailable",
            )
        if output not in self._outputs:
            raise CapabilityError(
                f"artifact output {output!r} is not descriptor-declared",
                code="artifact_output_ungranted",
            )
        if output in self._spent:
            raise CapabilityError(
                f"artifact output {output!r} was already committed by this attempt",
                code="artifact_output_repeated",
            )
        if not sources or len(sources) > MAX_ARTIFACT_SOURCES:
            raise CapabilityError(
                f"artifact source count is outside 1..={MAX_ARTIFACT_SOURCES}",
                code="artifact_source_count",
            )
        selected: dict[str, str] = {}
        allowed = {id(model): model for model in self._models.values()}
        for alias, model in sources.items():
            if id(model) not in allowed:
                raise CapabilityError(
                    f"artifact source {alias!r} is not one of this job's exact model inputs",
                    code="artifact_source_ungranted",
                )
            if not model.checkpoint_ref.startswith("sha256:"):
                raise CapabilityError(
                    f"artifact source {alias!r} has no exact Manifest identity",
                    code="artifact_source_identity",
                )
            selected[alias] = model.checkpoint_ref
        if not targets or len(targets) > MAX_ARTIFACT_COMPONENTS:
            raise CapabilityError(
                f"artifact target count is outside 1..={MAX_ARTIFACT_COMPONENTS}",
                code="artifact_target_count",
            )
        additions = sum(len(target.add) for target in targets.values())
        if additions > MAX_ARTIFACT_ADDITIONS:
            raise CapabilityError(
                f"artifact additions exceed {MAX_ARTIFACT_ADDITIONS}",
                code="artifact_addition_count",
            )

        maximum = self._outputs[output]
        receipt = self._sink(
            ArtifactCommit(
                output_slot=output,
                sources=selected,
                targets=dict(targets),
                configs=dict(configs or {}),
                order=tuple(order),
                max_new_bytes=maximum,
            )
        )
        if receipt.output_slot != output:
            raise CapabilityError(
                "the artifact transaction host returned another output slot",
                code="artifact_receipt_mismatch",
            )
        self._spent.add(output)
        return receipt
