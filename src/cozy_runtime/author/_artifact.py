"""Runtime-owned model artifact transaction on the author surface.

The job supplies a closed structural edit and byte streams.  It never sees a TensorFS
store, transaction id, destination repository, upload grant, or publication credential.
Runtime hands one immutable declaration plus bounded readers to the host transaction
implementation. TensorFS locks geometry/order/sources/output bounds before it pulls a byte;
no complete output is copied through the attempt spool.
"""

from __future__ import annotations

import re
from collections.abc import Buffer, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, NoReturn, Protocol, runtime_checkable

from cozy_runtime.author._errors import Cancelled, CapabilityError, ConformanceError
from cozy_runtime.author._model import Model

MAX_ARTIFACT_OUTPUTS = 16
MAX_ARTIFACT_SOURCES = 16
MAX_ARTIFACT_COMPONENTS = 64
MAX_ARTIFACT_ADDITIONS = 65_536
MAX_ARTIFACT_SOURCE_READ = 64 << 20
MAX_ARTIFACT_NEW_BYTES = (1 << 53) - 1
MAX_ARTIFACT_ENCODINGS = 32
_OUTPUT_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


@runtime_checkable
class Readable(Protocol):
    def read(self, size: int = -1) -> bytes: ...


ArtifactBytes = bytes | bytearray | memoryview | Readable


@dataclass(frozen=True, slots=True)
class ArtifactOutput:
    """One descriptor-declared model output, byte ceiling, and optional exact contract."""

    name: str
    max_new_bytes: int
    topology_digest: str = ""
    encodings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _OUTPUT_NAME.fullmatch(self.name) is None:
            raise ConformanceError(
                f"artifact output {self.name!r} is not a portable name",
                code="artifact_output_name",
                fields=[self.name],
            )
        if (
            not isinstance(self.max_new_bytes, int)
            or isinstance(self.max_new_bytes, bool)
            or not 0 <= self.max_new_bytes <= MAX_ARTIFACT_NEW_BYTES
        ):
            raise ConformanceError(
                f"artifact output {self.name} max_new_bytes must be in "
                f"0..={MAX_ARTIFACT_NEW_BYTES}",
                code="artifact_output_bound",
                fields=[self.name],
            )
        contracted = bool(self.topology_digest or self.encodings)
        if contracted and (
            not isinstance(self.topology_digest, str)
            or _DIGEST.fullmatch(self.topology_digest) is None
            or not isinstance(self.encodings, tuple)
            or not 1 <= len(self.encodings) <= MAX_ARTIFACT_ENCODINGS
            or tuple(sorted(set(self.encodings))) != self.encodings
            or any(
                not isinstance(value, str) or _DIGEST.fullmatch(value) is None
                for value in self.encodings
            )
        ):
            raise ConformanceError(
                f"artifact output {self.name} contract requires one exact topology digest "
                f"and 1..={MAX_ARTIFACT_ENCODINGS} sorted unique EncodingSpec digests",
                code="artifact_output_contract",
                fields=[self.name],
            )


@dataclass(frozen=True, slots=True)
class ArtifactPart:
    dtype: str
    shape: tuple[int, ...]
    data: ArtifactBytes | None = None

    def __post_init__(self) -> None:
        if not self.dtype or any(dimension <= 0 for dimension in self.shape):
            raise CapabilityError(
                "an artifact part needs a dtype and positive dimensions",
                code="artifact_part_declaration",
            )
        if self.data is not None and not isinstance(
            self.data, (bytes, bytearray, memoryview, Readable)
        ):
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
        if len(set(self.drop)) != len(self.drop):
            raise CapabilityError(
                "artifact target drop keys must be unique; replacing a tensor is spelled "
                "by dropping and adding the same key",
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
    checkpoint_evidence_canonical_bytes: bytes
    replayed: bool = False


class ArtifactHostTransaction(Protocol):
    """Runtime-private host operations behind one author-facing transaction."""

    replayed: bool
    receipt: ArtifactReceipt | None

    def source_read_into(
        self,
        source: str,
        component: str,
        key: str,
        role: str,
        offset: int,
        into: memoryview,
    ) -> None: ...

    def add_part(self, component: str, key: str, role: str, data: ArtifactBytes) -> None: ...

    def add_config(self, name: str, data: ArtifactBytes) -> None: ...

    def commit(self) -> ArtifactReceipt: ...

    def fence(self) -> None: ...


class ArtifactTransaction:
    """One opened, declaration-locked ArtifactSink output transaction."""

    __slots__ = (
        "_attempt",
        "_cancel",
        "_closed",
        "_configs",
        "_host",
        "_output",
        "_sources",
        "_targets",
    )

    def __init__(
        self,
        *,
        attempt: object,
        cancel: Callable[[], bool],
        output: str,
        sources: Mapping[str, str],
        targets: Mapping[str, ArtifactTarget],
        configs: Mapping[str, ArtifactConfig],
        host: ArtifactHostTransaction,
    ) -> None:
        self._attempt = attempt
        self._cancel = cancel
        self._output = output
        self._sources = frozenset(sources)
        self._targets = dict(targets)
        self._configs = dict(configs)
        self._host = host
        self._closed = False

    @property
    def replayed(self) -> bool:
        return self._host.replayed

    @property
    def receipt(self) -> ArtifactReceipt | None:
        """The prior receipt on replay; absent for a newly opened writer."""
        return self._host.receipt

    def source_read_into(
        self,
        source: str,
        component: str,
        key: str,
        role: str,
        offset: int,
        into: Buffer,
    ) -> None:
        """Read one exact source-part range into caller-owned bounded writable memory."""
        self._check("ArtifactTransaction.source_read_into")
        if source not in self._sources:
            self._fail(
                f"artifact source alias {source!r} is outside this transaction",
                code="artifact_source_ungranted",
            )
        if offset < 0:
            self._fail("artifact source offset is negative", code="artifact_source_range")
        try:
            view = memoryview(into)
        except TypeError as exc:
            self._fail(
                f"artifact source destination is not a writable buffer: {exc}",
                code="artifact_source_buffer",
            )
        if (
            view.readonly
            or not view.c_contiguous
            or not 0 < view.nbytes <= MAX_ARTIFACT_SOURCE_READ
        ):
            self._fail(
                f"artifact source destination is readonly, non-contiguous, empty, or over "
                f"{MAX_ARTIFACT_SOURCE_READ} bytes",
                code="artifact_source_buffer",
            )
        try:
            self._host.source_read_into(source, component, key, role, offset, view.cast("B"))
        except Exception as exc:
            self._host.fence()
            self._closed = True
            if isinstance(exc, CapabilityError):
                raise
            raise CapabilityError(
                f"TensorFS source read refused: {type(exc).__name__}: {exc}",
                code="artifact_source_refused",
            ) from exc

    def add_part(self, component: str, key: str, role: str, data: ArtifactBytes) -> None:
        self._check("ArtifactTransaction.add_part")
        target = self._targets.get(component)
        tensor = target.add.get(key) if target is not None else None
        if tensor is None or role not in tensor.parts:
            self._fail(
                f"artifact part {component}/{key}#{role} is undeclared",
                code="artifact_part_ungranted",
            )
        self._call(lambda: self._host.add_part(component, key, role, data), "artifact part")

    def add_config(self, name: str, data: ArtifactBytes) -> None:
        self._check("ArtifactTransaction.add_config")
        config = self._configs.get(name)
        if config is None or config.data is None:
            self._fail(
                f"artifact config {name!r} is undeclared",
                code="artifact_config_ungranted",
            )
        self._call(lambda: self._host.add_config(name, data), "artifact config")

    def commit(self) -> ArtifactReceipt:
        self._check("ArtifactTransaction.commit")
        try:
            receipt = self._host.commit()
        except Exception as exc:
            self._host.fence()
            self._closed = True
            if isinstance(exc, CapabilityError):
                raise
            raise CapabilityError(
                f"TensorFS artifact commit refused: {type(exc).__name__}: {exc}",
                code="artifact_tensorfs_refused",
            ) from exc
        self._closed = True
        return receipt

    def fence(self) -> None:
        if not self._closed:
            self._host.fence()
            self._closed = True

    def __enter__(self) -> ArtifactTransaction:
        self._check("ArtifactTransaction.__enter__")
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.fence()
        return False

    def _check(self, what: str) -> None:
        from cozy_runtime.author._services import Attempt

        attempt = self._attempt
        assert isinstance(attempt, Attempt)
        attempt.check_open(what)
        if self._closed:
            raise CapabilityError(
                f"{what} used after artifact output {self._output!r} closed",
                code="artifact_transaction_closed",
            )
        if self._cancel():
            self._host.fence()
            self._closed = True
            raise Cancelled(f"artifact output {self._output!r} was canceled")

    def _fail(self, detail: str, *, code: str) -> NoReturn:
        self._host.fence()
        self._closed = True
        raise CapabilityError(detail, code=code)

    def _call(self, operation: Callable[[], None], what: str) -> None:
        try:
            operation()
        except Exception as exc:
            self._host.fence()
            self._closed = True
            if isinstance(exc, CapabilityError):
                raise
            raise CapabilityError(
                f"TensorFS {what} refused: {type(exc).__name__}: {exc}",
                code="artifact_tensorfs_refused",
            ) from exc


class ArtifactSink:
    """Attempt-bound capability for one or more descriptor-declared model outputs."""

    __slots__ = ("_attempt", "_cancel", "_models", "_open", "_outputs", "_spent")

    def __init__(
        self,
        attempt: object,
        models: Mapping[str, Model[object]],
        outputs: Mapping[str, int],
        open_transaction: Callable[[ArtifactCommit], ArtifactHostTransaction] | None,
        cancel: Callable[[], bool] = lambda: False,
    ) -> None:
        self._attempt = attempt
        self._models = dict(models)
        self._outputs = dict(outputs)
        self._open = open_transaction
        self._cancel = cancel
        self._spent: set[str] = set()

    def open(
        self,
        output: str,
        *,
        sources: Mapping[str, Model[object]],
        targets: Mapping[str, ArtifactTarget],
        configs: Mapping[str, ArtifactConfig] | None = None,
        order: Sequence[tuple[str, str]],
    ) -> ArtifactTransaction:
        """Lock one exact declaration and return its bounded reader/writer capability."""
        from cozy_runtime.author._services import Attempt

        attempt = self._attempt
        assert isinstance(attempt, Attempt)
        attempt.check_open("ArtifactSink.derive")
        if self._open is None:
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
        transaction = ArtifactTransaction(
            attempt=attempt,
            cancel=self._cancel,
            output=output,
            sources=selected,
            targets=targets,
            configs=dict(configs or {}),
            host=self._open(
                ArtifactCommit(
                    output_slot=output,
                    sources=selected,
                    targets=dict(targets),
                    configs=dict(configs or {}),
                    order=tuple(order),
                    max_new_bytes=maximum,
                )
            ),
        )
        self._spent.add(output)
        return transaction

    def derive(
        self,
        output: str,
        *,
        sources: Mapping[str, Model[object]],
        targets: Mapping[str, ArtifactTarget],
        configs: Mapping[str, ArtifactConfig] | None = None,
        order: Sequence[tuple[str, str]],
    ) -> ArtifactReceipt:
        """One-shot convenience for jobs whose produced readers need no source reads."""
        with self.open(
            output, sources=sources, targets=targets, configs=configs, order=order
        ) as transaction:
            if transaction.replayed:
                receipt = transaction.receipt
                assert receipt is not None
                return receipt
            for component, target in sorted(targets.items()):
                for key, tensor in sorted(target.add.items()):
                    for role, part in sorted(tensor.parts.items()):
                        if part.data is None:
                            transaction.fence()
                            raise CapabilityError(
                                f"artifact part {component}/{key}#{role} has no data",
                                code="artifact_part_source",
                            )
                        transaction.add_part(component, key, role, part.data)
            for name, config in sorted((configs or {}).items()):
                if config.data is not None:
                    transaction.add_config(name, config.data)
            receipt = transaction.commit()
            if receipt.output_slot != output:
                raise CapabilityError(
                    "the artifact transaction host returned another output slot",
                    code="artifact_receipt_mismatch",
                )
            return receipt
