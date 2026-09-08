"""Runtime-owned model weights transaction on the author surface.

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

from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._errors import Cancelled, CapabilityError, ConformanceError
from cozy_runtime.author._model import Model

MAX_WEIGHTS_OUTPUTS = 16
MAX_WEIGHTS_SOURCES = 16
MAX_WEIGHTS_COMPONENTS = 64
MAX_WEIGHTS_ADDITIONS = 65_536
MAX_WEIGHTS_SOURCE_READ = 64 << 20
MAX_WEIGHTS_NEW_BYTES = (1 << 53) - 1
MAX_WEIGHTS_SOURCE_CONFIGS = 64
MAX_WEIGHTS_SOURCE_TENSORS = 65_536
MAX_WEIGHTS_SOURCE_ROLES = 16
_OUTPUT_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


@runtime_checkable
class Readable(Protocol):
    def read(self, size: int = -1) -> bytes: ...


WeightsBytes = bytes | bytearray | memoryview | Readable


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
class WeightsPartSource:
    """Inherit one exact encoded role from a declared source without moving bytes."""

    source: str
    component: str
    tensor: str
    role: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.source, self.component, self.tensor, self.role)
        ):
            raise CapabilityError(
                "a weights part source names source, component, tensor and role",
                code="weights_part_source",
            )


@dataclass(frozen=True, slots=True)
class WeightsPart:
    dtype: str
    shape: tuple[int, ...]
    data: WeightsBytes | None = None
    source: WeightsPartSource | None = None

    def __post_init__(self) -> None:
        if self.source is not None and (
            not isinstance(self.source, WeightsPartSource) or self.data is not None
        ):
            raise CapabilityError(
                "a weights part inherits a declared source or accepts bytes, never both",
                code="weights_part_source",
            )
        if not self.dtype or any(dimension <= 0 for dimension in self.shape):
            raise CapabilityError(
                "a weights part needs a dtype and positive dimensions",
                code="weights_part_declaration",
            )
        if self.data is not None and not isinstance(
            self.data, (bytes, bytearray, memoryview, Readable)
        ):
            raise CapabilityError(
                "a weights part is bytes or a readable stream",
                code="weights_part_source",
            )


@dataclass(frozen=True, slots=True)
class WeightsTensor:
    logical_dtype: str
    shape: tuple[int, ...]
    encoding: str
    parts: Mapping[str, WeightsPart]

    def __post_init__(self) -> None:
        if (
            not self.logical_dtype
            or not self.encoding
            or not self.parts
            or any(dimension <= 0 for dimension in self.shape)
        ):
            raise CapabilityError(
                "a weights tensor needs logical dtype, shape, encoding and parts",
                code="weights_tensor_declaration",
            )


@dataclass(frozen=True, slots=True)
class WeightsTarget:
    """Copy/derive/create one target component through TensorFS tfs-019."""

    source: str = ""
    source_component: str = ""
    drop: tuple[str, ...] = ()
    add: Mapping[str, WeightsTensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if bool(self.source) != bool(self.source_component):
            raise CapabilityError(
                "weights target source and source_component are present together",
                code="weights_target_source",
            )
        if not self.source and not self.add:
            raise CapabilityError(
                "a weights target either derives from a component or creates values",
                code="weights_target_empty",
            )
        if len(set(self.drop)) != len(self.drop):
            raise CapabilityError(
                "weights target drop keys must be unique; replacing a tensor is spelled "
                "by dropping and adding the same key",
                code="weights_target_keys",
            )


@dataclass(frozen=True, slots=True)
class WeightsConfig:
    """Copy, add, or derive one inline JCS config in the resulting CozyTensors header.

    ``length`` bounds a readable stream; it is operational and never enters the immutable
    TensorFS declaration. Byte-backed configs derive their length directly.
    """

    source: str = ""
    source_config: str = ""
    data: WeightsBytes | None = None
    length: int = 0

    def __post_init__(self) -> None:
        copied = bool(self.source or self.source_config)
        written = self.data is not None
        if bool(self.source) != bool(self.source_config) or not (copied or written):
            raise CapabilityError(
                "a weights config is an exact copy, addition, or source-derived replacement",
                code="weights_config_declaration",
            )
        if self.data is not None:
            known = len(self.data) if isinstance(self.data, (bytes, bytearray, memoryview)) else 0
            if (
                not isinstance(self.length, int)
                or isinstance(self.length, bool)
                or self.length < 0
                or (known and self.length not in (0, known))
                or (not known and not self.length)
            ):
                raise CapabilityError(
                    "a weights config stream needs its exact positive length",
                    code="weights_config_length",
                )
        elif self.length != 0:
            raise CapabilityError(
                "a copied weights config has no stream length",
                code="weights_config_length",
            )


@dataclass(frozen=True, slots=True)
class WeightsCommit:
    output_slot: str
    sources: Mapping[str, str]
    targets: Mapping[str, WeightsTarget]
    configs: Mapping[str, WeightsConfig]
    order: tuple[tuple[str, str], ...]
    max_new_bytes: int


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


@dataclass(frozen=True, slots=True)
class WeightsSourcePart:
    """One bounded role fact from an already-granted source Manifest."""

    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WeightsSourceTensor:
    """Logical geometry and stored roles, with no object or location information."""

    component: str
    key: str
    logical_dtype: str
    shape: tuple[int, ...]
    parts: tuple[WeightsSourcePart, ...]
    encoding: str = ""


@dataclass(frozen=True, slots=True)
class WeightsSource:
    """The closed structural view of one exact granted Manifest.

    Tuple order is the Manifest's construction order.  The view deliberately omits
    object references, offsets, store paths, and configuration bytes; those stay behind
    the Runtime-owned weights transaction.
    """

    configs: tuple[str, ...]
    tensors: tuple[WeightsSourceTensor, ...]


class WeightsHostTransaction(Protocol):
    """Runtime-private host operations behind one author-facing transaction."""

    replayed: bool
    receipt: WeightsReceipt | None

    @property
    def completed_parts(self) -> frozenset[tuple[str, str, str]]: ...

    @property
    def completed_configs(self) -> frozenset[str]: ...

    def checkpoint(self) -> None: ...

    def source_read_into(
        self,
        source: str,
        component: str,
        key: str,
        role: str,
        offset: int,
        into: memoryview,
    ) -> None: ...

    def add_part(self, component: str, key: str, role: str, data: WeightsBytes) -> None: ...

    def add_config(self, name: str, data: WeightsBytes, length: int) -> None: ...

    def commit(self) -> WeightsReceipt: ...

    def fence(self) -> None: ...


class WeightsTransaction:
    """One opened, declaration-locked WeightsSink output transaction."""

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
        targets: Mapping[str, WeightsTarget],
        configs: Mapping[str, WeightsConfig],
        host: WeightsHostTransaction,
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
    def receipt(self) -> WeightsReceipt | None:
        """The prior receipt on replay; absent for a newly opened writer."""
        return self._host.receipt

    @property
    def completed_parts(self) -> frozenset[tuple[str, str, str]]:
        """Verified added roles available for reuse, as ``(component, key, role)``."""
        self._check("WeightsTransaction.completed_parts")
        return self._host.completed_parts

    @property
    def completed_configs(self) -> frozenset[str]:
        """Verified added configurations available for reuse."""
        self._check("WeightsTransaction.completed_configs")
        return self._host.completed_configs

    def checkpoint(self) -> None:
        """Record a completed semantic unit for the owner's recovery pipeline."""
        self._check("WeightsTransaction.checkpoint")
        self._call(self._host.checkpoint, "weights checkpoint")

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
        self._check("WeightsTransaction.source_read_into")
        if source not in self._sources:
            self._fail(
                f"weights source alias {source!r} is outside this transaction",
                code="weights_source_ungranted",
            )
        if offset < 0:
            self._fail("weights source offset is negative", code="weights_source_range")
        try:
            view = memoryview(into)
        except TypeError as exc:
            self._fail(
                f"weights source destination is not a writable buffer: {exc}",
                code="weights_source_buffer",
            )
        if view.readonly or not view.c_contiguous or not 0 < view.nbytes <= MAX_WEIGHTS_SOURCE_READ:
            self._fail(
                f"weights source destination is readonly, non-contiguous, empty, or over "
                f"{MAX_WEIGHTS_SOURCE_READ} bytes",
                code="weights_source_buffer",
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
                code="weights_source_refused",
            ) from exc

    def add_part(self, component: str, key: str, role: str, data: WeightsBytes) -> None:
        self._check("WeightsTransaction.add_part")
        target = self._targets.get(component)
        tensor = target.add.get(key) if target is not None else None
        if tensor is None or role not in tensor.parts:
            self._fail(
                f"weights part {component}/{key}#{role} is undeclared",
                code="weights_part_ungranted",
            )
        if tensor.parts[role].source is not None:
            self._fail(
                f"weights part {component}/{key}#{role} inherits a source and cannot accept bytes",
                code="weights_part_source",
            )
        self._call(lambda: self._host.add_part(component, key, role, data), "weights part")

    def add_config(self, name: str, data: WeightsBytes) -> None:
        self._check("WeightsTransaction.add_config")
        config = self._configs.get(name)
        if config is None or config.data is None:
            self._fail(
                f"weights config {name!r} is undeclared",
                code="weights_config_ungranted",
            )
        length = (
            len(config.data)
            if isinstance(config.data, (bytes, bytearray, memoryview))
            else config.length
        )
        self._call(lambda: self._host.add_config(name, data, length), "weights config")

    def commit(self) -> WeightsReceipt:
        self._check("WeightsTransaction.commit")
        try:
            receipt = self._host.commit()
        except Exception as exc:
            self._host.fence()
            self._closed = True
            if isinstance(exc, CapabilityError):
                raise
            raise CapabilityError(
                f"TensorFS weights commit refused: {type(exc).__name__}: {exc}",
                code="weights_tensorfs_refused",
            ) from exc
        self._closed = True
        return receipt

    def fence(self) -> None:
        if not self._closed:
            self._host.fence()
            self._closed = True

    def __enter__(self) -> WeightsTransaction:
        self._check("WeightsTransaction.__enter__")
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
                f"{what} used after weights output {self._output!r} closed",
                code="weights_transaction_closed",
            )
        if self._cancel():
            self._host.fence()
            self._closed = True
            raise Cancelled(f"weights output {self._output!r} was canceled")

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
                code="weights_tensorfs_refused",
            ) from exc


class WeightsSink:
    """Attempt-bound capability for one or more interface-declared model outputs."""

    __slots__ = (
        "_attempt",
        "_cancel",
        "_models",
        "_open",
        "_outputs",
        "_source_config",
        "_source_structure",
        "_spent",
    )

    def __init__(
        self,
        attempt: object,
        models: Mapping[str, Model[object]],
        outputs: Mapping[str, int],
        open_transaction: Callable[[WeightsCommit], WeightsHostTransaction] | None,
        source_structure: Callable[[str], WeightsSource] | None = None,
        cancel: Callable[[], bool] = lambda: False,
        source_config: Callable[[str, str], bytes] | None = None,
    ) -> None:
        self._attempt = attempt
        self._source_config = source_config
        self._models = dict(models)
        self._outputs = dict(outputs)
        self._open = open_transaction
        self._source_structure = source_structure
        self._cancel = cancel
        self._spent: set[str] = set()

    @property
    def models(self) -> Mapping[str, Model[object]]:
        """This job's granted model inputs by parameter name — the exact objects
        `structure` and `open` accept as sources (read-only copy)."""
        return dict(self._models)

    def config(self, source: Model[object], name: str) -> bytes:
        """Read one bounded inline config from an exact model input granted to this job."""
        from cozy_runtime.author._services import Attempt

        attempt = self._attempt
        assert isinstance(attempt, Attempt)
        attempt.check_open("WeightsSink.config")
        if not any(source is model for model in self._models.values()):
            raise CapabilityError(
                "weights config source is not a granted model input",
                code="weights_source_ungranted",
            )
        if not source.checkpoint_ref.startswith("sha256:") or not isinstance(name, str) or not name:
            raise CapabilityError(
                "weights config requires an exact manifest and config name",
                code="weights_source_identity",
            )
        if self._cancel():
            raise Cancelled("weights config read was canceled")
        if self._source_config is None:
            raise CapabilityError(
                "this attempt has no weights config reader",
                code="weights_source_config_unavailable",
            )
        data = self._source_config(source.checkpoint_ref, name)
        if not isinstance(data, bytes) or not 0 < len(data) <= MAX_WEIGHTS_SOURCE_READ:
            raise CapabilityError(
                "weights config exceeds its byte bound", code="weights_source_config_invalid"
            )
        return data

    def structure(self, source: Model[object]) -> WeightsSource:
        """Return bounded tensor geometry for one exact source already granted to this job."""
        from cozy_runtime.author._services import Attempt

        attempt = self._attempt
        assert isinstance(attempt, Attempt)
        attempt.check_open("WeightsSink.structure")
        allowed = {id(model): model for model in self._models.values()}
        if id(source) not in allowed:
            raise CapabilityError(
                "weights structure source is not one of this job's exact model inputs",
                code="weights_source_ungranted",
            )
        if not source.checkpoint_ref.startswith("sha256:"):
            raise CapabilityError(
                "weights structure source has no exact Manifest identity",
                code="weights_source_identity",
            )
        if self._source_structure is None:
            raise CapabilityError(
                "this attempt has no weights source-structure host",
                code="weights_source_structure_unavailable",
            )
        if self._cancel():
            raise Cancelled("weights source-structure read was canceled")
        try:
            observed = self._source_structure(source.checkpoint_ref)
        except Exception as exc:
            if isinstance(exc, CapabilityError):
                raise
            raise CapabilityError(
                f"TensorFS source structure refused: {type(exc).__name__}: {exc}",
                code="weights_source_refused",
            ) from exc
        if (
            not isinstance(observed, WeightsSource)
            or not 0 <= len(observed.configs) <= MAX_WEIGHTS_SOURCE_CONFIGS
            or not 0 < len(observed.tensors) <= MAX_WEIGHTS_SOURCE_TENSORS
            or len(set(observed.configs)) != len(observed.configs)
        ):
            raise CapabilityError(
                "weights source structure exceeds its closed bounds",
                code="weights_source_structure_invalid",
            )
        seen: set[tuple[str, str]] = set()
        for tensor in observed.tensors:
            identity = (tensor.component, tensor.key)
            if (
                not tensor.component
                or not tensor.key
                or not tensor.logical_dtype
                or not tensor.parts
                or len(tensor.parts) > MAX_WEIGHTS_SOURCE_ROLES
                or any(dimension <= 0 for dimension in tensor.shape)
                or identity in seen
                or len({part.name for part in tensor.parts}) != len(tensor.parts)
                or any(
                    not part.name
                    or not part.dtype
                    or any(dimension <= 0 for dimension in part.shape)
                    for part in tensor.parts
                )
            ):
                raise CapabilityError(
                    "weights source structure contains invalid tensor geometry",
                    code="weights_source_structure_invalid",
                )
            seen.add(identity)
        return observed

    def open(
        self,
        output: str,
        *,
        sources: Mapping[str, Model[object]],
        targets: Mapping[str, WeightsTarget],
        configs: Mapping[str, WeightsConfig] | None = None,
        order: Sequence[tuple[str, str]],
    ) -> WeightsTransaction:
        """Lock one exact declaration and return its bounded reader/writer capability."""
        from cozy_runtime.author._services import Attempt

        attempt = self._attempt
        assert isinstance(attempt, Attempt)
        attempt.check_open("WeightsSink.derive")
        if self._open is None:
            raise CapabilityError(
                "this attempt has no weights transaction host",
                code="weights_sink_unavailable",
            )
        if output not in self._outputs:
            raise CapabilityError(
                f"weights output {output!r} is not interface-declared",
                code="weights_output_ungranted",
            )
        if output in self._spent:
            raise CapabilityError(
                f"weights output {output!r} was already committed by this attempt",
                code="weights_output_repeated",
            )
        if len(sources) > MAX_WEIGHTS_SOURCES:
            raise CapabilityError(
                f"weights source count is outside 0..={MAX_WEIGHTS_SOURCES}",
                code="weights_source_count",
            )
        selected: dict[str, str] = {}
        allowed = {id(model): model for model in self._models.values()}
        for alias, model in sources.items():
            if id(model) not in allowed:
                raise CapabilityError(
                    f"weights source {alias!r} is not one of this job's exact model inputs",
                    code="weights_source_ungranted",
                )
            if not model.checkpoint_ref.startswith("sha256:"):
                raise CapabilityError(
                    f"weights source {alias!r} has no exact Manifest identity",
                    code="weights_source_identity",
                )
            selected[alias] = model.checkpoint_ref
        if not targets or len(targets) > MAX_WEIGHTS_COMPONENTS:
            raise CapabilityError(
                f"weights target count is outside 1..={MAX_WEIGHTS_COMPONENTS}",
                code="weights_target_count",
            )
        additions = sum(len(target.add) for target in targets.values())
        if additions > MAX_WEIGHTS_ADDITIONS:
            raise CapabilityError(
                f"weights additions exceed {MAX_WEIGHTS_ADDITIONS}",
                code="weights_addition_count",
            )

        maximum = self._outputs[output]
        transaction = WeightsTransaction(
            attempt=attempt,
            cancel=self._cancel,
            output=output,
            sources=selected,
            targets=targets,
            configs=dict(configs or {}),
            host=self._open(
                WeightsCommit(
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
        targets: Mapping[str, WeightsTarget],
        configs: Mapping[str, WeightsConfig] | None = None,
        order: Sequence[tuple[str, str]],
    ) -> WeightsReceipt:
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
                        if part.source is not None:
                            continue
                        if part.data is None:
                            transaction.fence()
                            raise CapabilityError(
                                f"weights part {component}/{key}#{role} has no data",
                                code="weights_part_source",
                            )
                        transaction.add_part(component, key, role, part.data)
            for name, config in sorted((configs or {}).items()):
                if config.data is not None:
                    transaction.add_config(name, config.data)
            receipt = transaction.commit()
            if receipt.output_slot != output:
                raise CapabilityError(
                    "the weights transaction host returned another output slot",
                    code="weights_receipt_mismatch",
                )
            return receipt
