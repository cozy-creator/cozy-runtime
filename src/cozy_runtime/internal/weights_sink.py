"""The Runtime/host adapter for TensorFS derived weights transactions.

The author service emits one declaration plus bounded byte readers. This module is the
privileged half: it validates the immutable invocation grant, opens the real tfs-019
transaction before pulling bytes, streams directly into TensorFS, and returns exact receipt
bytes suitable for Worker Protocol's ``WeightsReceipt`` wrapper.

Local and private workers supply the same workspace callbacks: bind exact intent before
TensorFS opens, then durably record the exact native receipt before commit returns to package
code. Standalone native tests can use the lower-level transaction without a worker transport.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact, ObjectRef
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._weights import (
    MAX_WEIGHTS_SOURCE_READ,
    Readable,
    WeightsBytes,
    WeightsCommit,
    WeightsConfig,
    WeightsHostTransaction,
    WeightsReceipt,
    WeightsSource,
    WeightsSourcePart,
    WeightsSourceTensor,
)
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from tensorfs.derived import (
        Derivation,
        DerivedTransaction,
        SourceCapability,
        SourceInspection,
        Target,
    )

    from cozy_runtime.internal.base_observation import Observation
    from cozy_runtime.internal.hostfacts import HostFacts

_OUTPUT_SLOT = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_EXCHANGE_KINDS = frozenset({"declaration", "receipt"})


def work_fingerprint(
    spec: Mapping[str, Any],
    environment_content_digest: str,
    base: Observation,
    device: HostFacts,
) -> str:
    """Parent-authored numerical work identity, independent of attempt/custody clocks.

    The original exact InvocationSpec still scopes the transaction. Its deadline and
    publication grant are authority, not inputs to arithmetic. CUDA architecture/driver
    deliberately participate: AdaLN tables computed on another architecture are not
    presumed bit-identical. Device UUID, memory, paths and boot facts never participate.
    """
    from cozy_runtime.internal.base_observation import protected_distribution

    job = spec.get("job")
    if not isinstance(job, Mapping) or not environment_content_digest:
        raise CapabilityError(
            "weights work has no selected job environment", code="weights_work_identity"
        )
    if device.backend == "cuda" and (device.gpu_sm <= 0 or not device.driver_version):
        raise CapabilityError(
            "CUDA work has no measured architecture/driver", code="weights_work_identity"
        )
    facts = {
        "job": {key: job[key] for key in ("build_id", "job_descriptor_id")},
        "environment_digest": spec.get("environment_digest", ""),
        "environment_content_digest": environment_content_digest,
        "payload_digest": spec["payload_digest"],
        "inputs": spec.get("inputs", []),
        "outputs": spec.get("outputs", []),
        "interpreter": base.interpreter_identity,
        "distributions": {
            name: version for name, version in base.distributions if protected_distribution(name)
        },
        "torch_build": base.accelerator_build,
        "cuda_runtime": base.cuda_version,
        "device": {
            "backend": device.backend or "cpu",
            "architecture": device.gpu_sm,
            "driver": device.driver_version,
        },
    }
    return (
        "sha256:"
        + hashlib.sha256(
            b"cozy.runtime.derived-work/1\0" + canonical_json.encode(facts)
        ).hexdigest()
    )


@dataclass(frozen=True, slots=True)
class WeightsHostBinding:
    """The supervisor's durable answer to one exact TensorFS declaration."""

    transaction_id: str
    writer_epoch: int
    weights_receipt_digest: bytes = b""
    checkpoint: tuple[str, int] | None = None


@dataclass(frozen=True, slots=True)
class WeightsCheckpoint:
    """Mechanical TensorFS progress observed under one already-bound weights intent."""

    output_slot: str
    weights_transaction_id: str
    writer_epoch: int
    declaration_digest: str
    head: str
    head_length: int
    plan_digest: str
    index: int
    bytes: int


class _ExactReader:
    """Expose exactly one declared stream length to TensorFS, then prove EOF."""

    __slots__ = ("ended", "remaining", "source")

    def __init__(self, source: Readable, length: int) -> None:
        self.source = source
        self.remaining = length
        self.ended = False

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if self.remaining:
            want = self.remaining if size < 0 else min(size, self.remaining)
            chunk = self.source.read(want)
            if not isinstance(chunk, bytes) or not chunk or len(chunk) > want:
                raise CapabilityError(
                    "weights config stream ended before its declared length",
                    code="weights_config_length",
                )
            self.remaining -= len(chunk)
            return chunk
        if not self.ended:
            extra = self.source.read(1)
            if not isinstance(extra, bytes) or extra:
                raise CapabilityError(
                    "weights config stream exceeded its declared length",
                    code="weights_config_length",
                )
            self.ended = True
        return b""


def exchange_file(spool: Path, output_slot: str, kind: str) -> Path:
    """One worker-chosen spool file; no child-authored path crosses the seam."""
    if _OUTPUT_SLOT.fullmatch(output_slot) is None or kind not in _EXCHANGE_KINDS:
        raise CapabilityError("weights exchange identity is invalid", code="weights_host_identity")
    return spool / f"weights-{output_slot}-{kind}.canonical"


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
                f"weights {kind} replay changed exact bytes",
                code="weights_receipt_mismatch" if kind == "receipt" else "weights_intent_conflict",
            ) from None
    return target


def source_structure(header: Mapping[str, Any]) -> WeightsSource:
    """Author-safe geometry from one native-validated header, shared by readers/writers."""
    configs = header.get("configs") or {}
    components = header.get("components")
    if not isinstance(configs, Mapping) or not isinstance(components, Mapping):
        raise ValueError("header configs/components are not mappings")
    tensors: list[WeightsSourceTensor] = []
    for component, table in components.items():
        if not isinstance(table, Mapping):
            raise ValueError(f"component {component!r} is not a tensor mapping")
        for key, entry in table.items():
            if not isinstance(entry, Mapping):
                raise ValueError(f"tensor {component}/{key} is not a mapping")
            logical = entry.get("logical")
            parts = entry.get("parts")
            if not isinstance(logical, Mapping) or not isinstance(parts, Mapping):
                raise ValueError(f"tensor {component}/{key} has no logical/parts maps")
            tensors.append(
                WeightsSourceTensor(
                    component=str(component),
                    key=str(key),
                    logical_dtype=str(logical["logical_dtype"]),
                    encoding=str(entry["encoding"]),
                    shape=tuple(int(value) for value in logical["shape"]),
                    parts=tuple(
                        WeightsSourcePart(
                            name=str(role),
                            dtype=str(part["dtype"]),
                            shape=tuple(int(value) for value in part["shape"]),
                        )
                        for role, part in parts.items()
                    ),
                )
            )
    return WeightsSource(
        configs=tuple(str(name) for name in configs),
        tensors=tuple(tensors),
    )


def source_config(store: Any, manifest: str, length: int, name: str) -> bytes:
    """Read one config through native retained-source validation under the Store owner."""
    try:
        held = store.manifest(manifest)
        structure = source_structure(tensorfs.parse_header(bytes(held["header"])))
        if name not in structure.configs:
            raise CapabilityError(
                "weights source has no declared config with this name",
                code="weights_config_ungranted",
            )
        inspected = store.inspect_derived_source(
            manifest,
            length,
            list(dict.fromkeys(tensor.component for tensor in structure.tensors)),
            [name],
        )
        configs = inspected["configs"]
        if len(configs) != 1 or configs[0]["name"] != name:
            raise ValueError("native config selection changed")
        data = configs[0]["data"]
        if not isinstance(data, bytes) or not 0 < len(data) <= MAX_WEIGHTS_SOURCE_READ:
            raise ValueError("native config exceeds the source read bound")
        return data
    except CapabilityError:
        raise
    except Exception as exc:
        raise CapabilityError(
            f"TensorFS source config refused: {exc}", code="weights_source_refused"
        ) from exc


@dataclass(slots=True)
class _OpenedOutput:
    output_slot: str
    transaction_id: str
    declaration: object
    declaration_digest: str
    writer: Any
    receipt: WeightsReceipt | None
    writer_epoch: int
    checkpoint: tuple[str, int] | None


class WeightsTransactionHost:
    """One immutable invocation's real TensorFS weights authority."""

    def __init__(
        self,
        *,
        store: Any,
        owner_scope: str,
        request_id: str,
        invocation_spec_digest: str,
        work_fingerprint: str,
        writer_session_id: int,
        allowed_sources: Mapping[str, int],
        output_bounds: Mapping[str, int],
        bind_intent: Callable[[str, str, bytes, bytes], WeightsHostBinding] | None = None,
        record_receipt: Callable[[WeightsReceipt], None] | None = None,
        record_checkpoint: Callable[[WeightsCheckpoint], None] | None = None,
    ) -> None:
        if (
            not owner_scope
            or not request_id
            or not invocation_spec_digest.startswith("sha256:")
            or re.fullmatch(r"sha256:[0-9a-f]{64}", work_fingerprint) is None
            or writer_session_id <= 0
        ):
            raise CapabilityError(
                "weights host identity is incomplete", code="weights_host_identity"
            )
        self.store = store
        self.owner_scope = owner_scope
        self.request_id = request_id
        self.invocation_spec_digest = invocation_spec_digest
        self.work_fingerprint = work_fingerprint
        self.writer_session_id = writer_session_id
        self.allowed_sources = dict(allowed_sources)
        self.output_bounds = dict(output_bounds)
        self.bind_intent = bind_intent
        self.record_receipt = record_receipt
        self.record_checkpoint = record_checkpoint
        self._opened_outputs: dict[str, _OpenedOutput] = {}

    def open(self, request: WeightsCommit) -> WeightsHostTransaction:
        """Compatibility for active author consumers, over the same native binding."""
        from tensorfs.derived import Config, Derivation, Source

        if self.output_bounds.get(request.output_slot) != request.max_new_bytes:
            raise CapabilityError(
                "weights output differs from its immutable grant", code="weights_output_ungranted"
            )
        definition = Derivation(
            sources={
                alias: Source(manifest, self.allowed_sources.get(manifest, 0))
                for alias, manifest in request.sources.items()
            },
            # Legacy declarations expose the same structural fields. TensorFS
            # marshals them once; the compatibility path adds no geometry codec.
            targets=cast("Mapping[str, Target]", request.targets),
            configs={
                name: config if isinstance(config, Config) else Config(**self._config(config))
                for name, config in cast(Mapping[str, Any], request.configs).items()
            },
            order=request.order,
        )
        opened = self._begin(request.output_slot, definition)
        return OpenedWeightsTransaction(
            host=self,
            request=request,
            transaction_id=opened.transaction_id,
            declaration=opened.declaration,
            declaration_digest=opened.declaration_digest,
            writer=opened.writer,
            receipt=opened.receipt,
            writer_epoch=opened.writer_epoch,
            checkpoint=opened.checkpoint,
        )

    def open_output(self, output_slot: str, definition: Derivation) -> DerivedTransaction:
        """Bind one declared output and return TensorFS's actual scoped handle."""
        from tensorfs.derived import DerivedTransaction

        opened = self._begin(output_slot, definition)
        self._opened_outputs[opened.transaction_id] = opened
        if opened.receipt is not None:
            return DerivedTransaction.replay(canonical_json.decode(opened.receipt.tensorfs_receipt))
        if not isinstance(opened.writer, DerivedTransaction):
            opened.writer.fence()
            raise CapabilityError(
                "execution owner did not provide a scoped TensorFS handle",
                code="weights_host_unavailable",
            )
        return opened.writer

    def adopt_model(self, facts: Mapping[str, Any]) -> ModelArtifact:
        """Project an exact committed native output; a caller cannot mint a receipt."""
        transaction = str(facts.get("transaction_id", ""))
        opened = self._opened_outputs.get(transaction)
        if opened is None:
            raise CapabilityError(
                "model receipt was not produced by this output capability",
                code="weights_receipt_mismatch",
            )
        existing = self.store.derived_lookup(transaction)
        if existing.get("state") != "committed" or existing.get("receipt") != facts:
            raise CapabilityError(
                "model receipt differs from native committed custody",
                code="weights_receipt_mismatch",
            )
        self._receipt(dict(facts), transaction, opened.declaration)
        wrapped = self._wrap(
            opened.output_slot,
            transaction,
            dict(facts),
            replayed=opened.receipt is not None,
            request_id=self.request_id,
        )
        self.acknowledge(wrapped)
        return wrapped.artifact

    def inspect_source(
        self, manifest: str, components: Sequence[str], configs: Sequence[str]
    ) -> SourceInspection:
        """Inspect the admitted root through native parsing, without rebuilding geometry."""
        from tensorfs.derived import SourceInspection

        length = self.allowed_sources.get(manifest)
        if length is None:
            raise CapabilityError(
                "source is outside this invocation", code="weights_source_ungranted"
            )
        if not components or not configs:
            header = tensorfs.parse_header(bytes(self.store.manifest(manifest)["header"]))
            components = components or tuple(header["components"])
            configs = configs or tuple(header.get("configs", {}))
        return SourceInspection.from_native(
            self.store.inspect_derived_source(manifest, length, components, configs)
        )

    def source(self, manifest: str) -> SourceCapability:
        from tensorfs.derived import SourceCapability

        length = self.allowed_sources.get(manifest)
        if length is None:
            raise CapabilityError(
                "source is outside this invocation", code="weights_source_ungranted"
            )
        return SourceCapability(
            manifest,
            length,
            lambda components, configs: self.inspect_source(manifest, components, configs),
        )

    def _begin(self, output_slot: str, definition: Derivation) -> _OpenedOutput:
        maximum = self.output_bounds.get(output_slot)
        if maximum is None:
            raise CapabilityError("weights output is not declared", code="weights_output_ungranted")
        if any(
            self.allowed_sources.get(source.manifest) != source.length
            for source in definition.sources.values()
        ):
            raise CapabilityError(
                "derivation source is outside this invocation", code="weights_source_ungranted"
            )
        transaction_id = self.transaction_id(output_slot)
        arguments = definition.native_arguments(maximum)
        try:
            declaration = self.store.derived_declaration(
                *arguments,
                work_fingerprint=self.work_fingerprint,
            )
            if not isinstance(declaration, bytes) or not declaration:
                raise ValueError("native declaration has no canonical bytes")
            expected_declaration = canonical_json.decode(declaration)
        except Exception as exc:
            raise CapabilityError(
                f"TensorFS weights declaration refused: {type(exc).__name__}: {exc}",
                code="weights_tensorfs_refused",
            ) from exc
        declaration_digest = hashlib.sha256(declaration).digest()
        binding = (
            self.bind_intent(output_slot, transaction_id, declaration_digest, declaration)
            if self.bind_intent is not None
            else WeightsHostBinding(transaction_id, self.writer_session_id)
        )
        if binding.transaction_id != transaction_id or binding.writer_epoch <= 0:
            raise CapabilityError(
                "the weights host bound another transaction or an invalid writer epoch",
                code="weights_host_identity",
            )
        self.writer_session_id = binding.writer_epoch
        existing = self.store.derived_lookup(transaction_id)
        if existing.get("state") == "committed":
            disposition = existing.get("disposition")
            if not isinstance(disposition, dict) or disposition.get("kind") not in {
                "pending",
                "adopted",
            }:
                raise CapabilityError(
                    "weights receipt no longer has a retained transaction",
                    code="weights_transaction_closed",
                )
            facts = self._receipt(existing.get("receipt"), transaction_id, expected_declaration)
            try:
                manifest = facts["manifest"]
                # Native retention and verified closure, not receipt metadata, establish
                # that this completed result is reusable. This never reads payload bytes.
                self.store.inspect_derived_source(
                    "sha256:" + manifest["sha256"],
                    manifest["length"],
                    list(definition.targets),
                    list(definition.configs),
                )
            except Exception as exc:
                raise CapabilityError(
                    f"TensorFS completed result is not retained and complete: {exc}",
                    code="weights_receipt_unavailable",
                ) from exc
            receipt = self._wrap(
                output_slot,
                transaction_id,
                facts,
                replayed=True,
                request_id=self.request_id,
            )
            _, _, protocol_digest = protocol_receipt(
                receipt,
                owner_scope=self.owner_scope,
                request_id=self.request_id,
                invocation_spec_digest=self.invocation_spec_digest,
            )
            if binding.weights_receipt_digest:
                if binding.weights_receipt_digest != protocol_digest:
                    raise CapabilityError(
                        "the weights host replayed another receipt digest",
                        code="weights_receipt_mismatch",
                    )
            elif self.record_receipt is not None:
                # TensorFS committed before the host receipt ACK. Close that crash gap now.
                self.record_receipt(receipt)
            return _OpenedOutput(
                output_slot=output_slot,
                transaction_id=transaction_id,
                declaration=expected_declaration,
                declaration_digest=documents.spell(declaration_digest),
                writer=None,
                receipt=receipt,
                writer_epoch=binding.writer_epoch,
                checkpoint=binding.checkpoint,
            )
        if binding.weights_receipt_digest:
            raise CapabilityError(
                "the weights host replayed a receipt absent from local TensorFS",
                code="weights_receipt_mismatch",
            )
        if existing.get("state") == "abandoned":
            raise CapabilityError(
                f"weights transaction {transaction_id} was abandoned",
                code="weights_transaction_closed",
            )
        try:
            writer = self.store.begin_derived(
                transaction_id,
                binding.writer_epoch,
                *arguments,
                work_fingerprint=self.work_fingerprint,
                checkpoint=binding.checkpoint,
            )
        except Exception as exc:
            raise CapabilityError(
                f"TensorFS weights declaration refused: {type(exc).__name__}: {exc}",
                code="weights_tensorfs_refused",
            ) from exc
        return _OpenedOutput(
            output_slot=output_slot,
            transaction_id=transaction_id,
            declaration=expected_declaration,
            declaration_digest=documents.spell(declaration_digest),
            writer=writer,
            receipt=None,
            writer_epoch=binding.writer_epoch,
            checkpoint=binding.checkpoint,
        )

    def config(self, manifest: str, name: str) -> bytes:
        """Read selected inline config bytes through the native retained-source validator."""
        length = self.allowed_sources.get(manifest)
        if length is None:
            raise CapabilityError(
                "weights config source is outside the immutable grant",
                code="weights_source_ungranted",
            )
        return source_config(self.store, manifest, length, name)

    def structure(self, manifest: str) -> WeightsSource:
        """Project one granted Manifest to author-safe structural facts only."""
        if manifest not in self.allowed_sources:
            raise CapabilityError(
                f"weights source {manifest} is not in the immutable grant",
                code="weights_source_ungranted",
            )
        try:
            held = self.store.manifest(manifest)
            header = tensorfs.parse_header(bytes(held["header"]))
            return source_structure(header)
        except CapabilityError:
            raise
        except Exception as exc:
            raise CapabilityError(
                f"TensorFS source structure refused: {type(exc).__name__}: {exc}",
                code="weights_source_refused",
            ) from exc

    def transaction_id(self, output_slot: str) -> str:
        return weights_transaction_id(
            self.owner_scope, self.request_id, self.invocation_spec_digest, output_slot
        )

    def fence(self, output_slot: str) -> bool:
        return bool(
            self.store.derived_fence(self.transaction_id(output_slot), self.writer_session_id)
        )

    def acknowledge(self, receipt: WeightsReceipt) -> None:
        """Do not return a TensorFS commit to package code before the host records it."""
        if self.record_receipt is not None:
            self.record_receipt(receipt)

    def abandon(self, output_slot: str) -> WeightsReceipt | None:
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
                code="weights_receipt_absent",
            )
        return self._wrap(
            output_slot, transaction_id, facts, replayed=True, request_id=self.request_id
        )

    def release(self, output_slot: str) -> WeightsReceipt:
        """Release a committed transaction root after publication or rejection."""
        transaction_id = self.transaction_id(output_slot)
        result = self.store.derived_dispose(transaction_id)
        facts = result.get("receipt") if isinstance(result, dict) else None
        if not isinstance(facts, dict):
            raise CapabilityError(
                "TensorFS disposition returned no receipt", code="weights_receipt_absent"
            )
        return self._wrap(
            output_slot, transaction_id, facts, replayed=True, request_id=self.request_id
        )

    @staticmethod
    def _config(config: WeightsConfig) -> dict[str, Any]:
        if config.source:
            return {
                "kind": "derive" if config.data is not None else "copy",
                "source": config.source,
                "source_config": config.source_config,
            }
        return {"kind": "add"}

    @staticmethod
    def _reader(source: WeightsBytes, *, length: int | None = None) -> Readable:
        if isinstance(source, (bytes, bytearray, memoryview)):
            raw = bytes(source)
            if length is not None and len(raw) != length:
                raise CapabilityError(
                    "weights config bytes differ from their declared length",
                    code="weights_config_length",
                )
            return BytesIO(raw)
        if isinstance(source, Readable):
            return source if length is None else _ExactReader(source, length)
        raise CapabilityError("weights part source is not readable", code="weights_part_source")

    @staticmethod
    def _receipt(value: object, transaction_id: str, declaration: object) -> dict[str, Any]:
        if not isinstance(value, dict) or value.get("transaction_id") != transaction_id:
            raise CapabilityError(
                "TensorFS returned a receipt for another transaction",
                code="weights_receipt_mismatch",
            )
        if value.get("declaration") != declaration:
            raise CapabilityError(
                "TensorFS receipt declaration differs from the locked weights request",
                code="weights_receipt_mismatch",
            )
        return value

    @staticmethod
    def _wrap(
        output_slot: str,
        transaction_id: str,
        facts: dict[str, Any],
        *,
        replayed: bool,
        request_id: str,
    ) -> WeightsReceipt:
        raw = canonical_json.encode(facts)
        receipt_digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        return WeightsReceipt(
            output_slot=output_slot,
            weights_transaction_id=transaction_id,
            tensorfs_receipt_digest=receipt_digest,
            tensorfs_receipt=raw,
            replayed=replayed,
            _artifact=ModelArtifact(
                request_id,
                output_slot,
                ObjectRef("sha256:" + facts["manifest"]["sha256"], facts["manifest"]["length"]),
                receipt_digest,
            ),
        )


class OpenedWeightsTransaction:
    """One declaration-locked TensorFS writer exposed through the narrow host protocol."""

    def __init__(
        self,
        *,
        host: WeightsTransactionHost,
        request: WeightsCommit,
        transaction_id: str,
        declaration: object,
        declaration_digest: str,
        writer: Any,
        receipt: WeightsReceipt | None,
        writer_epoch: int,
        checkpoint: tuple[str, int] | None,
    ) -> None:
        self.host = host
        self.request = request
        self.transaction_id = transaction_id
        self.declaration = declaration
        self.declaration_digest = declaration_digest
        self.writer = writer
        self.receipt = receipt
        self.replayed = receipt is not None
        self.closed = self.replayed
        self.writer_epoch = writer_epoch
        self.checkpoint_head = checkpoint

    @property
    def completed_parts(self) -> frozenset[tuple[str, str, str]]:
        declared = frozenset(
            (component, key, role)
            for component, target in self.request.targets.items()
            for key, tensor in target.add.items()
            for role in tensor.parts
        )
        if self.replayed:
            return declared
        try:
            completed = frozenset(
                tuple(row) for row in self._writer("completed parts").completed_parts()
            )
            if not completed <= declared:
                raise ValueError("completed parts are outside the locked declaration")
            return completed
        except Exception as exc:
            self._fence_after(exc, "completed parts")

    @property
    def completed_configs(self) -> frozenset[str]:
        declared = frozenset(
            name for name, config in self.request.configs.items() if config.data is not None
        )
        if self.replayed:
            return declared
        try:
            completed = frozenset(self._writer("completed configs").completed_configs())
            if not completed <= declared:
                raise ValueError("completed configs are outside the locked declaration")
            return completed
        except Exception as exc:
            self._fence_after(exc, "completed configs")

    def checkpoint(self) -> None:
        if self.replayed:
            return
        try:
            from tensorfs.derived import DerivedTransaction

            writer = self._writer("checkpoint")
            facts = (
                writer.checkpoint()
                if isinstance(writer, DerivedTransaction)
                else writer.checkpoint(
                    self.host.request_id, self.request.output_slot, previous=self.checkpoint_head
                )
            )
            if not isinstance(facts, dict):
                raise ValueError("checkpoint has no exact TensorFS facts")
            if facts.get("head") is None:
                return
            for field in ("head", "plan_digest"):
                if (
                    not isinstance(facts.get(field), str)
                    or re.fullmatch(r"sha256:[0-9a-f]{64}", facts[field]) is None
                ):
                    raise ValueError("checkpoint has an invalid object identity")
            for field in ("head_length", "index", "bytes"):
                if type(facts.get(field)) is not int or not 0 <= facts[field] < 1 << 64:
                    raise ValueError("checkpoint has an invalid counter")
            if facts["head_length"] == 0:
                raise ValueError("checkpoint head is empty")
            head = (facts["head"], facts["head_length"])
            if head != self.checkpoint_head and self.host.record_checkpoint is not None:
                self.host.record_checkpoint(
                    WeightsCheckpoint(
                        output_slot=self.request.output_slot,
                        weights_transaction_id=self.transaction_id,
                        writer_epoch=self.writer_epoch,
                        declaration_digest=self.declaration_digest,
                        head=head[0],
                        head_length=head[1],
                        plan_digest=facts["plan_digest"],
                        index=facts["index"],
                        bytes=facts["bytes"],
                    )
                )
            self.checkpoint_head = head
        except Exception as exc:
            self._fence_after(exc, "checkpoint")

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

    def add_part(self, component: str, key: str, role: str, data: WeightsBytes) -> None:
        self._call(
            "weights part",
            lambda writer: writer.add_part(component, key, role, self.host._reader(data)),
        )

    def add_config(self, name: str, data: WeightsBytes, length: int) -> None:
        self._call(
            "weights config",
            lambda writer: writer.add_config(name, self.host._reader(data, length=length)),
        )

    def commit(self) -> WeightsReceipt:
        if self.receipt is not None:
            return self.receipt
        writer = self._writer("commit")
        try:
            facts = self.host._receipt(writer.commit(), self.transaction_id, self.declaration)
        except Exception as exc:
            self._fence_after(exc, "commit")
        self.closed = True
        self.receipt = self.host._wrap(
            self.request.output_slot,
            self.transaction_id,
            facts,
            replayed=False,
            request_id=self.host.request_id,
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
                f"weights transaction {self.transaction_id} is closed during {what}",
                code="weights_transaction_closed",
            )
        return self.writer

    def _call(self, what: str, operation: Any) -> None:
        writer = self._writer(what)
        try:
            operation(writer)
        except Exception as exc:
            self._fence_after(exc, what)

    def _fence_after(self, cause: Exception, what: str) -> NoReturn:
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
            code="weights_tensorfs_refused",
        ) from cause


def weights_transaction_id(
    owner_scope: str, request_id: str, invocation_spec_digest: str, output_slot: str
) -> str:
    """Runtime alone assigns this ID for creation and finalization; hosts record and echo it."""
    subject = canonical_json.encode(
        {
            "owner_authority_scope": owner_scope,
            "request_id": request_id,
            "invocation_spec_digest": invocation_spec_digest,
            "output_slot": output_slot,
        }
    )
    return "sha256:" + hashlib.sha256(b"cozy.runtime.artifact-transaction\0" + subject).hexdigest()


def protocol_receipt(
    receipt: WeightsReceipt,
    *,
    owner_scope: str,
    request_id: str,
    invocation_spec_digest: str,
) -> tuple[pb.WeightsReceiptRef, bytes, bytes]:
    """Wrap exact TensorFS receipt bytes in Worker Protocol WeightsReceipt/1."""
    body = pb.WeightsReceipt(
        owner_authority_scope=owner_scope,
        request_id=request_id,
        invocation_spec_digest=invocation_spec_digest,
        output_slot=receipt.output_slot,
        weights_transaction_id=receipt.weights_transaction_id,
        tensorfs_receipt_digest=receipt.tensorfs_receipt_digest,
        tensorfs_receipt_canonical_bytes=receipt.tensorfs_receipt,
    )
    canonical_bytes, digest = documents.identity(body)
    return (
        pb.WeightsReceiptRef(
            weights_receipt_digest=digest,
            weights_receipt_canonical_bytes=canonical_bytes,
        ),
        canonical_bytes,
        digest,
    )
