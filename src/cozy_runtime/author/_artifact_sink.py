"""The one author-facing transaction for durable job artifacts.

The values in this module describe WHAT a job is producing.  They deliberately contain no
store, path, publisher, repository, transaction id, writer id, or tool.  Runtime binds the
service to one verified InvocationSpec and turns the declaration into a fenced TensorFS
transaction; author code cannot construct that capability or widen it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable

from cozy_runtime.author._assets import Tree
from cozy_runtime.author._errors import CapabilityError

if TYPE_CHECKING:
    from cozy_runtime.author._services import Attempt


@runtime_checkable
class ArtifactReader(Protocol):
    """A bounded byte stream consumed by TensorFS's ordinary object writer."""

    def read(self, size: int = -1) -> bytes: ...


ArtifactBytes: TypeAlias = bytes | bytearray | memoryview | ArtifactReader
ARTIFACT_MIME_TYPE = "application/vnd.cozy.tensorfs.snapshot"


@dataclass(frozen=True, slots=True)
class ArtifactOutput:
    """One immutable job artifact slot and its maximum newly produced bytes."""

    output_id: str
    max_new_bytes: int


@dataclass(frozen=True, slots=True)
class ArtifactPart:
    """One declared stored role. Bytes are supplied later through the fenced writer."""

    role: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ArtifactValue:
    """One complete produced tensor/value in a target component."""

    key: str
    logical_dtype: str
    shape: tuple[int, ...]
    encoding: str
    parts: tuple[ArtifactPart, ...]


@dataclass(frozen=True, slots=True)
class CopyComponent:
    """Copy one exact source component map; payload objects carry by reference."""

    source: str
    component: str


@dataclass(frozen=True, slots=True)
class DeriveComponent:
    """Derive from one exact source component through a closed drop/add edit."""

    source: str
    component: str
    drop: tuple[str, ...] = ()
    add: tuple[ArtifactValue, ...] = ()


@dataclass(frozen=True, slots=True)
class CreateComponent:
    """Create a complete component from exactly the values listed here."""

    values: tuple[ArtifactValue, ...]


ArtifactComponent: TypeAlias = CopyComponent | DeriveComponent | CreateComponent


@dataclass(frozen=True, slots=True)
class CopyConfig:
    """Carry one exact source header ConfigRef into the target header."""

    source: str
    name: str


@dataclass(frozen=True, slots=True)
class CreateConfig:
    """Declare one exact newly produced config object."""

    media_type: str
    length: int


@dataclass(frozen=True, slots=True)
class DeriveConfig:
    """Read one selected source config and write one exact replacement."""

    source: str
    name: str
    media_type: str
    length: int


ArtifactConfig: TypeAlias = CopyConfig | CreateConfig | DeriveConfig


@dataclass(frozen=True, slots=True)
class ArtifactCommit:
    """Compact author-visible evidence that Runtime durably committed the slot.

    The complete receipt belongs to Runtime's terminal and never crosses into job code.
    These two identities come from TensorFS; the job cannot supply either one.
    """

    output_slot: str
    snapshot_digest: str
    header_digest: str
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class ArtifactSource:
    """TensorFS-verified effective identity of one grant-bound source alias."""

    alias: str
    snapshot_digest: str
    snapshot_length: int
    header_digest: str
    header_length: int
    components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArtifactComponentMetadata:
    name: str
    tensors: tuple[ArtifactValue, ...]


@dataclass(frozen=True, slots=True)
class ArtifactSourceConfig:
    name: str
    media_type: str
    data: bytes


@dataclass(frozen=True, slots=True)
class ArtifactInspection:
    source: ArtifactSource
    components: tuple[ArtifactComponentMetadata, ...]
    configs: tuple[ArtifactSourceConfig, ...]


@dataclass(frozen=True, slots=True)
class _ArtifactRequest:
    output_slot: str
    sources: Mapping[str, Tree]
    targets: Mapping[str, ArtifactComponent]
    configs: Mapping[str, ArtifactConfig]


@dataclass(frozen=True, slots=True)
class _ArtifactInspectRequest:
    source: Tree
    components: tuple[str, ...]
    configs: tuple[str, ...]


class _ArtifactWriterBackend(Protocol):
    sources: tuple[ArtifactSource, ...]
    replayed: bool

    def source_read_into(
        self,
        source: str,
        component: str,
        key: str,
        role: str,
        offset: int,
        into: object,
    ) -> None: ...

    def add_part(self, component: str, key: str, role: str, data: ArtifactBytes) -> None: ...
    def add_config(self, name: str, data: ArtifactBytes) -> None: ...
    def commit(self) -> ArtifactCommit: ...
    def fence(self) -> bool: ...


class ArtifactWriter:
    """One opened, attempt-bound writer. Leaving it uncommitted fences it."""

    __slots__ = ("_attempt", "_backend", "_closed")

    def __init__(self, attempt: Attempt, backend: _ArtifactWriterBackend) -> None:
        self._attempt = attempt
        self._backend = backend
        self._closed = False

    @property
    def sources(self) -> tuple[ArtifactSource, ...]:
        return self._backend.sources

    @property
    def replayed(self) -> bool:
        return self._backend.replayed

    def source_read_into(
        self,
        source: str,
        component: str,
        key: str,
        role: str,
        offset: int,
        into: object,
    ) -> None:
        self._check("ArtifactWriter.source_read_into")
        self._backend.source_read_into(source, component, key, role, offset, into)

    def add_part(self, component: str, key: str, role: str, data: ArtifactBytes) -> None:
        self._check("ArtifactWriter.add_part")
        self._backend.add_part(component, key, role, data)

    def add_config(self, name: str, data: ArtifactBytes) -> None:
        self._check("ArtifactWriter.add_config")
        self._backend.add_config(name, data)

    def commit(self) -> ArtifactCommit:
        self._check("ArtifactWriter.commit")
        result = self._backend.commit()
        self._closed = True
        return result

    def fence(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        return self._backend.fence()

    def __enter__(self) -> ArtifactWriter:
        self._check("ArtifactWriter.__enter__")
        return self

    def __exit__(self, *_error: object) -> Literal[False]:
        self.fence()
        return False

    def _check(self, what: str) -> None:
        self._attempt.check_open(what)
        if self._closed:
            raise CapabilityError(
                f"{what}: the artifact writer is already committed or fenced",
                code="artifact_writer_closed",
            )


class ArtifactSink:
    """Job-only, attempt-bound author service for one derived snapshot transaction."""

    __slots__ = ("_attempt", "_inspect", "_open")

    def __init__(
        self,
        attempt: Attempt,
        inspect: Callable[[_ArtifactInspectRequest], ArtifactInspection] | None = None,
        open_writer: Callable[[_ArtifactRequest], _ArtifactWriterBackend] | None = None,
    ) -> None:
        self._attempt = attempt
        self._inspect = inspect
        self._open = open_writer

    def inspect(
        self,
        source: Tree,
        *,
        components: Sequence[str],
        configs: Sequence[str] = (),
    ) -> ArtifactInspection:
        """Preview only explicitly selected metadata/config bytes from one exact Tree."""
        self._attempt.check_open("ArtifactSink.inspect")
        if self._inspect is None:
            raise CapabilityError(
                "ArtifactSink inspection is not bound to this verified InvocationSpec",
                code="artifact_sink_unavailable",
            )
        selected_components = _ordered_unique(tuple(components), "source components")
        selected_configs = _ordered_unique(tuple(configs), "source configs") if configs else ()
        if not selected_components and not selected_configs:
            raise CapabilityError(
                "ArtifactSink.inspect requires a component or config selector",
                code="artifact_inspection_empty",
            )
        return self._inspect(_ArtifactInspectRequest(source, selected_components, selected_configs))

    def open(
        self,
        output_slot: str,
        *,
        sources: Mapping[str, Tree],
        targets: Mapping[str, ArtifactComponent],
        configs: Mapping[str, ArtifactConfig],
    ) -> ArtifactWriter:
        """Open one closed target under one InvocationSpec-declared output slot.

        The declaration locks before the returned writer can read or write. `configs` is
        required even when empty, so config inheritance cannot appear as a convention.
        """
        self._attempt.check_open("ArtifactSink.open")
        if self._open is None:
            raise CapabilityError(
                "ArtifactSink is not bound to a verified job InvocationSpec on this attempt",
                code="artifact_sink_unavailable",
            )
        if not output_slot:
            raise CapabilityError("ArtifactSink needs a declared output slot", code="artifact_slot")
        if not sources:
            raise CapabilityError(
                "ArtifactSink needs at least one exact source snapshot", code="artifact_sources"
            )
        if not targets:
            raise CapabilityError(
                "ArtifactSink needs a non-empty closed target map", code="artifact_targets"
            )
        if any(not alias or not isinstance(tree, Tree) for alias, tree in sources.items()):
            raise CapabilityError(
                "ArtifactSink sources are non-empty aliases mapped to granted Tree values",
                code="artifact_sources",
            )
        if any(not name for name in targets) or any(not name for name in configs):
            raise CapabilityError(
                "ArtifactSink target and config names are non-empty", code="artifact_declaration"
            )
        return ArtifactWriter(
            self._attempt, self._open(_ArtifactRequest(output_slot, sources, targets, configs))
        )


def _ordered_unique(values: Sequence[str], what: str) -> tuple[str, ...]:
    """Canonicalize a set-shaped author declaration and refuse ambiguity."""
    ordered = tuple(sorted(values))
    if any(not value for value in ordered) or len(set(ordered)) != len(ordered):
        raise CapabilityError(f"{what} must be non-empty and unique", code="artifact_declaration")
    return ordered
