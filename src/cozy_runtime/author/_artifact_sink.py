"""The one author-facing transaction for durable job artifacts.

The values in this module describe WHAT a job is producing.  They deliberately contain no
store, path, publisher, repository, transaction id, writer id, or tool.  Runtime binds the
service to one verified InvocationSpec and turns the declaration into a fenced TensorFS
transaction; author code cannot construct that capability or widen it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from cozy_runtime.author._assets import Tree
from cozy_runtime.author._errors import CapabilityError

if TYPE_CHECKING:
    from cozy_runtime.author._services import Attempt


@runtime_checkable
class ArtifactReader(Protocol):
    """A bounded byte stream consumed by TensorFS's ordinary object writer."""

    def read(self, size: int = -1) -> bytes: ...


ArtifactBytes: TypeAlias = bytes | bytearray | memoryview | ArtifactReader


@dataclass(frozen=True, slots=True)
class ArtifactPart:
    """One produced stored role and the stream containing its exact bytes."""

    role: str
    dtype: str
    shape: tuple[int, ...]
    data: ArtifactBytes


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
    """Admit exact newly produced config bytes through the object plane."""

    data: ArtifactBytes


ArtifactConfig: TypeAlias = CopyConfig | CreateConfig


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
class _ArtifactRequest:
    output_slot: str
    sources: Mapping[str, Tree]
    targets: Mapping[str, ArtifactComponent]
    configs: Mapping[str, ArtifactConfig]


class ArtifactSink:
    """Job-only, attempt-bound author service for one derived snapshot transaction."""

    __slots__ = ("_attempt", "_commit")

    def __init__(
        self,
        attempt: Attempt,
        commit: Callable[[_ArtifactRequest], ArtifactCommit] | None = None,
    ) -> None:
        self._attempt = attempt
        self._commit = commit

    def commit(
        self,
        output_slot: str,
        *,
        sources: Mapping[str, Tree],
        targets: Mapping[str, ArtifactComponent],
        configs: Mapping[str, ArtifactConfig],
    ) -> ArtifactCommit:
        """Commit one closed target under one InvocationSpec-declared output slot.

        `configs` is required even when empty.  That makes the target header closed: config
        inheritance cannot appear later as an implicit convention.
        """
        self._attempt.check_open("ArtifactSink.commit")
        if self._commit is None:
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
        return self._commit(_ArtifactRequest(output_slot, sources, targets, configs))


def _ordered_unique(values: Sequence[str], what: str) -> tuple[str, ...]:
    """Canonicalize a set-shaped author declaration and refuse ambiguity."""
    ordered = tuple(sorted(values))
    if any(not value for value in ordered) or len(set(ordered)) != len(ordered):
        raise CapabilityError(f"{what} must be non-empty and unique", code="artifact_declaration")
    return ordered

