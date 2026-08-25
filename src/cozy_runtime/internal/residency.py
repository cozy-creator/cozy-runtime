"""Fill backends — the runtime's side of the one fill path, and the backend-blindness fence.

The endpoint declares WHICH components a method may touch; it never learns HOW their bytes
got there. Two backends implement the same `author.Backend` protocol: `StagedBackend`
stages through a host buffer and `ResidentBackend` places directly. The same endpoint code
runs on either and cannot observe which — there is no author-facing surface that names a
backend, a placement rung, or a staging event (§1.1/§3.2).

Both are launch STANDS-IN for the real thing: cr-005 brings the TensorFS stream and
cr-008b the component-staging plan, behind this same call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from cozy_runtime.author import TensorSpec
from cozy_runtime.author._loader import Census, TensorLike


@dataclass(slots=True)
class FillEvent:
    key: str
    nbytes: int
    route: str


@dataclass(slots=True)
class _RecordingBackend:
    route: str
    events: list[FillEvent] = field(default_factory=list)
    rolled_back: list[str] = field(default_factory=list)

    @property
    def filled_bytes(self) -> int:
        return sum(e.nbytes for e in self.events)

    def materialize(self, obj: object, walked: Census) -> Mapping[str, TensorLike]:
        """These stand-ins place nothing: the destinations the census walked are already
        the destinations they write through."""
        return walked.live

    def fill(self, key: str, spec: TensorSpec, destination: TensorLike) -> None:
        write = getattr(destination, "fill_", None)
        if callable(write):
            write(spec.nbytes)
        self.events.append(FillEvent(key, spec.nbytes, self.route))

    def commit(self) -> None:
        """No device copies to fence — the freeze point is cr-005's `StreamingFillBackend`."""
        return None

    def rollback(self, keys: Sequence[str]) -> None:
        """Construction is TRANSACTIONAL: a failure rolls fills back as a unit (§1.1)."""
        self.rolled_back.extend(keys)
        self.events = [e for e in self.events if e.key not in set(keys)]


@dataclass(slots=True)
class StagedBackend(_RecordingBackend):
    """Host-staged placement: bytes land in a staging buffer, then in the destination."""

    route: str = "staged"


@dataclass(slots=True)
class ResidentBackend(_RecordingBackend):
    """Direct resident placement — a different route, an identical author contract."""

    route: str = "resident"
