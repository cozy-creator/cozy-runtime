"""Per-service fakes — an SDK FEATURE package authors use, not test scaffolding (§1.6).

One fake per service the handler requests, never one fake world. Each is a CAPABILITY
ADAPTER over the production kernel: `fake_outputs()` is the real `Outputs` writing to a
real spool through the real encode and size-limit path, plus recording accessors. The
encode under test is the one production runs, so a fake cannot drift from the real thing
by construction — there is no parallel behavioral runtime here to maintain.

`fake_input()` closes the loop the other way: a saved OUTPUT asset becomes an input
handle hydrated through the real `bind`, so one attempt's result feeds the next the way
the executor feeds it.

`Settings` and `Secrets` need no fake wrapper: they carry exactly the validated value, so
`Settings(MyConfig(...))` IS the fake — a wrapper adding nothing is a surface to maintain.

`invoke_with_fakes()` runs a registered handler through the REAL kernel with fakes standing
in for the runtime's side of the contract: same decode, same overlay, same preflight, same
injection, same return validation, same output transaction.

`StagedBackend`/`ResidentBackend` are the weightless fill doubles a `Model` constructs
against with no store and no device.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cozy_runtime.author._app import Registration
from cozy_runtime.author._assets import Asset, GrantedInput, bind
from cozy_runtime.author._context import AdapterRef, Context, Device
from cozy_runtime.author._decode import DEFAULT_DECODE_LIMITS, DecodeLimits, MediaDecoder
from cozy_runtime.author._defaults import Clamp, Recipe
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._invoke import Invocation, InvocationResult, invoke, prepare
from cozy_runtime.author._loader import Census, TensorLike, TensorSpec
from cozy_runtime.author._model import Model
from cozy_runtime.author._observations import Observation
from cozy_runtime.author._services import (
    MAX_OUTPUT_BYTES,
    AdjustmentRow,
    Adjustments,
    Attempt,
    Egress,
    Outputs,
    Telemetry,
)


def _spool(name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"cozy-{name}-"))


def fake_attempt(
    request_id: str = "fake-attempt",
    *,
    spool: Path | None = None,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> Attempt:
    """A real attempt over a real spool — the transaction the services share.

    `spool` names the directory (created if absent) when the caller has to LOOK at what the
    attempt left behind: a partial-output cleanup or a spool-residue proof reads the
    directory, and it cannot read one it was never told.
    """
    if spool is None:
        return Attempt(request_id, _spool(request_id), max_output_bytes, settle_at_save=True)
    spool.mkdir(parents=True, exist_ok=True)
    return Attempt(request_id, spool, max_output_bytes, settle_at_save=True)


def fake_input[AssetT: Asset](
    source: AssetT,
    *,
    attempt: str,
    input_id: str = "input",
    order: int = 0,
    max_decoded_bytes: int = 0,
) -> AssetT:
    """`source`'s bytes as a fresh INPUT handle hydrated into `attempt`.

    The round trip a package author needs and cannot otherwise take: `out.save_*` returns
    an OUTPUT asset, and feeding it back to a handler — or to `MediaDecoder` — is what the
    executor does when one attempt's result becomes the next attempt's granted input.
    Hydration runs through the same `bind` the executor's `_hydrate` runs, with the same
    facts the worker established before acceptance, so a fake input cannot carry a
    provenance the real one would not have.
    """
    if source._local is None:
        raise CapabilityError(
            f"asset {source.ref!r} has no bytes to grant: an input stands for bytes the "
            "worker already fetched, verified and spooled",
            code="asset_unhydrated",
        )
    granted = GrantedInput(
        input_id=input_id,
        local=source._local,
        media_type=source.media_type,
        digest=source.digest,
        length=source.size_bytes,
        order=order,
    )
    bound = type(source)(source.ref)
    bind(
        bound,
        local=granted.local,
        attempt=attempt,
        media_type=granted.media_type,
        digest=granted.digest,
        length=granted.length,
        max_decoded_bytes=max_decoded_bytes,
        input_id=granted.input_id,
        order=granted.order,
    )
    return bound


def fake_context(
    *,
    request_id: str = "fake-attempt",
    seconds: float = 300.0,
    device: Device | None = None,
    cancelled: bool = False,
) -> Context:
    """A real `Context` for a handler: the attempt's id, deadline, device and cancellation."""
    return Context(request_id, time.monotonic() + seconds, device or Device(), lambda: cancelled)


class OutputsRecorder(Outputs):
    """The fake `Outputs`: the REAL save path plus assertions about what was saved."""

    __slots__ = ()

    @property
    def saved(self) -> tuple[Asset, ...]:
        return tuple(self._attempt.pending.values())

    def assert_saved(self, *, count: int | None = None, kind: str | None = None) -> None:
        if count is not None and len(self.saved) != count:
            raise AssertionError(f"saved {len(self.saved)} outputs, expected {count}")
        if kind is not None:
            wrong = [a.ref for a in self.saved if a.kind != kind]
            if wrong:
                raise AssertionError(f"outputs are not all {kind}: {', '.join(wrong)}")


class TelemetryRecorder(Telemetry):
    """The fake `Telemetry`: the real emit path, plus the recorded event stream."""

    __slots__ = ()

    @property
    def events(self) -> tuple[Observation, ...]:
        return tuple(self._attempt.ring)

    def stages(self) -> Mapping[str, float]:
        """The COMMITTED `stage_ms` facts, read off the O(1) accumulator rather than
        recomputed from rows — the ring is bounded and a long attempt sheds its oldest."""
        return {
            name: round(track.total_ms, 3)
            for name, track in self._attempt.attribution.stages.items()
        }


class AdjustmentsRecorder(Adjustments):
    """The fake `Adjustments`: the real row path, plus the recorded envelope rows."""

    __slots__ = ()

    @property
    def rows(self) -> tuple[AdjustmentRow, ...]:
        return tuple(self._attempt.rows)


def fake_outputs(attempt: Attempt | None = None, **kwargs: Any) -> OutputsRecorder:
    return OutputsRecorder(attempt or fake_attempt(**kwargs))


def fake_telemetry(attempt: Attempt | None = None, ctx: Context | None = None) -> TelemetryRecorder:
    return TelemetryRecorder(attempt or fake_attempt(), ctx or fake_context())


def fake_adjustments(attempt: Attempt | None = None) -> AdjustmentsRecorder:
    return AdjustmentsRecorder(attempt or fake_attempt())


def fake_egress(respond: Callable[..., Any], attempt: Attempt | None = None) -> Egress:
    """An `Egress` over a local responder. The real broker, allowlist and SSRF fail-closed
    path are cr-012's; this proves only injection and the capability."""
    return Egress(attempt or fake_attempt(), respond)


def fake_media_decoder(
    attempt: Attempt | None = None,
    *,
    active: bool = False,
    limits: DecodeLimits = DEFAULT_DECODE_LIMITS,
) -> MediaDecoder:
    """The real decoder over a caller-controlled attempt, without a parallel fake codec."""
    return MediaDecoder(attempt or fake_attempt(), active=lambda: active, limits=limits)


def invoke_with_fakes(
    registration: Registration,
    wire: Mapping[str, object],
    *,
    request_id: str = "fake-attempt",
    spool: Path | None = None,
    models: Mapping[str, Model[Any]] | None = None,
    settings: object = None,
    secrets: object = None,
    recipes: Sequence[Recipe] = (),
    clamps: Sequence[Clamp] = (),
    adapters: tuple[AdapterRef, ...] = (),
    seconds: float = 300.0,
    cancelled: bool = False,
    egress_broker: Callable[..., Any] | None = None,
    assets: Mapping[str, GrantedInput] | None = None,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> InvocationResult:
    """Run a handler through the REAL kernel with no GPU, no hub and no RecordOwner."""
    spool = spool or _spool(request_id)
    spool.mkdir(parents=True, exist_ok=True)
    return invoke(
        prepare(
            registration,
            wire,
            recipes=recipes,
            clamps=clamps,
            settings=settings,
            input_metadata=assets or {},
        ),
        Invocation(
            request_id=request_id,
            spool=spool,
            deadline=time.monotonic() + seconds,
            cancel=lambda: cancelled,
            models=models or {},
            recipes=recipes,
            clamps=clamps,
            settings=settings,
            secrets=secrets,
            adapters=adapters,
            egress_broker=egress_broker,
            assets=assets or {},
            max_output_bytes=max_output_bytes,
        ),
        # No worker stands behind a fake, so every registered frame is encoded at the save.
        Attempt(request_id, spool, max_output_bytes, settle_at_save=True),
    )


# --------------------------------------------------------------------------- fill
#
# WEIGHTLESS FILL BACKENDS. The package declares WHICH components a method may touch;
# it never learns HOW their bytes got there, and these two prove it: the same handler
# runs on either and cannot observe which. They are DOUBLES and they lived under
# `internal/` as if they were a runtime plane — production binds through the weight plane
# (`internal/weights.py`) and has never called these (codex audit, adopted 2026-08-25).


@dataclass(slots=True)
class FillEvent:
    key: str
    nbytes: int
    route: str


@dataclass(slots=True)
class _RecordingBackend:
    route: str
    events: list[FillEvent] = field(default_factory=list)
    poisoned: list[str] = field(default_factory=list)

    @property
    def filled_bytes(self) -> int:
        return sum(e.nbytes for e in self.events)

    def fit(self, walked: Census, *, encoded_leaves: str) -> None:
        """A stand-in holds no checkpoint header, so there is nothing to judge against."""
        return None

    def materialize(
        self, obj: object, walked: Census, *, encoded_leaves: str = "refuse"
    ) -> Mapping[str, TensorLike]:
        """These stand-ins place nothing: the destinations the census walked are already
        the destinations they write through — and a stand-in never replaces a leaf, so the
        consent it is handed is recorded by its signature and acted on by nothing."""
        return walked.live

    def fill(self, key: str, spec: TensorSpec, destination: TensorLike) -> None:
        write = getattr(destination, "fill_", None)
        if callable(write):
            write(spec.nbytes)
        self.events.append(FillEvent(key, spec.nbytes, self.route))

    def commit(self) -> None:
        """No device copies to fence: the weight plane (`internal/weights.py`) owns them."""
        return None

    def poison(self, keys: Sequence[str]) -> None:
        """Construction is TRANSACTIONAL: a failure abandons the fills as a unit (§1.1)."""
        self.poisoned.extend(keys)
        self.events = [e for e in self.events if e.key not in set(keys)]


@dataclass(slots=True)
class StagedBackend(_RecordingBackend):
    """Host-staged placement: bytes land in a staging buffer, then in the destination."""

    route: str = "staged"


@dataclass(slots=True)
class ResidentBackend(_RecordingBackend):
    """Direct resident placement — a different route, an identical author contract."""

    route: str = "resident"
