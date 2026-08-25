"""Per-service fakes — an SDK FEATURE endpoint authors use, not test scaffolding (§1.6).

One fake per service the handler requests, never one fake world. Each is a CAPABILITY
ADAPTER over the production kernel: `fake_outputs()` is the real `Outputs` writing to a
real spool through the real encode and size-limit path, plus recording accessors. The
encode under test is the one production runs, so a fake cannot drift from the real thing
by construction — there is no parallel behavioral runtime here to maintain.

`Settings` and `Secrets` need no fake wrapper: they carry exactly the validated value, so
`Settings(MyConfig(...))` IS the fake — a wrapper adding nothing is a surface to maintain.

`invoke_with_fakes()` runs a registered handler through the REAL kernel with fakes standing
in for the runtime's side of the contract: same decode, same overlay, same preflight, same
injection, same return validation, same output transaction.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from cozy_runtime.author._app import Registration
from cozy_runtime.author._assets import Asset
from cozy_runtime.author._context import AdapterRef, Context, Device
from cozy_runtime.author._defaults import Clamp, Recipe
from cozy_runtime.author._invoke import Invocation, ResultEnvelope, invoke
from cozy_runtime.author._model import Model
from cozy_runtime.author._services import (
    MAX_OUTPUT_BYTES,
    AdjustmentRow,
    Adjustments,
    Attempt,
    Egress,
    Outputs,
    Telemetry,
    TelemetryEvent,
)


def _spool(name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"cozy-{name}-"))


def fake_attempt(
    request_id: str = "fake-attempt", *, max_output_bytes: int = MAX_OUTPUT_BYTES
) -> Attempt:
    """A real attempt over a real temporary spool — the transaction the services share."""
    return Attempt(request_id, _spool(request_id), max_output_bytes)


def fake_context(
    *,
    request_id: str = "fake-attempt",
    seconds: float = 300.0,
    boot_warmup: bool = False,
    device: Device | None = None,
    cancelled: bool = False,
) -> Context:
    """A real `Context`. `boot_warmup` is a real fact, so a handler's warm-pass branch is
    exercised the same way the boot warm pass exercises it (cr-007)."""
    return Context(
        request_id,
        time.monotonic() + seconds,
        boot_warmup,
        device or Device(),
        lambda: cancelled,
    )


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
    def events(self) -> tuple[TelemetryEvent, ...]:
        return tuple(self._attempt.events)

    def stages(self) -> Mapping[str, float]:
        return {
            e.name: e.value for e in self.events if e.kind == "stage" and isinstance(e.value, float)
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


def invoke_with_fakes(
    registration: Registration,
    wire: Mapping[str, object],
    *,
    request_id: str = "fake-attempt",
    models: Mapping[str, Model[Any]] | None = None,
    settings: object = None,
    secrets: object = None,
    recipes: Sequence[Recipe] = (),
    clamps: Sequence[Clamp] = (),
    adapters: tuple[AdapterRef, ...] = (),
    boot_warmup: bool = False,
    seconds: float = 300.0,
    cancelled: bool = False,
    egress_broker: Callable[..., Any] | None = None,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> ResultEnvelope:
    """Run a handler through the REAL kernel with no GPU, no hub and no coordinator."""
    return invoke(
        registration,
        wire,
        Invocation(
            request_id=request_id,
            spool=_spool(request_id),
            deadline=time.monotonic() + seconds,
            boot_warmup=boot_warmup,
            cancel=lambda: cancelled,
            models=models or {},
            recipes=recipes,
            clamps=clamps,
            settings=settings,
            secrets=secrets,
            adapters=adapters,
            egress_broker=egress_broker,
            max_output_bytes=max_output_bytes,
        ),
    )
