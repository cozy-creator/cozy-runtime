"""Request-scoped activation capture options and observed execution facts."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Protocol

import msgspec

from cozy_runtime.author._assets import Tree
from cozy_runtime.author._errors import InvalidRequest


class ActivationCapture(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Capture selected declared components during an ordinary invocation.

    Empty ``steps`` captures the whole measured schedule. Explicit steps are zero-based,
    ascending, unique, and include zero. This option is independent of the package payload.
    """

    components: tuple[str, ...]
    steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.components
            or len(self.components) > 32
            or len(set(self.components)) != len(self.components)
            or any(not name or len(name) > 128 for name in self.components)
        ):
            raise InvalidRequest(
                "capture needs 1..32 unique component names", code="capture_component_undeclared"
            )
        if self.steps and (
            len(self.steps) > 4096
            or any(type(step) is not int or step < 0 for step in self.steps)
            or self.steps != tuple(sorted(set(self.steps)))
            or self.steps[0] != 0
        ):
            raise InvalidRequest(
                "capture steps must be unique, ascending, nonnegative and include zero",
                code="capture_steps",
            )


class ExecutionEnvironment(msgspec.Struct, frozen=True):
    """Facts observed by the worker and executor that actually handled this invocation."""

    runtime_version: str
    worker_image_digest: str = ""
    accelerator: str = ""
    driver: str = ""
    cuda: str = ""
    worker_boot_id: str = ""
    execution_lane: str = ""
    execution_contract_digest: str = ""
    kernel_symbol: str = ""


class ActivationCaptureResult(msgspec.Struct, frozen=True):
    output_id: str
    content_digest: str
    tree: Tree | None = None
    """Hydrated from the matching verified native grant, never serialized on the wire."""


class ExecutionObservation(msgspec.Struct, frozen=True):
    environment: ExecutionEnvironment
    capture: ActivationCaptureResult | None = None


class _CaptureClock(Protocol):
    def schedule(self, total: int) -> None: ...

    def completed(self, step: int) -> None: ...


_capture_clock: ContextVar[_CaptureClock | None] = ContextVar("cozy_capture_clock", default=None)
