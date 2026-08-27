"""Typed refusals. Every author-surface failure is one of these — never a bare ValueError.

Two clocks (§1.0): `ConformanceError` refuses at BUILD (describe), everything else refuses
at REQUEST time. The clock is the class, so a caller never has to read prose to know
whether an endpoint is broken or a request is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


class AuthorError(Exception):
    """Base of every typed refusal the author surface raises."""

    default_code = "author_error"

    def __init__(
        self, message: str, *, code: str | None = None, fields: Sequence[str] = ()
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.fields = tuple(fields)

    def __str__(self) -> str:
        where = f" [{', '.join(self.fields)}]" if self.fields else ""
        return f"{self.code}: {self.message}{where}"


class ConformanceError(AuthorError):
    """A BUILD-time refusal: the endpoint's own surface is invalid (§1.7).

    Same verdict class as a `mypy --strict` failure — describe is the second reader of the
    one signature, never a second contract.
    """

    default_code = "conformance"


class InvalidRequest(AuthorError):
    """The request does not satisfy the declared schema. API bounds REJECT (§1.2)."""

    default_code = "invalid_request"


class UnsupportedInput(InvalidRequest):
    """A well-formed request this deployment will not serve. Raised from preflight."""

    default_code = "unsupported_input"


class DefaultResolutionError(InvalidRequest):
    """The ModelDefault overlay could not be resolved. Recipes refuse AS A UNIT (§1.2)."""

    default_code = "default_resolution"


class PreflightViolation(AuthorError):
    """A preflight hook reached past its narrow contract (metadata only, never bytes)."""

    default_code = "preflight_violation"


class CapabilityError(AuthorError):
    """A service was used outside the capability or lifetime its signature declared."""

    default_code = "capability"


class OutputError(AuthorError):
    """The output transaction failed (§3.4 step 1): size, encode, or handle provenance."""

    default_code = "output"


class RuntimeFailure(AuthorError):
    """The Runtime broke an already-admitted attempt; the request and author are sound."""

    default_code = "runtime_failure"


class Cancelled(AuthorError):
    """The attempt was cancelled. THE cancellation spelling is `ctx.raise_if_cancelled()`."""

    default_code = "cancelled"


class DeadlineExceeded(Cancelled):
    """The attempt outlived its deadline — a cancellation cause, not a separate spelling."""

    default_code = "deadline"


# ------------------------------------------------------- the neutral set at the boundary

Terminal = Literal["succeeded", "refused", "failed", "canceled"]
Origin = Literal["author", "request", "runtime"]


@dataclass(frozen=True, slots=True)
class Outcome:
    """What ONE attempt reports at the invocation boundary — neutral FACTS only (§2).

    There is deliberately no `retryable`, `retry_after` or `fatal` member. RETRYABLE and
    FATAL are coordinator PROJECTIONS over these facts, never observations the worker or
    the author writes: each wrongly-classified retry buys a pod.
    """

    terminal: Terminal
    origin: Origin
    code: str
    message: str = ""
    fields: tuple[str, ...] = ()


#: The frozen vocabulary the descriptor publishes and the refusal layer maps.
NEUTRAL_ERROR_SET: Mapping[str, object] = {
    "author_raises": ["InvalidRequest", "UnsupportedInput", "Cancelled", "exception"],
    "terminals": ["succeeded", "refused", "failed", "canceled"],
    "origins": ["author", "request", "runtime"],
    "retryability": "a coordinator projection over these facts; never author-declared",
}

SUCCEEDED = Outcome("succeeded", "runtime", "ok")

#: The code a device out-of-memory escaping a handler reports under. It is RUNTIME-origin
#: on purpose: residency is the runtime's whole job here, the endpoint owns no memory
#: vocabulary at all (the `endpoint-owns-no-residency` fence proves it cannot even name
#: one), so blaming the author for the card running out is naming the one party that had no
#: say. Measured: cl-003 watched a plan-arithmetic defect arrive as
#: `AUTHOR_EXCEPTION/CAUSE_ORIGIN_AUTHOR` on an endpoint that made no memory call.
DEVICE_OOM = "device_out_of_memory"


def _is_device_oom(exc: BaseException) -> bool:
    """A torch device OOM, recognized WITHOUT importing torch.

    This module is the author kernel's and it is deliberately dependency-free — the
    worker imports it and the worker has no torch. The identity is therefore
    structural: the class's own module and name, which `torch.cuda.OutOfMemoryError` has had
    since it was split out of `RuntimeError`.
    """
    kind = type(exc)
    return kind.__name__ == "OutOfMemoryError" and kind.__module__.split(".")[0] == "torch"


def classify(exc: BaseException | None) -> Outcome:
    """Map what escaped a handler onto the neutral set.

    A retryability attribute on the exception is NEVER read — an author who sets one has
    written a fact nobody consumes, which is the point (§2).
    """
    if exc is None:
        return SUCCEEDED
    if isinstance(exc, Cancelled):
        origin: Origin = "runtime" if isinstance(exc, DeadlineExceeded) else "author"
        return Outcome("canceled", origin, exc.code, exc.message, exc.fields)
    if isinstance(exc, InvalidRequest):
        return Outcome("refused", "request", exc.code, exc.message, exc.fields)
    if isinstance(exc, RuntimeFailure):
        return Outcome("failed", "runtime", exc.code, exc.message, exc.fields)
    if isinstance(exc, AuthorError):
        return Outcome("failed", "author", exc.code, exc.message, exc.fields)
    if _is_device_oom(exc):
        return Outcome("failed", "runtime", DEVICE_OOM, str(exc).splitlines()[0][:400])
    return Outcome("failed", "author", "unhandled_exception", f"{type(exc).__name__}: {exc}")
