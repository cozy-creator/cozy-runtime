"""Typed refusals. Every author-surface failure is one of these — never a bare ValueError.

Two clocks (§1.0): `ConformanceError` refuses at BUILD (describe), everything else refuses
at REQUEST time. The clock is the class, so a caller never has to read prose to know
whether an endpoint is broken or a request is.
"""

from __future__ import annotations

from collections.abc import Sequence


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


class Cancelled(AuthorError):
    """The attempt was cancelled. THE cancellation spelling is `ctx.raise_if_cancelled()`."""

    default_code = "cancelled"


class DeadlineExceeded(Cancelled):
    """The attempt outlived its deadline — a cancellation cause, not a separate spelling."""

    default_code = "deadline"
