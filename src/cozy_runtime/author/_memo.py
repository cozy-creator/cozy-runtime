"""The result-cache declaration of a deterministic function."""

from collections.abc import Sequence
from dataclasses import dataclass

from ._errors import ConformanceError


@dataclass(frozen=True, slots=True, init=False)
class Memoize:
    """`memoize=True`, but a request is memoizable only when it explicitly provides each
    optional parameter named in `requires`; omitting one (or sending null) just runs it.
    Each named parameter defaults to None (`seed: int | None = None`)."""

    requires: tuple[str, ...]

    def __init__(self, *, requires: Sequence[str] = ()) -> None:
        if not isinstance(requires, (list, tuple)) or not all(
            isinstance(name, str) and name for name in requires
        ):
            raise ConformanceError(
                "Memoize requires a list or tuple of non-empty parameter names",
                code="invocable_memoize",
            )
        object.__setattr__(self, "requires", tuple(dict.fromkeys(requires)))


def memo_policy(memoize: bool | Memoize) -> Memoize | None:
    """One memo declaration: `True` is `Memoize()`, `False` is None."""
    if isinstance(memoize, Memoize):
        return memoize
    if type(memoize) is not bool:
        raise ConformanceError("memoize must be a boolean or Memoize", code="invocable_memoize")
    return Memoize() if memoize else None


def one_memo(invocable: Memoize | None, registration: Memoize | None) -> Memoize | None:
    """`memoize` is declared once: on `@invocable` or on the App registration."""
    if invocable is not None and registration is not None:
        raise ConformanceError(
            "declare memoize once, on invocable or App registration", code="invocable_memoize"
        )
    return registration if invocable is None else invocable
