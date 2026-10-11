"""Explicit, inert dependencies of a deterministic operation."""

from collections.abc import Callable, Sequence
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


@dataclass(frozen=True, slots=True)
class MemoResource:
    """One named package resource whose bytes affect the operation's result."""

    package: str
    name: str


@dataclass(frozen=True, slots=True)
class MemoDistribution:
    """A native dependency's installed version and immutable build provenance.

    This reads distribution metadata, never its file inventory or native binaries.
    Editable or unavailable metadata disables memo reuse, not execution.
    """

    name: str


type MemoDependency = Callable[..., object] | str | MemoResource | MemoDistribution


def declare(
    fn: Callable[..., object],
    *,
    memoize: bool,
    version: str | None,
    dependencies: tuple[MemoDependency, ...],
) -> None:
    if version is not None and (not memoize or not isinstance(version, str) or len(version) > 128):
        raise ConformanceError(
            "memo_version needs memoize=True and a bounded version", code="invocable_memoize"
        )
    if (
        not isinstance(dependencies, tuple)
        or len(dependencies) > 64
        or any(
            not callable(item) and not isinstance(item, (str, MemoResource, MemoDistribution))
            for item in dependencies
        )
        or (dependencies and not memoize)
    ):
        raise ConformanceError(
            "memo_dependencies needs memoize=True and bounded dependencies",
            code="invocable_memoize",
        )
    if version is not None or dependencies:
        declaration = (version or "", dependencies)
        previous = getattr(fn, "_cozy_memo_declaration", declaration)
        if previous != declaration:
            raise ConformanceError("declare memo behavior once", code="invocable_memoize")
        fn.__dict__["_cozy_memo_declaration"] = declaration
