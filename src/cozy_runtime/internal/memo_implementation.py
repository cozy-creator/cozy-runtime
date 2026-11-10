"""A memoized callable's code identity is its release: the build of the distribution that
ships it. Nothing is declared by the author."""

from __future__ import annotations

import functools
import importlib.metadata
import sys

from cozy_runtime.author._handler import Handler
from cozy_runtime.internal import installed
from cozy_runtime.internal.memo_identity import distribution_identity, release_identity


def describe(fn: Handler) -> dict[str, str]:
    """The operation identity of an entrypoint: its name and its distribution's build, as the
    static reader computes it (`static_interface._operation_identity`). An unreadable build (an
    editable install) disables the shortcut; it never refuses an invocation."""
    export = getattr(fn, "_cozy_export", None)
    if export is not None:
        fn = export.implementation
    owner = _owner(fn.__module__.split(".")[0])
    identity = release_identity(
        f"{fn.__module__}.{fn.__qualname__}", owner or "", _build(owner) if owner else None
    )
    if identity is None:
        return {"operation_identity_unavailable": "operation_release_unavailable"}
    return {"operation_identity": identity}


def stage(fn: Handler) -> str | None:
    """A memoized Model method's code identity in this process: its name and its distribution's
    installed files by their recorded hashes. None for an editable or unrecorded install."""
    owner = _owner(fn.__module__.split(".")[0])
    if owner is None:
        return None
    try:
        files = installed.record_digest(importlib.metadata.distribution(owner))
    except importlib.metadata.PackageNotFoundError:
        return None
    return f"{fn.__module__}.{fn.__qualname__}@{files}" if files else None


@functools.cache
def _owners() -> dict[str, list[str]]:
    return dict(importlib.metadata.packages_distributions())


def _owner(top: str) -> str | None:
    """The one distribution that ships the top-level module `top`."""
    found = sorted(set(_owners().get(top, ())))
    return found[0] if len(found) == 1 else None


def _build(name: str) -> str | None:
    """Explicit build metadata; never enumerate RECORD or hash binaries."""
    try:
        distribution = importlib.metadata.distribution(name)
        return distribution_identity(
            distribution.version,
            distribution.read_text("WHEEL"),
            distribution.read_text("direct_url.json"),
            sys.implementation.cache_tag,
        )
    except (importlib.metadata.PackageNotFoundError, OSError, TypeError, ValueError):
        return None
