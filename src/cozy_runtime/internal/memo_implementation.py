"""A memoized callable's code identity is its release: the build of the distribution that
ships it, or local code's own files. Nothing is declared by the author."""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import os
import sys
from pathlib import Path

import msgspec

from cozy_runtime.author._handler import Handler
from cozy_runtime.internal import installed
from cozy_runtime.internal.memo_identity import distribution_identity, release_identity

#: Local code past this many bytes in its package is not keyed by content: it serves unmemoized.
_SOURCE_BYTES = 64 << 20


def describe(fn: Handler) -> dict[str, str]:
    """The operation identity of an entrypoint: its name and its distribution's build, as the
    static reader computes it (`static_interface._operation_identity`). An unreadable build (an
    editable install) disables the shortcut; it never refuses an invocation."""
    owner = _owner(fn.__module__.split(".")[0])
    identity = release_identity(
        f"{fn.__module__}.{fn.__qualname__}", owner or "", _build(owner) if owner else None
    )
    if identity is None:
        return {"operation_identity_unavailable": "operation_release_unavailable"}
    return {"operation_identity": identity}


def stage(fn: Handler) -> str | None:
    """A memoized Model method's code identity in this process: its name and its distribution's
    installed files by their recorded hashes, else its package's files by content (`_source`).
    None when neither can be read."""
    top = fn.__module__.split(".")[0]
    files = _recorded(top) or _source(top)
    return f"{fn.__module__}.{fn.__qualname__}@{files}" if files else None


def _recorded(top: str) -> str:
    owner = _owner(top)
    if owner is None:
        return ""
    try:
        return installed.record_digest(importlib.metadata.distribution(owner))
    except importlib.metadata.PackageNotFoundError:
        return ""


@functools.cache
def _source(top: str) -> str:
    """Local code's identity: each file of the top-level package `top` by path and content. TensorD
    installs local code editable (its RECORD names only a path file) from a tree it never changes
    under a generation (cr-302)."""
    module = sys.modules.get(top)
    roots = [Path(p) for p in getattr(module, "__path__", None) or ()]
    if not roots and (origin := getattr(module, "__file__", None)):
        roots = [Path(origin)]
    rows: list[tuple[str, str]] = []
    total = 0
    try:
        for root in roots:
            for path in _files(root):
                data = path.read_bytes()
                total += len(data)
                if total > _SOURCE_BYTES:
                    return ""
                name = path.name if path == root else path.relative_to(root).as_posix()
                rows.append((name, hashlib.sha256(data).hexdigest()))
    except OSError:
        return ""
    if not rows:
        return ""
    return "sha256:" + hashlib.sha256(msgspec.json.encode(sorted(rows))).hexdigest()


def _files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    found: list[Path] = []
    for directory, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
        kept = (n for n in names if not n.startswith(".") and not n.endswith(".pyc"))
        found += (Path(directory) / n for n in kept)
    return found


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
