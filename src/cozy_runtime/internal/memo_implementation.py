"""Operation cache invalidation, independent of package installation and routing."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import importlib.util
import inspect
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from cozy_runtime.author._memo import MemoDependency, MemoDistribution, MemoResource

from cozy_runtime.internal.memo_identity import (
    MAX_SOURCE_BYTES,
    definitions_identity,
    distribution_identity,
    normalized,
)


def describe(fn: Callable[..., object]) -> dict[str, str]:
    """Hash Python behavior; authors may salt external behavior and declare helpers.

    Only this callable and those declared helpers enter its key. Install handles,
    caller scripts, package inventories and client SDKs never participate. Missing
    source disables the shortcut; it cannot refuse an otherwise valid invocation.
    """
    declaration: tuple[str, tuple[MemoDependency, ...]] = getattr(
        fn, "_cozy_memo_declaration", ("", ())
    )
    version, dependencies = declaration
    try:
        definitions = [_definition(fn)]
        for dependency in dependencies:
            if isinstance(dependency, MemoResource):
                name = PurePosixPath(dependency.name)
                if name.is_absolute() or ".." in name.parts:
                    return {"operation_identity_unavailable": "operation_resource_unavailable"}
                digest = hashlib.sha256()
                try:
                    with (
                        importlib.resources.files(dependency.package)
                        .joinpath(dependency.name)
                        .open("rb") as stream
                    ):
                        for chunk in iter(lambda: stream.read(1 << 20), b""):
                            digest.update(chunk)
                except (ImportError, OSError, TypeError, ValueError):
                    return {"operation_identity_unavailable": "operation_resource_unavailable"}
                definitions.append(
                    ("resource:" + dependency.package + "/" + dependency.name, digest.hexdigest())
                )
            elif isinstance(dependency, MemoDistribution):
                identity = _native_identity(dependency.name)
                if identity is None:
                    return {
                        "operation_identity_unavailable": "operation_native_identity_unavailable"
                    }
                definitions.append(("distribution:" + dependency.name, identity))
            elif isinstance(dependency, str):
                spec = importlib.util.find_spec(dependency)
                if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
                    raise ValueError("helper source unavailable")
                with Path(spec.origin).open("rb") as stream:
                    source = stream.read(MAX_SOURCE_BYTES + 1)
                if len(source) > MAX_SOURCE_BYTES:
                    raise ValueError("helper source exceeds bound")
                definitions.append((dependency, normalized(source)))
            else:
                definitions.append(_definition(dependency))
    except (ImportError, OSError, TypeError, ValueError, SyntaxError):
        return {"operation_identity_unavailable": "operation_source_unavailable"}
    return {"operation_identity": definitions_identity(version, definitions)}


def _definition(fn: Callable[..., object]) -> tuple[str, str]:
    export = getattr(fn, "_cozy_export", None)
    if export is not None:
        fn = export.implementation
    source = textwrap.dedent(inspect.getsource(fn))
    if len(source.encode()) > MAX_SOURCE_BYTES:
        raise ValueError("operation source exceeds bound")
    return (fn.__module__ + "." + fn.__qualname__, normalized(source, callable_only=True))


def _native_identity(name: str) -> str | None:
    """Cheap explicit native metadata; never enumerate RECORD or hash binaries."""
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
