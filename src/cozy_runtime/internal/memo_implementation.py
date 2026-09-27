"""Operation cache invalidation, independent of package installation and routing."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.resources
import importlib.util
import inspect
import json
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from cozy_runtime.author._memo import MemoDistribution, MemoResource

MAX_SOURCE_BYTES = 1 << 20


def describe(fn: Callable[..., object]) -> dict[str, str]:
    """Hash Python behavior; authors may salt external behavior and declare helpers.

    Only this callable and those declared helpers enter its key. Install handles,
    caller scripts, package inventories and client SDKs never participate. Missing
    source disables the shortcut; it cannot refuse an otherwise valid invocation.
    """
    version, dependencies = getattr(fn, "_cozy_memo_declaration", ("", ()))
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
                definitions.append((dependency, _normalized(source)))
            else:
                definitions.append(_definition(dependency))
        document = json.dumps(
            {"operation_version": version, "definitions": definitions},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    except (ImportError, OSError, TypeError, ValueError, SyntaxError):
        return {"operation_identity_unavailable": "operation_source_unavailable"}
    return {"operation_identity": "sha256:" + hashlib.sha256(document).hexdigest()}


def _definition(fn: Callable[..., object]) -> tuple[str, str]:
    export = getattr(fn, "_cozy_export", None)
    if export is not None:
        fn = export.implementation
    source = textwrap.dedent(inspect.getsource(fn))
    if len(source.encode()) > MAX_SOURCE_BYTES:
        raise ValueError("operation source exceeds bound")
    return (fn.__module__ + "." + fn.__qualname__, _normalized(source, callable_only=True))


def _normalized(source: str | bytes, *, callable_only: bool = False) -> str:
    node = ast.parse(source)
    if callable_only:
        if len(node.body) != 1 or not isinstance(
            node.body[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            raise ValueError("operation source is not one definition")
        node.body[0].decorator_list = []
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _native_identity(name: str) -> str | None:
    """Cheap explicit native metadata; never enumerate RECORD or hash binaries."""
    try:
        distribution = importlib.metadata.distribution(name)
        version = distribution.version
        wheel = distribution.read_text("WHEEL")
        if not version or not wheel:
            return None
        tags = sorted(line for line in wheel.splitlines() if line.startswith(("Tag:", "Build:")))
        if not tags:
            return None
        provenance: dict[str, object] = {}
        direct = distribution.read_text("direct_url.json")
        if direct:
            source = json.loads(direct)
            if not isinstance(source, dict) or "dir_info" in source:
                return None
            archive = source.get("archive_info", {})
            vcs = source.get("vcs_info", {})
            if not isinstance(archive, dict) or not isinstance(vcs, dict):
                return None
            if archive.get("hashes"):
                provenance["archive_hashes"] = archive["hashes"]
            elif archive.get("hash"):
                provenance["archive_hash"] = archive["hash"]
            if vcs.get("commit_id"):
                provenance["commit"] = vcs["commit_id"]
            # uv may record a local immutable wheel with archive_info={}.
            # Its version and wheel tags still supply the declared native build
            # contract; include archive hashes when the installer retains them.
            if "vcs_info" in source and not vcs.get("commit_id"):
                return None
            if "archive_info" not in source and "vcs_info" not in source:
                return None
        return json.dumps(
            {
                "version": version,
                "wheel": tags,
                "build": provenance,
                "abi": sys.implementation.cache_tag,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    except (importlib.metadata.PackageNotFoundError, OSError, TypeError, ValueError):
        return None
