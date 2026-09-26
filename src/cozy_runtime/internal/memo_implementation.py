"""Operation cache invalidation, independent of package installation and routing."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import textwrap
from collections.abc import Callable
from pathlib import Path

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
            if isinstance(dependency, str):
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
