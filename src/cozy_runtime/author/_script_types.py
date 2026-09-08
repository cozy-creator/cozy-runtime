"""Read a script's declared types and finite grants without executing its body."""

from __future__ import annotations

import ast
import builtins
import importlib
import tomllib
from typing import Any, get_origin

from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._loader import Loader
from cozy_runtime.author._model import Model
from cozy_runtime.author._weights import WeightsOutput


class _ScriptSource(Model[object]):
    """A derive-only source; inspecting its manifest does not construct inference."""

    def load(self, loader: Loader) -> None:
        pass


def annotation(node: ast.expr, tree: ast.Module) -> Any:
    """Resolve type expressions from explicit imports, never eval script statements."""
    if isinstance(node, ast.Constant):
        if node.value is None:
            return type(None)
        if isinstance(node.value, str):
            return annotation(ast.parse(node.value, mode="eval").body, tree)
    if isinstance(node, ast.Name):
        if node.id in ("list", "tuple", "dict", "str", "int", "float", "bool", "bytes", "object"):
            return getattr(builtins, node.id)
        for statement in tree.body:
            if isinstance(statement, ast.ImportFrom) and statement.module and not statement.level:
                for alias in statement.names:
                    if (alias.asname or alias.name) == node.id:
                        return getattr(importlib.import_module(statement.module), alias.name)
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    if (alias.asname or alias.name.split(".")[0]) == node.id:
                        return importlib.import_module(alias.name if alias.asname else node.id)
    if isinstance(node, ast.Attribute) and not node.attr.startswith("_"):
        return getattr(annotation(node.value, tree), node.attr)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return annotation(node.left, tree) | annotation(node.right, tree)
    if isinstance(node, ast.Subscript):
        base = annotation(node.value, tree)
        args = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        values = tuple(annotation(arg, tree) for arg in args)
        return base[values[0] if len(values) == 1 else values]
    raise ConformanceError(
        "script annotations must use imported types, not script expressions or local classes",
        code="script_main_signature",
    )


def parameter_type(node: ast.expr, tree: ast.Module) -> Any:
    value = annotation(node, tree)
    return _ScriptSource if value is Model or get_origin(value) is Model else value


def declarations(raw: bytes) -> tuple[dict[str, str], tuple[WeightsOutput, ...]]:
    block: list[str] = []
    opened = found = False
    for line in raw.decode("utf-8").splitlines():
        if line == "# /// script":
            if opened or found:
                raise ValueError("multiple script metadata blocks")
            opened = True
        elif opened and line == "# ///":
            opened, found = False, True
        elif opened:
            if not (line == "#" or line.startswith("# ")):
                raise ValueError("script metadata must contain comment lines")
            block.append(line[2:])
    if opened:
        raise ValueError("unclosed script metadata")
    cozy = tomllib.loads("\n".join(block)).get("tool", {}).get("cozy", {})
    if not isinstance(cozy, dict) or set(cozy) - {"models", "weights"}:
        raise ValueError("script tool.cozy accepts models and weights only")
    models = cozy.get("models", {})
    weights = cozy.get("weights", {})
    if not isinstance(models, dict) or any(
        not name.isidentifier() or not isinstance(value, str) or not value
        for name, value in models.items()
    ):
        raise ValueError("script models must map parameter names to default references")
    if (
        not isinstance(weights, dict)
        or len(weights) > 16
        or any(
            not name.isidentifier() or type(value) is not int or not 0 < value <= (1 << 63) - 1
            for name, value in weights.items()
        )
    ):
        raise ValueError("script weights must map output names to finite positive byte limits")
    return models, tuple(
        WeightsOutput(name, max_new_bytes=limit) for name, limit in weights.items()
    )
