"""Read a script's declared types and finite grants without executing its body."""

from __future__ import annotations

import ast
import builtins
import importlib
from typing import Any, get_origin

from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._loader import Loader
from cozy_runtime.author._markers import AssetBound
from cozy_runtime.author._model import Model


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
    if isinstance(node, ast.Call) and annotation(node.func, tree) is AssetBound:
        names = [keyword.arg for keyword in node.keywords]
        if None not in names and len(set(names)) == len(names):
            return AssetBound(
                *(ast.literal_eval(arg) for arg in node.args),
                **{
                    keyword.arg: ast.literal_eval(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg is not None
                },
            )
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
