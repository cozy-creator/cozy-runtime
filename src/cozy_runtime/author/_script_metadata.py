"""Pure syntax and finite declarations shared by script execution and static describe."""

from __future__ import annotations

import ast
import tomllib

from cozy_runtime.author._assets import Asset, Tree
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._walker import strip
from cozy_runtime.author._weights import WeightsOutput


def _contains_yield(node: ast.AST) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
        return False  # A nested generator does not make main itself a generator.
    if isinstance(node, (ast.Yield, ast.YieldFrom)):
        return True
    return any(_contains_yield(child) for child in ast.iter_child_nodes(node))


def main_definition(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef:
    mains = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
    ]
    if len(mains) != 1:
        raise ConformanceError(
            "script must define one main() or main(ctx)", code="script_main_missing"
        )
    if any(_contains_yield(statement) for statement in mains[0].body):
        raise ConformanceError(
            "script main must execute normally, not yield a generator",
            code="script_main_signature",
        )
    definition = mains[0]
    args = definition.args
    positional = [*args.posonlyargs, *args.args]
    parameters = [*positional, *args.kwonlyargs]
    if (
        args.vararg
        or args.kwarg
        or args.defaults
        or any(args.kw_defaults)
        or len(positional) > 1
        or (positional and positional[0].arg != "ctx")
        or any(p.arg.startswith("_script_") for p in parameters)
    ):
        raise ConformanceError(
            "script main accepts optional ctx and typed keyword-only injected parameters",
            code="script_main_signature",
        )
    return definition


def declarations(raw: bytes) -> tuple[dict[str, str], tuple[WeightsOutput, ...], bool]:
    """A script's `[tool.cozy]` models, weights outputs and accelerator. A script composes
    package calls on CPU unless it declares `accelerator = true`."""
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
    tool = tomllib.loads("\n".join(block)).get("tool", {})
    if not isinstance(tool, dict):
        raise ValueError("script tool metadata must be a table")
    cozy = tool.get("cozy", {})
    if not isinstance(cozy, dict) or set(cozy) - {"models", "weights", "accelerator"}:
        raise ValueError("script tool.cozy accepts models, weights and accelerator only")
    accelerator = cozy.get("accelerator", False)
    if type(accelerator) is not bool:
        raise ValueError("script accelerator must be a boolean")
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
    return (
        models,
        tuple(WeightsOutput(name, max_new_bytes=limit) for name, limit in weights.items()),
        accelerator,
    )


def byte_parameter(value: object) -> bool:
    base, _ = strip(value)
    return isinstance(base, type) and issubclass(base, (Asset, Tree))
