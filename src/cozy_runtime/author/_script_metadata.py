"""Pure syntax and finite declarations shared by script execution and static describe."""

from __future__ import annotations

import ast
import operator
import tomllib
from collections.abc import Callable
from typing import Any

from cozy_runtime.author._assets import Asset, Tree
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._model_defaults import model_defaults
from cozy_runtime.author._walker import strip
from cozy_runtime.author._weights import WeightsOutput

_ARITHMETIC: dict[type[ast.operator], Callable[[Any, Any], object]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_INTEGER: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
    ast.BitXor: operator.xor,
}


class Unfolded(ValueError):
    """The node the constant folder stops at, and why."""

    def __init__(self, node: ast.AST, why: str) -> None:
        super().__init__(why)
        self.node = node


def fold_constant(node: ast.expr, other: Callable[[ast.expr], Any]) -> Any:
    """A literal/arithmetic expression's value, executing nothing; `other` values names and calls.

    The one evaluator: script execution and the static package reader both fold through it.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        items = [fold_constant(item, other) for item in node.elts]
        return tuple(items) if isinstance(node, ast.Tuple) else items
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            raise Unfolded(node, "a dict splat")
        return {
            fold_constant(key, other): fold_constant(value, other)
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Invert)):
        operand = fold_constant(node.operand, other)
        if isinstance(operand, bool) or not isinstance(operand, (int, float)):
            raise Unfolded(node, "unary arithmetic on a non-number")
        if isinstance(node.op, ast.USub):
            return -operand
        return +operand if isinstance(node.op, ast.UAdd) else ~int(operand)
    if isinstance(node, ast.BinOp):
        left, right = fold_constant(node.left, other), fold_constant(node.right, other)
        if isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str):
            return left + right
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (left, right)):
            raise Unfolded(node, "arithmetic on non-numbers")
        if (arithmetic := _ARITHMETIC.get(type(node.op))) is not None:
            return arithmetic(left, right)
        if isinstance(left, int) and isinstance(right, int) and type(node.op) in _INTEGER:
            return _INTEGER[type(node.op)](left, right)
        raise Unfolded(node, "an operator the constant folder does not evaluate")
    return other(node)


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


type Rungs = list[dict[str, Any]]


def declarations(raw: bytes) -> tuple[dict[str, Rungs], tuple[WeightsOutput, ...], bool]:
    """A script's `[tool.cozy]`: its model default ladders (a lane string is one `*` rung), its
    weights outputs and accelerator. A script without Model slots composes on CPU unless it
    declares `accelerator = true`."""
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
    declared = cozy.get("models", {})
    if not isinstance(declared, dict):
        raise ValueError("script models must map parameter names to default ladders")
    models = {
        name: [{"gpu": "*", "lane": value}] if isinstance(value, str) else value
        for name, value in declared.items()
    }
    model_defaults(models)
    weights = cozy.get("weights", {})
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
