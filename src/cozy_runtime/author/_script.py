"""Ordinary client scripts, adapted to the existing admitted execution boundary."""

from __future__ import annotations

import ast
import asyncio
import importlib
import importlib.util
import inspect
from pathlib import Path

import msgspec

from cozy_runtime.author._app import App
from cozy_runtime.author._context import Context
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._services import Telemetry


class ScriptContext:
    """Optional cancellation and reporting tools passed to a script's main(ctx)."""

    def __init__(self, context: Context, telemetry: Telemetry):
        self._context = context
        self._telemetry = telemetry

    @property
    def request_id(self) -> str:
        return self._context.request_id

    @property
    def cancelled(self) -> bool:
        return self._context.cancelled

    def raise_if_cancelled(self) -> None:
        self._context.raise_if_cancelled()

    def log(self, message: str, *, level: str = "info") -> None:
        self._telemetry.log(message, level=level)

    def progress(self, fraction: float, *, stage: str | None = None) -> None:
        self._telemetry.progress(fraction, stage=stage)

    def metric(self, name: str, value: float, *, unit: str | None = None) -> None:
        self._telemetry.metric(name, value, unit=unit)


class _ScriptRequest(msgspec.Struct):
    pass


class _ScriptCompleted(msgspec.Struct):
    pass


def _contains_yield(node: ast.AST) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
        return False  # A nested generator does not make main itself a generator.
    if isinstance(node, (ast.Yield, ast.YieldFrom)):
        return True
    return any(_contains_yield(child) for child in ast.iter_child_nodes(node))


def script_app(module: str) -> App:
    """Creator's generated adapter. Discovery reads syntax without executing the script.

    The script and imports are executed only inside the admitted remote/local
    attempt. Authors provide main() or main(ctx), synchronous or asynchronous.
    """
    if not module.isidentifier():
        raise ConformanceError("script adapter requires one module", code="script_main_missing")
    found = importlib.util.find_spec(module)
    if found is None or not found.origin:
        raise ConformanceError("script module is absent", code="script_main_missing")
    with Path(found.origin).open("rb") as source:
        raw = source.read((32 << 20) + 1)
    if len(raw) > 32 << 20:
        raise ConformanceError("script exceeds its source bound", code="script_source_too_large")
    try:
        tree = ast.parse(raw, filename=found.origin)
    except SyntaxError as exc:
        raise ConformanceError("script syntax is invalid", code="script_syntax") from exc
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
    args = mains[0].args
    positional = [*args.posonlyargs, *args.args]
    parameters = [*positional, *args.kwonlyargs]
    if (
        args.vararg
        or args.kwarg
        or len(parameters) > 1
        or (parameters and parameters[0].arg != "ctx")
    ):
        raise ConformanceError(
            "script entrypoint must be main() or main(ctx)", code="script_main_signature"
        )
    with_context = bool(parameters)
    keyword_context = bool(args.kwonlyargs)
    app = App()

    @app.job
    def main(ctx: Context, payload: _ScriptRequest, telemetry: Telemetry) -> _ScriptCompleted:
        ctx.raise_if_cancelled()
        function = getattr(importlib.import_module(module), "main", None)
        if not callable(function):
            raise ConformanceError("script main is not callable", code="script_main_signature")
        context = ScriptContext(ctx, telemetry)
        if with_context:
            result = function(ctx=context) if keyword_context else function(context)
        else:
            result = function()
        if inspect.isgenerator(result) or inspect.isasyncgen(result):
            raise ConformanceError(
                "script main returned an unexecuted generator", code="script_main_signature"
            )
        if inspect.isawaitable(result):

            async def wait() -> None:
                await result

            asyncio.run(wait())
        ctx.raise_if_cancelled()
        return _ScriptCompleted()

    return app
