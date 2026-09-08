"""Ordinary client scripts, adapted to the existing admitted execution boundary."""

from __future__ import annotations

import ast
import asyncio
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any

import msgspec

from cozy_runtime.author._app import App
from cozy_runtime.author._context import Context
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._script_metadata import declarations, main_definition
from cozy_runtime.author._script_types import annotation, parameter_type
from cozy_runtime.author._services import Telemetry
from cozy_runtime.author._signature import _classify


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
    definition = main_definition(tree)
    args = definition.args
    positional = [*args.posonlyargs, *args.args]
    parameters = [*positional, *args.kwonlyargs]
    types: dict[str, Any] = {}
    try:
        models, weights = declarations(raw)
        for parameter in parameters:
            if parameter.arg == "ctx":
                continue
            if parameter.annotation is None:
                raise ValueError("injected parameters need a type")
            value = parameter_type(parameter.annotation, tree)
            if _classify(parameter.arg, value, "script main").role not in {"model", "service"}:
                raise ValueError("script parameter must declare a model or service")
            types[parameter.arg] = value
        if set(models) - {
            name
            for name, value in types.items()
            if _classify(name, value, "script main").role == "model"
        }:
            raise ValueError("model default names an undeclared model parameter")
        returned = annotation(definition.returns, tree) if definition.returns else type(None)
    except Exception as exc:
        raise ConformanceError(
            "script main types or tool.cozy declarations are invalid",
            code="script_main_signature",
        ) from exc
    result_type = (
        msgspec.defstruct("_ScriptResult", [("value", returned)], module=__name__)
        if returned is not type(None)
        else _ScriptCompleted
    )
    telemetry_name = next(
        (name for name, value in types.items() if value is Telemetry), "_script_telemetry"
    )
    emits_media = any(
        _classify(name, value, "script main").capability == "save" for name, value in types.items()
    )
    app = App()

    def main(**injected: Any) -> Any:
        ctx, telemetry = injected.pop("_script_context"), injected[telemetry_name]
        if telemetry_name == "_script_telemetry":
            injected.pop(telemetry_name)
        injected.pop("_script_request")
        ctx.raise_if_cancelled()
        result: Any = None
        try:
            function = getattr(importlib.import_module(module), "main", None)
            if not callable(function):
                raise ConformanceError("script main is not callable", code="script_main_signature")
            context = ScriptContext(ctx, telemetry)
            if positional:
                result = function(context, **injected)
            else:
                if any(p.arg == "ctx" for p in parameters):
                    injected["ctx"] = context
                result = function(**injected)
            if inspect.isgenerator(result) or inspect.isasyncgen(result):
                raise ConformanceError(
                    "script main returned an unexecuted generator", code="script_main_signature"
                )
            if inspect.isawaitable(result):
                pending = result

                async def wait() -> Any:
                    return await pending

                result = asyncio.run(wait())
        except SystemExit as exc:
            if exc.code is not None and exc.code != 0:
                raise ConformanceError(
                    "script exited with a nonzero status", code="script_exit"
                ) from exc
        ctx.raise_if_cancelled()
        return result_type(result) if returned is not type(None) else result_type()

    # Reuse normal signature analysis, injection, grants and output validation.
    # No custom invocation path or authority is introduced by a plain script.
    types = {
        "_script_context": Context,
        "_script_request": _ScriptRequest,
        telemetry_name: Telemetry,
        **types,
    }
    main.__annotations__ = {**types, "return": result_type}
    main.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=value)
            for name, value in types.items()
        ],
        return_annotation=result_type,
    )
    app.job(weights=weights, emits_media=emits_media)(main)
    return app
