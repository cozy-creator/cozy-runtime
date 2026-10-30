"""Managed-call declarations from function ASTs and the closed author vocabulary."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any, get_origin

import msgspec

from cozy_runtime import author
from cozy_runtime.author._files import Files
from cozy_runtime.author._services import SERVICES
from cozy_runtime.author._signature import (
    Declaration,
    Slot,
    _classify,
    _wire_default,
    _wire_type,
    assets_binding,
    check_parameter_markers,
)
from cozy_runtime.author._walker import is_struct, strip

if TYPE_CHECKING:
    from cozy_runtime.internal.static_interface import Reader, Registration


def build(reader: Reader, registration: Registration) -> Declaration:
    module, fn = registration.module, registration.fn
    positional = [*fn.args.posonlyargs, *fn.args.args]
    if (
        not isinstance(fn, ast.AsyncFunctionDef)
        or not positional
        or fn.args.vararg
        or fn.args.kwarg
    ):
        reader.refuse(
            module,
            fn,
            "an invocable must be async with Context first and no variadic parameters",
            "static_unsupported",
        )
    if fn.args.defaults:
        reader.refuse(
            module, fn, "injected invocable parameters cannot have defaults", "static_unsupported"
        )
    defaults = {
        arg.arg: value
        for arg, value in zip(fn.args.kwonlyargs, fn.args.kw_defaults, strict=True)
        if value is not None
    }
    fields: list[Any] = []
    parameters: list[str] = []
    explicit_defaults: dict[str, object] = {}
    slots: list[Slot] = []
    contexts: list[str] = []
    services: list[str] = []
    assets = None
    for index, arg in enumerate([*positional, *fn.args.kwonlyargs]):
        if arg.annotation is None:
            reader.refuse(module, arg, "an unannotated invocable parameter")
        annotation = arg.annotation
        override = registration.model_classes.get(arg.arg)
        found = (override, False) if override else reader.model_annotation(module, annotation)
        if found is not None:
            if arg.arg in defaults:
                reader.refuse(module, arg, "a model injection cannot have a default")
            slots.append(Slot(arg.arg, *found))
            fields.append((arg.arg, author.ModelArtifact | None, None))
            parameters.append(arg.arg)
            continue
        translated = reader.translate(module, annotation)
        check_parameter_markers(translated)
        base, _ = strip(translated)
        injected = (
            base is author.Context
            or (isinstance(base, type) and any(issubclass(base, cls) for cls in SERVICES.values()))
            or get_origin(base) in (author.Settings, author.Secrets)
        )
        if injected:
            if arg.arg in defaults:
                reader.refuse(module, arg, "injected invocable parameters cannot have defaults")
            parameter = _classify(arg.arg, translated, f"{module.name}:{fn.name}")
            if parameter.role == "context":
                if index != 0:
                    reader.refuse(module, arg, "Context must be the first invocable parameter")
                contexts.append(arg.arg)
            if parameter.capability:
                services.append(parameter.capability)
            continue
        if arg not in fn.args.kwonlyargs:
            reader.refuse(module, arg, "invocable operation arguments must be keyword-only")
        parameters.append(arg.arg)
        if base is Files or get_origin(base) is Files:
            if assets is not None or arg.arg in defaults:
                reader.refuse(module, arg, "one Files parameter without a default is supported")
            assets = assets_binding(arg.arg, translated)
            fields.append((arg.arg, assets.wire_type))
            continue
        _wire_type(translated)
        if arg.arg in defaults:
            default = reader.fold(module, defaults[arg.arg])
            _wire_default(default)
            try:
                msgspec.convert(default, type=translated, strict=True)
            except (TypeError, ValueError, msgspec.ValidationError) as error:
                reader.refuse(
                    module, defaults[arg.arg], f"invocable default differs from its type: {error}"
                )
            fields.append((arg.arg, translated, default))
            explicit_defaults[arg.arg] = default
        else:
            fields.append((arg.arg, translated))
    if not contexts:
        reader.refuse(module, fn, "an invocable must have exactly one Context, first")
    payload = msgspec.defstruct(
        f"{fn.name}Request",
        fields,
        module=module.name,
        kw_only=True,
        frozen=True,
        forbid_unknown_fields=True,
    )
    if fn.returns is None:
        reader.refuse(module, fn, "an invocable needs a result annotation")
    result = reader.translate(module, fn.returns)
    _wire_type(result)
    result, _ = strip(result)
    if not is_struct(result):
        reader.refuse(module, fn.returns, "an invocable result must be a msgspec.Struct")
    return registration.declaration(
        payload_type=payload,
        result_type=result,
        contexts=tuple(contexts),
        services=tuple(services),
        slots=tuple(slots),
        assets=assets,
        parameters=tuple(parameters),
        parameter_defaults=explicit_defaults,
    )
