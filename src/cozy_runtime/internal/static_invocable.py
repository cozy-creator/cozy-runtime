"""Managed-call declarations from function ASTs and the closed author vocabulary."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any, cast, get_origin

import msgspec

import cozy_runtime.internal.invocable_interface as invocable_interface
import cozy_runtime.internal.package_interface as package_interface
from cozy_runtime import author
from cozy_runtime.author._assets import Assets
from cozy_runtime.author._describe import assert_declaration_shape
from cozy_runtime.author._model_defaults import check_model_arguments
from cozy_runtime.author._services import JOB_ONLY, SERVICES, SERVING_ONLY, UNGRANTABLE
from cozy_runtime.author._signature import _classify, _wire_default, _wire_type, assets_binding
from cozy_runtime.author._walker import is_struct, strip
from cozy_runtime.internal.canonical import Json

if TYPE_CHECKING:
    from cozy_runtime.internal.static_interface import Reader, Registration


def build(reader: Reader, registration: Registration) -> dict[str, Json]:
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
    models: list[dict[str, Json]] = []
    contexts: list[str] = []
    capabilities: list[str] = []
    assets = None
    for index, arg in enumerate([*positional, *fn.args.kwonlyargs]):
        if arg.annotation is None:
            reader.refuse(module, arg, "an unannotated invocable parameter")
        annotation = arg.annotation
        model = reader.model_annotation(module, annotation)
        if model is not None:
            if arg.arg in defaults:
                reader.refuse(module, arg, "a model injection cannot have a default")
            models.append(
                package_interface.model_slot(
                    f"{registration.name}.models.{arg.arg}",
                    model.name,
                    model.encoded_leaves,
                    model.fusion,
                    model.components,
                    model.sequence_parallel,
                    defaults=registration.model_defaults.get(arg.arg, ()),
                )
            )
            fields.append((arg.arg, author.ModelArtifact | None, None))
            parameters.append(arg.arg)
            continue
        translated = reader.translate(module, annotation)
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
                capabilities.append(parameter.capability)
            continue
        if arg not in fn.args.kwonlyargs:
            reader.refuse(module, arg, "invocable operation arguments must be keyword-only")
        parameters.append(arg.arg)
        if base is Assets or get_origin(base) is Assets:
            if assets is not None or arg.arg in defaults:
                reader.refuse(module, arg, "one Assets parameter without a default is supported")
            assets = assets_binding(arg.arg, translated)
            fields.append((arg.arg, assets.wire_type))
            if assets.decoded:
                capabilities.append("media_decode")
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
    if len(contexts) != 1 or contexts[0] != positional[0].arg:
        reader.refuse(module, fn, "an invocable must have exactly one Context, first")
    if len(capabilities) != len(set(capabilities)):
        reader.refuse(module, fn, "an injected service is requested twice")
    forbidden = (SERVING_ONLY if registration.kind == "job" else JOB_ONLY) | set(UNGRANTABLE)
    if set(capabilities) & forbidden:
        reader.refuse(
            module, fn, "invocable requests a capability unavailable to its callable kind"
        )
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
    assert_declaration_shape(payload)
    assert_declaration_shape(result)
    doc = package_interface.callable_doc(
        name=registration.name,
        kind=registration.kind,
        payload_type=payload,
        result_type=result,
        models=models,
        assets=assets,
        publishes=registration.publishes,
        weights_outputs=registration.weights,
        accelerator=registration.accelerator,
    )
    if registration.weights and "weights" not in capabilities:
        capabilities.append("weights")
    check_model_arguments(
        registration.model_defaults,
        [str(slot["path"]).rsplit(".", 1)[-1] for slot in models],
    )
    metadata = invocable_interface.from_types(
        payload,
        result,
        module=module.name,
        export=fn.name,
        context=contexts[0],
        parameters=parameters,
        parameter_defaults=explicit_defaults,
        memoize=registration.memoize,
        capabilities=capabilities,
    )
    doc["invocable"] = cast(Json, metadata)
    if registration.kind == "entrypoint":
        invocable_interface.project_serving_models(doc)
    invocable_interface.validate(
        msgspec.convert(doc["invocable"], invocable_interface.Invocable, strict=True),
        doc["request"],
        doc["result"],
        kind=registration.kind,
    )
    return doc
