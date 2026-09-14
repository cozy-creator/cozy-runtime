"""Source-stable Python names for the existing canonical callable schemas."""

from __future__ import annotations

import enum
import inspect
import keyword
import types
import typing
from collections.abc import Mapping, Sequence
from typing import Any, get_args, get_origin

import msgspec

from cozy_runtime.author import ConformanceError, Surface
from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._walker import is_struct, strip
from cozy_runtime.internal import canonical, schema


def metadata(surface: Surface) -> dict[str, Any]:
    if surface.kind == "entrypoint" and not surface.invocable:
        return serving_metadata(
            surface.payload_type,
            surface.result_type,
            module=surface.fn.__module__,
            export=surface.fn.__name__,
            context=next((p.name for p in surface.params if p.role == "context"), ""),
            capabilities=sorted(surface.capabilities),
            memoize=surface.memoize,
        )
    parameters = [
        parameter.name
        for parameter in surface.params
        if parameter.role in ("wire", "model", "assets")
    ]
    signature = inspect.signature(surface.fn)
    defaults = {
        name: signature.parameters[name].default
        for name in parameters
        if signature.parameters[name].default is not inspect.Parameter.empty
    }
    return from_types(
        surface.payload_type,
        surface.result_type,
        module=surface.fn.__module__,
        export=surface.fn.__name__,
        context=next(p.name for p in surface.params if p.role == "context"),
        parameters=parameters,
        parameter_defaults=defaults,
        memoize=surface.memoize,
        capabilities=sorted(surface.capabilities),
    )


def serving_metadata(
    payload_type: object,
    result_type: object,
    *,
    module: str,
    export: str,
    context: str,
    capabilities: Sequence[str],
    memoize: bool = False,
) -> dict[str, Any]:
    """Caller names over the existing serving wire schema; never change its handler."""
    return from_types(
        payload_type,
        result_type,
        module=module,
        export=export,
        context=context,
        parameters=[
            field.encode_name
            for field in msgspec.structs.fields(typing.cast(type[msgspec.Struct], payload_type))
        ],
        parameter_defaults={},
        memoize=memoize,
        capabilities=capabilities,
        serving=True,
    )


def from_types(
    payload_type: object,
    result_type: object,
    *,
    module: str,
    export: str,
    context: str,
    parameters: Sequence[str],
    parameter_defaults: Mapping[str, object],
    memoize: bool,
    capabilities: Sequence[str],
    serving: bool = False,
) -> dict[str, Any]:
    """Portable names/defaults from resolved types, shared by both interface readers."""
    defaults: dict[str, Any] = {}
    names: dict[str, str] = {}
    enums: dict[str, list[str]] = {}

    def walk(annotation: object, path: str) -> None:
        base, _ = strip(annotation)
        if base is ModelArtifact:
            return
        origin = get_origin(base)
        if isinstance(base, type) and is_struct(base):
            names[path] = base.__name__
            for field in msgspec.structs.fields(base):
                if (not serving and field.name != field.encode_name) or not _identifier(
                    field.encode_name
                ):
                    raise ConformanceError(
                        "invocable struct fields need portable Python names", code="invocable_field"
                    )
                nested = path + "/" + field.encode_name
                walk(field.type, nested)
                if field.default is not msgspec.NODEFAULT:
                    default = msgspec.to_builtins(field.default)
                    canonical.write(default)
                    defaults[nested] = default
                elif serving and field.default_factory in (list, dict):
                    defaults[nested] = field.default_factory()
        elif isinstance(base, type) and issubclass(base, enum.Enum):
            names[path] = base.__name__
            members = sorted(base, key=lambda member: canonical.write(member.value))
            enums[path] = [member.name for member in members]
        elif origin is list or (
            origin is tuple and len(get_args(base)) == 2 and get_args(base)[1] is Ellipsis
        ):
            walk(get_args(base)[0], path + "/[]")
        elif serving and origin is dict:
            walk(get_args(base)[0], path + "/map/key")
            walk(get_args(base)[1], path + "/map/value")
        elif serving and origin is tuple:
            args = get_args(base)
            if len(args) == 2 and args[1] is Ellipsis:
                walk(args[0], path + "/[]")
            else:
                for index, arg in enumerate(args):
                    walk(arg, path + f"/tuple/{index}")
        elif origin in (types.UnionType, typing.Union):
            for index, member in enumerate(
                sorted(get_args(base), key=lambda item: canonical.write(schema.render(item)))
            ):
                walk(member, path + f"/union/{index}")

    walk(payload_type, "request")
    walk(result_type, "result")
    defaults.update({"request/" + name: value for name, value in parameter_defaults.items()})
    return {
        "module": module,
        "export": export,
        "context": context,
        "parameters": list(parameters),
        "defaults": defaults,
        "type_names": names,
        "enum_members": enums,
        "memoize": memoize,
        "capabilities": sorted(capabilities),
    }


def _identifier(value: object) -> bool:
    return isinstance(value, str) and value.isidentifier() and not keyword.iskeyword(value)


def project_serving_models(document: dict[str, Any]) -> None:
    """Expose injected Model parameters once, in the serving Model plane.

    Only the trusted builders of registered @invocable entrypoints call this.
    The source proxy retains its original typed flat request for self calls.
    Ordinary request structs with colliding field names still fail generation.
    """
    parameters = {model["path"].rpartition(".")[2] for model in document.get("models", [])}
    if not parameters:
        return
    fields = document["request"]["fields"]
    for parameter in parameters:
        field = next((field for field in fields if field["name"] == parameter), None)
        if field is None or field["type"] != {"union": ["null", {"input": "model"}]}:
            raise ConformanceError(
                "serving Model parameter lost its typed declaration", code="invocable_interface"
            )
    document["request"]["fields"] = [field for field in fields if field["name"] not in parameters]
    metadata = document["invocable"]
    metadata["parameters"] = [name for name in metadata["parameters"] if name not in parameters]
    for field in ("defaults", "type_names", "enum_members"):
        metadata[field] = {
            path: value
            for path, value in metadata[field].items()
            if not any(
                path == "request/" + parameter or path.startswith("request/" + parameter + "/")
                for parameter in parameters
            )
        }


def validate(value: object, request: Any, result: Any, *, kind: str = "job") -> dict[str, Any]:
    """Validate every token the pure generator can turn into Python source."""
    keys = {
        "module",
        "export",
        "context",
        "parameters",
        "defaults",
        "type_names",
        "enum_members",
        "memoize",
        "capabilities",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ConformanceError(
            "invocable metadata has unknown or missing fields", code="invocable_interface"
        )
    if type(value["memoize"]) is not bool:
        raise ConformanceError(
            "invocable memoization declaration must be a boolean", code="invocable_interface"
        )
    capabilities = value["capabilities"]
    if (
        not isinstance(capabilities, list)
        or not all(isinstance(capability, str) for capability in capabilities)
        or capabilities != sorted(set(capabilities))
    ):
        raise ConformanceError(
            "invocable capabilities must be sorted unique names", code="invocable_interface"
        )
    if value["memoize"] and set(capabilities) & {"egress", "secrets"}:
        raise ConformanceError(
            "memoized interface declares external or secret-dependent work",
            code="invocable_memoize_effect",
        )
    if (
        not isinstance(value["module"], str)
        or not all(_identifier(part) for part in value["module"].split("."))
        or not _identifier(value["export"])
        or not (_identifier(value["context"]) or (kind == "entrypoint" and value["context"] == ""))
    ):
        raise ConformanceError(
            "invocable export is not a Python module function", code="invocable_interface"
        )
    parameters = value["parameters"]
    if (
        not isinstance(parameters, list)
        or not all(_identifier(name) for name in parameters)
        or len(set(parameters)) != len(parameters)
        or value["context"] in parameters
    ):
        raise ConformanceError(
            "invocable parameters are not unique Python names", code="invocable_interface"
        )
    if (
        not isinstance(request, dict)
        or not isinstance(result, dict)
        or not isinstance(request.get("fields"), list)
        or (not isinstance(result.get("fields"), list) and result != {"input": "model"})
    ):
        raise ConformanceError(
            "invocable request and result must be closed structs", code="invocable_interface"
        )
    if parameters != [field.get("name") for field in request["fields"]]:
        raise ConformanceError(
            "invocable parameters differ from request fields", code="invocable_interface"
        )
    for key in ("defaults", "type_names", "enum_members"):
        if not isinstance(value[key], dict):
            raise ConformanceError(
                "invocable name/default metadata must be an object", code="invocable_interface"
            )
    nodes: dict[str, Any] = {}

    def walk(node: Any, path: str) -> None:
        nodes[path] = node
        if isinstance(node, dict):
            if "fields" in node:
                if path not in value["type_names"]:
                    raise ConformanceError(
                        "invocable struct is missing its Python name", code="invocable_interface"
                    )
                for field in node["fields"]:
                    walk(field["type"], path + "/" + field["name"])
            elif "list" in node:
                walk(node["list"], path + "/[]")
            elif "union" in node:
                for index, member in enumerate(node["union"]):
                    walk(member, path + f"/union/{index}")
            elif kind == "entrypoint" and "map" in node:
                walk(node["map"]["key"], path + "/map/key")
                walk(node["map"]["value"], path + "/map/value")
            elif kind == "entrypoint" and "tuple" in node:
                for index, member in enumerate(node["tuple"]):
                    walk(member, path + f"/tuple/{index}")
            elif kind == "entrypoint" and "opaque" in node:
                pass
            elif "opaque" in node or "map" in node or "tuple" in node:
                raise ConformanceError(
                    "invocable schema is not a supported wire type", code="invocable_interface"
                )

    walk(request, "request")
    walk(result, "result")
    for path, name in value["type_names"].items():
        if (
            path not in nodes
            or not _identifier(name)
            or not isinstance(nodes[path], dict)
            or not ("fields" in nodes[path] or "literal" in nodes[path])
        ):
            raise ConformanceError(
                "invocable type name has no matching schema", code="invocable_interface"
            )
    for path, members in value["enum_members"].items():
        node = nodes.get(path)
        if (
            not isinstance(node, dict)
            or path not in value["type_names"]
            or "literal" not in node
            or not isinstance(members, list)
            or len(members) != len(node["literal"])
            or not all(_identifier(member) for member in members)
            or len(set(members)) != len(members)
        ):
            raise ConformanceError(
                "invocable enumeration names differ from its values", code="invocable_interface"
            )
    for path, default in value["defaults"].items():
        allowed = (
            (str, int, float, bool, list, dict) if kind == "entrypoint" else (str, int, float, bool)
        )
        empty_sequence = (
            default in ([], ()) and isinstance(nodes.get(path), dict) and "list" in nodes[path]
        )
        if path not in nodes or not (default is None or type(default) in allowed or empty_sequence):
            raise ConformanceError(
                "invocable default is not a canonical immutable field value",
                code="invocable_interface",
            )
        canonical.write(default)
    return value
