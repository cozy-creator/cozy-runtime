"""Source-stable Python names for the existing canonical callable schemas."""

from __future__ import annotations

import enum
import inspect
import keyword
import types
import typing
from typing import Any, get_args, get_origin

import msgspec

from cozy_runtime.author import ConformanceError, Surface
from cozy_runtime.author._walker import is_struct, strip
from cozy_runtime.internal import canonical, schema


def metadata(surface: Surface) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    names: dict[str, str] = {}
    enums: dict[str, list[str]] = {}

    def walk(annotation: object, path: str) -> None:
        base, _ = strip(annotation)
        origin = get_origin(base)
        if isinstance(base, type) and is_struct(base):
            names[path] = base.__name__
            for field in msgspec.structs.fields(base):
                if field.name != field.encode_name or not _identifier(field.name):
                    raise ConformanceError(
                        "invocable struct fields need portable Python names", code="invocable_field"
                    )
                nested = path + "/" + field.name
                walk(field.type, nested)
                if field.default is not msgspec.NODEFAULT:
                    canonical.write(field.default)
                    defaults[nested] = field.default
        elif isinstance(base, type) and issubclass(base, enum.Enum):
            names[path] = base.__name__
            members = sorted(base, key=lambda member: canonical.write(member.value))
            enums[path] = [member.name for member in members]
        elif origin is list:
            walk(get_args(base)[0], path + "/[]")
        elif origin in (types.UnionType, typing.Union):
            for index, member in enumerate(
                sorted(get_args(base), key=lambda item: canonical.write(schema.render(item)))
            ):
                walk(member, path + f"/union/{index}")

    walk(surface.payload_type, "request")
    walk(surface.result_type, "result")
    parameters = [parameter.name for parameter in surface.params if parameter.role == "wire"]
    signature = inspect.signature(surface.fn)
    for name in parameters:
        default = signature.parameters[name].default
        if default is not inspect.Parameter.empty:
            defaults["request/" + name] = default
    return {
        "module": surface.fn.__module__,
        "export": surface.fn.__name__,
        "context": next(
            parameter.name for parameter in surface.params if parameter.role == "context"
        ),
        "parameters": parameters,
        "defaults": defaults,
        "type_names": names,
        "enum_members": enums,
        "reusable": surface.reusable,
        "capabilities": sorted(surface.capabilities),
    }


def _identifier(value: object) -> bool:
    return isinstance(value, str) and value.isidentifier() and not keyword.iskeyword(value)


def validate(value: object, request: Any, result: Any) -> dict[str, Any]:
    """Validate every token the pure generator can turn into Python source."""
    keys = {
        "module",
        "export",
        "context",
        "parameters",
        "defaults",
        "type_names",
        "enum_members",
        "reusable",
        "capabilities",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ConformanceError(
            "invocable metadata has unknown or missing fields", code="invocable_interface"
        )
    if type(value["reusable"]) is not bool:
        raise ConformanceError(
            "invocable reuse declaration must be a boolean", code="invocable_interface"
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
    if value["reusable"] and set(capabilities) & {"egress", "secrets"}:
        raise ConformanceError(
            "reusable interface declares external or secret-dependent work",
            code="invocable_reusable_effect",
        )
    if (
        not isinstance(value["module"], str)
        or not all(_identifier(part) for part in value["module"].split("."))
        or not _identifier(value["export"])
        or not _identifier(value["context"])
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
        or not isinstance(result.get("fields"), list)
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
        if path not in nodes or not (default is None or type(default) in (str, int, float, bool)):
            raise ConformanceError(
                "invocable default is not a canonical immutable field value",
                code="invocable_interface",
            )
        canonical.write(default)
    return value
