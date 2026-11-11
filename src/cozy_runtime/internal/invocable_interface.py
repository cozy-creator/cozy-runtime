"""Source-stable Python names for the existing canonical callable schemas."""

from __future__ import annotations

import enum
import keyword
import re
import types
import typing
from collections.abc import Mapping, Sequence
from typing import get_args, get_origin

import msgspec

from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._tool import Prompt, Seed
from cozy_runtime.author._walker import is_struct, strip
from cozy_runtime.internal import canonical, schema


def from_types(
    payload_type: object,
    result_type: object,
    *,
    module: str,
    export: str,
    context: str,
    parameters: Sequence[str] | None,
    parameter_defaults: Mapping[str, object],
    memoize: bool,
    capabilities: Sequence[str],
) -> dict[str, canonical.Json]:
    """Portable names/defaults from resolved types, shared by both interface readers.
    `parameters` None: a handler, whose caller names are its request struct's fields."""
    serving = parameters is None
    if parameters is None:
        payload = typing.cast(type[msgspec.Struct], payload_type)
        parameters = [field.encode_name for field in msgspec.structs.fields(payload)]
    defaults: dict[str, canonical.Json] = {}
    names: dict[str, str] = {}
    enums: dict[str, list[str]] = {}

    def walk(annotation: object, path: str) -> None:
        base, _ = strip(annotation)
        if base is ModelArtifact or (get_origin(base) or base) is Prompt:
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
                    default = msgspec.to_builtins(field.default, enc_hook=schema.wire)
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
    seeds = _unset_seeds(payload_type) if memoize else []
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
        **({"memo_requires": seeds} if seeds else {}),
    }


def _unset_seeds(payload_type: object) -> list[str]:
    """A request's optional seeds (`seed: Seed | None = None`): a request that leaves one unset
    asks for a new draw, so the machine runs it without the memo."""
    base, _ = strip(payload_type)
    if not is_struct(base):
        return []
    return [
        field.encode_name
        for field in msgspec.structs.fields(typing.cast(type[msgspec.Struct], base))
        if field.default is None and Seed in get_args(strip(field.type)[0])
    ]


def check_memo_requires(value: Invocable, request: object) -> None:
    """Each memo condition names an optional request parameter a caller may omit."""
    if value.memo_requires is msgspec.UNSET:
        return
    if not value.memoize:
        raise ConformanceError("memo_requires needs memoize", code="invocable_memoize")
    rows = request.get("fields", []) if isinstance(request, dict) else []
    fields = {field.get("name"): field for field in rows if isinstance(field, dict)}
    for name in value.memo_requires:
        if name not in value.parameters:
            raise ConformanceError(
                f'memoize requires "{name}", which is not a parameter of {value.export} '
                f"(parameters: {', '.join(value.parameters)})",
                code="invocable_memoize",
            )
        field = fields.get(name, {})
        if field.get("wire") not in ("optional", "omissible"):
            raise ConformanceError(
                f'memoize requires "{name}", a required parameter of {value.export}: only an '
                "optional parameter (one with a default) can be omitted",
                code="invocable_memoize",
            )


def _identifier(value: object) -> bool:
    return isinstance(value, str) and value.isidentifier() and not keyword.iskeyword(value)


def project_serving_models(document: dict[str, canonical.Json]) -> None:
    """Expose injected Model parameters once, in the serving Model plane.

    Only the interface builder calls it, for a function whose request is its own parameters.
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


class Invocable(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Source-stable caller names/defaults for one callable; newer writers' keys are ignored."""

    module: str
    export: str
    context: str
    parameters: tuple[str, ...]
    defaults: dict[str, object]
    type_names: dict[str, str]
    enum_members: dict[str, tuple[str, ...]]
    memoize: bool
    capabilities: tuple[str, ...]
    memo_requires: tuple[str, ...] | msgspec.UnsetType = msgspec.UNSET
    """Optional parameters a request must explicitly provide to be memoizable."""
    operation_identity: str | msgspec.UnsetType = msgspec.UNSET


def validate(value: Invocable, request: object, result: object) -> None:
    """Validate every token the pure generator can turn into Python source."""
    if value.operation_identity is not msgspec.UNSET and (
        re.fullmatch(r"sha256:[0-9a-f]{64}", value.operation_identity) is None
    ):
        raise ConformanceError("invalid operation identity", code="invocable_interface")
    if value.memoize and set(value.capabilities) & {"egress", "secrets"}:
        raise ConformanceError(
            "memoized interface declares external or secret-dependent work",
            code="invocable_memoize_effect",
        )
    if (
        not all(_identifier(part) for part in value.module.split("."))
        or not _identifier(value.export)
        or not (_identifier(value.context) or value.context == "")
    ):
        raise ConformanceError(
            "invocable export is not a Python module function", code="invocable_interface"
        )
    parameters = value.parameters
    if (
        not all(_identifier(name) for name in parameters)
        or len(set(parameters)) != len(parameters)
        or value.context in parameters
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
    if list(parameters) != [field.get("name") for field in request["fields"]]:
        raise ConformanceError(
            "invocable parameters differ from request fields", code="invocable_interface"
        )
    check_memo_requires(value, request)
    nodes: dict[str, canonical.Json] = {}

    def walk(node: canonical.Json, path: str) -> None:
        nodes[path] = node
        if isinstance(node, dict):
            node = node.get("pickup", node)
            if "fields" in node:
                if path not in value.type_names:
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
            elif "map" in node:
                walk(node["map"]["key"], path + "/map/key")
                walk(node["map"]["value"], path + "/map/value")
            elif "tuple" in node:
                for index, member in enumerate(node["tuple"]):
                    walk(member, path + f"/tuple/{index}")

    walk(request, "request")
    walk(result, "result")
    # Names and defaults for paths this schema does not have are never read: ignore them.
    for path, name in value.type_names.items():
        if path not in nodes:
            continue
        if (
            not _identifier(name)
            or not isinstance(nodes[path], dict)
            or not ({"fields", "literal", "pickup"} & set(nodes[path]))
        ):
            raise ConformanceError(
                "invocable type name has no matching schema", code="invocable_interface"
            )
    for path, members in value.enum_members.items():
        node = nodes.get(path)
        if node is None:
            continue
        if (
            not isinstance(node, dict)
            or path not in value.type_names
            or "literal" not in node
            or len(members) != len(node["literal"])
            or not all(_identifier(member) for member in members)
            or len(set(members)) != len(members)
        ):
            raise ConformanceError(
                "invocable enumeration names differ from its values", code="invocable_interface"
            )
    for path, default in value.defaults.items():
        allowed = (str, int, float, bool, list, dict)
        empty_sequence = (
            default in ([], ()) and isinstance(nodes.get(path), dict) and "list" in nodes[path]
        )
        if path not in nodes:
            continue
        if not (default is None or type(default) in allowed or empty_sequence):
            raise ConformanceError(
                "invocable default is not a canonical immutable field value",
                code="invocable_interface",
            )
        canonical.write(default)
