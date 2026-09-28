"""Pure generation of ordinary typed caller code from a PackageInterface."""

from __future__ import annotations

import ast
import hashlib
from collections import defaultdict

import msgspec

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.invocable_interface import Invocable
from cozy_runtime.internal.package_interface import CallableDoc


def generate(interface: bytes) -> dict[str, bytes]:
    """Generate code without importing the package or its implementation dependencies."""
    document = package_interface.parse(interface, source="interface generator")
    digest = "sha256:" + hashlib.sha256(interface).hexdigest()
    modules: dict[str, list[tuple[package_interface.Kind, CallableDoc, Invocable]]] = defaultdict(
        list
    )
    for kind, entry in document.callables():
        invocable = entry.invocable
        if invocable is not msgspec.UNSET and not entry.internal:
            modules[invocable.module].append((kind, entry, invocable))
    if not modules:
        raise ConformanceError(
            "package declares no invocable exports", code="interface_exports_absent"
        )
    files: dict[str, bytes] = {}
    for module, entries in sorted(modules.items()):
        code = _module(module, entries, digest)
        # A package-local marker never collides with another interface wheel's
        # top-level py.typed. Public import spelling remains the exact module.
        path = module.replace(".", "/") + "/__init__.py"
        files[path] = code.encode()
        stub = ast.parse(code)
        for node in ast.walk(stub):
            if isinstance(node, ast.FunctionDef):
                node.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
        files[path.removesuffix(".py") + ".pyi"] = (ast.unparse(stub) + "\n").encode()
        parts = module.split(".")
        for index in range(1, len(parts)):
            files.setdefault("/".join(parts[:index]) + "/__init__.py", b"")
        files[parts[0] + "/py.typed"] = b""
    return files


def _module(
    module: str,
    entries: list[tuple[package_interface.Kind, CallableDoc, Invocable]],
    digest: str,
) -> str:
    imports = [
        "from __future__ import annotations",
        "from collections.abc import Awaitable",
        "from typing import Annotated, Any, Literal, cast",
        "from enum import Enum",
        "import msgspec",
        "from cozy_runtime.author import "
        "AssetBound, Assets, ImagePreparation, Image, DecodedAudio, DecodedVideo, "
        "ImageAsset, AudioAsset, VideoAsset, FileAsset, Tree",
        "from cozy_runtime.author import ModelArtifact",
        "from cozy_runtime.author._calls import _CallType, _invoke_proxy",
        "",
    ]
    if any(kind == "entrypoint" for kind, _, _ in entries):
        imports.append("from cozy_runtime.author import ActivationCapture, PendingCall")
    declarations: dict[str, tuple[bytes, str]] = {}
    functions: list[str] = []
    bindings: list[str] = []
    exports: set[str] = set()
    for kind, entry, metadata in sorted(entries, key=lambda item: item[2].export):
        serving = kind == "entrypoint"
        defaults, names, enums = metadata.defaults, metadata.type_names, metadata.enum_members

        def expression(
            node: Json,
            path: str,
            tag_field: str | None = None,
            *,
            names: dict[str, str] = names,
            defaults: dict[str, object] = defaults,
            enums: dict[str, tuple[str, ...]] = enums,
        ) -> str:
            if isinstance(node, str):
                return {
                    "str": "str",
                    "int": "int",
                    "float": "float",
                    "bool": "bool",
                    "null": "None",
                }[node]
            if "fields" in node:
                name = names[path]
                if name.startswith("__cozy_") or name in exports:
                    raise ConformanceError(
                        "interface type collides with a generated export",
                        code="interface_name_conflict",
                    )
                # Open wire types: a callee may add result fields a caller does not read yet.
                options = ["kw_only=True", "frozen=True"]
                if "tag" in node:
                    options += [
                        "tag=" + repr(node["tag"]),
                        "tag_field=" + repr(node.get("tag_field", tag_field or "type")),
                    ]
                fields: list[str] = []
                for field in node["fields"]:
                    field_path = path + "/" + field["name"]
                    annotation = expression(field["type"], field_path)
                    if field.get("constraints"):
                        annotation = (
                            f"Annotated[{annotation}, msgspec.Meta(**{field['constraints']!r})]"
                        )
                    bound = {
                        name: limit
                        for name, limit in field.get("asset_bound", {}).items()
                        if name in package_interface.ASSET_BOUND_KEYS
                    }
                    if bound:
                        annotation = f"Annotated[{annotation}, AssetBound(**{bound!r})]"
                    default = ""
                    if field_path in defaults:
                        value = defaults[field_path]
                        default = (
                            f" = msgspec.field(default_factory=lambda: {value!r})"
                            if isinstance(value, (list, dict))
                            else " = " + literal(field["type"], field_path, value)
                        )
                    fields.append(f"    {field['name']}: {annotation}{default}")
                source = (
                    "class "
                    + name
                    + "(msgspec.Struct, "
                    + ", ".join(options)
                    + "):\n"
                    + "\n".join(fields or ["    pass"])
                    + "\n"
                )
                _declare(declarations, name, source)
                return name
            if "literal" in node:
                if path in enums:
                    name = names[path]
                    rows = [
                        f"    {member} = {value!r}"
                        for member, value in zip(enums[path], node["literal"], strict=True)
                    ]
                    _declare(declarations, name, f"class {name}(Enum):\n" + "\n".join(rows) + "\n")
                    return name
                return "Literal[" + ", ".join(repr(value) for value in node["literal"]) + "]"
            if "list" in node:
                return "list[" + expression(node["list"], path + "/[]") + "]"
            if "map" in node:
                key = expression(node["map"]["key"], path + "/map/key")
                value = expression(node["map"]["value"], path + "/map/value")
                return f"dict[{key}, {value}]"
            if "tuple" in node:
                return (
                    "tuple["
                    + ", ".join(
                        expression(member, path + f"/tuple/{index}")
                        for index, member in enumerate(node["tuple"])
                    )
                    + "]"
                )
            if "union" in node:
                return " | ".join(
                    expression(member, path + f"/union/{index}", node.get("tag_field"))
                    for index, member in enumerate(node["union"])
                )
            if "asset" in node:
                return {
                    "image": "ImageAsset",
                    "audio": "AudioAsset",
                    "video": "VideoAsset",
                    "file": "FileAsset",
                }[node["asset"]]
            if node.get("input") == "tree":
                return "Tree"
            if node.get("input") == "model":
                return "ModelArtifact"
            if node.get("opaque") == "Any":
                return "Any"
            raise ConformanceError(
                "interface schema has no portable Python type", code="interface_type"
            )

        def literal(
            node: Json,
            path: str,
            value: object,
            *,
            names: dict[str, str] = names,
            enums: dict[str, tuple[str, ...]] = enums,
        ) -> str:
            if path in enums:
                return names[path] + "." + enums[path][int(node["literal"].index(value))]
            return repr(value)

        request_name = expression(entry.request, "request")
        result_name = expression(entry.result, "result")
        export = metadata.export
        if export in exports or export in declarations:
            raise ConformanceError(
                "interface repeats a Python export", code="interface_name_conflict"
            )
        exports.add(export)
        parameters: list[str] = []
        if metadata.parameters or serving:
            parameters.append("*")
        request = entry.request
        assert isinstance(request, dict) and isinstance(request["fields"], list)
        fields = {field["name"]: field for field in request["fields"] if isinstance(field, dict)}
        for name in metadata.parameters:
            path = "request/" + name
            annotation = expression(fields[name]["type"], path)
            if entry.assets is not msgspec.UNSET and entry.assets.parameter == name:
                kinds = []
                for row in entry.assets.kinds:
                    cls = (
                        {"image": "Image", "audio": "DecodedAudio", "video": "DecodedVideo"}[
                            row.kind
                        ]
                        if entry.assets.view == "decoded"
                        else expression({"asset": row.kind}, path)
                    )
                    bound = {
                        key: value
                        for key, value in (
                            ("max_bytes", row.max_bytes),
                            ("max_decoded_bytes", row.max_decoded_bytes),
                            ("media_types", row.media_types),
                        )
                        if value is not msgspec.UNSET
                    }
                    if any(bound.values()):
                        cls = f"Annotated[{cls}, AssetBound(**{bound!r})]"
                    kinds.append(cls)
                annotation = "Assets[" + " | ".join(kinds) + "]"
                preparation = next(
                    (row.prepare for row in entry.assets.kinds if row.prepare is not msgspec.UNSET),
                    None,
                )
                if preparation is not None:
                    caps = {
                        key: value
                        for key, value in (
                            ("max_edge", preparation.max_edge),
                            ("max_pixels", preparation.max_pixels),
                        )
                        if value is not msgspec.UNSET
                    }
                    annotation = f"Annotated[{annotation}, ImagePreparation(**{caps!r})]"
            default = (
                " = " + literal(fields[name]["type"], path, defaults[path])
                if path in defaults
                else ""
            )
            if isinstance(defaults.get(path), (list, dict)):
                annotation += " | msgspec.UnsetType"
                default = " = msgspec.UNSET"
            parameters.append(f"{name}: {annotation}{default}")
        arguments = "{" + ", ".join(f"{name!r}: {name}" for name in metadata.parameters) + "}"
        if any(
            isinstance(defaults.get("request/" + name), (list, dict))
            for name in metadata.parameters
        ):
            arguments = (
                "{key: value for key, value in "
                + arguments
                + ".items() if value is not msgspec.UNSET}"
            )
        options = ""
        returned = f"Awaitable[{result_name}]"
        if serving:
            model_names = [slot.parameter for slot in entry.models]
            all_names = [*metadata.parameters, *model_names, "capture"]
            if len(set(all_names)) != len(all_names):
                raise ConformanceError(
                    "serving request fields collide with model slots or capture options",
                    code="interface_name_conflict",
                )
            # Model arguments override injected slots. None leaves selection to the
            # host, which knows both owner bindings and the callee's authored defaults.
            model_overrides = [f"{name}: ModelArtifact | None = None" for name in model_names]
            parameters += model_overrides
            parameters.append("capture: ActivationCapture | None = None")
            models_name, call_name = f"__cozy_{export}Models", f"__cozy_{export}Call"
            model_fields = [f"    {field}" for field in model_overrides]
            _declare(
                declarations,
                models_name,
                f"class {models_name}(msgspec.Struct, kw_only=True, frozen=True):\n"
                + "\n".join(model_fields or ["    pass"])
                + "\n",
            )
            _declare(
                declarations,
                call_name,
                f"class {call_name}(msgspec.Struct, kw_only=True, frozen=True):\n"
                f"    payload: {request_name}\n    models: {models_name}\n",
            )
            models = "{" + ", ".join(f"{name!r}: {name}" for name in model_names) + "}"
            arguments = "{'payload': " + arguments + ", 'models': " + models + "}"
            request_name = call_name
            options = ", capture=capture"
            returned = f"PendingCall[{result_name}]"
        functions.append(
            f"def {export}({', '.join(parameters)}) -> {returned}:\n"
            f"    return cast({returned}, "
            f"_invoke_proxy({module!r}, {export!r}, {arguments}{options}))\n"
        )
        bindings.append(
            f"    {export!r}: _CallType({digest!r}, {module!r}, {export!r}, "
            f"{request_name}, {result_name}),"
        )
    return "\n".join(
        imports
        + [item[1] for item in declarations.values()]
        + functions
        + ["__cozy_bindings__ = {", *bindings, "}", ""]
    )


def _declare(declarations: dict[str, tuple[bytes, str]], name: str, source: str) -> None:
    encoded = source.encode()
    if name in declarations and declarations[name][0] != encoded:
        raise ConformanceError(
            "interface types share a name but differ in meaning", code="interface_name_conflict"
        )
    declarations[name] = (encoded, source)
