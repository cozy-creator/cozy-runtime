"""Pure generation of ordinary typed caller code from a PackageInterface."""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import io
import re
import zipfile
from collections import defaultdict
from typing import Any

from packaging.version import Version

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import canonical, invocable_interface, package_interface

GENERATOR_ABI = "cozy.interface-generator/4"
_DISTRIBUTION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def generate(interface: bytes) -> dict[str, bytes]:
    """Generate code without importing the package or its implementation dependencies."""
    document = package_interface.read_bytes(interface, source="interface generator")
    digest = "sha256:" + hashlib.sha256(interface).hexdigest()
    modules: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for collection, kind in (("entrypoints", "entrypoint"), ("jobs", "job")):
        for entry in document[collection]:
            if "invocable" in entry:
                metadata = invocable_interface.validate(
                    entry["invocable"], entry["request"], entry["result"], kind=kind
                )
                modules[metadata["module"]].append({**entry, "_kind": kind})
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


def _module(module: str, entries: list[dict[str, Any]], digest: str) -> str:
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
    if any(entry["_kind"] == "entrypoint" for entry in entries):
        imports.append("from cozy_runtime.author import ActivationCapture, PendingCall")
    declarations: dict[str, tuple[bytes, str]] = {}
    functions: list[str] = []
    bindings: list[str] = []
    exports: set[str] = set()
    for entry in sorted(entries, key=lambda item: item["invocable"]["export"]):
        metadata = entry["invocable"]
        serving = entry["_kind"] == "entrypoint"
        defaults, names, enums = (
            metadata["defaults"],
            metadata["type_names"],
            metadata["enum_members"],
        )

        def expression(
            node: Any,
            path: str,
            tag_field: str | None = None,
            *,
            names: dict[str, str] = names,
            defaults: dict[str, Any] = defaults,
            enums: dict[str, list[str]] = enums,
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
                options = ["kw_only=True", "frozen=True", "forbid_unknown_fields=True"]
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
                    if field.get("asset_bound"):
                        annotation = (
                            f"Annotated[{annotation}, AssetBound(**{field['asset_bound']!r})]"
                        )
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
            node: Any,
            path: str,
            value: Any,
            *,
            names: dict[str, str] = names,
            enums: dict[str, list[str]] = enums,
        ) -> str:
            if path in enums:
                return names[path] + "." + enums[path][int(node["literal"].index(value))]
            return repr(value)

        request_name = expression(entry["request"], "request")
        result_name = expression(entry["result"], "result")
        export = metadata["export"]
        if export in exports or export in declarations:
            raise ConformanceError(
                "interface repeats a Python export", code="interface_name_conflict"
            )
        exports.add(export)
        parameters: list[str] = []
        if metadata["parameters"] or serving:
            parameters.append("*")
        fields = {field["name"]: field for field in entry["request"]["fields"]}
        for name in metadata["parameters"]:
            path = "request/" + name
            annotation = expression(fields[name]["type"], path)
            if entry.get("assets", {}).get("parameter") == name:
                kinds = []
                for row in entry["assets"]["kinds"]:
                    cls = (
                        {"image": "Image", "audio": "DecodedAudio", "video": "DecodedVideo"}[
                            row["kind"]
                        ]
                        if entry["assets"].get("view") == "decoded"
                        else expression({"asset": row["kind"]}, path)
                    )
                    bound = {
                        key: value
                        for key, value in row.items()
                        if key not in {"kind", "max_count", "prepare"}
                    }
                    bound["media_types"] = tuple(bound["media_types"])
                    if any(bound.values()):
                        cls = f"Annotated[{cls}, AssetBound(**{bound!r})]"
                    kinds.append(cls)
                annotation = "Assets[" + " | ".join(kinds) + "]"
                preparation = next(
                    (row["prepare"] for row in entry["assets"]["kinds"] if "prepare" in row), None
                )
                if preparation is not None:
                    caps = {key: value for key, value in preparation.items() if key != "profile"}
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
        arguments = "{" + ", ".join(f"{name!r}: {name}" for name in metadata["parameters"]) + "}"
        if any(
            isinstance(defaults.get("request/" + name), (list, dict))
            for name in metadata["parameters"]
        ):
            arguments = (
                "{key: value for key, value in "
                + arguments
                + ".items() if value is not msgspec.UNSET}"
            )
        options = ""
        returned = f"Awaitable[{result_name}]"
        if serving:
            model_names = [slot["path"].rpartition(".")[2] for slot in entry.get("models", [])]
            all_names = [*metadata["parameters"], *model_names, "capture"]
            if len(set(all_names)) != len(all_names):
                raise ConformanceError(
                    "serving request fields collide with model slots or capture options",
                    code="interface_name_conflict",
                )
            parameters += [f"{name}: ModelArtifact" for name in model_names]
            parameters.append("capture: ActivationCapture | None = None")
            models_name, call_name = f"__cozy_{export}Models", f"__cozy_{export}Call"
            model_fields = [f"    {name}: ModelArtifact" for name in model_names]
            _declare(
                declarations,
                models_name,
                f"class {models_name}(msgspec.Struct, kw_only=True, frozen=True, "
                "forbid_unknown_fields=True):\n"
                + "\n".join(model_fields or ["    pass"])
                + "\n",
            )
            _declare(
                declarations,
                call_name,
                f"class {call_name}(msgspec.Struct, kw_only=True, frozen=True, "
                "forbid_unknown_fields=True):\n"
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
            f"_invoke_proxy({digest!r}, {module!r}, {export!r}, {arguments}{options}))\n"
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


def build(
    interface: bytes, *, distribution: str, version: str, implementation_digest: str
) -> tuple[str, bytes]:
    """Package deterministic source and its exact implementation join into one wheel."""
    if (
        not _DISTRIBUTION.fullmatch(distribution)
        or str(Version(version)) != version
        or not canonical.DIGEST_RE.fullmatch(implementation_digest)
    ):
        raise ConformanceError("interface wheel binding is not exact", code="interface_binding")
    normalized = re.sub(r"[-_.]+", "_", distribution).lower()
    dist = f"{normalized}-{version}.dist-info"
    files = generate(interface)
    files[dist + "/package-interface.json"] = interface
    files[dist + "/METADATA"] = (
        "Metadata-Version: 2.4\n" + f"Name: {distribution}\nVersion: {version}\n"
        "Requires-Python: >=3.12,<3.13\nRequires-Dist: cozy-runtime>=0.4.0\n"
        f"Cozy-Interface-Generator: {GENERATOR_ABI}\n"
        f"Cozy-Implementation-Digest: {implementation_digest}\n\n"
    ).encode()
    files[dist + "/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: cozy-runtime\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    rows = [
        [
            path,
            "sha256="
            + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(),
            str(len(data)),
        ]
        for path, data in sorted(files.items())
    ]
    rows.append([dist + "/RECORD", "", ""])
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    files[dist + "/RECORD"] = record.getvalue().encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, data in sorted(files.items()):
            info = zipfile.ZipInfo(path, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return f"{normalized}-{version}-py3-none-any.whl", output.getvalue()
