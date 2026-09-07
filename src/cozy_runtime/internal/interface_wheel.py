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

GENERATOR_ABI = "cozy.interface-generator/1"
_DISTRIBUTION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def generate(interface: bytes) -> dict[str, bytes]:
    """Generate code without importing the package or its implementation dependencies."""
    document = package_interface.read_bytes(interface, source="interface generator")
    digest = "sha256:" + hashlib.sha256(interface).hexdigest()
    modules: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in [*document["entrypoints"], *document["jobs"]]:
        if "invocable" in entry:
            metadata = invocable_interface.validate(
                entry["invocable"], entry["request"], entry["result"]
            )
            modules[metadata["module"]].append(entry)
    if not modules:
        raise ConformanceError(
            "package declares no invocable exports", code="interface_exports_absent"
        )
    files: dict[str, bytes] = {"py.typed": b""}
    for module, entries in sorted(modules.items()):
        code = _module(module, entries, digest)
        is_package = any(other.startswith(module + ".") for other in modules)
        path = module.replace(".", "/") + ("/__init__.py" if is_package else ".py")
        files[path] = code.encode()
        stub = ast.parse(code)
        for node in ast.walk(stub):
            if isinstance(node, ast.FunctionDef):
                node.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
        files[path.removesuffix(".py") + ".pyi"] = (ast.unparse(stub) + "\n").encode()
        parts = module.split(".")
        for index in range(1, len(parts)):
            files.setdefault("/".join(parts[:index]) + "/__init__.py", b"")
        files[parts[0] + "/py.typed" if len(parts) > 1 or is_package else "py.typed"] = b""
    return files


def _module(module: str, entries: list[dict[str, Any]], digest: str) -> str:
    imports = [
        "from __future__ import annotations",
        "from collections.abc import Awaitable",
        "from typing import Annotated, Literal, cast",
        "from enum import Enum",
        "import msgspec",
        "from cozy_runtime.author import "
        "Context, AssetBound, ImageAsset, AudioAsset, VideoAsset, FileAsset, Tree",
        "from cozy_runtime.author._calls import _CallType, _invoke_proxy",
        "",
    ]
    declarations: dict[str, tuple[bytes, str]] = {}
    functions: list[str] = []
    bindings: list[str] = []
    exports: set[str] = set()
    for entry in sorted(entries, key=lambda item: item["invocable"]["export"]):
        metadata = entry["invocable"]
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
                    default = (
                        " = " + literal(field["type"], field_path, defaults[field_path])
                        if field_path in defaults
                        else ""
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
        export, context = metadata["export"], metadata["context"]
        if export in exports or export in declarations:
            raise ConformanceError(
                "interface repeats a Python export", code="interface_name_conflict"
            )
        exports.add(export)
        parameters = [f"{context}: Context", "/"]
        if metadata["parameters"]:
            parameters.append("*")
        fields = {field["name"]: field for field in entry["request"]["fields"]}
        for name in metadata["parameters"]:
            path = "request/" + name
            annotation = expression(fields[name]["type"], path)
            default = (
                " = " + literal(fields[name]["type"], path, defaults[path])
                if path in defaults
                else ""
            )
            parameters.append(f"{name}: {annotation}{default}")
        arguments = "{" + ", ".join(f"{name!r}: {name}" for name in metadata["parameters"]) + "}"
        functions.append(
            f"def {export}({', '.join(parameters)}) -> Awaitable[{result_name}]:\n"
            f"    return cast(Awaitable[{result_name}], "
            f"_invoke_proxy({context}, {digest!r}, {module!r}, {export!r}, {arguments}))\n"
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
        "Requires-Python: >=3.12,<3.13\nRequires-Dist: cozy-runtime>=0.2.31\n"
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
