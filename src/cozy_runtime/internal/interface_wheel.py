"""Pure generation of ordinary typed caller code from a PackageInterface."""

from __future__ import annotations

import ast
import base64
import csv
import email.parser
import hashlib
import io
import re
import zipfile
from collections import defaultdict
from typing import Any

from packaging.tags import parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import invocable_interface, package_interface

GENERATOR_ABI = "cozy.interface-generator/9"
MAX_IMPLEMENTATION_WHEEL = 256 << 20
_DISTRIBUTION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_COMPANION = re.compile(r"(?:.+/)?_cozy_callers_[0-9a-f]{16}_[0-9]+\.py")
_COMPANION_IMPORT = re.compile(rb"\n\nfrom \.?_cozy_callers_[0-9a-f]{16}_[0-9]+ import [^\n]*\n\Z")


def generate(interface: bytes) -> dict[str, bytes]:
    """Generate code without importing the package or its implementation dependencies."""
    document = package_interface.read_bytes(interface, source="interface generator")
    digest = "sha256:" + hashlib.sha256(interface).hexdigest()
    modules: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for collection, kind in (("entrypoints", "entrypoint"), ("jobs", "job")):
        for entry in document[collection]:
            if "invocable" in entry and not entry.get("internal", False):
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
                        if key in package_interface.ASSET_BOUND_KEYS
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


def build(
    interface: bytes,
    *,
    distribution: str,
    version: str,
    implementation_wheel: bytes | None = None,
    implementation_filename: str | None = None,
) -> tuple[str, bytes]:
    """Package deterministic interface source and an optional implementation overlay."""
    if not _DISTRIBUTION.fullmatch(distribution) or str(Version(version)) != version:
        raise ConformanceError("interface wheel binding is not exact", code="interface_binding")
    normalized = re.sub(r"[-_.]+", "_", distribution).lower()
    dist = f"{normalized}-{version}.dist-info"
    files = generate(interface)
    attributes: dict[str, int] = {}
    filename = f"{normalized}-{version}-py3-none-any.whl"
    metadata = (
        "Metadata-Version: 2.4\n" + f"Name: {distribution}\nVersion: {version}\n"
        "Requires-Python: >=3.12\nRequires-Dist: cozy-runtime>=0.4.0\n"
    ).encode()
    if implementation_wheel is not None:
        if implementation_filename is None:
            raise ConformanceError(
                "implementation wheel filename is required", code="interface_binding"
            )
        files, metadata, attributes = _overlay(
            interface, implementation_wheel, distribution, version
        )
        _wheel_identity(implementation_filename, files[dist + "/WHEEL"], distribution, version)
        filename = implementation_filename
    files[dist + "/package-interface.json"] = interface
    layout = "overlay" if implementation_wheel is not None else "generated"
    header, _, description = metadata.partition(b"\n\n")
    files[dist + "/METADATA"] = (
        header.rstrip(b"\n")
        + (
            f"\nCozy-Interface-Generator: {GENERATOR_ABI}\nCozy-Interface-Layout: {layout}\n\n"
        ).encode()
        + description
    )
    if implementation_wheel is None:
        files[dist + "/WHEEL"] = (
            b"Wheel-Version: 1.0\nGenerator: cozy-runtime\n"
            b"Root-Is-Purelib: true\nTag: py3-none-any\n"
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
            info.external_attr = attributes.get(path, 0o100644 << 16)
            archive.writestr(info, data)
    return filename, output.getvalue()


def _overlay(
    interface: bytes, implementation: bytes, distribution: str, version: str
) -> tuple[dict[str, bytes], bytes, dict[str, int]]:
    """Retain library code/data/dependencies and replace only managed callable exports."""
    if len(implementation) > MAX_IMPLEMENTATION_WHEEL:
        raise ConformanceError("implementation wheel exceeds its bound", code="interface_binding")
    with zipfile.ZipFile(io.BytesIO(implementation)) as archive:
        members = archive.infolist()
        if sum(row.file_size for row in members) > MAX_IMPLEMENTATION_WHEEL:
            raise ConformanceError(
                "implementation contents exceed their bound", code="interface_binding"
            )
        files = {}
        attributes = {}
        for row in members:
            path = row.filename
            if row.is_dir():
                continue
            if (
                path in files
                or path.startswith("/")
                or "\\" in path
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or ((row.external_attr >> 16) & 0o170000) == 0o120000
            ):
                raise ConformanceError(
                    "implementation wheel has invalid members", code="interface_binding"
                )
            files[path] = archive.read(row)
            attributes[path] = row.external_attr
    metadata_paths = [name for name in files if name.endswith(".dist-info/METADATA")]
    if len(metadata_paths) != 1:
        raise ConformanceError(
            "implementation wheel has no unique metadata", code="interface_binding"
        )
    metadata_path = metadata_paths[0]
    metadata = files[metadata_path].replace(b"\r\n", b"\n")
    expected_dist = re.sub(r"[-_.]+", "_", distribution).lower() + "-" + version + ".dist-info"
    if metadata_path != expected_dist + "/METADATA":
        raise ConformanceError(
            "implementation metadata directory differs", code="interface_binding"
        )
    if expected_dist + "/WHEEL" not in files:
        raise ConformanceError(
            "implementation wheel has no WHEEL metadata", code="interface_binding"
        )
    headers = email.parser.BytesParser().parsebytes(metadata)

    def normalize(name: str) -> str:
        return re.sub(r"[-_.]+", "_", name).lower()

    if (
        normalize(str(headers.get("Name", ""))) != normalize(distribution)
        or headers.get("Version") != version
    ):
        raise ConformanceError("implementation wheel identity differs", code="interface_binding")
    # Description text is a body, not header text. Preserve it after adding bindings.
    if headers.get("Cozy-Interface-Generator") is not None:
        # A published callable wheel already carries a managed import surface from
        # whichever generator built it. Private capture rebinds that source: remove exactly
        # the reserved caller companions and their terminal imports, then regenerate them
        # for this interface below. Every other package source stays as published.
        if headers.get("Cozy-Interface-Layout") != "overlay":
            raise ConformanceError(
                "a generated interface wheel is not an implementation", code="interface_binding"
            )
        for name in [name for name in files if _COMPANION.fullmatch(name)]:
            del files[name]
        for name, data in files.items():
            if name.endswith(".py"):
                files[name] = _COMPANION_IMPORT.sub(b"", data)
        header, _, description = metadata.partition(b"\n\n")
        metadata = (
            b"\n".join(
                row
                for row in header.split(b"\n")
                if not row.lower().startswith(
                    (
                        b"cozy-interface-generator:",
                        b"cozy-implementation-digest:",
                        b"cozy-interface-layout:",
                    )
                )
            )
            + b"\n\n"
            + description
        )
    for target, (generated, source, suffix) in overlay_sources(interface, set(files)).items():
        if generated in files:
            raise ConformanceError(
                "caller module collides with implementation", code="interface_binding"
            )
        files[generated] = source
        files[target] += suffix
    # Rebuilt below; every other source, extension, asset and dependency header stays.
    for name in list(files):
        if name.startswith(metadata_path.rpartition("/")[0] + "/") and name.rpartition("/")[2] in {
            "RECORD",
            "RECORD.jws",
            "RECORD.p7s",
        }:
            del files[name]
    return files, metadata, attributes


def _wheel_identity(filename: str, raw: bytes, distribution: str, version: str) -> None:
    """An overlay cannot change an implementation's platform or optional build tag."""
    if "/" in filename or "\\" in filename:
        raise ConformanceError("implementation wheel filename is invalid", code="interface_binding")
    try:
        name, release, build, filename_tags = parse_wheel_filename(filename)
        headers = email.parser.BytesParser().parsebytes(raw)
        tags = set().union(*(parse_tag(value) for value in headers.get_all("Tag", [])))
        if (
            name != canonicalize_name(distribution)
            or release != Version(version)
            or tags != filename_tags
            or headers.get_all("Root-Is-Purelib") not in (["true"], ["false"])
            or not str(headers.get("Wheel-Version", "")).startswith("1.")
            or (
                headers.get("Build") is not None
                and (not build or headers.get("Build") != filename.split("-")[-4])
            )
        ):
            raise ValueError("filename and WHEEL metadata disagree")
    except ValueError as exc:
        raise ConformanceError(
            f"implementation wheel identity differs: {exc}", code="interface_binding"
        ) from exc


def overlay_sources(interface: bytes, available: set[str]) -> dict[str, tuple[str, bytes, bytes]]:
    """Expected trusted companion source and terminal import for each library module."""
    document = package_interface.read_bytes(interface, source="caller overlay")
    digest = "sha256:" + hashlib.sha256(interface).hexdigest()
    modules: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for collection, kind in (("entrypoints", "entrypoint"), ("jobs", "job")):
        for entry in document[collection]:
            if "invocable" in entry and not entry.get("internal", False):
                modules[entry["invocable"]["module"]].append({**entry, "_kind": kind})
    result: dict[str, tuple[str, bytes, bytes]] = {}
    for index, (module, entries) in enumerate(sorted(modules.items())):
        path = module.replace(".", "/")
        candidates = [name for name in (path + ".py", path + "/__init__.py") if name in available]
        if len(candidates) != 1:
            raise ConformanceError(
                "managed module has no unique Python implementation", code="interface_binding"
            )
        target = candidates[0]
        companion = "_cozy_callers_" + digest[7:23] + "_" + str(index)
        directory = target.rpartition("/")[0]
        generated = (directory + "/" if directory else "") + companion + ".py"
        source = _module(module, entries, digest).encode()
        relative = "." if directory else ""
        exports = [row["invocable"]["export"] for row in entries if row["_kind"] == "entrypoint"]
        aliases = ", ".join(
            [*(name + " as " + name for name in exports), "__cozy_bindings__ as __cozy_bindings__"]
        )
        suffix = ("\n\nfrom " + relative + companion + " import " + aliases + "\n").encode()
        result[target] = (generated, source, suffix)
    return result
