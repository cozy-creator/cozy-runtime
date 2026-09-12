"""Fixed Runtime-owned operation surface and its inert, metadata-only carrier."""

from __future__ import annotations

import csv
import hashlib
import zipfile
from email.parser import BytesParser
from functools import lru_cache
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal.package_interface import canonical_bytes
from cozy_runtime.internal.static_interface import build_builtin

NAME = "cozy-runtime-operations"
MODULE = "cozy_runtime.derive.operations"
APPLICATION = MODULE + ":app"
FLOOR = "0.16.8"
WHEEL = (
    b"Wheel-Version: 1.0\nGenerator: cozy-runtime-builtin\n"
    b"Root-Is-Purelib: true\nTag: py3-none-any\n"
)
ENTRY_POINTS = ("[cozy.application]\ndefault = " + APPLICATION + "\n").encode()
MEMBERS = frozenset({"METADATA", "WHEEL", "entry_points.txt", "RECORD"})


@lru_cache(maxsize=1)
def document() -> bytes:
    return canonical_bytes(build_builtin("operations"))


def digest() -> str:
    return "sha256:" + hashlib.sha256(document()).hexdigest()


def _refuse() -> ConformanceError:
    return ConformanceError(
        "Runtime operation carrier differs from its fixed contract", code="builtin_carrier"
    )


def validate_metadata(metadata: bytes, wheel: bytes, entry: bytes, version: str) -> None:
    try:
        if str(Version(version)) != version or Version(version) < Version(FLOOR):
            raise _refuse()
    except InvalidVersion as exc:
        raise _refuse() from exc
    if max(len(metadata), len(wheel), len(entry)) > 8192 or wheel != WHEEL or entry != ENTRY_POINTS:
        raise _refuse()
    headers = BytesParser().parsebytes(metadata, headersonly=True)
    expected = {
        "Metadata-Version": "2.4",
        "Name": NAME,
        "Version": version,
        "Requires-Python": ">=3.12,<3.13",
        "Cozy-Builtin": "operations",
        "Cozy-Builtin-Interface": digest(),
    }
    if set(headers.keys()) != {*expected, "Requires-Dist"}:
        raise _refuse()
    if any(headers.get_all(key) != [value] for key, value in expected.items()):
        raise _refuse()
    requirements = headers.get_all("Requires-Dist", [])
    try:
        if len(requirements) != 2 or {str(Requirement(row)) for row in requirements} != {
            str(Requirement("cozy-runtime>=0.16.8,<1")),
            str(Requirement("numpy>=1.26")),
        }:
            raise _refuse()
    except ValueError as exc:
        raise _refuse() from exc


def validate_carrier(path: Path, version: str) -> None:
    root = "cozy_runtime_operations-" + version + ".dist-info/"
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 65536:
        raise _refuse()
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if (
                len(names) != 4
                or set(names) != {root + name for name in MEMBERS}
                or any(row.file_size > 8192 for row in archive.infolist())
            ):
                raise _refuse()
            validate_metadata(
                archive.read(root + "METADATA"),
                archive.read(root + "WHEEL"),
                archive.read(root + "entry_points.txt"),
                version,
            )
            # The ordinary wheel admission owns RECORD hashes, path safety and tags.
    except (OSError, zipfile.BadZipFile) as exc:
        raise _refuse() from exc


def validate_installed(directory: Path) -> None:
    suffix = directory.name.removeprefix("cozy_runtime_operations-").removesuffix(".dist-info")
    if directory.name != "cozy_runtime_operations-" + suffix + ".dist-info":
        raise _refuse()
    allowed = MEMBERS | {"INSTALLER", "REQUESTED", "direct_url.json"}
    if directory.is_symlink() or any(
        row.is_symlink() or not row.is_file() or row.name not in allowed
        for row in directory.iterdir()
    ):
        raise _refuse()

    def read(name: str) -> bytes:
        path = directory / name
        if not path.is_file() or path.stat().st_size > 8192:
            raise _refuse()
        return path.read_bytes()

    validate_metadata(read("METADATA"), read("WHEEL"), read("entry_points.txt"), suffix)
    rows = list(csv.reader(read("RECORD").decode().splitlines()))
    if any(
        len(row) != 3
        or not row[0].startswith(directory.name + "/")
        or row[0].removeprefix(directory.name + "/") not in allowed
        for row in rows
    ):
        raise _refuse()


def rows() -> list[dict[str, Any]]:
    return [
        {
            "interface_digest": digest(),
            "module": MODULE,
            "export": "quantize",
            "builtin": "operations",
        }
    ]


def validate_binding(interface: str, module: str, export: str) -> None:
    if (interface, module, export) != (digest(), MODULE, "quantize"):
        raise _refuse()
