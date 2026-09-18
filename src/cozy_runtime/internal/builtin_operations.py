"""Fixed Runtime-owned operation surface and its inert, metadata-only carrier."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import zipfile
from email.parser import BytesParser
from functools import lru_cache
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

from cozy_runtime._interpreter import supported_minors
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


def carrier(version: str) -> tuple[str, bytes]:
    """Runtime's inert carrier; the admitted implementation stays in the SDK."""
    directory = f"cozy_runtime_operations-{version}.dist-info"
    metadata = (
        f"Metadata-Version: 2.4\nName: {NAME}\nVersion: {version}\n"
        f"Requires-Python: >={supported_minors()[0]}\nRequires-Dist: cozy-runtime>={FLOOR},<1\n"
        "Requires-Dist: numpy>=1.26\nCozy-Builtin: operations\n"
        f"Cozy-Builtin-Interface: {digest()}\n"
    ).encode()
    validate_metadata(metadata, WHEEL, ENTRY_POINTS, version)
    files = {"METADATA": metadata, "WHEEL": WHEEL, "entry_points.txt": ENTRY_POINTS}
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, raw in sorted(files.items()):
        encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        writer.writerow((f"{directory}/{name}", "sha256=" + encoded, len(raw)))
    writer.writerow((f"{directory}/RECORD", "", ""))
    files["RECORD"] = record.getvalue().encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, raw in sorted(files.items()):
            entry = zipfile.ZipInfo(f"{directory}/{name}", (2000, 1, 1, 0, 0, 0))
            entry.create_system, entry.external_attr = 3, 0o100644 << 16
            archive.writestr(entry, raw)
    return f"cozy_runtime_operations-{version}-py3-none-any.whl", output.getvalue()


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
        "Requires-Python": ">=" + supported_minors()[0],
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
    allowed = MEMBERS | {"INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json"}
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


def supports_environment(python: str) -> bool:
    """Do not send a new broker row to an older captured Runtime interpreter."""
    executable = Path(python)
    prefix = executable.parent.parent
    config = prefix / "pyvenv.cfg"
    if not executable.is_absolute() or not config.is_file() or config.stat().st_size > 8192:
        return False
    roots = list((prefix / "lib").glob("python*/site-packages"))
    metadata = [path for root in roots for path in root.glob("cozy_runtime-*.dist-info/METADATA")]
    if not metadata:
        # The normal remote generation inherits the current worker image's base.
        # Local SDK snapshots instead carry their own Runtime metadata above.
        settings = dict(
            line.split("=", 1) for line in config.read_text().splitlines() if "=" in line
        )
        if not any(
            key.strip() == "include-system-site-packages" and value.strip() == "true"
            for key, value in settings.items()
        ):
            return False
        metadata = list(Path(__file__).parents[2].glob("cozy_runtime-*.dist-info/METADATA"))
    if len(metadata) != 1 or metadata[0].stat().st_size > 8192:
        return False
    headers = BytesParser().parsebytes(metadata[0].read_bytes(), headersonly=True)
    try:
        return headers.get("Name") == "cozy-runtime" and Version(
            headers.get("Version", "")
        ) >= Version(FLOOR)
    except InvalidVersion:
        return False


def validate_binding(interface: str, module: str, export: str) -> None:
    if (interface, module, export) != (digest(), MODULE, "quantize"):
        raise _refuse()
