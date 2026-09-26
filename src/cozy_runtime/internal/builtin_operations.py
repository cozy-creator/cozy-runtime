"""Fixed Runtime-owned operation surface and its inert, metadata-only carrier."""

from __future__ import annotations

from email.parser import BytesParser
from functools import lru_cache
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal.package_interface import canonical_bytes

NAME = "cozy-runtime-operations"
MODULE = "cozy_runtime.derive.operations"
EXPORTS = ("quantize", "prepare_model")
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
    """Describe actual installed Runtime code in the package/executor process."""
    from cozy_runtime.internal.discovery import discover_installed
    from cozy_runtime.internal.package_interface import build

    return canonical_bytes(build(discover_installed(APPLICATION)))


def rows() -> list[dict[str, Any]]:
    return [
        {
            "module": MODULE,
            "export": export,
            "builtin": "operations",
        }
        for export in EXPORTS
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
    if module != MODULE or export not in EXPORTS:
        raise ConformanceError("unknown Runtime operation", code="builtin_operation")
