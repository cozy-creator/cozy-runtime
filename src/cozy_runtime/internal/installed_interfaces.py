"""Discover invocable interface metadata in an ordinary installed package environment."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import canonical, package_interface
from cozy_runtime.internal.package_installation import InstalledEnvironment


def read(installed: InstalledEnvironment) -> list[dict[str, Any]]:
    directories = sorted(installed.site_packages.glob("*.dist-info"))
    if len(directories) > 4096:
        raise ConformanceError(
            "installed distribution inventory exceeds its bound", code="child_environment"
        )
    interfaces: list[dict[str, Any]] = []
    exports: set[tuple[str, str]] = set()
    for directory in directories:
        path = directory / "package-interface.json"
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            body = stream.read(canonical.DOC_MAX_BYTES + 1)
        if len(body) > canonical.DOC_MAX_BYTES:
            raise ConformanceError("installed interface exceeds its bound", code="child_interface")
        descriptor = package_interface.read_bytes(body)
        # uv owns installed package bytes. Discovery reads the declared interface;
        # it does not re-generate and compare installed Python source at invocation.
        for entry in [*descriptor["entrypoints"], *descriptor["jobs"]]:
            invocable = entry.get("invocable")
            if invocable is None or entry.get("internal", False):
                continue
            key = (invocable["module"], invocable["export"])
            if key in exports:
                raise ConformanceError(
                    "two interface dependencies claim the same import", code="child_interface"
                )
            exports.add(key)
            interfaces.append(
                {
                    "module": key[0],
                    "export": key[1],
                    "kind": "job" if entry in descriptor["jobs"] else "entrypoint",
                    "interface_path": str(path),
                }
            )
    if len(interfaces) > 256:
        raise ConformanceError(
            "installed invocable export inventory exceeds its bound", code="child_interface"
        )
    return interfaces


def for_job(python: str, installation_id: str) -> list[dict[str, Any]]:
    """Read the explicitly selected venv; never run its interpreter or inherit imports."""
    executable = Path(python)
    generation = executable.parent.parent
    if not executable.is_absolute() or not (generation / "pyvenv.cfg").is_file():
        return []
    # Runtime and every admitted implementation use the same CPython ABI. uv's
    # venv layout is platform-defined; no package interpreter needs to run merely
    # to locate immutable metadata on the preparation/control process.
    site = generation / (
        "Lib/site-packages"
        if sys.platform == "win32"
        else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    return read(InstalledEnvironment(installation_id, generation, site, executable, "", "", True))
