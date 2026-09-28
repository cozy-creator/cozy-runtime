"""Discover invocable interface metadata in an ordinary installed package environment."""

from __future__ import annotations

import sys
from pathlib import Path

import msgspec

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import canonical, package_interface
from cozy_runtime.internal.package_installation import InstalledEnvironment


def read(installed: InstalledEnvironment) -> list[dict[str, str]]:
    directories = sorted(installed.site_packages.glob("*.dist-info"))
    if len(directories) > 4096:
        raise ConformanceError(
            "installed distribution inventory exceeds its bound", code="child_environment"
        )
    interfaces: list[dict[str, str]] = []
    exports: set[tuple[str, str]] = set()
    for directory in directories:
        path = directory / "package-interface.json"
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            body = stream.read(canonical.DOC_MAX_BYTES + 1)
        if len(body) > canonical.DOC_MAX_BYTES:
            raise ConformanceError("installed interface exceeds its bound", code="child_interface")
        descriptor = package_interface.parse(body)
        # uv owns installed package bytes. Discovery reads the declared interface;
        # it does not re-generate and compare installed Python source at invocation.
        for kind, entry in descriptor.callables():
            invocable = entry.invocable
            if invocable is msgspec.UNSET or entry.internal:
                continue
            key = (invocable.module, invocable.export)
            if key in exports:
                raise ConformanceError(
                    "two interface dependencies claim the same import", code="child_interface"
                )
            exports.add(key)
            interfaces.append(
                {"module": key[0], "export": key[1], "kind": kind, "interface_path": str(path)}
            )
    if len(interfaces) > 256:
        raise ConformanceError(
            "installed invocable export inventory exceeds its bound", code="child_interface"
        )
    return interfaces


def for_job(python: str, installation_id: str) -> list[dict[str, str]]:
    """Read the explicitly selected venv; never run its interpreter or inherit imports."""
    executable = Path(python)
    generation = executable.parent.parent
    if not executable.is_absolute() or not (generation / "pyvenv.cfg").is_file():
        return []
    # The selected venv can use a different Python minor than this controller.
    # Read its own layout metadata without starting package Python or .pth hooks.
    if sys.platform == "win32":
        site = generation / "Lib/site-packages"
    else:
        settings = {
            key.strip(): value.strip()
            for line in (generation / "pyvenv.cfg").read_text().splitlines()
            if "=" in line
            for key, value in [line.split("=", 1)]
        }
        version = settings.get("version_info", settings.get("version", ""))
        minor = ".".join(version.split(".")[:2])
        site = generation / f"lib/python{minor}/site-packages"
    return read(InstalledEnvironment(installation_id, generation, site, executable, "", "", True))
