"""Read exact generated interface dependencies from a sealed package environment."""

from __future__ import annotations

import hashlib
import sys
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import canonical, interface_wheel, package_interface
from cozy_runtime.internal.package_environment import InstalledEnvironment


def read(installed: InstalledEnvironment) -> list[dict[str, Any]]:
    if not installed.content_digest:
        raise ConformanceError(
            "package calls require a sealed environment", code="child_environment"
        )
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
        with (directory / "METADATA").open("rb") as stream:
            raw = stream.read((1 << 20) + 1)
        if len(raw) > 1 << 20:
            raise ConformanceError("installed metadata exceeds its bound", code="child_environment")
        metadata = BytesParser().parsebytes(raw)
        implementation = metadata.get("Cozy-Implementation-Digest")
        generator = metadata.get("Cozy-Interface-Generator")
        if implementation is None and generator is None:
            continue
        if (
            generator != interface_wheel.GENERATOR_ABI
            or implementation is None
            or not canonical.DIGEST_RE.fullmatch(implementation)
        ):
            raise ConformanceError(
                "installed interface has no exact supported implementation join",
                code="child_interface",
            )
        with path.open("rb") as stream:
            body = stream.read(canonical.DOC_MAX_BYTES + 1)
        if len(body) > canonical.DOC_MAX_BYTES:
            raise ConformanceError("installed interface exceeds its bound", code="child_interface")
        descriptor = package_interface.read_bytes(body)
        # A hostile dependency cannot turn a metadata header into invoke authority:
        # its ordinary code must equal the trusted deterministic generator's output.
        for relative, expected in interface_wheel.generate(body).items():
            target = installed.site_packages / relative
            with target.open("rb") as stream:
                actual = stream.read(len(expected) + 1)
            if actual != expected:
                raise ConformanceError(
                    "installed proxy differs from trusted generator", code="child_interface"
                )
        digest = "sha256:" + hashlib.sha256(body).hexdigest()
        for entry in [*descriptor["entrypoints"], *descriptor["jobs"]]:
            invocable = entry.get("invocable")
            if invocable is None:
                continue
            key = (invocable["module"], invocable["export"])
            if key in exports:
                raise ConformanceError(
                    "two interface dependencies claim the same import", code="child_interface"
                )
            exports.add(key)
            interfaces.append(
                {
                    "interface_digest": digest,
                    "module": key[0],
                    "export": key[1],
                    "implementation_digest": implementation,
                    "interface_path": str(path),
                }
            )
    if len(interfaces) > 256:
        raise ConformanceError(
            "installed invocable export inventory exceeds its bound", code="child_interface"
        )
    return interfaces


def for_job(python: str, content_digest: str) -> list[dict[str, Any]]:
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
    return read(InstalledEnvironment(generation, site, executable, b"", "", content_digest, True))
