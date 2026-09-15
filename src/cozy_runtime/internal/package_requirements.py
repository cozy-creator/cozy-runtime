"""Validate a joined package environment from metadata, without importing package code."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from cozy_runtime.internal import base_observation
from cozy_runtime.internal.package_environment import (
    MAX_METADATA_BYTES,
    MAX_PACKAGE_WHEELS,
    EnvironmentRefusal,
    _marker_environment,
    _name,
)


class DependencyConflict(EnvironmentRefusal):
    """Machine-readable facts accompany the existing readable dependency refusal."""

    def __init__(self, package: str, requirement: Requirement, installed: str) -> None:
        self.facts = {
            "package": package,
            "distribution": _name(requirement.name),
            "required": str(requirement.specifier),
            "installed": installed,
        }
        super().__init__(
            "package_environment_dependency_base_conflict",
            f"{package} requires {requirement}, but the execution environment provides "
            f"{requirement.name} {installed}; update the worker or select a compatible package",
        )


@dataclass(frozen=True)
class _Distribution:
    version: str
    python: str
    requirements: tuple[Requirement, ...]


def _distribution(version: str, python: str, requirements: list[str]) -> _Distribution:
    try:
        Version(version)
        SpecifierSet(python)
        return _Distribution(version, python, tuple(Requirement(raw) for raw in requirements))
    except (InvalidVersion, InvalidSpecifier, InvalidRequirement) as exc:
        raise EnvironmentRefusal("package_environment_requirement_invalid", str(exc)) from exc


def validate(
    site_packages: Path,
    python: Path,
    base: base_observation.Observation,
) -> None:
    """Check every installed root and its selected extras against actual inherited providers.

    Only the trusted base interpreter runs; the application's interpreter and modules do
    not. Reading METADATA is small and repeated on reuse because the base can be updated
    independently of the immutable package generation. Unrelated base packages are not roots.
    """
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import importlib.metadata as m,json; "
            "print(json.dumps([[d.metadata.get('Name',''),d.version,"
            "d.metadata.get('Requires-Python',''),d.metadata.get_all('Requires-Dist',[])] "
            "for d in m.distributions()]))",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        if result.returncode or len(result.stdout) > MAX_METADATA_BYTES:
            raise ValueError("base dependency metadata is unavailable or oversized")
        rows: Any = json.loads(result.stdout)
        if not isinstance(rows, list) or len(rows) > MAX_PACKAGE_WHEELS:
            raise ValueError("base dependency metadata is invalid or oversized")
        available: dict[str, _Distribution] = {}
        for name, version, requires_python, requirements in rows:
            name = _name(name)
            # Mirror the measured base's precedence, not arbitrary sys.path ordering.
            if base.distribution_versions.get(name) == version and name not in available:
                available[name] = _distribution(version, requires_python, requirements)
        if available.keys() != base.distribution_versions.keys():
            raise ValueError("worker dependencies changed since its base observation")
    except (ValueError, TypeError) as exc:
        raise EnvironmentRefusal("package_environment_base_changed", str(exc)) from exc

    roots: set[str] = set()
    infos = sorted(site_packages.glob("*.dist-info"))
    if len(infos) > MAX_PACKAGE_WHEELS:
        raise EnvironmentRefusal(
            "package_environment_requirement_invalid", "too many distributions"
        )
    for info in infos:
        path = info / "METADATA"
        if (
            info.is_symlink()
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > MAX_METADATA_BYTES
        ):
            raise EnvironmentRefusal("package_environment_requirement_invalid", str(path))
        headers = BytesParser().parsebytes(path.read_bytes(), headersonly=True)
        name = _name(str(headers.get("Name", "")))
        if not name or name in roots:
            raise EnvironmentRefusal(
                "package_environment_requirement_invalid", f"duplicate or absent name: {name}"
            )
        roots.add(name)
        available[name] = _distribution(
            str(headers.get("Version", "")),
            str(headers.get("Requires-Python", "")),
            headers.get_all("Requires-Dist", []),
        )

    environment = _marker_environment(base)
    pending = [(name, Requirement(name)) for name in sorted(roots)]
    visited: dict[str, set[str]] = {}
    while pending:
        parent, requirement = pending.pop()
        name = _name(requirement.name)
        distribution = available.get(name)
        if distribution is None:
            raise EnvironmentRefusal(
                "package_environment_dependency_missing", f"{parent} requires {requirement}"
            )
        if not requirement.specifier.contains(distribution.version, prereleases=True):
            raise DependencyConflict(parent, requirement, distribution.version)
        if distribution.python and not SpecifierSet(distribution.python).contains(
            base.python_full_version, prereleases=True
        ):
            raise EnvironmentRefusal(
                "package_environment_python_incompatible",
                f"{name} requires Python {distribution.python}, "
                f"but the worker uses {base.python_full_version}",
            )
        extras = {"", *requirement.extras} - visited.get(name, set())
        if not extras:
            continue
        visited.setdefault(name, set()).update(extras)
        for dependency in distribution.requirements:
            if dependency.marker is None or any(
                dependency.marker.evaluate({**environment, "extra": extra}) for extra in extras
            ):
                pending.append((name, dependency))
