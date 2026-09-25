"""Validate a complete private environment without importing package code."""

from __future__ import annotations

from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path

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
            f"{requirement.name} {installed}; "
            "prepare a complete compatible private dependency closure",
        )


class WorkerSdkOutOfRange(DependencyConflict):
    """The package's declared range excludes the worker's own SDK, which it must run on."""

    def __init__(self, package: str, requirement: Requirement, worker: str, locked: str) -> None:
        super().__init__(package, requirement, worker)
        self.code = "package_environment_worker_sdk_out_of_range"
        self.detail = (
            f"{package} requires {requirement}, but this worker runs "
            f"{requirement.name} {worker} (the release lock pinned {locked}); "
            "use a worker image whose SDK satisfies the package, or republish the package"
        )
        self.args = (f"{self.code}: {self.detail}",)


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
    """Read private METADATA and validate transitive dependencies and selected extras."""
    available: dict[str, _Distribution] = {}

    roots: set[str] = set()
    infos = sorted(site_packages.glob("*.dist-info"))
    if not infos or len(infos) > MAX_PACKAGE_WHEELS:
        raise EnvironmentRefusal(
            "package_environment_requirement_invalid", "package metadata is absent or oversized"
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

    validate_closure(available, roots, base)


def validate_closure(
    available: dict[str, _Distribution], roots: set[str], base: base_observation.Observation
) -> None:
    """Traverse selected extras and transitive requirements within a private closure."""
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
