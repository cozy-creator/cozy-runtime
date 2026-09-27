"""Retain Runtime's own installed environment for its library operations."""

from __future__ import annotations

import importlib.metadata
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.internal import builtin_operations, package_installation


def sdk_identity() -> dict[str, str]:
    return {"runtime_version": importlib.metadata.version("cozy-runtime")}


@dataclass(frozen=True)
class Preparation:
    installed: package_installation.InstalledEnvironment
    sdk: dict[str, str]

    def document(self) -> dict[str, Any]:
        return {
            "environment_python": str(self.installed.python),
            "installation_id": self.installed.installation_id,
            "runtime_version": self.sdk["runtime_version"],
            "package_interface": canonical_json.decode(builtin_operations.document()),
        }


def prepare(
    root: Path,
    *,
    dependency_cache: Path | None = None,
) -> Preparation:
    sdk = sdk_identity()
    installed = package_installation.retain_environment(
        root,
        Path(sys.executable),
        package="runtime/operations",
        release=sdk["runtime_version"],
    )
    return Preparation(installed, sdk)
