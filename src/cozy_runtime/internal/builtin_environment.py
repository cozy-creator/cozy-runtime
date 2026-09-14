"""Retain an exact Runtime SDK and prepare its optional numerical environment."""

from __future__ import annotations

import importlib.metadata
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import builtin_operations, numerical_environment, package_environment


def sdk_identity() -> dict[str, str]:
    version = importlib.metadata.version("cozy-runtime")
    records = numerical_environment._installed_records({"cozy-runtime": version})
    return {
        "runtime_version": version,
        "implementation_digest": "sha256:" + records["cozy-runtime"]["record_digest"],
        "interface_digest": builtin_operations.digest(),
    }


@dataclass(frozen=True)
class Preparation:
    installed: package_environment.InstalledEnvironment
    environment_digest: str
    sdk: dict[str, str]
    closure: str

    def document(self) -> dict[str, Any]:
        return {
            "environment_python": str(self.installed.python),
            "environment_digest": self.environment_digest,
            "content_digest": self.installed.content_digest,
            "receipt_digest": self.installed.receipt_digest,
            "closure": self.closure,
            "runtime_version": self.sdk["runtime_version"],
            "implementation_digest": self.sdk["implementation_digest"],
            "package_interface": canonical_json.decode(builtin_operations.document()),
        }


def prepare(
    wheel_path: Path,
    root: Path,
    *,
    capture: bool = False,
    dependency_cache: Path | None = None,
) -> Preparation:
    """Capture this exact SDK, adding the optional numerical closure only on use."""
    sdk = sdk_identity()
    builtin_operations.validate_carrier(wheel_path, sdk["runtime_version"])
    python = Path(sys.executable)
    base = package_environment.observe_base(python)
    wheel, verified = package_environment.measure_wheel(wheel_path, "project", base)
    identity = canonical_json.digest(
        {
            "builtin": "operations",
            "wheel": wheel.digest,
            "interface": sdk["interface_digest"],
            "runtime_implementation": sdk["implementation_digest"],
        }
    )
    source = package_environment.Input(
        identity,
        wheel.digest,
        (wheel,),
        (package_environment.BlobGrant(wheel.digest, wheel.length, wheel_path, verified),),
    )
    installed = package_environment._materialize_dependency_snapshot(
        source,
        root,
        python,
        python,
        base,
        builtin_capture=capture,
        dependency_cache=dependency_cache,
    )
    builtin_operations.validate_installed(
        installed.site_packages / f"cozy_runtime_operations-{sdk['runtime_version']}.dist-info"
    )
    actual = package_environment.observe_base(installed.python)
    records, _ = numerical_environment._prepared_distributions(actual, installed.python)
    if "sha256:" + records["cozy-runtime"]["record_digest"] != sdk["implementation_digest"]:
        raise ConformanceError(
            "Runtime changed while its builtin was prepared", code="builtin_sdk_changed"
        )
    return Preparation(
        installed,
        identity,
        sdk,
        "\n".join(name + "==" + version for name, version in actual.distributions),
    )
