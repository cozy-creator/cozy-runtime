"""Retain Runtime's own installed environment for its library operations."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import sys
from pathlib import Path

from cozy_runtime.internal import package_installation
from cozy_runtime.internal.package_environment import EnvironmentRefusal


def prepare(root: Path) -> package_installation.InstalledEnvironment:
    """This Runtime's interpreter as its operations' installation: one per installed Runtime,
    named by its files, so every process of it reopens the interface described beside it."""
    runtime = importlib.metadata.distribution("cozy-runtime")
    identity = f"{sys.executable}\0{runtime.read_text('RECORD') or runtime.version}"
    identifier = "runtime-operations-" + hashlib.sha256(identity.encode()).hexdigest()[:32]
    with contextlib.suppress(EnvironmentRefusal):
        return package_installation.open_installation(root, identifier)
    return package_installation.retain_environment(
        root,
        Path(sys.executable),
        package="runtime/operations",
        release=runtime.version,
        installation_id=identifier,
    )
