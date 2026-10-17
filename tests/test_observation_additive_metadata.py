"""Installed Runtime observation bridges tolerate additional producer diagnostics."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

from cozy_runtime.internal import base_observation, native_wheel, package_environment


def test_a_bare_interpreter_is_observed_as_this_machines_linux_arch(tmp_path: Path) -> None:
    """x86_64 and aarch64 alike: the observation, lock markers and ELF check name this CPU."""
    venv = tmp_path / "venv"
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(venv)], check=True, capture_output=True
    )
    observed = base_observation.from_python(venv / "bin/python")
    assert observed.os_arch == base_observation.OS_ARCH[platform.machine().lower()]
    assert (
        package_environment._marker_environment(observed)["platform_machine"] == platform.machine()
    )
    native_wheel._ELF(Path(sys.executable).resolve(), require_shared=False)
