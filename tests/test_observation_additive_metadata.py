"""Installed Runtime observation bridges tolerate additional producer diagnostics."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import base_observation, native_wheel, package_environment


def test_observation_accepts_additions_but_retains_required_facts_and_types() -> None:
    current = base_observation.Observation(
        python_abi="cp312",
        os_arch="linux/amd64",
        libc="glibc2.39",
        accelerator_backend="none",
        accelerator_build="cpu",
        torch_release="none",
        cuda_version="none",
        uv_version="0.10.0",
        distributions=(("cozy-runtime", "0.18.34"),),
        import_roots=(("cozy", ("cozy-runtime",)),),
        python_full_version="3.12.12",
    )
    body = base_observation._body(current)
    extended = json.loads(json.dumps({**body, "future_diagnostic": {"available": True}}))
    assert base_observation._parse(extended) == current
    for name in body:
        missing = dict(extended)
        missing.pop(name)
        with pytest.raises(base_observation.ObservationRefusal):
            base_observation._parse(missing)
    for name, invalid in (
        ("python_abi", 312),
        ("distributions", [["cozy-runtime", 34]]),
        ("import_roots", [["cozy", [34]]]),
    ):
        with pytest.raises(base_observation.ObservationRefusal):
            base_observation._parse({**extended, name: invalid})


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
