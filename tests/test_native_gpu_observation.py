"""`observe_gpu` reads nvidia-smi's real answer, whatever depth the driver line reports.

Live incident (2026-09-02, RTX 4070 Laptop, driver 595.84): the driver pattern demanded
MAJOR.MINOR.PATCH, so a two-part driver refused every native dependency wheel on this host
as `package_environment_native_gpu_unavailable` — the message was the GPU's own identity.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import pytest

from cozy_runtime.internal.native_wheel import NativeWheelRefusal, observe_gpu

RECORDED = {
    "two-part": "NVIDIA GeForce RTX 4070 Laptop GPU, 595.84, 8.9",
    "three-part": "NVIDIA H200, 570.86.15, 9.0",
}


def _nvidia_smi_saying(tmp_path: Path, line: str, monkeypatch: pytest.MonkeyPatch) -> None:
    shim = tmp_path / "nvidia-smi"
    shim.write_text(f"#!/bin/sh\nprintf '%s\\n' '{line}'\n")
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")


@pytest.mark.parametrize("line", RECORDED.values(), ids=RECORDED.keys())
def test_recorded_driver_depths_observe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: str
) -> None:
    _nvidia_smi_saying(tmp_path, line, monkeypatch)
    name, driver, capability = (item.strip() for item in line.split(","))
    observed = observe_gpu(0)
    assert observed["name"] == name
    assert observed["driver_version"] == driver
    assert observed["sm"] == int(capability.replace(".", ""))


def test_a_non_numeric_driver_still_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _nvidia_smi_saying(tmp_path, "NVIDIA H200, [N/A], 9.0", monkeypatch)
    with pytest.raises(NativeWheelRefusal) as caught:
        observe_gpu(0)
    assert caught.value.code == "package_environment_native_gpu_unavailable"


def test_this_hosts_real_driver_observes() -> None:
    if shutil.which("nvidia-smi") is None:
        pytest.skip("no nvidia-smi on this host")
    try:
        observed = observe_gpu(0)
    except NativeWheelRefusal as caught:
        if "No devices were found" in caught.detail or "not found" in caught.detail.lower():
            pytest.skip(caught.detail)
        raise
    assert observed["sm"] >= 50
