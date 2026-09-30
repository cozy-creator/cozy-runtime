"""`CUDA_VISIBLE_DEVICES` is where a machine's GPUs are: the Runtime's inventory honours it."""

from __future__ import annotations

from pathlib import Path

import pytest

from cozy_runtime.internal import readiness

ROWS = "0, Card, GPU-a, 00000000:01:00.0, 8192, 1.0\n1, Card, GPU-b, 00000000:02:00.0, 8192, 1.0\n"


def _driver(tmp_path: Path) -> tuple[tuple[tuple[str, str], ...], Path]:
    """A recording nvidia-smi with two cards, first on the child PATH."""
    asked = tmp_path / "asked"
    tool = tmp_path / "bin" / "nvidia-smi"
    tool.parent.mkdir()
    tool.write_text(f"#!/bin/sh\necho asked >> {asked}\nprintf '{ROWS}'\n")
    tool.chmod(0o755)
    return (("PATH", f"{tool.parent}:/usr/bin:/bin"),), asked


@pytest.mark.parametrize(
    "visible,indexes",
    [(None, [0, 1]), ("1", [1]), ("GPU-a", [0]), ("1,GPU-a", [0, 1]), ("7", [])],
)
def test_inventory_keeps_the_visible_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, visible: str | None, indexes: list[int]
) -> None:
    env, _ = _driver(tmp_path)
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    # A CUDA base whose operator names devices the driver lacks serves CPU; it is not refused.
    gpus = readiness.runtime_gpus(env, required=visible is None)
    assert [gpu["device_index"] for gpu in gpus] == indexes


@pytest.mark.parametrize("visible", ["", " , "])
def test_hidden_devices_serve_cpu_without_asking_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, visible: str
) -> None:
    env, asked = _driver(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert readiness.runtime_gpus(env, required=True) == []
    assert not asked.exists()
