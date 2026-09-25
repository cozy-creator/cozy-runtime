"""The real machine builtin prepares through a live supervisor admission channel."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_rented_builtin_keeps_frozen_sdk_and_generic_snapshot_refusal(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    wheels = tmp_path / "wheels"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheels)],
        cwd=root,
        check=True,
        capture_output=True,
    )
    wheel = next(wheels.glob("*.whl"))
    core = tmp_path / "core"
    subprocess.run(
        ["uv", "venv", str(core), "--python", sys.executable], check=True, capture_output=True
    )
    subprocess.run(
        ["uv", "pip", "install", "--python", str(core / "bin/python"), str(wheel), "uv==0.12.11"],
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        [
            str(core / "bin/python"),
            str(root / "tests/testdata/rented_builtin_prepare.py"),
            str(tmp_path / "worker"),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(core / "bin") + os.pathsep + os.environ.get("PATH", ""),
            "CUDA_VISIBLE_DEVICES": "",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RENTED BUILTIN PASS" in result.stdout
