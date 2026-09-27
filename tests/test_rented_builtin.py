"""A rented worker captures builtins from the Runtime's own installation under storage admission."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from test_end_to_end import NO_EXECUTOR


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_rented_builtin_captures_the_runtime_installation(tmp_path: Path) -> None:
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
