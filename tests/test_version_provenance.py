"""Installed Runtime reports wheel metadata and embedded provenance, never ambient Git."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = "a" * 40
PROJECT_VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def test_installed_wheel_version_is_independent_of_caller_repository(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    git = shutil.which("git")
    assert uv is not None and git is not None

    source = tmp_path / "source"
    shutil.copytree(
        ROOT,
        source,
        ignore=shutil.ignore_patterns(".git", ".venv", "dist", "__pycache__", ".mypy_cache"),
    )
    subprocess.run(
        [sys.executable, str(source / "scripts/stamp-build-provenance.py"), STAMP],
        check=True,
    )
    wheelhouse = tmp_path / "wheelhouse"
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(wheelhouse), str(source)],
        check=True,
        capture_output=True,
    )
    wheel = next(wheelhouse.glob(f"cozy_runtime-{PROJECT_VERSION}-*.whl"))
    environment = tmp_path / "environment"
    subprocess.run(
        [uv, "venv", "--python", "3.12", str(environment)],
        check=True,
        capture_output=True,
    )
    python = environment / "bin" / "python"
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), str(wheel)],
        check=True,
        capture_output=True,
    )

    caller = tmp_path / "caller"
    caller.mkdir()
    subprocess.run([git, "init", "-q"], cwd=caller, check=True)
    (caller / "README").write_text("ambient caller repository\n")
    subprocess.run([git, "add", "README"], cwd=caller, check=True)
    subprocess.run(
        [
            git,
            "-c",
            "user.name=Caller",
            "-c",
            "user.email=caller@example.invalid",
            "commit",
            "-qm",
            "caller",
        ],
        cwd=caller,
        check=True,
    )
    caller_commit = subprocess.run(
        [git, "rev-parse", "--short=12", "HEAD"],
        cwd=caller,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    result = subprocess.run(
        [str(python), "-m", "cozy_runtime.cli.main", "--json", "version"],
        cwd=caller,
        check=True,
        capture_output=True,
        text=True,
    )
    reported = json.loads(result.stdout)
    assert reported["distribution"] == PROJECT_VERSION
    assert reported["tag"] == f"v{PROJECT_VERSION}"
    assert reported["commit"] == STAMP[:12]
    assert reported["commit"] != caller_commit
