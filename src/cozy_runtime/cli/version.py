"""`cozy-runtime version` — tag + commit + the two independent semvers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from cozy_runtime import PYTHON_CONTRACT_VERSION, WIRE_PROTOCOL_VERSION, __version__
from cozy_runtime.cli.io import Result
from cozy_runtime.internal.config import RuntimeConfig

_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(_SOURCE_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def run(config: RuntimeConfig, full: bool = False) -> Result:
    tag = _git("describe", "--tags", "--abbrev=0") or "untagged"
    commit = _git("rev-parse", "--short=12", "HEAD") or "unknown"
    rows: tuple[tuple[str, str], ...] = (
        ("distribution", __version__),
        ("tag", tag),
        ("commit", commit),
        ("wire_protocol", WIRE_PROTOCOL_VERSION),
        ("python_contract", PYTHON_CONTRACT_VERSION),
    )
    if full:
        present = config.credentials.present()
        rows += (
            ("cozy_home", str(config.cozy_home)),
            ("credentials", ", ".join(present) if present else "0 present"),
        )
    return Result(
        rows=rows,
        next=("cozy-runtime new <name> — scaffold an endpoint",),
    )
