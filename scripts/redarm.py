#!/usr/bin/env python3
"""Fence prover: PLANT each violation for real, observe the refusal, remove the plant.

Not a test — it mutates the working tree and restores it. Static arms run `scripts/fence.py`
and expect it red; live arms execute the real runtime function and expect its typed refusal.
Run it whenever a fence changes: `uv run python scripts/redarm.py`.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "cozy_runtime"


@dataclass(frozen=True, slots=True)
class Arm:
    fence: str
    path: pathlib.Path
    plant: str


STATIC_ARMS = (
    Arm("env-authority", SRC / "internal" / "child_env.py", "import os\nS = os.environ.get('X')\n"),
    Arm("package-imports", SRC / "author" / "_app.py", "import cozy_runtime.internal.config\n"),
    Arm("package-imports", SRC / "internal" / "config.py", "from cozy_runtime.cli import io\n"),
    Arm("forbidden-dependency", SRC / "internal" / "config.py", "import varena\n"),
    Arm("forbidden-dependency", ROOT / "pyproject.toml", 'dependencies = ["tensorhub"]'),
    Arm("single-config", SRC / "cli" / "version.py", "def read_config() -> None: ...\n"),
    Arm("single-config", SRC / "cli" / "io.py", "CONFIG = {'smuggled': True}\n"),
    Arm("no-interactive", SRC / "cli" / "new.py", "ANSWER = input('name? ')\n"),
)

LIVE_ARMS = (
    (
        "child-env-allowlist",
        "from cozy_runtime.internal.child_env import project_child_env\n"
        "project_child_env({'CUDA_LAUNCH_BLOCKING': '1'})\n",
        "not in the reviewed device-child env allowlist",
    ),
    (
        "one-config-authority",
        "from cozy_runtime.internal.config import read_config\n"
        "read_config({}); read_config({})\n",
        "exactly one config authority",
    ),
)


def fence() -> tuple[int, str]:
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fence.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    return out.returncode, out.stdout + out.stderr


def run_static(arm: Arm) -> str:
    original = arm.path.read_text()
    try:
        if arm.path.name == "pyproject.toml":
            planted = original.replace("dependencies = []", arm.plant, 1)
        else:
            planted = original + "\n" + arm.plant
        arm.path.write_text(planted)
        code, output = fence()
    finally:
        arm.path.write_text(original)
    if code == 0:
        return f"NOT PROVEN — {arm.fence} stayed green with a planted violation"
    lines = output.splitlines()
    line = next((ln.strip() for ln in lines if arm.path.name in ln), lines[0])
    return f"red — {arm.fence} ({arm.path.relative_to(ROOT)}): {line}"


def run_live(name: str, source: str, expect: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, cwd=ROOT, check=False
    )
    if out.returncode == 0 or expect not in out.stderr:
        return f"NOT PROVEN — {name} did not refuse (exit {out.returncode})"
    detail = next(ln for ln in out.stderr.splitlines() if expect in ln)
    return f"red — {name}: {detail.strip()}"


def main() -> int:
    code, _ = fence()
    if code != 0:
        print("refusing to plant: the tree is already fence-red", file=sys.stderr)
        return 1
    results = [run_static(a) for a in STATIC_ARMS]
    results += [run_live(*a) for a in LIVE_ARMS]
    for r in results:
        print(r)
    unproven = [r for r in results if r.startswith("NOT PROVEN")]
    code, _ = fence()
    if code != 0:
        print("PLANT NOT REMOVED — tree left fence-red", file=sys.stderr)
        return 1
    print(f"{len(results) - len(unproven)}/{len(results)} red arms observed; tree restored green")
    return 1 if unproven else 0


if __name__ == "__main__":
    raise SystemExit(main())
