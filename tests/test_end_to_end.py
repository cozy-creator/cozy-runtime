"""The end-to-end CI suite: three real runs, no mocks.

Every test below starts a REAL process against a REAL build. Nothing here is stubbed,
patched, or doubled — a test that passes because a fake agreed with it is worse than no
test, which is why this repository had none for so long. What it costs is about a minute,
which is what a build of the wheel plus two venvs costs, because that is the thing under
test.

    uv run pytest
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT = 900


def _run(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def _green(result: subprocess.CompletedProcess[str], expected: str) -> None:
    if result.returncode != 0:
        pytest.fail(
            f"exit {result.returncode}\n--- stdout ---\n{result.stdout[-8000:]}"
            f"\n--- stderr ---\n{result.stderr[-8000:]}"
        )
    assert expected in result.stdout, result.stdout[-4000:]


def test_author_surface_runs_end_to_end() -> None:
    """The real invocation kernel and construction plane over the tiny-model packages.

    Decode, resolve, preflight, hydrate, invoke, validate, finalize; the descriptor's
    canonical bytes; the neutral author-error set at the invocation boundary; and a
    byte-identical reproduction of one release across two freshly built venvs.
    """
    _green(
        _run("uv", "run", "python", "scripts/verify.py"),
        "all sections executed against the real kernel",
    )


@pytest.mark.skipif(shutil.which("cc") is None, reason="the network cut needs a C compiler")
def test_installed_cli_exits_and_refuses() -> None:
    """The PUBLIC CLI, from a freshly built wheel installed into a fresh venv.

    Run from `/` with no `PYTHONPATH` and stdin closed, so nothing about the source tree can
    reach it: the exit-code matrix, exact Descriptor/1 bytes replayed twice, typed media
    refusals, and an offline invocation with `connect(2)` cut out from under the process.
    """
    _green(
        _run("sh", "proofs/cli/run.sh"),
        "CLI proof: installed-wheel exits",
    )


def test_package_environment_materializes_and_refuses() -> None:
    """`materialize()` — the function production calls — against adversarial fixtures.

    Byte-identical cold and rebuild receipts, each venv importing the exact project and
    dependency wheels, and the refusal matrix: a bit-flipped wheel, a zip bomb, lying
    `import_roots`, duplicate package roots, plus the receipt race and the
    no-partial-output rule.
    """
    _green(
        _run("sh", "proofs/environment/run.sh"),
        "environment proof: installed public CLI produced byte-identical",
    )
