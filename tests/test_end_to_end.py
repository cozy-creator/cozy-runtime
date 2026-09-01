"""The end-to-end suite: real processes, real builds, no mocks.

Two runs, both against `examples/marco-polo` — a real weightless package with one
entrypoint and one job. Nothing here is stubbed, patched, or doubled: a test that passes
because a fake agreed with it is worse than no test.

    uv run pytest
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "examples" / "marco-polo"
TIMEOUT = 420


def _run(*argv: str, cwd: Path | str = ROOT, expect: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        stdin=subprocess.DEVNULL,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )
    if result.returncode != expect:
        pytest.fail(
            f"{argv} exited {result.returncode}, wanted {expect}\n"
            f"--- stdout ---\n{result.stdout[-6000:]}\n--- stderr ---\n{result.stderr[-6000:]}"
        )
    return result


@pytest.fixture(scope="session")
def installed_cli() -> Iterator[str]:
    """The PUBLIC CLI, built into a wheel and installed into a venv that has nothing else.

    This is the shape a user gets: `pip install` of the built artifact, not the source tree.
    Anything reachable only from a checkout is invisible here, which is the point.
    """
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to build and install the wheel under test"
    with tempfile.TemporaryDirectory(prefix="cozy-e2e.") as raw:
        root = Path(raw)
        _run(uv, "build", "--wheel", "--out-dir", str(root / "dist"))
        _run(uv, "build", "--wheel", "--out-dir", str(root / "dist"), str(PACKAGE))
        wheels = sorted(str(path) for path in (root / "dist").glob("*.whl"))
        assert len(wheels) == 2, wheels
        _run(uv, "venv", "--python", sys.executable, str(root / "venv"))
        _run(uv, "pip", "install", "--python", str(root / "venv" / "bin" / "python"), *wheels)
        yield str(root / "venv" / "bin" / "cozy-runtime")


def test_installed_cli_describes_and_serves(installed_cli: str) -> None:
    """describe -> exact replay -> invoke, from `/`, with no PYTHONPATH and stdin closed.

    The descriptor is a PUBLICATION output, so two derivations must be byte-identical; the
    invocation goes through the real kernel — decode, resolve, preflight, hydrate, invoke,
    validate, finalize — and returns the package's own result.
    """
    cli = (installed_cli, "--dir", str(PACKAGE))
    first = _run(*cli, "--json", "describe", cwd="/")
    second = _run(*cli, "--json", "describe", cwd="/")
    assert first.stdout == second.stdout
    assert json.loads(first.stdout)["application"] == "marco_polo_package:app"

    with tempfile.TemporaryDirectory(prefix="cz.", dir="/tmp") as out:
        served = _run(*cli, "--json", "run", "marco", "message=marco", "--out", out, cwd="/")
    body = json.loads(served.stdout)
    assert body["result"] == {"message": "polo"}
    assert body["status"] == "OUTCOME_STATUS_SUCCEEDED"

    # The exit-code matrix (`internal/exits.py`), which is the contract a caller reads.
    _run(*cli, "frobnicate", cwd="/", expect=2)
    _run(*cli, "run", "nosuchfn", "message=marco", "--out", "/tmp", cwd="/", expect=4)
    _run(*cli, "run", "marco", "message=nope", "--out", "/tmp", cwd="/", expect=3)
    _run(*cli, "run", "marco", "bogus=1", "--out", "/tmp", cwd="/", expect=3)


def test_installed_cli_runs_a_job_through_the_local_worker(installed_cli: str) -> None:
    """A JOB, which is a real two-process run: worker control plus a spawned executor.

    Not a second code path — the same acceptance boundary, plan and attempt kernel a served
    request takes, over a package that declares no model. The short `--out` root is
    deliberate: the executor's control socket lives under it and `sockaddr_un.sun_path`
    holds 108 bytes.
    """
    with tempfile.TemporaryDirectory(prefix="cz.", dir="/tmp") as out:
        done = _run(
            installed_cli,
            "--dir",
            str(PACKAGE),
            "--json",
            "job",
            "marco_job",
            "message=marco",
            "--out",
            out,
            cwd="/",
        )
    body = json.loads(done.stdout)
    assert body["status"] == "OUTCOME_STATUS_SUCCEEDED", body
    assert body["result"] == {"message": "polo"}
