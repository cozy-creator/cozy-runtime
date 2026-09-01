"""The end-to-end suite: real processes, real builds, no mocks.

Everything below runs against `examples/marco-polo`, a real weightless package with one
entrypoint and one job, built into a wheel and installed into a venv that has nothing else.
Nothing is stubbed, patched, or doubled: a test that passes because a fake agreed with it is
worse than no test.

The two legs that SPAWN AN EXECUTOR skip where the host cannot contain one, and say which
host fact stopped them — see `_executor_containment`. They run on an ordinary developer box.

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


def _executor_containment() -> str:
    """Why this host cannot spawn an executor, or "" when it can.

    Exactly the decision `internal/proctree.create_executor_scope` makes, read rather than
    performed so the probe creates nothing. A worker may only start an executor inside a
    cgroup it owns; where the unified hierarchy is writable but the worker's OWN cgroup is
    not delegated — a GitHub-hosted runner, most locked-down CI — there is no scope to make
    and the runtime refuses by design. The alternative process-tree backend wants the rental
    container topology (root worker, distinct non-root executor uid, bounded memory.max),
    which a hosted runner does not present either.
    """
    if sys.platform != "linux":
        return f"executor containment is Linux-only; this is {sys.platform}"
    root = Path("/sys/fs/cgroup")
    if not (root / "cgroup.controllers").is_file():
        return "no unified cgroup-v2 hierarchy is mounted"
    if bool(os.statvfs(root).f_flag & os.ST_RDONLY):
        return "the cgroup-v2 mount is read-only; the process-tree backend wants a rental pod"
    try:
        entries = Path("/proc/self/cgroup").read_text().splitlines()
        relative = next(line.split("::", 1)[1] for line in entries if line.startswith("0::"))
    except (OSError, StopIteration, IndexError):
        return "this process's cgroup-v2 delegation is unreadable"
    own = root / relative.strip("/")
    if not os.access(own, os.W_OK, effective_ids=True):
        return f"this process's own cgroup {own} is not delegated, so no child scope can be made"
    return ""


NO_EXECUTOR = _executor_containment()
needs_executor = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")


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


def test_installed_cli_describes_and_refuses(installed_cli: str) -> None:
    """describe -> exact replay -> the exit-code matrix, from `/`, no PYTHONPATH, stdin closed.

    The descriptor is a PUBLICATION output, so two derivations must be byte-identical, and
    the exit codes (`internal/exits.py`) are the contract a caller actually reads. None of
    this needs a device or an executor.
    """
    cli = (installed_cli, "--dir", str(PACKAGE))
    first = _run(*cli, "--json", "describe", cwd="/")
    second = _run(*cli, "--json", "describe", cwd="/")
    assert first.stdout == second.stdout
    assert json.loads(first.stdout)["application"] == "marco_polo_package:app"

    _run(*cli, "frobnicate", cwd="/", expect=2)
    _run(*cli, "run", "nosuchfn", "message=marco", "--out", "/tmp", cwd="/", expect=4)
    _run(*cli, "run", "marco", "message=nope", "--out", "/tmp", cwd="/", expect=3)
    _run(*cli, "run", "marco", "bogus=1", "--out", "/tmp", cwd="/", expect=3)


@needs_executor
def test_installed_cli_serves_a_request(installed_cli: str) -> None:
    """One served request, all the way through: worker control plus a spawned executor.

    decode -> resolve -> preflight -> hydrate -> invoke -> validate -> finalize, returning
    the package's own result over the real protocol bytes.
    """
    with tempfile.TemporaryDirectory(prefix="cz.", dir="/tmp") as out:
        served = _run(
            installed_cli,
            "--dir",
            str(PACKAGE),
            "--json",
            "run",
            "marco",
            "message=marco",
            "--out",
            out,
            cwd="/",
        )
    body = json.loads(served.stdout)
    assert body["result"] == {"message": "polo"}
    assert body["status"] == "OUTCOME_STATUS_SUCCEEDED"


@needs_executor
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
