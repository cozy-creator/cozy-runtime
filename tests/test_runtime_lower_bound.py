"""A synced project locked to Runtime 0.18.67 that declares `>=0.18.67` runs this machine's
Runtime; an installation an older installer froze at 0.18.67 is rebuilt as a new generation.

Real: PyPI's cozy-runtime releases, uv, the product installer. The lock is made once, the way
the project locked on 0.18.67's release day, and cached; PyPI unreachable on a cold cache is a
skip. The executor seam over such an installation is `test_cross_version_seam`.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

import msgspec
import pytest
from packaging.version import Version

from cozy_runtime.internal import package_installation
from cozy_runtime.internal.config import package_install_environment

LOCKED = "0.18.67"
CACHE = (
    Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    / "cozy-runtime-tests"
    / "runtime-lower-bound"
)
PYTHON = Path(getattr(sys, "_base_executable", sys.executable))
PYPROJECT = f"""[project]
name = "bound-probe"
version = "1.0.0"
requires-python = ">=3.12,<3.13"
dependencies = ["cozy-runtime>={LOCKED}"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
only-include = ["bound_probe.py"]
"""


class _File(msgspec.Struct):
    upload_time_iso_8601: datetime.datetime


class _Index(msgspec.Struct):
    releases: dict[str, list[_File]]


def _published() -> _Index:
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/cozy-runtime/json", timeout=30) as r:
            return msgspec.json.decode(r.read(), type=_Index)
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"PyPI is unreachable: {exc}")


def _source(project: Path) -> None:
    project.mkdir()
    (project / "pyproject.toml").write_text(PYPROJECT)
    (project / "bound_probe.py").write_text("import cozy_runtime\n")
    lock = CACHE / f"uv-{LOCKED}.lock"
    if not lock.is_file():
        uploaded = max(row.upload_time_iso_8601 for row in _published().releases[LOCKED])
        done = subprocess.run(
            [
                "uv",
                "lock",
                "--project",
                str(project),
                "--python",
                str(PYTHON),
                "--exclude-newer",
                (uploaded + datetime.timedelta(minutes=1)).isoformat(),
            ],
            env=package_install_environment(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0, done.stderr[-4000:]
        CACHE.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / "uv.lock", lock)
    shutil.copyfile(lock, project / "uv.lock")
    assert f'name = "cozy-runtime"\nversion = "{LOCKED}"' in lock.read_text()


def _archive(project: Path, archive: Path) -> None:
    with tarfile.open(archive, "w") as tar:
        for name in ("pyproject.toml", "uv.lock", "bound_probe.py"):
            data = (project / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _runtime(python: Path) -> str:
    return subprocess.run(
        [str(python), "-c", "import cozy_runtime; print(cozy_runtime.__version__)"],
        env=package_install_environment(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_a_lower_bound_installs_the_machines_runtime_and_a_frozen_one_is_rebuilt(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _source(project)
    archive = tmp_path / "source.tar"
    _archive(project, archive)
    mine, published = importlib.metadata.version("cozy-runtime"), _published().releases
    expected = mine if mine in published else max(published, key=Version)
    root = tmp_path / "installs"

    installed = package_installation.install(
        root,
        package="local/bound-probe",
        release="1.0.0",
        python=PYTHON,
        installation_id="bound",
        source_archive=archive,
    )
    assert _runtime(installed.python) == expected
    record = json.loads((root / "installations/bound/installation.json").read_text())
    assert record["sdk"]["cozy-runtime"] == expected
    assert record["machine"] == package_installation.machine_sdk()

    # What an installer before this rule left: the lock synced exactly, a bare record.
    legacy = root / "installations" / "legacy"
    legacy.mkdir()
    shutil.copytree(project, legacy / "source")
    venv = {**package_install_environment(), "UV_PROJECT_ENVIRONMENT": str(legacy / "venv")}
    subprocess.run(
        [
            "uv",
            "sync",
            "--frozen",
            "--no-dev",
            "--python",
            str(PYTHON),
            "--project",
            str(legacy / "source"),
        ],
        env=venv,
        capture_output=True,
        check=True,
    )
    (legacy / "installation.json").write_text(
        json.dumps({"package": "local/bound-probe", "release": "1.0.0"})
    )
    frozen = package_installation.open_installation(root, "legacy")
    assert _runtime(frozen.python) == LOCKED

    rebuilt = package_installation.refresh(root, "legacy")
    assert rebuilt.generation != frozen.generation
    assert _runtime(rebuilt.python) == expected
    assert _runtime(frozen.python) == LOCKED, "a running executor's generation is kept"
    assert package_installation.refresh(root, "legacy").generation == rebuilt.generation


def test_a_published_installation_follows_the_machines_runtime(tmp_path: Path) -> None:
    """A published release's installation keeps its lock. Built on a machine that ran another
    Runtime (it was updated since), installing it again rebuilds its SDK as a new generation
    at this machine's Runtime; the rest of the lock stays as installed."""
    project = tmp_path / "project"
    _source(project)
    exported = subprocess.run(
        ["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--project", str(project)],
        env=package_install_environment(),
        capture_output=True,
        check=True,
    ).stdout
    mine, published = importlib.metadata.version("cozy-runtime"), _published().releases
    expected = mine if mine in published else max(published, key=Version)
    root = tmp_path / "installs"

    def install() -> package_installation.InstalledEnvironment:
        return package_installation.install(
            root,
            package="proof/bound-probe",
            release="1.0.0",
            python=PYTHON,
            installation_id="published",
            requirements=exported,
        )

    first = install()
    assert _runtime(first.python) == expected
    assert not package_installation.stale(root, "published")
    # As a machine on the lock's Runtime left it.
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(first.python),
            "--no-deps",
            f"cozy-runtime=={LOCKED}",
        ],
        env=package_install_environment(),
        capture_output=True,
        check=True,
    )
    path = root / "installations/published/installation.json"
    record = json.loads(path.read_text())
    record["machine"] = {"cozy-runtime": LOCKED}
    path.write_text(json.dumps(record))
    assert package_installation.stale(root, "published")

    again = install()
    assert again.generation != first.generation
    assert _runtime(again.python) == expected
    assert not package_installation.stale(root, "published")
    assert install().generation == again.generation
