"""A locked package environment runs the worker's own SDK, never the Runtime its lock pinned.

Real uv installs, real venvs and a real executor spawn: the lock pins an older fake
`cozy-runtime`; the worker SDK (a separate, regular-installed venv) supplies another one.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

import cozy_runtime
from cozy_runtime.internal import package_requirements
from cozy_runtime.internal import package_environment as environment
from cozy_runtime.internal.worker import lanes
from cozy_runtime.internal.worker.child import ExecutorRuntimeMismatch
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from test_device_lanes import _config, _workspace
from test_end_to_end import NO_EXECUTOR
from test_publish_derivation import _SimpleIndex
from test_unpublished_exact_dependencies import _wheel

DIGEST = "sha256:" + "61" * 32


def _locked(index: _SimpleIndex, wheels: list[Path]) -> environment.LockedRequirements:
    pins = []
    for wheel in wheels:
        name, version = wheel.name.split("-")[:2]
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        pins.append(f"{name.replace('_', '-')}=={version} --hash=sha256:{digest}")
    return environment.read_locked_requirements(
        (f"--index-url {index.url}\n" + "\n".join(sorted(pins)) + "\n").encode()
    )


def _worker_sdk(root: Path, version: str) -> Path:
    """A worker interpreter with a regular (non-editable) cozy-runtime `version` installed."""
    wheels = root / f"sdk-wheels-{version}"
    wheels.mkdir(parents=True)
    runtime = _wheel(wheels, "cozy_runtime", (), f"VALUE={version!r}\n".encode(), version=version)
    venv = root / f"sdk-{version}"
    uv = shutil.which("uv") or "uv"
    subprocess.run([uv, "venv", "-q", "--python", sys.executable, str(venv)], check=True)
    subprocess.run(
        [uv, "pip", "install", "-q", "--no-deps", "--python", str(venv / "bin/python"), runtime],
        check=True,
    )
    return venv / "bin" / "python"


def _fixture(tmp_path: Path, requirement: str) -> tuple[_SimpleIndex, environment.LockedRequirements]:
    project = _wheel(tmp_path, "sdk_project", (requirement,), b"VALUE='project'\n")
    pinned = _wheel(tmp_path, "cozy_runtime", (), b"VALUE='locked'\n", version="1.0")
    index = _SimpleIndex(tmp_path / "index", [project, pinned])
    return index, _locked(index, [project, pinned])


def _imported(installed: environment.InstalledEnvironment) -> str:
    return subprocess.run(
        [str(installed.python), "-c", "import cozy_runtime; print(cozy_runtime.VALUE)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_locked_environment_runs_the_worker_sdk_not_the_lock_pin(tmp_path: Path) -> None:
    index, locked = _fixture(tmp_path, "cozy-runtime>=1.0")
    try:
        sdk = _worker_sdk(tmp_path, "2.0")
        installed = environment.materialize_locked(
            locked, locked.rows, tmp_path / "installs", environment_digest=DIGEST, sdk_python=sdk
        )
        assert _imported(installed) == "2.0"
        versions = {p.name for p in installed.site_packages.glob("cozy_runtime-*.dist-info")}
        assert versions == {"cozy_runtime-2.0.dist-info"}
        identity = (installed.generation / environment.CONTENT_IDENTITY_NAME).read_text()
        assert '"worker_sdk"' in identity and '"2.0"' in identity
    finally:
        index.close()


def test_environment_identity_follows_the_worker_sdk(tmp_path: Path) -> None:
    index, locked = _fixture(tmp_path, "cozy-runtime>=1.0")
    root = tmp_path / "installs"
    try:
        two, three = _worker_sdk(tmp_path, "2.0"), _worker_sdk(tmp_path, "3.0")

        def make(sdk: Path) -> environment.InstalledEnvironment:
            return environment.materialize_locked(
                locked, locked.rows, root, environment_digest=DIGEST, sdk_python=sdk
            )

        first = make(two)
        replay = make(two)
        assert replay.reused and replay.content_digest == first.content_digest
        upgraded = make(three)
        assert not upgraded.reused
        assert upgraded.content_digest != first.content_digest
        assert _imported(first) == "2.0" and _imported(upgraded) == "3.0"
    finally:
        index.close()


def test_worker_sdk_outside_the_declared_range_refuses_typed(tmp_path: Path) -> None:
    index, locked = _fixture(tmp_path, "cozy-runtime>=1.0,<2")
    root = tmp_path / "installs"
    try:
        sdk = _worker_sdk(tmp_path, "2.0")
        with pytest.raises(package_requirements.WorkerSdkOutOfRange) as caught:
            environment.materialize_locked(
                locked, locked.rows, root, environment_digest=DIGEST, sdk_python=sdk
            )
        assert caught.value.code == "package_environment_worker_sdk_out_of_range"
        assert caught.value.facts == {
            "package": "sdk-project",
            "distribution": "cozy-runtime",
            "required": "<2,>=1.0",
            "installed": "2.0",
        }
        assert "lock pinned 1.0" in str(caught.value)
        assert not list((root / "receipts").glob("*.json"))
    finally:
        index.close()


def test_an_editable_worker_sdk_is_not_copied(tmp_path: Path) -> None:
    # This checkout's Runtime is editable: the lock's row stays, the handshake guards it.
    index, locked = _fixture(tmp_path, "cozy-runtime>=1.0")
    try:
        installed = environment.materialize_locked(
            locked, locked.rows, tmp_path / "installs", environment_digest=DIGEST
        )
        if "cozy-runtime" in environment.worker_sdk(Path(sys.executable)):
            pytest.skip("this interpreter's Runtime is a regular install")
        assert _imported(installed) == "locked"
    finally:
        index.close()


def _shadowing_python(root: Path) -> Path:
    """An interpreter that imports this checkout's Runtime but whose metadata says 0.0.1:
    the shape of an environment whose lock carried another Runtime."""
    venv = root / "shadow"
    uv = shutil.which("uv") or "uv"
    subprocess.run([uv, "venv", "-q", "--python", sys.executable, str(venv)], check=True)
    site = next(venv.glob("lib/python*/site-packages"))
    info = site / "cozy_runtime-0.0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.3\nName: cozy-runtime\nVersion: 0.0.1\n")
    source = Path(cozy_runtime.__file__).parents[1]
    (site / "zz_worker_checkout.pth").write_text(
        f"{source}\n{sysconfig.get_path('purelib')}\n"
    )
    return venv / "bin" / "python"


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_executor_handshake_refuses_another_runtime() -> None:
    with _workspace() as root:
        worker = Worker(
            _config(root / "home"),
            WorkerOptions(root=root / "worker", devices="0"),
            InMemoryControlHost(),
        )
        try:
            lane = lanes.DeviceLane("lane-0", (0,), "0", worker_pid=os.getpid())
            same = worker.supervision.spawn(imposed=worker.imposed(lane))
            assert same.hello["runtime_version"] == cozy_runtime.__version__
            worker.supervision.retire_current(same, "handshake proof")
            worker.supervision.python = str(_shadowing_python(root))
            with pytest.raises(ExecutorRuntimeMismatch) as caught:
                worker.supervision.spawn(imposed=worker.imposed(lane))
            assert "cozy-runtime 0.0.1" in str(caught.value)
            assert f"this worker runs {cozy_runtime.__version__}" in str(caught.value)
        finally:
            worker.supervision.close()
