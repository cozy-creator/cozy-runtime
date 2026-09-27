"""Published preparation reuses one environment per release, lock and ABI and collects the rest."""

from __future__ import annotations

import functools
import hashlib
import http.server
import sys
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from cozy_runtime.internal import canonical, package_installation, package_interface
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import worker_pb2 as pb

INTERFACE = canonical.write(
    {
        "application": "reuse_fixture:app",
        "format": package_interface.SCHEMA,
        "entrypoints": [],
        "jobs": [
            {
                "name": "main",
                "request": {"fields": []},
                "result": {"fields": []},
                "publishes": False,
            }
        ],
    }
)


def _wheel(directory: Path, name: str, files: dict[str, bytes]) -> Path:
    info = f"{name}-1.0.dist-info"
    wheel = directory / f"{name}-1.0-py3-none-any.whl"
    members = {
        **files,
        f"{info}/METADATA": f"Metadata-Version: 2.3\nName: {name}\nVersion: 1.0\n".encode(),
        f"{info}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for member, body in members.items():
            archive.writestr(member, body)
        archive.writestr(f"{info}/RECORD", "".join(f"{member},,\n" for member in members))
    return wheel


@pytest.fixture
def index(tmp_path: Path) -> Iterator[tuple[str, str]]:
    project = _wheel(
        tmp_path,
        "reuse_fixture",
        {
            "reuse_fixture.py": b"VALUE = 1\n",
            "reuse_fixture-1.0.dist-info/package-interface.json": INTERFACE,
        },
    )
    helper = _wheel(tmp_path, "zz_helper", {"zz_helper.py": b"VALUE = 2\n"})
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def row(wheel: Path) -> str:
        name = wheel.name.split("-")[0].replace("_", "-")
        url = f"http://127.0.0.1:{server.server_port}/{wheel.name}"
        return f"{name} @ {url} --hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}"

    try:
        yield row(project) + "\n", row(project) + "\n" + row(helper) + "\n"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _prepare(worker: Worker, root: Path, lock: str) -> str:
    result = worker.prepare_package_set(
        pb.PreparePackageSetRequest(
            install_root=str(root / "install"),
            download_delegation=canonical.write(
                {
                    "format": "cozy.worker.v1.DownloadDelegation/1",
                    "models": [],
                    "packages": [{"package": "test/reuse-fixture", "release": "1.0"}],
                }
            ),
            locked_requirements=lock.encode(),
        )
    )
    return result.installed_package.installation_id


def test_prepare_reuses_one_environment_and_collects_superseded_ones(
    tmp_path: Path, index: tuple[str, str]
) -> None:
    only_project, with_helper = index
    worker = Worker(
        RuntimeConfig(cozy_home=tmp_path / "home", credentials=Credentials()),
        WorkerOptions(
            root=tmp_path / "worker",
            install_root=tmp_path / "install",
            artifact_cache=tmp_path / "artifacts",
            tensorfs_root=tmp_path / "store",
            python=sys.executable,
            worker_id="reuse",
        ),
        InMemoryControlHost(),
    )
    installations = tmp_path / "install" / "installations"
    plans = tmp_path / "home" / "job-plans"
    try:
        first = _prepare(worker, tmp_path, only_project)
        venv = installations / first / "venv"
        marker = venv / "reuse-marker"
        marker.write_text("the same venv")
        assert _prepare(worker, tmp_path, only_project) == first
        assert marker.read_text() == "the same venv"
        assert sorted(path.name for path in installations.glob("release-*")) == [first]

        # A different lock is a different environment; the unreferenced old one goes.
        second = _prepare(worker, tmp_path, with_helper)
        assert second != first
        assert sorted(path.name for path in installations.glob("release-*")) == [second]
        assert not (plans / first).exists() and (plans / second).is_dir()
        assert {key[1] for key in worker.prepared_installations} == {second}

        # An environment a live placement serves from is kept.
        again = _prepare(worker, tmp_path, only_project)
        assert again == first
        worker.placement = next(
            value for key, value in worker.prepared_installations.items() if key[1] == first
        )
        assert _prepare(worker, tmp_path, with_helper) == second
        assert sorted(path.name for path in installations.glob("release-*")) == sorted(
            [first, second]
        )
    finally:
        worker.placement = None
        worker.shutdown()


def test_published_install_is_the_lock_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither the package's nor the host's uv settings change what a published lock installs."""
    served = tmp_path / "index"
    directory = served / "simple" / "reuse-fixture"
    directory.mkdir(parents=True)
    project = _wheel(
        directory,
        "reuse_fixture",
        {
            "reuse_fixture.py": b"VALUE = 1\n",
            "reuse_fixture-1.0.dist-info/package-interface.json": INTERFACE,
        },
    )
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    lock = (
        f"--index-url http://127.0.0.1:{server.server_port}/simple\n"
        f"reuse-fixture==1.0 --hash=sha256:{hashlib.sha256(project.read_bytes()).hexdigest()}\n"
    )

    dead = "http://127.0.0.1:9/simple"
    pins = tmp_path / "pins.txt"
    pins.write_text("reuse-fixture==2.0\n")
    hostile = f'index-url = "{dead}"\nconstraint-dependencies = ["reuse-fixture==2.0"]\n'
    (tmp_path / "install").mkdir()
    # The author's project settings, discoverable from the installation's directory.
    (tmp_path / "install" / "pyproject.toml").write_text(
        '[project]\nname = "reuse-fixture"\nversion = "1.0"\n\n'
        '[tool.uv]\nconstraint-dependencies = ["reuse-fixture==2.0"]\noffline = true\n\n'
        f'[tool.uv.pip]\nindex-url = "{dead}"\n'
    )
    # The host's own user and explicit uv configuration and environment.
    (tmp_path / "xdg" / "uv").mkdir(parents=True)
    (tmp_path / "xdg" / "uv" / "uv.toml").write_text(hostile)
    (tmp_path / "host-uv.toml").write_text(hostile)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("UV_CONFIG_FILE", str(tmp_path / "host-uv.toml"))
    monkeypatch.setenv("UV_INDEX_URL", dead)
    monkeypatch.setenv("UV_DEFAULT_INDEX", dead)
    monkeypatch.setenv("UV_CONSTRAINT", str(pins))
    monkeypatch.setenv("UV_OVERRIDE", str(pins))
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("PIP_INDEX_URL", dead)

    worker = Worker(
        RuntimeConfig(cozy_home=tmp_path / "home", credentials=Credentials()),
        WorkerOptions(
            root=tmp_path / "worker",
            install_root=tmp_path / "install",
            artifact_cache=tmp_path / "artifacts",
            tensorfs_root=tmp_path / "store",
            python=sys.executable,
            worker_id="lock-alone",
        ),
        InMemoryControlHost(),
    )
    try:
        installation = _prepare(worker, tmp_path, lock)
        installed = package_installation.open_installation(tmp_path / "install", installation)
        record = next(installed.site_packages.glob("reuse_fixture-*.dist-info")) / "RECORD"
        assert "reuse_fixture.py" in record.read_text()
        assert (installed.site_packages / "reuse_fixture.py").read_bytes() == b"VALUE = 1\n"
    finally:
        worker.shutdown()
        server.shutdown()
        server.server_close()
        thread.join()
