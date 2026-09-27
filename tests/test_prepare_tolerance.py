"""Published preparation installs from facts it can use and ignores harmless skew."""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest

from cozy_runtime.internal import canonical, package_interface
from cozy_runtime.internal.worker.package_prepare import PreparationRefusal, prepare_package_set
from cozy_runtime.protocol import worker_pb2 as pb
from test_derived_config_runtime import _released_source


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


def test_published_prepare_tolerates_lock_order_captured_patch_and_damaged_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interface = canonical.write(
        {
            "application": "skew_fixture:app",
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
    project = _wheel(
        tmp_path,
        "skew_fixture",
        {
            "skew_fixture.py": b"import zz_helper\nVALUE = zz_helper.VALUE\n",
            "skew_fixture-1.0.dist-info/package-interface.json": interface,
        },
    )
    helper = _wheel(tmp_path, "zz_helper", {"zz_helper.py": b"VALUE = 7\n"})
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def row(wheel: Path) -> str:
        name = wheel.name.split("-")[0].replace("_", "-")
        url = f"http://127.0.0.1:{server.server_port}/{wheel.name}"
        return f"{name} @ {url} --hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}"

    # A newer exporter's order, with one row emitted twice.
    locked = "\n".join([row(helper), row(project), row(helper)]) + "\n"
    # The client captured another patch of this interpreter's minor.
    captured = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro + 1}"
    monkeypatch.setenv("COZY_PACKAGE_PYTHONS", sys.executable)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    damaged = artifacts / hashlib.sha256(interface).hexdigest()
    damaged.write_bytes(b"torn write")
    damaged.chmod(0o444)
    try:
        result = prepare_package_set(
            pb.PreparePackageSetRequest(
                install_root=str(tmp_path / "install"),
                application="stale_client:app",
                model_slot_paths=["main.models.gone"],
                python_version=captured,
                download_delegation=canonical.write(
                    {
                        "format": "cozy.worker.v1.DownloadDelegation/1",
                        "models": [],
                        "packages": [{"package": "test/skew-fixture", "release": "1.0"}],
                    }
                ),
                locked_requirements=locked.encode(),
            ),
            artifact_cache=artifacts,
            tensorfs_root=tmp_path / "store",
            install_root=tmp_path / "install",
            python=Path(sys.executable),
            verified=lambda *_: None,
            job_plan_root=tmp_path / "plans",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    (placement,) = json.loads(result.placement_set.placement_set_canonical_bytes)["placements"]
    assert placement["installation_id"] == result.installed_package.installation_id
    assert json.loads(result.installed_package.package_interface)["application"] == (
        "skew_fixture:app"
    )
    assert damaged.read_bytes() == interface
    python = next((tmp_path / "install" / "installations").glob("*/venv/bin/python"))
    output = subprocess.check_output(
        [
            str(python),
            "-I",
            "-c",
            "import sys, skew_fixture; print(skew_fixture.VALUE, '%d.%d' % sys.version_info[:2])",
        ],
        text=True,
    )
    assert output.split() == ["7", f"{sys.version_info.major}.{sys.version_info.minor}"]


def test_published_prepare_dedupes_identical_selections_and_names_conflicts(
    tmp_path: Path,
) -> None:
    store, manifest, _length = _released_source(tmp_path / "models")
    interface = canonical.write(
        {
            "application": "slot_fixture:app",
            "format": package_interface.SCHEMA,
            "entrypoints": [],
            "jobs": [
                {
                    "name": "main",
                    "request": {"fields": []},
                    "result": {"fields": []},
                    "publishes": False,
                    "models": [
                        {"path": "main.models.source", "class": "Source", "component_use": {}}
                    ],
                }
            ],
        }
    )
    wheel = _wheel(
        tmp_path,
        "slot_fixture",
        {
            "slot_fixture.py": b"VALUE = 1\n",
            "slot_fixture-1.0.dist-info/package-interface.json": interface,
        },
    )
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    row = {
        "package": "test/slot-fixture",
        "slot": "main.models.source",
        "model": "test/model",
        "release": "1",
        "lane": "main",
        "manifest": manifest,
    }

    def prepare(case: str, models: list[dict[str, str]]) -> pb.PreparePackageSetResult:
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        url = f"http://127.0.0.1:{server.server_port}/{wheel.name}"
        return prepare_package_set(
            pb.PreparePackageSetRequest(
                install_root=str(tmp_path / case),
                download_delegation=canonical.write(
                    {
                        "format": "cozy.worker.v1.DownloadDelegation/1",
                        "models": models,
                        "packages": [{"package": "test/slot-fixture", "release": "1.0"}],
                    }
                ),
                locked_requirements=f"slot-fixture @ {url} --hash=sha256:{digest}\n".encode(),
            ),
            artifact_cache=tmp_path / "artifacts",
            tensorfs_root=Path(store.root),
            install_root=tmp_path / case,
            python=Path(sys.executable),
            verified=lambda *_: None,
            job_plan_root=tmp_path / "plans" / case,
        )

    try:
        result = prepare("repeated", [row, dict(row)])
        (placement,) = json.loads(result.placement_set.placement_set_canonical_bytes)["placements"]
        assert [model["repo"] for model in placement["models"]] == ["test/model"]
        other = {**row, "manifest": "sha256:" + "0" * 64}
        with pytest.raises(PreparationRefusal) as refused:
            prepare("conflict", [row, other])
        assert refused.value.code == "package_prepare_model_selection_mismatch"
        assert "selected twice" in refused.value.detail and "select one model" in (
            refused.value.detail
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
