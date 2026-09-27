"""A published package owns its environment despite stale host image observations."""

from __future__ import annotations

import base64
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

from cozy_runtime.internal import canonical, package_installation, package_interface
from cozy_runtime.internal.worker.package_prepare import PreparationRefusal, prepare_package_set
from cozy_runtime.protocol import worker_pb2 as pb


def test_real_install_ignores_advisory_inventory_and_application_but_checks_wheel(
    tmp_path: Path,
) -> None:
    interface = canonical.write(
        {
            "application": "inventory_fixture:app",
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
    info = "inventory_fixture-1.0.dist-info"
    wheel = tmp_path / "inventory_fixture-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        files = {
            "inventory_fixture.py": b"VALUE = 42\n",
            f"{info}/METADATA": b"Metadata-Version: 2.3\nName: inventory-fixture\nVersion: 1.0\n",
            f"{info}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            f"{info}/package-interface.json": interface,
        }
        for name, body in files.items():
            archive.writestr(name, body)
        archive.writestr(f"{info}/RECORD", "".join(f"{name},,\n" for name in files))
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    installed: list[package_installation.InstalledEnvironment] = []

    def prepare(
        case: str,
        inventory: pb.ImageInventory | None,
        *,
        wheel_hash: str = digest,
        application: str = "inventory_fixture:app",
    ) -> pb.PreparePackageSetResult:
        root = tmp_path / case
        return prepare_package_set(
            pb.PreparePackageSetRequest(
                install_root=str(root),
                application=application,
                download_delegation=canonical.write(
                    {
                        "format": "cozy.worker.v1.DownloadDelegation/1",
                        "models": [],
                        "packages": [{"package": "test/inventory-fixture", "release": "1.0"}],
                    }
                ),
                image_inventory=inventory,
                locked_requirements=(
                    f"inventory-fixture @ http://127.0.0.1:{server.server_port}/{wheel.name} "
                    f"--hash=sha256:{wheel_hash}\n"
                ).encode(),
            ),
            artifact_cache=tmp_path / "artifacts",
            tensorfs_root=tmp_path / "store",
            install_root=root,
            python=Path(sys.executable),
            verified=lambda *_: None,
            job_plan_root=tmp_path / "plans" / case,
            materialized=installed.append,
        )

    try:
        stale = pb.ImageInventory(
            profile="older-image",
            python="3.12.0",
            interpreters=[pb.PythonInterpreter(version="3.12.0", abi="cp312")],
            distributions=[pb.ImageDistribution(distribution="cozy-runtime", version="0.1.0")],
        )
        for case, inventory in (
            ("stale", stale),
            ("missing", None),
            ("empty", pb.ImageInventory()),
        ):
            result = prepare(case, inventory)
            assert json.loads(result.placement_set.placement_set_canonical_bytes)["placements"]
            python = next((tmp_path / case / "installations").glob("*/venv/bin/python"))
            output = subprocess.check_output(
                [
                    str(python),
                    "-I",
                    "-c",
                    "import inventory_fixture; print(inventory_fixture.VALUE)",
                ],
                text=True,
            )
            assert output.strip() == "42"
        adopted = prepare("wrong-application", stale, application="other:app")
        body = json.loads(adopted.placement_set.placement_set_canonical_bytes)
        interface = base64.b64decode(body["placements"][0]["package_interface"])
        assert json.loads(interface)["application"] == "inventory_fixture:app"
        with pytest.raises(PreparationRefusal, match="package_installation_uv_failed"):
            prepare("wrong-hash", stale, wheel_hash="0" * 64)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
