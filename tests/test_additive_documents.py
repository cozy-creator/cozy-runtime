"""Documents from newer or older writers keep working when they carry additive fields."""

from __future__ import annotations

import base64
import functools
import hashlib
import http.server
import json
import sys
import threading
import zipfile
from pathlib import Path

import pytest

from cozy_runtime.internal import canonical, package_interface
from cozy_runtime.internal.worker.package_prepare import prepare_package_set
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.session import read_placement_set
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def _interface() -> bytes:
    return canonical.write(
        {
            "application": "additive_fixture:app",
            "format": package_interface.SCHEMA,
            "entrypoints": [],
            "jobs": [
                {
                    "name": "main",
                    "request": {
                        "fields": [{"name": "steps", "type": "int", "since": "a later writer"}],
                        "doc": "an additive schema attribute",
                    },
                    "result": {"fields": []},
                    "publishes": False,
                    "timeout_hint": 30,
                }
            ],
            "generator": "a newer Runtime",
        }
    )


def _wheel(directory: Path, interface: bytes) -> Path:
    info = "additive_fixture-1.0.dist-info"
    wheel = directory / "additive_fixture-1.0-py3-none-any.whl"
    members = {
        "additive_fixture.py": b"VALUE = 1\n",
        f"{info}/METADATA": b"Metadata-Version: 2.3\nName: additive-fixture\nVersion: 1.0\n",
        f"{info}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{info}/package-interface.json": interface,
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)
        archive.writestr(f"{info}/RECORD", "".join(f"{name},,\n" for name in members))
    return wheel


def test_prepare_and_desired_state_accept_additive_fields(tmp_path: Path) -> None:
    interface = _interface()
    wheel = _wheel(tmp_path, interface)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    try:
        result = prepare_package_set(
            pb.PreparePackageSetRequest(
                install_root=str(tmp_path / "install"),
                application="additive_fixture:app",
                download_delegation=canonical.write(
                    {
                        "format": "cozy.worker.v1.DownloadDelegation/1",
                        "models": [],
                        "packages": [{"package": "test/additive-fixture", "release": "1.0"}],
                        "priority": "a field this worker has never heard of",
                    }
                ),
                locked_requirements=(
                    f"additive-fixture @ http://127.0.0.1:{server.server_port}/{wheel.name} "
                    f"--hash=sha256:{digest}\n"
                ).encode(),
            ),
            artifact_cache=tmp_path / "artifacts",
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
    assert result.installed_package.package_interface == interface

    # A placement set replayed through another Runtime version with additive fields.
    document = json.loads(result.placement_set.placement_set_canonical_bytes)
    document["placements"][0]["serving_hint"] = "warm"
    document["rollout"] = {"wave": 2}
    raw = canonical.write(document)
    (placement,) = read_placement_set(
        pb.DesiredPlacementSet(
            placement_set_digest=documents.digest_of(raw), placement_set_canonical_bytes=raw
        )
    )
    assert base64.b64decode(placement["package_interface"]) == interface

    # A job plan written by another Runtime version with a field this one does not read.
    (plan,) = (tmp_path / "plans").glob("*/*.json")
    record = json.loads(plan.read_bytes())
    binding = JobBinding.read({**record, "scheduler_class": "batch"})
    assert binding.job == "main" and binding.application == "additive_fixture:app"


def test_missing_required_interface_fields_still_refuse() -> None:
    body = json.loads(_interface())
    del body["jobs"][0]["result"]
    with pytest.raises(package_interface.StalePackageInterface, match="missing result"):
        package_interface.read_bytes(canonical.write(body))
    del body["jobs"]
    with pytest.raises(package_interface.StalePackageInterface, match="missing jobs"):
        package_interface.read_bytes(canonical.write(body))
