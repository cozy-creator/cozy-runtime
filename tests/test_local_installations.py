"""Real local installation survives Runtime restart without retaining wheel carriers."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from conftest import image_python
from cozy_runtime.internal import child_env
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.local import LocalRecordOwner, LocalRequest
from cozy_runtime.internal.worker.package_prepare import PreparationRefusal
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR
from test_job_preparation_isolation import package


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_installation_reopens_after_runtime_restart_without_carriers() -> None:
    with tempfile.TemporaryDirectory(prefix="cz-install.") as raw:
        root = Path(raw)
        environment, request = package(root, "retained")
        _, other = package(root, "another")
        config = RuntimeConfig(
            cozy_home=root / "home",
            credentials=Credentials(),
            child_base_env=tuple(
                sorted(
                    (k, v)
                    for k, v in os.environ.items()
                    if not child_env.erased(k) and k != "PYTHONPATH"
                )
            ),
        )

        def start() -> Worker:
            owner = LocalRecordOwner(
                LocalRequest(
                    entrypoint="retained", payload={}, outputs=(), kind="job", request_id="root"
                ),
                {},
                root / "run",
                package_installation_id="",
            )
            return Worker(
                config,
                WorkerOptions(
                    root=root / "worker",
                    devices="",
                    accelerator_backend="none",
                    python=str(image_python()),
                    install_root=environment,
                    artifact_cache=root / "artifacts",
                    tensorfs_root=root / "store",
                ),
                owner.host,
            )

        installation_id = request.package.installation_id
        worker = start()
        try:
            first = worker.prepare_local_package(request)
            installed = worker.local_operations[(request.operation_id, installation_id)].installed
            assert installed is not None
            # An installation ID names one package release; another package cannot adopt it.
            other.package.installation_id = installation_id
            with pytest.raises(PreparationRefusal, match="package_installation_conflict"):
                worker.prepare_local_package(other)
        finally:
            worker.shutdown()

        # Runtime memory and job plans are gone; only the installed generation survives.
        # A Host replay after recording the installation has no wheel paths.
        shutil.rmtree(root / "home/job-plans")
        for row in request.files:
            Path(row.path).unlink()
            row.path = ""
        worker = start()
        try:
            assert not worker.local_operations
            assert worker.prepare_local_package(request) == first
            restored = worker.local_operations[(request.operation_id, installation_id)]
            assert restored.installed is not None
            assert restored.installed.generation == installed.generation
            assert list((root / "home/job-plans").glob("*/*.json"))
            absent = pb.PrepareLocalPackageRequest()
            absent.CopyFrom(request)
            absent.package.installation_id = "local-never-installed"
            with pytest.raises(PreparationRefusal, match="local_package_reuse_unavailable"):
                worker.prepare_local_package(absent)
        finally:
            worker.shutdown()
