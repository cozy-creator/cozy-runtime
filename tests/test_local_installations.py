"""Real local installation survives Runtime restart without retaining wheel carriers."""

from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from conftest import image_python
from cozy_runtime.internal import child_env
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.local import LocalRecordOwner, LocalRequest
from cozy_runtime.internal.worker.package_prepare import PreparationRefusal
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR
from test_job_preparation_isolation import measure, package


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_installed_revision_reopens_after_runtime_restart_without_carriers() -> None:
    with tempfile.TemporaryDirectory(prefix="cz-install.") as raw:
        root = Path(raw)
        environment, request = package(root, "retained")
        config = RuntimeConfig(
            cozy_home=root / "home",
            credentials=Credentials(),
            child_base_env=tuple(
                sorted(
                    (k, v)
                    for k, v in os.environ.items()
                    if not child_env.erased(k)
                    and k not in ("PYTHONPATH", "COZY_BOOTSTRAP_CREDENTIAL")
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
                package_code_digest="",
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

        worker = start()
        try:
            measure(worker, request)
            before = time.monotonic()
            first = worker.prepare_local_package(request)
            cold_seconds = time.monotonic() - before
            revision = documents.spell(request.package.local_revision_digest)
            installed = worker.local_operations[(request.operation_id, revision)].installed
            assert installed is not None
            for wheel in request.wheels:
                Path(wheel.path).unlink()
                wheel.path = ""
            before = time.monotonic()
            assert worker.prepare_local_package(request) == first
            warm_seconds = time.monotonic() - before
        finally:
            worker.shutdown()

        # Runtime memory and job plans are gone; only installed code and its
        # installation receipts survive. A Host replay has no wheel paths.
        shutil.rmtree(root / "home/job-plans")
        worker = start()
        try:
            assert not worker.local_operations and not worker.prepared_revisions
            before = time.monotonic()
            assert worker.prepare_local_package(request) == first
            restart_seconds = time.monotonic() - before
            restored = worker.local_operations[(request.operation_id, revision)]
            assert restored.installed is not None
            assert restored.installed.generation == installed.generation
            assert worker.prepared_revisions[(restored.placement_id, revision)] is restored
            assert list((root / "home/job-plans").glob("*/*.json"))

            # A fresh transfer operation can bind the same immutable installation.
            other = pb.PrepareLocalPackageRequest()
            other.CopyFrom(request)
            other.operation_id = "fresh-root-install-reference"
            second = worker.prepare_local_package(other)
            assert second != first
            assert (
                worker.local_operations[(other.operation_id, revision)].installed
                == restored.installed
            )

            changed = pb.PrepareLocalPackageRequest()
            changed.CopyFrom(request)
            changed.package.local_revision_digest = b"x" * 32
            with pytest.raises(PreparationRefusal, match="local_package_reuse_unavailable"):
                worker.prepare_local_package(changed)
            changed.CopyFrom(request)
            changed.wheels[0].length += 1
            with pytest.raises(PreparationRefusal, match="local_package_revision_mismatch"):
                worker.prepare_local_package(changed)

            # Current base identity is part of reopening an installation.
            base = worker.base
            worker.base = dataclasses.replace(base, python_full_version="0.0.0")
            with pytest.raises(PreparationRefusal, match="local_package_reuse_unavailable"):
                worker.prepare_local_package(request)
            worker.base = base

            receipt = environment / "receipts" / (installed.receipt_digest[7:] + ".json")
            saved = receipt.read_bytes()
            receipt.chmod(0o600)
            receipt.write_bytes(b"corrupt")
            receipt.chmod(0o444)
            with pytest.raises(PreparationRefusal, match="local_package_reuse_unavailable"):
                worker.prepare_local_package(request)
            receipt.chmod(0o600)
            receipt.write_bytes(saved)
            receipt.chmod(0o444)

            moved = installed.generation.with_name(installed.generation.name + "-missing")
            installed.generation.rename(moved)
            with pytest.raises(PreparationRefusal, match="local_package_reuse_unavailable"):
                worker.prepare_local_package(request)
            moved.rename(installed.generation)
            assert worker.prepare_local_package(request) == first
            print(
                f"local installation: prepare={cold_seconds:.3f}s "
                f"reopen={warm_seconds:.3f}s restart_reopen={restart_seconds:.3f}s"
            )
        finally:
            worker.shutdown()
