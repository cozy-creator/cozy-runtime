"""Real local installation survives Runtime restart without retaining wheel carriers."""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from conftest import image_python
from cozy_runtime.internal import (
    canonical,
    child_env,
    package_environment,
    placement_materialization,
)
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.local import LocalRecordOwner, LocalRequest
from cozy_runtime.internal.worker.machine_child_target import Target, computation
from cozy_runtime.internal.worker.package_prepare import PreparationRefusal
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR, _run, _wheel_rows
from test_installed_dependencies import _distribution
from test_job_preparation_isolation import measure, package


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
@pytest.mark.parametrize("snapshot", [False, True], ids=["image-base", "local-sdk"])
def test_installed_revision_reopens_after_runtime_restart_without_carriers(snapshot: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="cz-install.") as raw:
        root = Path(raw)
        environment, request = package(root, "retained")
        _, other_request = package(root, "another")
        if not snapshot:
            # The wire lane transfers the complete wheel closure. It cannot borrow
            # Runtime from the image through system site-packages.
            runtime_wheel = next(
                image_python().parent.parent.parent.glob("wheels/cozy_runtime-*.whl")
            )
            sdk_wheels = root / "sdk-wheels"
            _run(
                str(image_python()),
                "-m",
                "pip",
                "download",
                "--only-binary=:all:",
                "--dest",
                str(sdk_wheels),
                str(runtime_wheel),
            )
            _, dependency = package(root, "dependency")
            dependency_path = Path(dependency.wheels[0].path)
            for closure in (request, other_request):
                retained_wheels = Path(closure.wheels[0].path).parent
                shutil.copyfile(dependency_path, retained_wheels / dependency_path.name)
                for wheel in sdk_wheels.glob("*.whl"):
                    shutil.copyfile(wheel, retained_wheels / wheel.name)
                closure.ClearField("wheels")
                closure.wheels.extend(_wheel_rows(list(retained_wheels.glob("*.whl"))))
        dependency_python = None
        if snapshot:
            uv = shutil.which("uv")
            assert uv is not None
            sdk = root / "sdk"
            _run(uv, "venv", "--python", str(image_python()), str(sdk))
            dependency_python = sdk / "bin/python"
            runtime_wheel = next(
                image_python().parent.parent.parent.glob("wheels/cozy_runtime-*.whl")
            )
            _run(uv, "pip", "install", "--python", str(dependency_python), str(runtime_wheel))
            _distribution(
                next(sdk.glob("lib/python*/site-packages")),
                "installation_dependency",
                {"installation_dependency.py": b"value=1\n"},
            )
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
                    environment_python=dependency_python,
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
            if snapshot:
                # Same project wheels and dependency versions, different frozen SDK bytes.
                sdk_site = next((root / "sdk").glob("lib/python*/site-packages"))
                _distribution(
                    sdk_site,
                    "installation_dependency",
                    {"installation_dependency.py": b"value=2\n"},
                )
                placement = documents.read(
                    first.placement_set.placement_set_canonical_bytes, pb.PlacementSet
                )["placements"][0]
                material = placement_materialization.package_environment_input(
                    package_name=request.package.package,
                    release=request.package.release,
                    project_wheel=placement["development"]["project_wheel"],
                    environment=placement["environment"],
                    environment_digest=placement["environment_digest"],
                    wheel_paths={
                        documents.spell(row.digest): Path(row.path) for row in request.wheels
                    },
                )
                changed_snapshot = package_environment.materialize(
                    material,
                    environment,
                    python=image_python(),
                    base=worker.base,
                    dependency_python=dependency_python,
                )
                assert changed_snapshot.content_digest != installed.content_digest
                assert package_environment.observe_base(changed_snapshot.python).distributions == (
                    package_environment.observe_base(installed.python).distributions
                )
                _assert_memo_identity(installed, changed_snapshot, revision)
                _distribution(
                    sdk_site,
                    "installation_dependency",
                    {"installation_dependency.py": b"value=1\n"},
                )
            measure(worker, other_request)
            worker.prepare_local_package(other_request)
            other_installed = worker.local_operations[
                (
                    other_request.operation_id,
                    documents.spell(other_request.package.local_revision_digest),
                )
            ].installed
            assert other_installed is not None
            _assert_memo_identity(installed, other_installed, revision)
            assert installed.generation != other_installed.generation
            assert (
                package_environment.parse_environment_receipt(installed.receipt_bytes)[
                    "environment_digest"
                ]
                == package_environment.parse_environment_receipt(other_installed.receipt_bytes)[
                    "environment_digest"
                ]
            )
            installation_path = environment / "local-installations" / (revision[7:] + ".json")
            inventory = canonical.parse_canonical(installation_path.read_bytes())
            assert "receipt_digest" not in inventory
            assert ("dependency_snapshot" in inventory) is snapshot
            if not snapshot:
                # A second valid generation has the same project, but drops one
                # supplied dependency. Its valid receipt must not cross-bind either.
                original_inventory = installation_path.read_bytes()
                placement = inventory["placement_set"]["placements"][0]
                placement["environment"]["local_wheels"] = [
                    row
                    for row in placement["environment"]["local_wheels"]
                    if row["distribution"] != "prepare-dependency"
                ]
                placement["environment_digest"] = documents.spell(
                    documents.embedded_document_digest(
                        placement["environment"], "cozy.worker.v1.Environment"
                    )
                )
                material = placement_materialization.package_environment_input(
                    package_name=request.package.package,
                    release=request.package.release,
                    project_wheel=placement["development"]["project_wheel"],
                    environment=placement["environment"],
                    environment_digest=placement["environment_digest"],
                    wheel_paths={
                        documents.spell(row.digest): Path(row.path) for row in request.wheels
                    },
                )
                missing_dependency = package_environment.materialize(
                    material, environment, python=image_python(), base=worker.base
                )
                assert missing_dependency.content_digest != installed.content_digest
                assert (
                    package_environment.open_generation(
                        environment,
                        receipt_digest=missing_dependency.receipt_digest,
                        environment_digest=placement["environment_digest"],
                        python=image_python(),
                    ).generation
                    == missing_dependency.generation
                )
                _replace(installation_path, canonical.write(inventory))
                with pytest.raises(PreparationRefusal, match="local_package_revision_mismatch"):
                    worker.prepare_local_package(request)
                _replace(installation_path, original_inventory)
                inventory = canonical.parse_canonical(original_inventory)
            original_inventory = installation_path.read_bytes()
            inventory["placement_set"]["placements"][0]["development"]["project_wheel"][
                "version"
            ] = "9.0.0"
            _replace(installation_path, canonical.write(inventory))
            with pytest.raises(PreparationRefusal, match="local_package_revision_mismatch"):
                worker.prepare_local_package(request)
            _replace(installation_path, original_inventory)
            inventory = canonical.parse_canonical(original_inventory)
            carrier_paths = [Path(wheel.path) for wheel in request.wheels]
            carriers = [path.read_bytes() for path in carrier_paths]
            for wheel in request.wheels:
                Path(wheel.path).unlink()
                wheel.path = ""
            before = time.monotonic()
            assert worker.prepare_local_package(request) == first
            warm_seconds = time.monotonic() - before
            # The current operation has no carriers, but metadata absence is repairable.
            placement = inventory["placement_set"]["placements"][0]
            interface_path = root / "artifacts" / placement["package_interface"]["digest"][7:]
            interface_path.unlink()
            with pytest.raises(PreparationRefusal, match="local_package_reuse_unavailable"):
                worker.prepare_local_package(request)
            repair = pb.PrepareLocalPackageRequest()
            repair.CopyFrom(request)
            for wheel, path, contents in zip(repair.wheels, carrier_paths, carriers, strict=True):
                path.write_bytes(contents)
                wheel.path = str(path)
            assert worker.prepare_local_package(repair) == first
            for path in carrier_paths:
                path.unlink()
        finally:
            worker.shutdown()

        # Runtime memory and job plans are gone; only installed code and its
        # installation receipts survive. A Host replay has no wheel paths.
        shutil.rmtree(root / "home/job-plans")
        worker = start()
        if snapshot:
            # Worker startup observes the configured base once. Reopening the
            # installation itself must not need the source SDK again.
            shutil.rmtree(root / "sdk")
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
            # A valid other-project receipt shares the Environment digest. An old or
            # corrupt pointer in the inventory must never select that executable code.
            saved_inventory = installation_path.read_bytes()
            inventory["receipt_digest"] = other_installed.receipt_digest
            _replace(installation_path, canonical.write(inventory))
            assert worker.prepare_local_package(request) == first
            assert (
                worker.local_operations[(request.operation_id, revision)].installed
                == restored.installed
            )
            receipt = environment / "receipts" / (installed.receipt_digest[7:] + ".json")
            hidden = receipt.with_suffix(".missing")
            receipt.rename(hidden)
            with pytest.raises(PreparationRefusal, match="local_package_reuse_unavailable"):
                worker.prepare_local_package(request)
            hidden.rename(receipt)
            other_placement = canonical.parse_canonical(
                (
                    environment
                    / "local-installations"
                    / (other_request.package.local_revision_digest.hex() + ".json")
                ).read_bytes()
            )["placement_set"]["placements"][0]
            inventory["placement_set"]["placements"][0]["development"]["project_wheel"] = (
                other_placement["development"]["project_wheel"]
            )
            _replace(installation_path, canonical.write(inventory))
            with pytest.raises(PreparationRefusal, match="local_package_revision_mismatch"):
                worker.prepare_local_package(request)
            _replace(installation_path, saved_inventory)

            plan = next((root / "home/job-plans" / revision[7:]).glob("*.json"))
            plan_raw = plan.read_bytes()
            _replace(plan, b"corrupt")
            with pytest.raises(PreparationRefusal, match="local_package_job_plan_conflict"):
                worker.prepare_local_package(request)
            assert plan.read_bytes() == b"corrupt"
            _replace(plan, plan_raw)

            if snapshot:
                from cozy_runtime.internal import _installed_dependencies

                snapshot_path = installed.generation / _installed_dependencies.MANIFEST_NAME
                snapshot_raw = snapshot_path.read_bytes()
                _replace(snapshot_path, b"corrupt")
                with pytest.raises(PreparationRefusal, match="local_package_reuse_unavailable"):
                    worker.prepare_local_package(request)
                _replace(snapshot_path, snapshot_raw)
                inventory = canonical.parse_canonical(saved_inventory)
                inventory["dependency_snapshot"]["digest"] = "sha256:" + "1" * 64
                _replace(installation_path, canonical.write(inventory))
                with pytest.raises(PreparationRefusal, match="local_package_reuse_unavailable"):
                    worker.prepare_local_package(request)
                _replace(installation_path, saved_inventory)

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
            if snapshot:
                moved.rename(installed.generation)
            else:
                repair = pb.PrepareLocalPackageRequest()
                repair.CopyFrom(request)
                repair.operation_id = "repair-missing-generation"
                repair_wheels = environment / ".stage" / repair.operation_id / "wheels"
                repair_wheels.mkdir(parents=True)
                for wheel, contents in zip(repair.wheels, carriers, strict=True):
                    path = repair_wheels / wheel.filename
                    path.write_bytes(contents)
                    wheel.path = str(path)
                # Existing partial generations are not an eviction and cannot be
                # replaced while another execution may still own their files.
                installed.generation.mkdir(mode=0o555)
                with pytest.raises(PreparationRefusal, match="package_environment_content_absent"):
                    worker.prepare_local_package(repair)
                assert not list(installed.generation.iterdir())
                installed.generation.rmdir()  # only the empty test-owned directory above

                other_inode = other_installed.generation.stat().st_ino
                before_repair = time.monotonic()
                worker.prepare_local_package(repair)
                repaired = worker.local_operations[(repair.operation_id, revision)].installed
                assert repaired is not None
                assert repaired.content_digest == installed.content_digest
                assert repaired.receipt_digest == installed.receipt_digest
                assert repaired.generation == installed.generation and not repaired.reused
                assert receipt.read_bytes() == saved
                assert other_installed.generation.stat().st_ino == other_inode
                assert moved.is_dir()
                assert plan.read_bytes() == plan_raw
                imported = subprocess.run(
                    [
                        str(repaired.python),
                        "-I",
                        "-c",
                        "import prepare_retained; print(prepare_retained.__file__)",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                assert Path(imported.stdout.strip()).parent == repaired.site_packages
                print(f"missing generation repaired in {time.monotonic() - before_repair:.3f}s")
                for wheel in repair.wheels:
                    Path(wheel.path).unlink()
                    wheel.path = ""
                assert worker.prepare_local_package(repair).placement_set
            assert worker.prepare_local_package(request) == first
            print(
                f"local installation ({snapshot=}): prepare={cold_seconds:.3f}s "
                f"reopen={warm_seconds:.3f}s restart_reopen={restart_seconds:.3f}s"
            )
        finally:
            worker.shutdown()


def _replace(path: Path, raw: bytes) -> None:
    mode = path.stat().st_mode & 0o777
    path.chmod(0o600)
    path.write_bytes(raw)
    path.chmod(mode)


def _assert_memo_identity(
    installed: package_environment.InstalledEnvironment,
    changed: package_environment.InstalledEnvironment,
    revision: str,
) -> None:
    target = Target(
        revision=revision,
        entrypoint="retained",
        interface_digest=revision,
        declaration={},
        prepared_revision={"receipt_digest": installed.receipt_digest},
        binding=None,
        digest=revision,
    )
    numerical = b"n" * 32
    key = computation(target, {"value": 1}, numerical)
    another_root = dataclasses.replace(
        target,
        prepared_revision={
            "receipt_digest": installed.receipt_digest,
            "root": "another-root",
            "caller_revision": "sha256:" + "c" * 64,
            "operation": "fresh-operation",
        },
    )
    assert computation(another_root, {"value": 1}, numerical) == key
    changed_target = dataclasses.replace(
        target, prepared_revision={"receipt_digest": changed.receipt_digest}
    )
    assert computation(changed_target, {"value": 1}, numerical) != key
