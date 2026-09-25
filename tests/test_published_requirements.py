"""Published packages must satisfy the actual SDK before any application import."""

from __future__ import annotations

import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import pytest

from cozy_runtime.internal import canonical, package_requirements
from cozy_runtime.internal import package_environment as environment
from cozy_runtime.internal.worker.control import _PreparationServicer
from cozy_runtime.internal.worker.package_prepare import prepare_package_set
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from test_publish_derivation import _SimpleIndex
from test_unpublished_exact_dependencies import _wheel


def _locked(index: _SimpleIndex, wheels: list[Path]) -> environment.LockedRequirements:
    pins = [
        f"{wheel.name.split('-')[0].replace('_', '-')}==1.0 "
        f"--hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}"
        for wheel in wheels
    ]
    return environment.read_locked_requirements(
        (f"--index-url {index.url}\n" + "\n".join(sorted(pins)) + "\n").encode()
    )


def _legacy_generation(root: Path, locked: environment.LockedRequirements) -> None:
    """An already-installed receipt from before published metadata admission existed."""
    base = environment.observe_base(Path(sys.executable))
    identity = canonical.write(
        environment._locked_content_identity(
            locked.digest, base.interpreter_identity, base.uv_version
        )
    )
    content = "sha256:" + hashlib.sha256(identity).hexdigest()
    receipt = canonical.write(
        environment._locked_receipt(DIGEST, content, locked.digest, locked.length)
    )
    generation = root / "contents" / content.removeprefix("sha256:")
    generation.parent.mkdir(parents=True)
    (root / "receipts").mkdir()
    for boundary in (root, root / "contents", root / "receipts"):
        boundary.chmod(0o700)
    environment._install_locked(
        environment._uv_binary(),
        Path(sys.executable),
        generation,
        root / "cache",
        locked.indexes,
        locked.rows,
    )
    (generation / environment.CONTENT_IDENTITY_NAME).write_bytes(identity)
    environment._make_read_only(generation)
    generation.chmod(0o555)
    target = root / "receipts" / (hashlib.sha256(receipt).hexdigest() + ".json")
    target.write_bytes(receipt)
    target.chmod(0o444)


DIGEST = "sha256:" + "37" * 32


@pytest.mark.parametrize("reused", [False, True])
def test_incompatible_runtime_refuses_fresh_and_legacy_cache_before_import(
    tmp_path: Path, reused: bool
) -> None:
    sentinel = tmp_path / "imported"
    wheel = _wheel(
        tmp_path,
        "requires_new_runtime",
        ("cozy-runtime[media]>=999",),
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('imported')\n".encode(),
    )
    runtime = _wheel(tmp_path, "cozy_runtime", (), b"VALUE=1\n")
    index = _SimpleIndex(tmp_path / "index", [wheel, runtime])
    try:
        locked = _locked(index, [wheel, runtime])
        root = tmp_path / "installs"
        if reused:
            _legacy_generation(root, locked)
        with pytest.raises(package_requirements.DependencyConflict) as caught:
            environment.materialize_locked(locked, locked.rows, root, environment_digest=DIGEST)
        assert caught.value.code == "package_environment_dependency_base_conflict"
        assert caught.value.facts == {
            "package": "requires-new-runtime",
            "distribution": "cozy-runtime",
            "required": ">=999",
            "installed": "1.0",
        }
        assert not sentinel.exists()
        if not reused:
            assert not list((root / "receipts").glob("*.json"))
    finally:
        index.close()


def test_selected_extra_rechecks_transitive_constraint(tmp_path: Path) -> None:
    # Root order initially visits helper without extras; following [media] must revisit it.
    project = _wheel(tmp_path, "a_project", ("z-helper[media]==1.0",), b"VALUE=42\n")
    helper = _wheel(tmp_path, "z_helper", ("cozy-runtime>=999; extra == 'media'",), b"VALUE=42\n")
    runtime = _wheel(tmp_path, "cozy_runtime", (), b"VALUE=1\n")
    index = _SimpleIndex(tmp_path / "index", [project, helper, runtime])
    try:
        locked = _locked(index, [project, helper, runtime])
        with pytest.raises(package_requirements.DependencyConflict) as caught:
            environment.materialize_locked(
                locked, locked.rows, tmp_path / "installs", environment_digest=DIGEST
            )
        assert caught.value.facts["package"] == "z-helper"
        assert caught.value.facts["required"] == ">=999"
    finally:
        index.close()


@pytest.mark.parametrize("python", [">=999", ">=3.12,<3.13"])
def test_python_markers_and_compatible_cache_reuse(tmp_path: Path, python: str) -> None:
    wheel = _wheel(
        tmp_path,
        "compatible_project",
        (
            "cozy-runtime>=0.1",
            "absent-dependency; sys_platform == 'win32'",
            "absent-extra; extra == 'unselected'",
        ),
        b"VALUE=42\n",
        requires_python=python,
    )
    runtime = _wheel(tmp_path, "cozy_runtime", (), b"VALUE=1\n")
    index = _SimpleIndex(tmp_path / "index", [wheel, runtime])
    try:
        locked = _locked(index, [wheel, runtime])
        root = tmp_path / "installs"
        if python == ">=999":
            with pytest.raises(environment.EnvironmentRefusal, match="python_incompatible"):
                environment.materialize_locked(locked, locked.rows, root, environment_digest=DIGEST)
            return
        installed = environment.materialize_locked(
            locked, locked.rows, root, environment_digest=DIGEST
        )
        inode = installed.generation.stat().st_ino
        replay = environment.materialize_locked(
            locked, locked.rows, root, environment_digest=DIGEST
        )
        assert replay.reused and replay.generation.stat().st_ino == inode
        assert list(replay.site_packages.glob("cozy_runtime*"))
        assert replay.generation.stat().st_uid == os.geteuid()
    finally:
        index.close()


def test_preparation_rpc_preserves_requirement_facts(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "requires_new_runtime",
        ("cozy-runtime>=999",),
        b"raise AssertionError('imported')\n",
    )
    runtime = _wheel(tmp_path, "cozy_runtime", (), b"VALUE=1\n")
    index = _SimpleIndex(tmp_path / "index", [wheel, runtime])
    base = environment.observe_base()
    root = tmp_path / "installs"
    request = pb.PreparePackageSetRequest(
        install_root=str(root),
        application="requires_new_runtime:app",
        download_delegation=canonical.write(
            {
                "format": "cozy.worker.v1.DownloadDelegation/1",
                "packages": [{"package": "proof/requires-new-runtime", "release": "1.0"}],
                "models": [],
            }
        ),
        image_inventory=pb.ImageInventory(
            profile="proof",
            python=base.python_full_version,
            distributions=[
                pb.ImageDistribution(distribution=n, version=v) for n, v in base.distributions
            ],
        ),
        package_interface=b"{}",
        locked_requirements=(
            f"--index-url {index.url}\n"
            + "\n".join(row.text for row in _locked(index, [wheel, runtime]).rows)
            + "\n"
        ).encode(),
    )

    def prepare(value: pb.PreparePackageSetRequest) -> pb.PreparePackageSetResult:
        return prepare_package_set(
            value,
            artifact_cache=tmp_path / "artifacts",
            tensorfs_root=tmp_path / "store",
            install_root=root,
            python=Path(sys.executable),
            verified=lambda *args: None,
            base=base,
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            server = grpc.server(pool)
            rpc.add_RuntimePreparationServicer_to_server(
                _PreparationServicer(prepare, None, None, None), server
            )
            port = server.add_insecure_port("127.0.0.1:0")
            server.start()
            try:
                with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                    with pytest.raises(grpc.RpcError) as caught:
                        rpc.RuntimePreparationStub(channel).PreparePackageSet(request)
                    assert caught.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                    assert dict(caught.value.trailing_metadata()) == {
                        "cozy-requirement-package": "requires-new-runtime",
                        "cozy-requirement-distribution": "cozy-runtime",
                        "cozy-requirement-required": ">=999",
                        "cozy-requirement-installed": "1.0",
                    }
            finally:
                server.stop(0).wait()
    finally:
        index.close()
