"""The fixed worker opens one real host-local TensorFS Store before readiness."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import tensorfs

from cozy_runtime.internal import host_paths
from cozy_runtime.internal.config import WorkerHostConfig
from cozy_runtime.internal.placement_materialization import prepare_boot
from cozy_runtime.internal.refusal import LaunchRefusal


def test_prepare_boot_opens_the_supervisor_prepared_local_store(tmp_path: Path) -> None:
    assert Path("/var/lib/tensorfs") == host_paths.Layout().tensorfs_root
    assert Path("/var/lib/cozy/installs") == host_paths.Layout().install_root
    assert host_paths.layout(str(tmp_path)).tensorfs_root == tmp_path / "var/lib/tensorfs"
    fixed = tmp_path / "fixed-worker-roots"
    store_root = tmp_path / "tensorfs"
    tensorfs.Store.ensure(str(store_root)).prepare_readers()

    prepared_root = prepare_boot(
        cast(WorkerHostConfig, object()),
        boot_root=fixed / "boot",
        tensorfs_root=store_root,
    )

    assert prepared_root == store_root


def test_prepare_boot_does_not_create_a_missing_store(tmp_path: Path) -> None:
    store_root = tmp_path / "tensorfs"

    with pytest.raises(LaunchRefusal, match="provision_store_invalid"):
        prepare_boot(
            cast(WorkerHostConfig, object()),
            boot_root=tmp_path / "boot",
            tensorfs_root=store_root,
        )
    assert not store_root.exists()


def test_prepare_boot_refuses_a_symlinked_tensorfs_root(tmp_path: Path) -> None:
    nfs = tmp_path / "repo-cache"
    tensorfs.Store.ensure(str(nfs))
    store_root = tmp_path / "tensorfs"
    store_root.symlink_to(nfs, target_is_directory=True)

    with pytest.raises(LaunchRefusal, match="provision_store_invalid"):
        prepare_boot(
            cast(WorkerHostConfig, object()),
            boot_root=tmp_path / "boot",
            tensorfs_root=store_root,
        )
