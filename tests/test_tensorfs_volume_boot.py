"""The fixed worker accepts its TensorFS root redirected by the root supervisor."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import cast

import pytest

from cozy_runtime.internal.config import WorkerHostConfig
from cozy_runtime.internal.placement_materialization import prepare_boot
from cozy_runtime.internal.refusal import LaunchRefusal


def test_prepare_boot_follows_root_supervisor_tensorfs_symlink(tmp_path: Path) -> None:
    mount = tmp_path / "provider-volume"
    mount.mkdir()
    tensorfs_target = mount / "tensorfs"
    tensorfs_target.mkdir()

    fixed = tmp_path / "fixed-worker-roots"
    fixed.mkdir()
    tensorfs = fixed / "tensorfs"
    staging = fixed / "tmp" / "transfers"
    tensorfs.symlink_to(tensorfs_target, target_is_directory=True)

    prepared = prepare_boot(
        cast(WorkerHostConfig, object()),
        boot_root=fixed / "boot",
        tensorfs_root=tensorfs,
        transfer_staging_root=staging,
    )

    assert prepared.tensorfs_root == tensorfs
    assert prepared.transfer_staging == staging
    assert tensorfs.is_symlink()
    assert staging.is_dir() and not staging.is_symlink()
    assert stat.S_IMODE(tensorfs_target.stat().st_mode) == 0o755
    assert stat.S_IMODE(staging.stat().st_mode) == 0o711
    assert (tensorfs_target / "tensorfs.sqlite").is_file()


def test_prepare_boot_still_refuses_transfer_staging_symlink(tmp_path: Path) -> None:
    staging_target = tmp_path / "foreign-staging"
    staging_target.mkdir()
    staging = tmp_path / "transfers"
    staging.symlink_to(staging_target, target_is_directory=True)

    with pytest.raises(LaunchRefusal, match="provision_store_invalid"):
        prepare_boot(
            cast(WorkerHostConfig, object()),
            boot_root=tmp_path / "boot",
            tensorfs_root=tmp_path / "tensorfs",
            transfer_staging_root=staging,
        )
