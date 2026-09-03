"""The fixed worker opens one real host-local TensorFS Store before readiness."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import cast

import pytest

from cozy_runtime.internal import host_paths
from cozy_runtime.internal.config import WorkerHostConfig
from cozy_runtime.internal.placement_materialization import prepare_boot
from cozy_runtime.internal.refusal import LaunchRefusal


def test_prepare_boot_initializes_the_explicit_local_tensorfs_root(tmp_path: Path) -> None:
    assert Path("/var/lib/tensorfs") == host_paths.TENSORFS_ROOT
    assert Path("/var/lib/cozy/installs") == host_paths.ENVIRONMENT_ROOT
    fixed = tmp_path / "fixed-worker-roots"
    tensorfs = tmp_path / "tensorfs"

    prepared_root = prepare_boot(
        cast(WorkerHostConfig, object()),
        boot_root=fixed / "boot",
        tensorfs_root=tensorfs,
    )

    assert prepared_root == tensorfs
    assert not tensorfs.is_symlink()
    assert stat.S_IMODE(tensorfs.stat().st_mode) == 0o755
    assert (tensorfs / "tensorfs.sqlite").is_file()


def test_prepare_boot_refuses_a_symlinked_tensorfs_root(tmp_path: Path) -> None:
    nfs = tmp_path / "repo-cache"
    nfs.mkdir()
    tensorfs = tmp_path / "tensorfs"
    tensorfs.symlink_to(nfs, target_is_directory=True)

    with pytest.raises(LaunchRefusal, match="provision_store_invalid"):
        prepare_boot(
            cast(WorkerHostConfig, object()),
            boot_root=tmp_path / "boot",
            tensorfs_root=tensorfs,
        )
