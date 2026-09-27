"""Local admission keeps finite recovery outside the shared write exclusion."""

from __future__ import annotations

import fcntl
import gc
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from weakref import WeakValueDictionary

import pytest

from cozy_runtime.internal import local_storage_admission as local
from cozy_runtime.internal import storage_admission as storage


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(storage, "_channel", None)
    monkeypatch.setattr(storage, "_reclaimers", WeakValueDictionary())
    yield


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / ".cozy-workspace").mkdir()
    return tmp_path


def test_registration_does_not_keep_stopped_workers_or_replace_live_peer(root: Path) -> None:
    first = storage.register_workspace(root, lambda _target: None)
    second = storage.register_workspace(root, lambda _target: None)
    assert storage.pressure_enabled()
    first.close()
    assert storage.pressure_enabled()
    del second
    gc.collect()
    assert not storage.pressure_enabled()


@pytest.mark.parametrize("sufficient_after_recovery", [True, False])
def test_local_recovery_runs_without_exclusion_and_rechecks_once(
    root: Path, monkeypatch: pytest.MonkeyPatch, sufficient_after_recovery: bool
) -> None:
    observations: list[tuple[storage.Write, ...]] = []
    recovered: list[int] = []
    shortage = {"available": 90, "required": 100, "reserve": 20}

    def shortfall(writes: Sequence[storage.Write]) -> dict[str, int] | None:
        observations.append(tuple(writes))
        return None if recovered and sufficient_after_recovery else shortage

    def recover(target: int) -> None:
        # A callback behind the failed admission's flock would deadlock native
        # work. Independently acquiring it proves the exception unwound first.
        fd = os.open(root / ".cozy-workspace/storage-admission.lock", os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        recovered.append(target)

    monkeypatch.setattr(local, "shortfall", shortfall)
    registration = storage.register_workspace(root, recover)
    try:
        if sufficient_after_recovery:
            with storage.admit(storage.Write(root, 100, 1)):
                (root / "entered").write_bytes(b"ok")
        else:
            with (
                pytest.raises(storage.StorageRefusal),
                storage.admit(storage.Write(root, 100, 1)),
            ):
                pytest.fail("busy or ineffective recovery admitted the write")
        assert recovered == [30]
        assert len(observations) == 2
        assert (root / "entered").exists() == sufficient_after_recovery
    finally:
        registration.close()


def test_unknown_capacity_refuses_without_pruning_or_running_author_write(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recovered: list[int] = []

    def unknown(_path: object) -> os.statvfs_result:
        raise OSError("mount observation failed")

    monkeypatch.setattr(os, "statvfs", unknown)
    registration = storage.register_workspace(root, recovered.append)
    try:
        with (
            pytest.raises(storage.StorageRefusal, match="capacity is unavailable"),
            storage.admit(storage.Write(root, 100, 1)),
        ):
            pytest.fail("unobserved capacity admitted a write")
        assert recovered == []
    finally:
        registration.close()


def test_filesystem_without_inode_table_is_admitted_on_bytes(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = os.statvfs(root)

    def btrfs(path: str | os.PathLike[str]) -> os.statvfs_result:
        # btrfs reports no inode table: f_files, f_ffree and f_favail are all zero.
        fields = list(real)
        fields[5:8] = [0, 0, 0]
        return os.statvfs_result(fields)

    monkeypatch.setattr(os, "statvfs", btrfs)
    registration = storage.register_workspace(root, lambda _target: None)
    assert storage.pressure_enabled() and local.pressure_target(root) >= 0
    with storage.admit(storage.Write(root / "blob", 1 << 20, 4)):
        (root / "blob").write_bytes(b"x")
    too_large = real.f_bavail * real.f_frsize + 1
    with (
        pytest.raises(storage.StorageRefusal) as refused,
        storage.admit(storage.Write(root / "huge", too_large, 1)),
    ):
        pass
    assert refused.value.code == "insufficient_storage"
    registration.close()
