"""Local counterpart of the Host's closed filesystem-capacity contract.

This exclusion accounts for pending writes only. TensorFS and the workspace journal
remain the sole owners of byte custody and eviction eligibility.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .storage_admission import Write

POLICY = json.loads(Path(__file__).with_name("storage-pressure-policy.json").read_bytes())


def existing_directory(path: Path) -> Path:
    while not path.exists():
        if path == path.parent:
            raise OSError("write filesystem has no existing parent")
        path = path.parent
    return path


def device(path: Path) -> int:
    return existing_directory(path).stat().st_dev


def pressure_target(path: Path) -> int:
    """Optional cleanup target at the same low/high watermarks as the Host."""
    observed = os.statvfs(existing_directory(path))
    total = observed.f_blocks * observed.f_frsize
    available = observed.f_bavail * observed.f_frsize
    if total <= 0 or not 0 <= available <= total:
        raise OSError("filesystem capacity is unavailable")
    # Filesystems without an inode table (btrfs reports zero) have no inode pressure.
    if observed.f_files > 0 and observed.f_favail <= max(
        POLICY["minimum_free_inodes"], observed.f_files // POLICY["enter_free_capacity_divisor"]
    ):
        return (1 << 63) - 1
    if available <= total // POLICY["enter_free_capacity_divisor"]:
        return int(max(1, total // POLICY["leave_free_capacity_divisor"] + 1 - available))
    return 0


def shortfall(writes: Sequence[Write]) -> dict[str, int] | None:
    groups: dict[int, tuple[Path, int, int]] = {}
    for write in writes:
        path = existing_directory(write.path)
        identity = path.stat().st_dev
        _, size, inodes = groups.get(identity, (path, 0, 0))
        groups[identity] = path, size + write.bytes, inodes + write.inodes
    for path, size, inodes in groups.values():
        observed = os.statvfs(path)
        total = observed.f_blocks * observed.f_frsize
        available = observed.f_bavail * observed.f_frsize
        count, free = observed.f_files, observed.f_favail
        reserve = max(POLICY["minimum_free_bytes"], total // POLICY["reserve_capacity_divisor"])
        # A filesystem that reports no inode table (btrfs) is admitted on bytes alone.
        inode_reserve = POLICY["minimum_free_inodes"] if count else 0
        if (
            not 0 < reserve < total
            or not 0 <= available <= total
            or (count and not inode_reserve < count)
            or not 0 <= free <= count
        ):
            raise OSError("filesystem capacity is unknown or cannot honor the storage reserve")
        if size > available - reserve or (count and inodes > free - inode_reserve):
            return {
                "available": available,
                "required": size,
                "reserve": reserve,
                "reserved": 0,
                "available_inodes": free,
                "required_inodes": inodes,
                "reserve_inodes": inode_reserve,
                "reserved_inodes": 0,
                "protected": -1,
            }
    return None


@contextmanager
def acquire(root: Path, writes: Sequence[Write]) -> Iterator[dict[str, object]]:
    from .storage_admission import StorageRefusal

    descriptor = os.open(
        root / ".cozy-workspace/storage-admission.lock",
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            facts = shortfall(writes)
        except OSError as exc:
            raise StorageRefusal(
                f"local filesystem capacity is unavailable: {exc}",
                code="storage_capacity_unavailable",
            ) from exc
        if facts is not None:
            raise StorageRefusal("local filesystem refused Runtime disk write", facts)
        yield {"ok": True}
    finally:
        os.close(descriptor)
