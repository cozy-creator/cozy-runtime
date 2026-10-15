"""The CPUs this process may use: its affinity under its cgroup's quota, v2 (`cpu.max` at its
cgroup and every ancestor) or v1 (`cpu.cfs_quota_us`). Never the host's cores: a vast pod on a
128-core EPYC with a 15-CPU v1 quota ran `ninja -j 128` (2026-10-15)."""

from __future__ import annotations

import functools
import os
from pathlib import Path

CGROUP = Path("/sys/fs/cgroup")


@functools.cache
def cpus() -> int:
    """Every CPU this process may use: its affinity and its cgroup's quota, at least one."""
    return max(1, min([len(os.sched_getaffinity(0)), *_v2_quotas(), *_v1_quota()]))


def _whole(quota: int, period: int) -> list[int]:
    """A quota in whole CPUs, rounded up; none for an unlimited one."""
    return [-(-quota // period)] if quota > 0 and period > 0 else []


def _v2_quotas() -> list[int]:
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return []
    own = next((line[3:] for line in lines if line.startswith("0::")), None)
    if own is None:
        return []
    directory = CGROUP / own.strip().lstrip("/")
    quotas = []
    for at in (directory, *directory.parents):
        if not at.is_relative_to(CGROUP):
            break
        try:
            quota, period = (at / "cpu.max").read_text().split()
            quotas += _whole(int(quota), int(period)) if quota != "max" else []
        except (OSError, ValueError):
            continue
    return quotas


def _v1_quota() -> list[int]:
    try:
        quota = int((CGROUP / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((CGROUP / "cpu" / "cpu.cfs_period_us").read_text())
    except (OSError, ValueError):
        return []
    return _whole(quota, period)
