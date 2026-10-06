"""Memory facts the planner reads: a cgroup's page cache and shared memory in either cgroup
version, and the executor's plane document as telemetry on the ledger.
"""

from __future__ import annotations

from pathlib import Path

from cozy_runtime.internal import proctree

GiB = 1 << 30


def test_a_cgroups_page_cache_and_shared_memory_are_read_in_either_version(
    tmp_path: Path,
) -> None:
    """A rented pod is cgroup v1 (its subtree's counters under `total_`), a desktop v2."""
    (tmp_path / "memory.stat").write_text(
        "cache 900\nshmem 7\ntotal_inactive_file 300\ntotal_active_file 200\ntotal_shmem 100\n"
    )
    v1 = {"inactive_file": 300, "active_file": 200, "shmem": 100}
    assert proctree._cgroup_stat(tmp_path) == v1
    (tmp_path / "memory.stat").write_text("anon 5\nfile 900\nshmem 100\ninactive_file 300\n")
    assert proctree._cgroup_stat(tmp_path) == {"inactive_file": 300, "active_file": 0, "shmem": 100}
