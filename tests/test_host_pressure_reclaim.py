"""Host admission honors reclaim pressure and never trusts claimed freed bytes."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from cozy_runtime.internal import memory, proctree


@pytest.mark.parametrize(
    ("high", "maximum", "expected"), [(1200, 1600, 100), (1600, 1200, 100), (1000, 1600, 0)]
)
def test_parent_high_and_max_both_bound_host_credit(
    monkeypatch: pytest.MonkeyPatch, high: int, maximum: int, expected: int
) -> None:
    files = {
        "/proc/meminfo": "MemAvailable: 10 kB\n",
        "/proc/self/cgroup": "0::/parent/child\n",
        "/sys/fs/cgroup/parent/child/memory.max": "max",
        "/sys/fs/cgroup/parent/child/memory.high": "max",
        "/sys/fs/cgroup/parent/memory.max": str(maximum),
        "/sys/fs/cgroup/parent/memory.high": str(high),
        "/sys/fs/cgroup/parent/memory.current": "1150",
        "/sys/fs/cgroup/parent/memory.stat": "inactive_file 50\n",
        "/sys/fs/cgroup/memory.max": "max",
        "/sys/fs/cgroup/memory.high": "max",
    }
    monkeypatch.setattr(
        Path, "is_file", lambda path: str(path) in files or str(path).endswith("cgroup.controllers")
    )
    monkeypatch.setattr(Path, "read_text", lambda path: files[str(path)])
    assert proctree.available_host_bytes() == expected


@dataclass
class Cache:
    slot: str
    current: list[int]
    relief: int
    status: str = "idle"
    last_used: int = 0
    asked: list[int] = field(default_factory=list)

    def state(self) -> str:
        return self.status

    def reclaim_host(self, nbytes: int) -> tuple[int, int]:
        self.asked.append(nbytes)
        self.current[0] -= self.relief
        return self.relief, max(0, self.current[0])


def test_host_victims_follow_queue_across_devices_and_preserve_ram_when_room_exists() -> None:
    current = [90]
    policy = memory.Machine(
        {}, lambda _: None, lambda _: None, host_available=lambda: 100 - current[0]
    )
    soon = Cache("sdxl", current, 40)
    later = Cache("later", current, 20)
    active = Cache("anima", current, 90, "active")
    policy.update_queue(
        [
            memory.Demand("sdxl", "a", "fp16", (1,), 0, 10),
            memory.Demand("later", "b", "fp16", (0,), 2, 1),
        ]
    )
    assert policy.reserve_host("small", 5, requester="anima", tenants=[soon, later, active])
    assert not soon.asked and not later.asked
    policy.release_host("small")
    assert policy.reserve_host("copy", 25, requester="anima", tenants=[soon, later, active])
    assert later.asked == [25] and not soon.asked and not active.asked
    assert policy.host_reserved == {"copy": 25}


def test_claimed_release_without_observed_relief_never_grants_or_retries_forever() -> None:
    current = [100]
    policy = memory.Machine(
        {}, lambda _: None, lambda _: None, host_available=lambda: 100 - current[0]
    )
    retained = Cache("held", current, 0)
    assert not policy.reserve_host("copy", 25, requester="active", tenants=[retained])
    assert retained.asked == [25] and not policy.host_reserved


def test_reclaim_makes_finite_progress_through_existing_high_debt() -> None:
    current = [150]
    policy = memory.Machine(
        {}, lambda _: None, lambda _: None, host_available=lambda: max(0, 100 - current[0])
    )
    retained = Cache("held", current, 25)
    assert policy.reserve_host("copy", 25, requester="active", tenants=[retained])
    assert retained.asked == [25, 25, 25]
    assert current == [75] and policy.host_reserved == {"copy": 25}


def test_ledger_exports_managed_host_storage_separately_from_process_rss() -> None:
    import os

    from cozy_runtime.internal.worker.ledger import Ledger

    ledger = Ledger(worker_pid=os.getpid())
    ledger.observe_memory({"host_bytes": 128, "host_resident_bytes": 96, "host_cached_bytes": 64})
    counters = {counter.name: counter for counter in ledger.counters()}
    assert counters["host_managed"].bytes == 128  # Shared backing is not double charged.
    assert counters["host_resident"].bytes == 96 and counters["host_cached"].bytes == 64
    assert counters["rss_worker"].bytes > 128
