"""Host admission honors reclaim pressure and never trusts claimed freed bytes."""

import threading
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, RLock
from time import monotonic, sleep
from typing import Any, cast

import pytest

from cozy_runtime.internal import memory, proctree


def test_host_pressure_counts_file_pages_charged_to_the_kernel_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {
        "/proc/self/cgroup": "0::/parent/child\n",
        "/sys/fs/cgroup/parent/child/memory.current": "1100",
        "/sys/fs/cgroup/parent/child/memory.high": "1000",
        "/sys/fs/cgroup/parent/child/memory.max": "2000",
        "/sys/fs/cgroup/parent/child/memory.stat": "inactive_file 200\n",
        "/sys/fs/cgroup/parent/memory.current": "1300",
        "/sys/fs/cgroup/parent/memory.high": "1000",
        "/sys/fs/cgroup/parent/memory.max": "2000",
        "/sys/fs/cgroup/parent/memory.stat": "inactive_file 100\n",
        "/sys/fs/cgroup/memory.current": "1300",
        "/sys/fs/cgroup/memory.high": "max",
        "/sys/fs/cgroup/memory.max": "max",
        "/sys/fs/cgroup/memory.stat": "inactive_file 0\n",
    }
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: str(path) == "/sys/fs/cgroup/cgroup.controllers" or str(path) in files,
    )
    monkeypatch.setattr(Path, "read_text", lambda path: files[str(path)])
    pressure = proctree.host_pressure()
    assert pressure is not None
    assert pressure.scope.endswith("/parent")
    assert pressure.current_bytes == 1300
    assert pressure.limit_bytes == 1000
    assert pressure.reclaimable_file_bytes == 100
    assert pressure.debt_bytes == 300


@pytest.mark.parametrize(
    ("current", "debt", "available"), [(900, 0, 300), (1000, 0, 200), (1100, 100, 100)]
)
def test_cache_remains_admission_credit_without_hiding_kernel_pressure(
    current: int, debt: int, available: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {
        "/proc/meminfo": "MemAvailable: 1000 kB\nMemFree: 1000 kB\n",
        "/proc/self/cgroup": "0::/worker\n",
        "/sys/fs/cgroup/worker/memory.current": str(current),
        "/sys/fs/cgroup/worker/memory.high": "1000",
        "/sys/fs/cgroup/worker/memory.max": "2000",
        "/sys/fs/cgroup/worker/memory.stat": "inactive_file 200\n",
    }
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: str(path) == "/sys/fs/cgroup/cgroup.controllers" or str(path) in files,
    )
    monkeypatch.setattr(Path, "read_text", lambda path: files[str(path)])
    pressure = proctree.host_pressure()
    assert pressure is not None and pressure.debt_bytes == debt
    assert pressure.reclaimable_file_bytes == 200
    assert proctree.available_host_bytes() == available
    assert proctree.optional_host_bytes() == max(1000 - current, 0)


def test_host_pressure_v1_large_limit_leaves_parent_debt(monkeypatch: pytest.MonkeyPatch) -> None:
    files = {
        "/proc/self/cgroup": "7:memory:/parent/child\n",
        "/sys/fs/cgroup/memory/parent/child/memory.usage_in_bytes": "1300",
        "/sys/fs/cgroup/memory/parent/child/memory.limit_in_bytes": str(1 << 60),
        "/sys/fs/cgroup/memory/parent/child/memory.stat": "total_inactive_file 0\n",
        "/sys/fs/cgroup/memory/parent/memory.usage_in_bytes": "1300",
        "/sys/fs/cgroup/memory/parent/memory.limit_in_bytes": "1000",
        "/sys/fs/cgroup/memory/parent/memory.stat": "total_inactive_file 100\n",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes": "1300",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": str(1 << 60),
        "/sys/fs/cgroup/memory/memory.stat": "total_inactive_file 0\n",
    }
    mount = proctree._CgroupMount(Path("/"), Path("/sys/fs/cgroup/memory"), frozenset({"memory"}))
    monkeypatch.setattr(Path, "is_file", lambda _: False)
    monkeypatch.setattr(Path, "read_text", lambda path: files[str(path)])
    monkeypatch.setattr(
        proctree,
        "_cgroup_v1_controller_path",
        lambda _: Path("/sys/fs/cgroup/memory/parent/child"),
    )
    monkeypatch.setattr(proctree, "_cgroup_v1_mounts", lambda: (mount,))
    pressure = proctree.host_pressure()
    assert pressure is not None
    assert pressure.scope.endswith("/parent")
    assert pressure.debt_bytes == 300


def test_unlimited_ancestor_without_counters_does_not_hide_child_debt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {
        "/proc/self/cgroup": "0::/child\n",
        "/sys/fs/cgroup/child/memory.current": "1200",
        "/sys/fs/cgroup/child/memory.high": "1000",
        "/sys/fs/cgroup/child/memory.max": "2000",
        "/sys/fs/cgroup/child/memory.stat": "inactive_file 0\n",
        "/sys/fs/cgroup/memory.high": "max",
        "/sys/fs/cgroup/memory.max": "max",
    }
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: str(path) == "/sys/fs/cgroup/cgroup.controllers" or str(path) in files,
    )
    monkeypatch.setattr(Path, "read_text", lambda path: files[str(path)])
    pressure = proctree.host_pressure()
    assert pressure is not None
    assert pressure.scope.endswith("/child")
    assert pressure.debt_bytes == 200


@dataclass
class PressureCache:
    slot: str
    held: int
    pressure: list[proctree.HostPressure]
    generation: int | None = 1
    status: str = "idle"
    last_used: int = 0
    calls: list[int] = field(default_factory=list)

    def state(self) -> str:
        return self.status

    @property
    def reclaim_generation(self) -> int | None:
        return self.generation

    def reclaim_host(self, nbytes: int) -> tuple[int, int]:
        self.calls.append(nbytes)
        released = min(10, self.held)
        self.held -= released
        fact = self.pressure[0]
        self.pressure[0] = proctree.HostPressure(
            fact.scope,
            fact.current_bytes - released,
            fact.limit_bytes,
            max(fact.debt_bytes - released, 0),
            fact.reclaimable_file_bytes,
        )
        return released, self.held


def test_pressure_relief_reclaims_only_until_measured_debt_is_gone() -> None:
    pressure = [proctree.HostPressure("/worker", 120, 100, 20)]
    policy = memory.Machine({}, lambda _: None, lambda _: None, host_pressure=lambda: pressure[0])
    cache = PressureCache("idle", 40, pressure)
    result = policy.relieve_host_pressure([cache])
    assert result == {
        "initial_debt_bytes": 20,
        "reclaimed_managed_bytes": 20,
        "attempts": 2,
        "debt_bytes": 0,
        "pressure_scope": "/worker",
        "initial_pressure_scope": "/worker",
        "generation": "1",
    }
    assert cache.calls == [20, 10]


def test_zero_progress_victim_does_not_block_later_idle_victim() -> None:
    pressure = [proctree.HostPressure("/worker", 120, 100, 20)]
    policy = memory.Machine({}, lambda _: None, lambda _: None, host_pressure=lambda: pressure[0])
    protected = PressureCache("protected", 0, pressure, last_used=-1)
    eligible = PressureCache("eligible", 20, pressure)
    result = policy.relieve_host_pressure([protected, eligible])
    assert result is not None
    assert result["reclaimed_managed_bytes"] == 20
    assert result["debt_bytes"] == 0
    assert protected.calls == [20]
    assert eligible.calls == [20, 10]


@pytest.mark.parametrize(
    ("change", "expected_calls"),
    [("active", 0), ("no_generation", 0), ("replaced", 1), ("unknown", 1), ("no_progress", 2)],
)
def test_pressure_relief_requires_idle_generation_and_monotone_progress(
    change: str, expected_calls: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    pressure = [proctree.HostPressure("/worker", 120, 100, 20)]
    policy = memory.Machine({}, lambda _: None, lambda _: None, host_pressure=lambda: pressure[0])
    cache = PressureCache("idle", 40, pressure)
    original = cache.reclaim_host
    if change == "active":
        cache.status = "active"
    if change == "no_generation":
        cache.generation = None

    def reclaim(nbytes: int) -> tuple[int, int]:
        if change == "no_progress":
            cache.calls.append(nbytes)
            return 1, cache.held  # A claimed release cannot sustain an endless loop.
        result = original(nbytes)
        if change == "replaced":
            cache.generation = 2
        if change == "unknown":
            policy.host_pressure = lambda: None
        return result

    monkeypatch.setattr(cache, "reclaim_host", reclaim)
    policy.relieve_host_pressure([cache])
    assert len(cache.calls) == expected_calls


def test_pressure_relief_keeps_cache_when_usage_is_at_the_limit() -> None:
    pressure = [proctree.HostPressure("/worker", 100, 100, 0, 20)]
    policy = memory.Machine({}, lambda _: None, lambda _: None, host_pressure=lambda: pressure[0])
    cache = PressureCache("idle", 40, pressure)
    assert policy.relieve_host_pressure([cache]) is None
    assert not cache.calls


def test_pressure_relief_remeasures_physical_relief_without_logical_release() -> None:
    pressure = [proctree.HostPressure("/worker", 120, 100, 20)]

    class PhysicalOnly(PressureCache):
        def reclaim_host(self, nbytes: int) -> tuple[int, int]:
            self.calls.append(nbytes)
            fact = self.pressure[0]
            self.pressure[0] = proctree.HostPressure(fact.scope, 100, 100, 0)
            return 0, self.held

    policy = memory.Machine({}, lambda _: None, lambda _: None, host_pressure=lambda: pressure[0])
    cache = PhysicalOnly("idle", 40, pressure)
    later = PressureCache("later", 40, pressure, last_used=1)
    assert policy.relieve_host_pressure([cache, later]) == {
        "initial_debt_bytes": 20,
        "reclaimed_managed_bytes": 0,
        "attempts": 1,
        "debt_bytes": 0,
        "pressure_scope": "/worker",
        "initial_pressure_scope": "/worker",
        "generation": "",
    }
    assert cache.calls == [20]
    assert not later.calls


def test_pressure_schedule_does_not_block_caller_or_duplicate_work() -> None:
    started, release = Event(), Event()

    class Fake:
        _host_pressure_schedule_lock = RLock()
        _host_pressure_scheduled = False
        control_lock = RLock()

        class Memory:
            def relieve_host_pressure(self, tenants: object) -> None:
                started.set()
                release.wait(5)
                return None

        memory = Memory()

        def _tenants(self, _: object) -> tuple[object, ...]:
            return ()

        def _lane(self, _: str, body: object, *, fatal: bool) -> object:
            return body

    fake = Fake()
    started_at = monotonic()
    # Bind the production method to a minimal worker-shaped object.
    from cozy_runtime.internal.worker.session import Worker

    Worker._schedule_host_pressure_relief(cast(Any, fake))
    Worker._schedule_host_pressure_relief(cast(Any, fake))
    assert monotonic() - started_at < 1
    assert started.wait(1)

    def scheduled() -> bool:
        return fake._host_pressure_scheduled

    assert scheduled()
    release.set()
    for _ in range(100):
        if not scheduled():
            break
        sleep(0.01)
    assert not scheduled()


def test_pressure_schedule_resets_after_thread_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Fake:
        _host_pressure_schedule_lock = RLock()
        _host_pressure_scheduled = False

        def _lane(self, _: str, body: object, *, fatal: bool) -> object:
            return body

    class BrokenThread:
        def __init__(self, **_: object) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("thread start failed")

    from cozy_runtime.internal.worker import session

    monkeypatch.setattr(threading, "Thread", BrokenThread)
    fake = Fake()
    with pytest.raises(RuntimeError, match="thread start failed"):
        session.Worker._schedule_host_pressure_relief(cast(Any, fake))
    assert not fake._host_pressure_scheduled


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
