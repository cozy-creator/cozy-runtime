"""Fresh kernel activity can justify finite idle cache evacuation at zero debt."""

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from cozy_runtime.internal import memory, proctree


def fact(
    high: int = 1, refault: int = 1, *, inode: int = 3, debt: int = 0
) -> proctree.HostPressure:
    samples = (
        proctree.HostReclaimSample("/child", 1, 2, 100, 0, 0),
        proctree.HostReclaimSample("/parent", 1, inode, 100, high, refault),
    )
    return proctree.HostPressure("/child", 100 + debt, 100, debt, 0, samples)


@dataclass
class Cache:
    slot: str
    held: int = 40
    status: str = "idle"
    reclaim_generation: int | None = 1
    last_used: int = 0
    calls: list[int] = field(default_factory=list)
    callback: Callable[[int], None] = lambda _: None
    progress: bool = True

    def state(self) -> str:
        return self.status

    def reclaim_host(
        self, nbytes: int, *, expected_generation: object | None = None
    ) -> tuple[int, int]:
        self.calls.append(nbytes)
        released = min(nbytes, 10, self.held) if self.progress else 0
        self.held -= released
        self.callback(nbytes)
        return released, self.held


def setup() -> tuple[memory.Machine, list[proctree.HostPressure | None], Cache, Cache]:
    observed: list[proctree.HostPressure | None] = [fact()]
    policy = memory.Machine({}, lambda _: None, lambda _: None, host_pressure=lambda: observed[0])
    active, idle = Cache("active", status="active"), Cache("idle")
    assert policy.relieve_host_pressure([active, idle]) is None
    return policy, observed, active, idle


def test_zero_debt_parent_churn_evacuates_one_generation_not_all_idle() -> None:
    policy, observed, active, idle = setup()
    later = Cache("later", last_used=1)
    observed[0] = fact(60001, 3660001)
    result = policy.relieve_host_pressure([active, idle, later])
    assert result is not None and result["reclaimed_managed_bytes"] == 40
    assert idle.calls == [0, 40, 30, 20, 10]
    assert not active.calls and not later.calls
    # Repeated snapshots never reuse the historical cumulative activity.
    assert policy.relieve_host_pressure([active, later]) is None
    assert not later.calls


@pytest.mark.parametrize(
    "mode", ["high_only", "refault_only", "reset", "incarnation", "missing", "no_active"]
)
def test_missing_or_noncoincident_evidence_preserves_cache(mode: str) -> None:
    policy, observed, active, idle = setup()
    observed[0] = {
        "high_only": fact(2, 1),
        "refault_only": fact(1, 2),
        "reset": fact(0, 2),
        "incarnation": fact(2, 2, inode=4),
        "missing": replace(fact(2, 2), reclaim_samples=()),
        "no_active": fact(2, 2),
    }[mode]
    if mode == "no_active":
        active.status = "idle"
    assert policy.relieve_host_pressure([active, idle]) is None
    assert not idle.calls and not active.calls


@pytest.mark.parametrize(
    "change", ["active", "generation", "unknown", "counter_reset", "scope_change"]
)
def test_probe_revalidates_generation_state_and_pressure(change: str) -> None:
    policy, observed, active, idle = setup()
    observed[0] = fact(2, 2)

    def changed(_: int) -> None:
        if change == "active":
            idle.status = "active"
        elif change == "generation":
            idle.reclaim_generation = 2
        elif change == "unknown":
            observed[0] = None
        elif change == "counter_reset":
            observed[0] = fact(0, 0)
        else:
            observed[0] = fact(2, 2, inode=5)

    idle.callback = changed
    policy.relieve_host_pressure([active, idle])
    assert idle.calls == [0] and idle.held == 40


def test_no_progress_is_finite_and_does_not_hide_later_eligible_victim() -> None:
    policy, observed, active, idle = setup()
    idle.progress = False
    later = Cache("later", last_used=1)
    observed[0] = fact(2, 2)
    policy.relieve_host_pressure([active, idle, later])
    assert idle.calls == [0, 40] and later.held == 0


def test_dead_and_unknown_generations_are_not_probed() -> None:
    policy, observed, active, idle = setup()
    observed[0] = fact(2, 2)
    idle.status = "dead"
    unknown = Cache("unknown", reclaim_generation=None)
    policy.relieve_host_pressure([active, idle, unknown])
    assert not idle.calls and not unknown.calls


def test_debt_clearing_does_not_stop_churn_episode_or_supply_admission_credit() -> None:
    policy, observed, active, idle = setup()
    observed[0] = fact(2, 2, debt=1)
    idle.callback = lambda _: observed.__setitem__(0, fact(2, 2))
    policy.host_available = lambda: 0
    result = policy.relieve_host_pressure([active, idle])
    assert result is not None and result["initial_debt_bytes"] == 1
    assert idle.held == 0 and policy.host_reserved == {} and policy.host_available() == 0


@pytest.mark.parametrize("missing", ["old_kernel", "unreadable", "changed_identity", "negative"])
def test_sampler_keeps_both_zero_debt_ancestors_and_optional_old_kernel_fallback(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    files = {"/proc/self/cgroup": "0::/parent/child\n"}
    for scope in ("/sys/fs/cgroup/parent/child", "/sys/fs/cgroup/parent"):
        files.update(
            {
                scope + "/memory.high": "100",
                scope + "/memory.max": "max",
                scope + "/memory.current": "100",
                scope + "/memory.stat": "inactive_file 0\nworkingset_refault_file 20\n",
                scope + "/memory.events": "high 10\n",
            }
        )
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: str(path) in files or str(path) == "/sys/fs/cgroup/cgroup.controllers",
    )
    monkeypatch.setattr(Path, "read_text", lambda path: files[str(path)])
    monkeypatch.setattr(Path, "stat", lambda path: SimpleNamespace(st_dev=1, st_ino=len(str(path))))
    observed = proctree.host_pressure()
    assert observed is not None and observed.debt_bytes == 0
    assert len(observed.reclaim_samples) == 2
    if missing == "old_kernel":
        del files["/sys/fs/cgroup/parent/memory.events"]
        files["/sys/fs/cgroup/parent/child/memory.stat"] = "inactive_file 0\n"
    elif missing == "unreadable":

        def read(path: Path) -> str:
            if path.name == "memory.events":
                raise PermissionError("unreadable optional evidence")
            return files[str(path)]

        monkeypatch.setattr(Path, "read_text", read)
    elif missing == "changed_identity":
        identities = iter(range(100))
        monkeypatch.setattr(
            Path, "stat", lambda _: SimpleNamespace(st_dev=1, st_ino=next(identities))
        )
    else:
        for name in tuple(files):
            if name.endswith("memory.events"):
                files[name] = "high -1\n"
    observed = proctree.host_pressure()
    assert observed is not None and not observed.reclaim_samples


def test_worker_refuses_replacement_before_dispatch_and_during_idle_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typing import Any, cast
    from cozy_runtime.internal.worker.session import Worker

    calls = []
    executor = SimpleNamespace(
        pid=50, epoch=3, hello={"memory": ["reclaim_host"]}, poisoned=False, alive=lambda: True
    )
    current = [executor]
    row = SimpleNamespace(executor=lambda: current[0])
    tenant = SimpleNamespace(row=row, slot="idle")
    worker = SimpleNamespace(engine=SimpleNamespace(live={}))
    assert (
        Worker._reclaim_host(cast(Any, worker), cast(Any, tenant), 40, expected_generation=(50, 2))
        is None
    )
    monkeypatch.setattr(proctree, "rss_bytes", lambda _: 0)

    def call(
        request: object, *, eligible: Callable[[], bool], on_reply: object
    ) -> dict[str, object]:
        calls.append(request)
        current[0] = SimpleNamespace(pid=51, epoch=4)
        assert not eligible()
        return {"ok": False}

    executor.call_if_idle = call
    assert (
        Worker._reclaim_host(cast(Any, worker), cast(Any, tenant), 40, expected_generation=(50, 3))
        is None
    )
    assert len(calls) == 1


def test_zero_probe_does_not_touch_planner_backing() -> None:
    # Production entry's short circuit precedes all tier/staging accesses.
    owner = memory.Planner.__new__(memory.Planner)
    assert owner.reclaim_host(0) == 0


@pytest.mark.parametrize("protected", [False, True])
def test_churn_reclaims_real_cpu_store_backing_but_preserves_retry_state(protected: bool) -> None:
    from test_host_reclaim_tensors import resident

    owner, held = resident("idle")
    original = held.part.root.weight
    if protected:
        held.part.root._cozy_retry_tensors = lambda: (held.part.root.weight,)
    policy, observed, active, _ = setup()

    class RealCache(Cache):
        def reclaim_host(
            self, nbytes: int, *, expected_generation: object | None = None
        ) -> tuple[int, int]:
            assert expected_generation == self.reclaim_generation
            self.calls.append(nbytes)
            released = owner.reclaim_host(nbytes)
            return released, owner.host_memory()["managed_host_bytes"]

    idle = RealCache("idle")
    observed[0] = fact(2, 2)
    policy.relieve_host_pressure([active, idle])
    assert idle.calls == [0, 128]
    assert held.part.root.weight.is_meta is not protected
    assert not original.is_meta  # a retained reference prevents any claimed RSS credit
    assert not active.calls


def test_max_only_pressure_at_zero_debt_evacuates_idle_cache() -> None:
    policy, observed, active, idle = setup()

    def maximum(value: int) -> proctree.HostPressure:
        sample = proctree.HostReclaimSample("/parent", 1, 3, 100, None, value, value, None, 100)
        return proctree.HostPressure("/parent", 99, 100, 0, 0, (sample,))

    observed[0] = maximum(10)
    assert policy.relieve_host_pressure([active, idle]) is None
    observed[0] = maximum(11)
    policy.relieve_host_pressure([active, idle])
    assert idle.held == 0


@pytest.mark.parametrize("change", ["high_reset", "max_reset", "max_missing", "limit_kind"])
def test_distinct_event_resets_and_limit_kind_changes_cannot_hide_in_sum(change: str) -> None:
    policy, observed, active, idle = setup()
    sample = proctree.HostReclaimSample("/parent", 1, 3, 100, 10, 10, 10, 100, 200)
    observed[0] = replace(fact(), reclaim_samples=(sample,))
    assert policy.relieve_host_pressure([active, idle]) is None
    updated = replace(sample, high=20, maximum=20, refault_file=20)
    if change == "high_reset":
        updated = replace(updated, high=1)
    elif change == "max_reset":
        updated = replace(updated, maximum=1)
    elif change == "max_missing":
        updated = replace(updated, maximum=None)
    else:
        updated = replace(updated, high_limit=None, max_limit=100)
    observed[0] = replace(fact(), reclaim_samples=(updated,))
    assert policy.relieve_host_pressure([active, idle]) is None
    assert not idle.calls


def test_sampler_reads_max_only_finite_limit_and_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    files = {
        "/proc/self/cgroup": "0::/worker\n",
        "/sys/fs/cgroup/worker/memory.high": "max",
        "/sys/fs/cgroup/worker/memory.max": "100",
        "/sys/fs/cgroup/worker/memory.current": "99",
        "/sys/fs/cgroup/worker/memory.events": "max 12\n",
        "/sys/fs/cgroup/worker/memory.stat": "workingset_refault_file 30\ninactive_file 1\n",
    }
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: str(path) in files or str(path) == "/sys/fs/cgroup/cgroup.controllers",
    )
    monkeypatch.setattr(Path, "read_text", lambda path: files[str(path)])
    monkeypatch.setattr(Path, "stat", lambda path: SimpleNamespace(st_dev=1, st_ino=len(str(path))))
    observed = proctree.host_pressure()
    assert observed is not None and observed.debt_bytes == 0
    (sample,) = observed.reclaim_samples
    assert sample.high is None and sample.maximum == 12
    assert sample.high_limit is None and sample.max_limit == 100
