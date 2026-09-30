"""CPU policy checks against mutable tenant facts; no executor or accelerator is created."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass, field

import pytest

from cozy_runtime.internal.memory import Demand, Machine


@dataclass
class Resident:
    slot: str
    held: dict[int, int]
    overhead: dict[int, int]
    status: str = "idle"
    last_used: int = 0
    actions: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)

    @property
    def ordinals(self) -> tuple[int, ...]:
        return tuple(self.overhead)

    def state(self) -> str:
        return self.status

    def vacate(self, ordinals: tuple[int, ...]) -> bool:
        assert self.status != "active"
        self.actions.append(("vacate", ordinals))
        for ordinal in ordinals:
            self.held[ordinal] = self.overhead[ordinal]
        return True

    def end(self, why: str) -> bool:
        assert self.status != "active"
        self.actions.append(("end", self.ordinals))
        self.held = dict.fromkeys(self.ordinals, 0)
        return True


def machine(tenants: list[Resident], budget: int = 100) -> Machine:
    def driver(ordinal: int) -> tuple[int, int]:
        used = sum(t.held.get(ordinal, 0) for t in tenants)
        return 100 - used, 100

    return Machine({-1: budget}, driver, lambda _: None)


def test_unseen_admission_reclaims_weight_cache_but_preserves_contexts() -> None:
    me = Resident("me", {0: 10}, {0: 10})
    warm = Resident("warm", {0: 70}, {0: 10})
    policy = machine([me, warm])
    assert policy.admit(me, [me, warm], None) == {0: 90}
    assert warm.actions == [("vacate", (0,))]
    assert policy.room(me, 0, 0, None, [me, warm], measured=False) == 90
    assert warm.actions == [("vacate", (0,))]


def test_explicit_legacy_room_vacates_weights_without_ending_processes() -> None:
    me = Resident("me", {0: 10}, {0: 10})
    warm = Resident("warm", {0: 70}, {0: 10})
    policy = machine([me, warm])
    assert policy.room(me, 0, None, None, [me, warm], measured=False) == 90
    assert warm.actions == [("vacate", (0,))]
    # Legacy accounting can lack an initial footprint; a successful vacate reports it.
    warm.held.clear()
    policy.room(me, 0, None, None, [me, warm], measured=False)
    assert warm.actions == [("vacate", (0,)), ("vacate", (0,))]


def test_next_use_then_measured_reload_cost_choose_victims() -> None:
    me = Resident("me", {0: 10}, {0: 10})
    soon = Resident("soon", {0: 20}, {0: 5}, last_used=0)
    expensive = Resident("expensive", {0: 20}, {0: 5}, last_used=1)
    cheap = Resident("cheap", {0: 20}, {0: 5}, last_used=2)
    tenants = [me, soon, expensive, cheap]
    policy = machine(tenants)
    policy.update_queue(
        [
            Demand("soon", "a", "fp16", (0,), 0, 1),
            Demand("expensive", "b", "fp16", (0,), 2, 20),
            Demand("cheap", "c", "fp16", (0,), 2, 2),
        ]
    )
    assert policy.room(me, 0, 0, 50, tenants, measured=True) == 55
    assert cheap.actions == [("vacate", (0,))]
    assert not soon.actions and not expensive.actions


def test_active_group_is_charged_but_never_reclaimed() -> None:
    me = Resident("me", {0: 10}, {0: 10})
    active = Resident("group", {0: 70, 1: 70}, {0: 10, 1: 10}, status="active")
    warm = Resident("warm", {0: 10}, {0: 5})
    tenants = [me, active, warm]
    policy = machine(tenants)
    assert policy.admit(me, tenants, 50) == {0: 20}
    assert not active.actions and not warm.actions


def test_impossible_follower_prevents_partial_leader_eviction() -> None:
    me = Resident("me", {0: 10, 1: 10}, {0: 10, 1: 10})
    warm = Resident("warm", {0: 70}, {0: 10})
    active = Resident("active", {1: 80}, {1: 10}, status="active")
    tenants = [me, warm, active]
    policy = machine(tenants)
    assert policy.admit(me, tenants, {0: 80, 1: 50}) == {0: 30, 1: 20}
    assert not warm.actions and not active.actions


def test_measured_room_can_reclaim_idle_context_after_weights() -> None:
    me = Resident("me", {0: 20}, {0: 20})
    neighbour = Resident("neighbour", {0: 30}, {0: 25})
    tenants = [me, neighbour]
    policy = machine(tenants, budget=60)
    assert policy.room(me, 0, 0, 50, tenants, measured=False) == 30
    assert not neighbour.actions  # Even the optimistic weight-only bound cannot fit.
    assert policy.room(me, 0, 0, 50, tenants, measured=True) == 60
    assert neighbour.actions == [("vacate", (0,)), ("end", (0,))]


def test_quote_samples_each_device_once_and_keeps_rank_facts_distinct() -> None:
    me = Resident("me", {0: 10, 1: 20}, {0: 5, 1: 5})
    neighbour = Resident("other", {0: 40, 1: 60}, {0: 5, 1: 5})
    samples: list[int] = []

    def driver(ordinal: int) -> tuple[int, int]:
        samples.append(ordinal)
        return {0: (40, 100), 1: (10, 100)}[ordinal]

    policy = Machine({-1: 80}, driver, lambda _: None)
    assert policy.quote(me, [me, neighbour]) == {0: 40, 1: 20}
    assert samples == [0, 1]
    neighbour.held.pop(1)
    assert policy.quote(me, [me, neighbour]) == {0: 40, 1: 10}
    assert 1 not in neighbour.held


def test_prefetch_frontier_retains_exact_representations_and_is_immutable() -> None:
    policy = Machine({}, lambda _: None, lambda _: None)
    demand = Demand("one", "digest", "fp16", (1, 0, 1), 1, 3)
    with pytest.raises(FrozenInstanceError):
        demand.tenant = "changed"  # type: ignore[misc]
    policy.update_queue(
        [
            Demand("later", "later", "fp16", (0,), 2),
            demand,
            Demand("two", "digest", "fp16", (0, 1), 1, 3),
            Demand("three", "digest", "int8", (0, 1), 1, 2),
        ]
    )
    frontier = policy.prefetch_frontier()
    assert len(frontier) == 2
    assert {(d.artifact, d.representation, d.devices) for d in frontier} == {
        ("digest", "fp16", (0, 1)),
        ("digest", "int8", (0, 1)),
    }
    policy.update_queue([])
    assert policy.prefetch_frontier() == ()


def test_pending_host_promises_are_atomic_and_do_not_overadmit() -> None:
    policy = Machine({}, lambda _: None, lambda _: None, host_available=lambda: 100)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: policy.reserve_host(str(i), 30), range(8)))
    assert sum(results) == 3
    assert sum(policy.host_reserved.values()) == 90
    for owner in tuple(policy.host_reserved):
        policy.release_host(owner)
    assert policy.reserve_host("next", 100)
    assert not policy.reserve_host("over", 1)


def test_completed_host_copy_is_accounted_by_live_available_memory() -> None:
    available = 100
    policy = Machine({}, lambda _: None, lambda _: None, host_available=lambda: available)
    assert policy.reserve_host("copy", 50)
    available = 60  # Forty bytes actually allocated; ten transient bytes were unnecessary.
    policy.release_host("copy")
    policy.release_host("copy")
    assert policy.reserve_host("next", 60)
    assert not policy.reserve_host("over", 1)


def test_unknown_host_room_does_not_fabricate_capacity() -> None:
    policy = Machine({}, lambda _: None, lambda _: None)
    assert policy.reserve_host("empty", 0)
    assert not policy.reserve_host("copy", 1)
    policy.host_available = lambda: -1
    assert not policy.reserve_host("copy", 1)
    with pytest.raises(ValueError):
        policy.reserve_host("copy", -1)
    assert not policy.host_reserved


def test_worker_lane_records_feed_the_same_coherent_quote() -> None:
    from cozy_runtime.internal.worker.lanes import LaneSet
    from cozy_runtime.internal.worker.session import _Tenant

    lanes = LaneSet.from_envelope("a,b", worker_pid=1)
    group = lanes.group((0, 1))
    mine, neighbour = group.row("mine"), group.row("neighbour")
    mine.held.update({0: 10, 1: 20})
    mine.overhead.update({0: 3, 1: 4})
    neighbour.held.update({0: 50, 1: 30})
    neighbour.overhead.update({0: 5, 1: 5})
    # Quoting consumes genuine Worker lane facts; it never invokes Worker operations.
    me = _Tenant("mine", group, mine, None)  # type: ignore[arg-type]
    other = _Tenant("neighbour", group, neighbour, None)  # type: ignore[arg-type]
    policy = Machine({-1: 80}, lambda ordinal: ({0: 40, 1: 50}[ordinal], 100), lambda _: None)
    assert policy.quote(me, [me, other]) == {0: 30, 1: 50}
    neighbour.held.pop(1)
    assert policy.quote(me, [me, other]) == {0: 30, 1: 50}
    policy.budget.clear()
    assert policy.quote(me, [me, other]) == {0: 50, 1: 70}
    assert 1 not in neighbour.held  # Physical room did not manufacture a legacy footprint.


@pytest.mark.parametrize("free,expected", [(100, 50), (70, 20), (40, -10)])
def test_unknown_legacy_usage_is_charged_once_against_the_quota(
    free: int, expected: int
) -> None:
    me = Resident("me", {0: 10}, {0: 10})
    peers = [Resident("old-a", {}, {0: 0}), Resident("old-b", {}, {0: 0})]
    policy = Machine({-1: 60}, lambda _: (free, 120), lambda _: None)
    assert policy.quote(me, [me, *peers]) == {0: expected}
    assert all(not peer.held and not peer.actions for peer in peers)
    policy.driver = lambda _: None
    assert policy.quote(me, [me, *peers]) == {0: None}


def test_prefetch_frontiers_are_independent_but_do_not_pass_overlapping_groups() -> None:
    policy = Machine({}, lambda _: None, lambda _: None)
    policy.update_queue(
        [
            Demand("first", "a", "fp16", (0,), 0),
            Demand("waiting-group", "b", "fp16", (0, 1), 1),
            Demand("behind-group", "c", "fp16", (1,), 2),
            Demand("independent", "d", "fp16", (2,), 3),
            Demand("unknown-first", "e", "fp16", (), 4),
            Demand("unknown-second", "f", "fp16", (), 5),
        ]
    )
    assert [d.tenant for d in policy.prefetch_frontier()] == [
        "first",
        "independent",
        "unknown-first",
    ]


def test_rank_room_vector_preflights_driver_free_as_well_as_grants() -> None:
    me = Resident("me", {0: 10, 1: 10}, {0: 10, 1: 10})
    warm = Resident("warm", {0: 70}, {0: 10})
    active = Resident("active", {1: 80}, {1: 10}, status="active")
    tenants = [me, warm, active]
    policy = machine(tenants)
    assert policy.room_vector(me, {0: (60, None, True), 1: (50, None, True)}, tenants) == {
        0: 30,
        1: 20,
    }
    assert not warm.actions
    with pytest.raises(ValueError):
        policy.room_vector(me, {2: (1, None, True)}, tenants)


def test_reaped_host_namespace_cleanup_releases_only_its_pending_claims() -> None:
    policy = Machine({}, lambda _: None, lambda _: None, host_available=lambda: 100)
    assert policy.reserve_host("executor:1:7:copy", 20)
    assert policy.reserve_host("executor:1:7:pending", 30)
    assert policy.reserve_host("executor:1:70:successor", 20)
    assert policy.reserve_host("prefetch:artifact", 20)
    policy.release_host_prefix("executor:1:7:")
    assert policy.host_reserved == {"executor:1:70:successor": 20, "prefetch:artifact": 20}
    with pytest.raises(ValueError):
        policy.release_host_prefix("")


def test_unknown_physical_quote_never_exceeds_device_total() -> None:
    me = Resident("me", {0: 90}, {0: 10})
    unknown = Resident("legacy", {}, {0: 10})
    policy = Machine({}, lambda _: (50, 100), lambda _: None)
    assert policy.quote(me, [me, unknown]) == {0: 100}
    assert unknown.held == {}
