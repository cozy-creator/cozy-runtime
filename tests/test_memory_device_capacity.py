"""Allocator reserve and driver free share one consumed-overhead accounting model; no CUDA."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest

from cozy_runtime.author._executor_requests import RankRoom
from cozy_runtime.internal.memory import Held, MemoryRefusal, Planner
from cozy_runtime.internal.parallel.group import OpReady


@dataclass
class Device:
    free: int
    reserved: int
    allocated: int
    overhead: int
    grant: int | None
    total: int = 2000


def planner(device: Device) -> tuple[Planner, list[float]]:
    fractions: list[float] = []
    cuda = SimpleNamespace(
        mem_get_info=lambda _: (device.free, device.total),
        memory_reserved=lambda _: device.reserved,
        memory_allocated=lambda _: device.allocated,
        set_per_process_memory_fraction=lambda value, _: fractions.append(value),
    )
    owner = Planner(SimpleNamespace(cuda=cuda, _C=SimpleNamespace()), SimpleNamespace(type="cuda"))
    owner.overhead, owner.grant = device.overhead, device.grant
    return owner, fractions


def test_real_1858_scope_demand_fits_without_charging_context_twice() -> None:
    device = Device(458489856, 415236096, 358960640, 195035136, 1068761088, 8186822656)
    owner, fractions = planner(device)
    owner._cap()
    assert device.reserved - device.allocated == 56275456  # remains charged, not credited
    assert owner.allocatable() == 458489856
    assert round(fractions[-1] * device.total) == 873725952
    assert owner._room_from([], 311167232) == []  # the refused scope's actual common demand
    # The corrected check changes no scope, weight placement, request shape or operation.
    assert not owner.scopes and not owner.operations


@pytest.mark.parametrize(
    ("device", "room", "cap"),
    [
        (Device(800, 400, 300, 100, 550), 50, 450),  # grant binds
        (Device(30, 400, 300, 100, 1500), 30, 430),  # physical driver free binds
        (Device(800, 400, 300, 100, 450), -50, 350),  # signed over-grant reserve
        (Device(800, 400, 300, 100, None), 800, 1200),
        (Device(800, 40, 30, 100, 50), -90, 0),  # context alone exceeds grant
        (Device(0, 400, 300, 100, 1500), 0, 400),
    ],
)
def test_cap_and_reserve_room_use_the_same_constraints(device: Device, room: int, cap: int) -> None:
    owner, fractions = planner(device)
    owner._cap()
    assert owner.allocatable() == room
    assert round(fractions[-1] * device.total) == cap


def test_fragmented_cache_is_neither_available_capacity_nor_invoice_credit() -> None:
    device = Device(80, 700, 100, 100, 1000)
    owner, fractions = planner(device)
    assert owner.allocatable() == 80
    requests: list[tuple[int, int, bool]] = []

    def request(free: int, grant: int, measured: bool) -> int:
        requests.append((free, grant, measured))
        return 1000

    owner.room = request
    assert owner._ask_room(300, [], measured=True)
    assert requests == [(300, 1100, True)]  # not (0,500): the600B cache holes are still reserve
    assert owner.allocatable() == 80  # a reply with no physical change provides no fabricated gain
    assert round(fractions[-1] * device.total) == 780
    assert not owner._ask_room(300, [], measured=True)


def test_owned_reclaim_is_priced_in_both_physical_and_total_invoices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, _ = planner(Device(80, 700, 100, 100, 1000))
    victim = cast(Held, object())
    monkeypatch.setattr(owner, "_evictable", lambda _: 50)
    requests: list[tuple[int, int, bool]] = []

    def request(free: int, grant: int, measured: bool) -> int:
        requests.append((free, grant, measured))
        return 1000

    owner.room = request
    owner._ask_room(300, [victim], measured=True)
    assert requests == [(250, 1050, True)]


def test_observed_nonallocator_growth_still_reduces_grant_capacity() -> None:
    device = Device(500, 400, 300, 100, 1000)
    owner, fractions = planner(device)
    owner.process_bytes = lambda: device.reserved + device.overhead
    assert owner.allocatable() == 500
    device.overhead, device.free = 180, 420
    owner._cap()
    assert owner.overhead == 180 and owner.allocatable() == 420
    assert fractions[-1] * device.total == 820
    assert owner.driver_process_bytes == 580
    # A later lower observation is not permission to discard the consumed high-water fact.
    device.overhead, device.free = 120, 480
    owner._cap()
    assert owner.overhead == 180 and owner.allocatable() == 420


@pytest.mark.parametrize(
    ("after_free", "after_grant", "expected"), [(50, 570, True), (49, 570, False), (50, 569, False)]
)
def test_group_preparation_keeps_physical_and_ownership_minima_distinct(
    after_free: int,
    after_grant: int,
    expected: bool,
) -> None:
    devices = [Device(1000, 480, 400, 40, 500), Device(20, 480, 400, 80, 2000)]
    owners = [planner(device)[0] for device in devices]
    statuses = []
    for rank, owner in enumerate(owners):
        facts = owner.preparation_bytes(
            MemoryRefusal("device_shortfall", "need", 50, owner.allocatable())
        )
        assert facts["free_wanted"] == 50
        assert facts["grant_wanted"] == [570, 610][rank]
        statuses.append(
            OpReady(
                operation=1,
                rank=rank,
                ok=False,
                code="device_shortfall",
                needed=facts["needed"],
                available=facts["available"],
                free_bytes=facts["free_bytes"],
                grant_bytes=facts["grant_bytes"],
                free_wanted=facts["free_wanted"],
                grant_wanted=facts["grant_wanted"],
            )
        )
    requests: list[tuple[RankRoom, ...]] = []

    def request(needs: tuple[RankRoom, ...]) -> tuple[int, ...]:
        requests.append(needs)
        devices[1].free = after_free
        return after_grant, 2000

    def apply(grants: tuple[int, ...]) -> None:
        for owner, grant in zip(owners, grants, strict=True):
            owner.grant = grant

    gained: dict[int, tuple[int, int]] = {}
    assert (
        owners[0].group_room(
            statuses, gained, 2, lambda: tuple(d.free for d in devices), request, apply
        )
        is expected
    )
    assert [(r.free_bytes, r.grant_bytes) for r in requests[0]] == [(50, 570), (50, 610)]
    if expected:
        assert [owner.allocatable() for owner in owners] == [50, 50]


@pytest.mark.parametrize(("observed", "needed"), [(0, 64), (128, 128)])
def test_oom_preparation_uses_allocation_demand_without_another_context_reserve(
    observed: int,
    needed: int,
) -> None:
    owner, _ = planner(Device(20, 480, 200, 80, 1000))
    owner.oom_asked = observed
    failure = RuntimeError("CUDA error: out of memory. Tried to allocate 64 B")
    facts = owner.preparation_bytes(failure)
    assert facts["needed"] == needed and facts["available"] == 20
    assert facts["free_wanted"] == needed
    assert facts["grant_wanted"] == 480 + 80 + needed


def test_external_physical_allocation_reduces_room_without_becoming_our_overhead() -> None:
    device = Device(500, 400, 300, 100, 1000)
    owner, fractions = planner(device)
    owner.process_bytes = lambda: device.reserved + device.overhead
    owner._cap()
    assert owner.allocatable() == 500
    device.free = 30  # Another process used470B; none of our reserve or context changed.
    owner._cap()
    assert owner.overhead == 100 and owner.driver_process_bytes == 500
    assert owner.allocatable() == 30 and round(fractions[-1] * device.total) == 430
