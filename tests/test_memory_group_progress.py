"""Group retry needs complete monotone progress from the same measurement source."""

from types import SimpleNamespace

import pytest

from cozy_runtime.author._executor_requests import Room
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.parallel.group import OpReady


@pytest.mark.parametrize(
    ("after", "grants", "prior", "expected"),
    [
        ((100, 100), (100, 100), {}, False),  # NVML's larger number is not CUDA gain.
        ((120, 139), (100, 100), {}, False),  # Every failed rank must cover its shortage.
        ((120, 140), (100, 100), {}, False),
        ((120, 140), (100, 100), {0: (120, 100), 1: (140, 100)}, False),
        ((100, 100), (120, 140), {}, False),
        ((120, 140), (120, 140), {}, True),
        ((100, 100), (120, 140), {0: (100, 120), 1: (100, 140)}, False),
    ],
)
def test_group_room_requires_full_new_gain_on_all_failed_ranks(after, grants, prior, expected):
    leader = iter((100, after[0]))
    follower = iter((100, after[1]))
    requests = []
    selected = []
    planner = SimpleNamespace(cuda=True, room_asks=0, _driver=lambda: (next(leader), 1000))
    group = SimpleNamespace(call=lambda _: [{"device_free_bytes": next(follower)}])

    def durable(request, _):
        requests.append(request)
        assert request.measured and all(need.measured for need in request.rank_needs)
        return Room(ok=True, memory_grants=grants, memory_free=(10000, 10000))

    executor = SimpleNamespace(
        rank=0,
        world=2,
        group=group,
        planner=planner,
        _durable=durable,
        _rank_memory=lambda: {},
        _grant=lambda _, values: selected.append(values),
    )
    statuses = [
        OpReady(
            operation=1,
            rank=rank,
            ok=False,
            code="device_shortfall",
            free_bytes=100,
            free_wanted=100 + short,
            grant_bytes=100,
            grant_wanted=100 + short,
        )
        for rank, short in enumerate((20, 40))
    ]
    progress = dict(prior)
    assert Executor._ask_group_room(executor, statuses, progress) is expected
    blocked = bool(prior) and any(
        100 + short <= prior[rank][0] and 100 + short <= prior[rank][1]
        for rank, short in enumerate((20, 40))
        if rank in prior
    )
    assert len(requests) == (0 if blocked else 1)
    assert selected == ([] if blocked else [grants])
    if expected:
        assert progress == {rank: (after[rank], grants[rank]) for rank in range(2)}
    else:
        assert progress == prior


@pytest.mark.parametrize(
    ("free", "grant", "after_free", "after_grant", "expected"),
    [
        (1000, 500, 1000, 530, True),
        (1000, 500, 1000, 529, False),
        (20, 2000, 50, 2000, True),
        (20, 2000, 49, 2000, False),
    ],
)
def test_preparation_prices_only_the_binding_constraint(
    free, grant, after_free, after_grant, expected
):
    from cozy_runtime.internal.memory import MemoryRefusal, Planner

    facts = SimpleNamespace(
        cuda=True,
        grant=grant,
        overhead=0,
        _driver=lambda: (free, 4000),
        _reserved=lambda: 480,
    )
    numbers = Planner.preparation_bytes(facts, MemoryRefusal("device_shortfall", "need", 50, 20))
    assert (numbers["free_wanted"], numbers["grant_wanted"]) == (50, 530)
    leader = iter((free, after_free))
    requests = []

    def durable(request, _):
        requests.append(request)
        assert request.measured and all(need.measured for need in request.rank_needs)
        return Room(ok=True, memory_grants=(after_grant, 4000))

    executor = SimpleNamespace(
        rank=0,
        world=2,
        group=SimpleNamespace(call=lambda _: [{"device_free_bytes": 4000}]),
        planner=SimpleNamespace(cuda=True, room_asks=0, _driver=lambda: (next(leader), 4000)),
        _durable=durable,
        _rank_memory=lambda: {},
        _grant=lambda *_: None,
    )
    status = OpReady(operation=1, rank=0, ok=False, code="device_shortfall", **numbers)
    progress = {}
    assert Executor._ask_group_room(executor, [status], progress) is expected
    assert (requests[0].rank_needs[0].free_bytes, requests[0].rank_needs[0].grant_bytes) == (
        50,
        530,
    )
    if expected:
        executor.planner._driver = lambda: (after_free, 4000)
        assert not Executor._ask_group_room(executor, [status], progress)
        assert len(requests) == 1  # An unchanged minimum cannot restart an abort loop.
