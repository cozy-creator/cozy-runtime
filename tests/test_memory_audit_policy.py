"""Audit regressions use actual CPU tensors and measured-policy fixtures, never CUDA."""

from types import MethodType
from typing import Any, cast

import pytest

from cozy_runtime.author import pure
from cozy_runtime.author._errors import DEVICE_OOM
from cozy_runtime.internal import attention_sol, memory, tiers
from test_machine_memory_policy import Resident, machine

torch = pytest.importorskip("torch")


def test_infeasible_rich_room_request_does_not_consume_minimum() -> None:
    requests = []

    def room(free: int, grant: int, measured: bool) -> int:
        requests.append((free, grant, measured))
        return 100

    owner = memory.Planner(torch, torch.device("cpu"), room=room)
    owner.grant = 100
    assert owner._ask_room(1000, [], measured=False)
    assert owner._ask_room(80, [], measured=True)
    assert not owner._ask_room(80, [], measured=True)
    assert [request[0] for request in requests] == [1000, 80]


def test_unknown_admission_reclaims_contexts_only_for_current_minimum() -> None:
    me = Resident("me", {0: 30}, {0: 30})
    old = Resident("old", {0: 40}, {0: 40}, last_used=0)
    newer = Resident("newer", {0: 20}, {0: 20}, last_used=1)
    tenants = [me, old, newer]
    policy = machine(tenants, budget=60)
    assert policy.admit(me, tenants, None) == {0: 40}
    assert old.actions == [("end", (0,))]
    assert not newer.actions
    # A historical high cannot end the remaining warm process.
    assert policy.admit(me, tenants, 60) == {0: 40}
    assert not newer.actions


def test_failed_residency_excluded_even_when_working_lower_bound_is_weak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    root = torch.nn.Module()
    root.anchor = torch.nn.Parameter(torch.empty(0))
    root._no_split_modules = ["Linear"]
    root.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(4)])
    held = owner.adopt("c", "model", root, None, tiers.HOST)
    key = memory.OpKey(held.kind, "forward", ())
    owner.records[key] = memory.Record(ok=1, fail=2, fail_keep=3, fail_capacity=100)
    monkeypatch.setattr(owner, "allocatable", lambda: 100)
    assert owner._plan(held, key, True, False).keep == 2
    monkeypatch.setattr(owner, "allocatable", lambda: 101)
    assert owner._plan(held, key, True, False).keep == 4


@pytest.mark.parametrize("alias", ["module", "parameter", "storage"])
def test_cross_component_alias_refuses_before_any_rebinding(alias: str) -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    a = torch.nn.Linear(2, 2)
    b = a if alias == "module" else torch.nn.Linear(2, 2)
    if alias == "parameter":
        b.weight = a.weight
    elif alias == "storage":
        b.weight = torch.nn.Parameter(a.weight.view_as(a.weight))
    owner.adopt("a", "model", a, None, tiers.HOST)
    before = a.weight.clone()
    with pytest.raises(memory.MemoryRefusal, match="shares a module, tensor or storage"):
        owner.adopt("b", "model", b, None, tiers.HOST)
    assert torch.equal(a.weight, before)
    assert list(owner.held) == ["a/model"]


def test_sol_wrapper_does_not_launder_foreign_forward() -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    root = pure(torch.nn.Identity())
    root.calls = 0

    def stateful(self: Any, value: Any) -> Any:
        self.calls += 1
        return value + self.calls

    root.forward = MethodType(stateful, root)
    held = owner.adopt("c", "model", root, None, tiers.HOST)
    assert not owner._state_contract(held, [])
    attention_sol.install_site(root, "model", "root")
    assert not owner._state_contract(held, [])


def test_decode_selection_is_finite_and_restores_rng() -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    root = pure(torch.nn.Identity())
    owner.adopt("c", "vae", root, None, tiers.HOST)
    rng = torch.get_rng_state().clone()
    calls = []

    def probe(side: int) -> None:
        assert torch.equal(torch.get_rng_state(), rng)
        calls.append(side)
        torch.rand(3)
        if side != 4:
            raise memory.MemoryRefusal(memory.SHORTFALL, "measured probe shortfall", 10, 5)

    assert root._cozy_select_decode_mode((0, 8, 4, 2), probe) == 4
    assert calls == [0, 8, 4]
    assert torch.equal(torch.get_rng_state(), rng)
    assert "selected request decode tile side 4" in owner.log[-1]


@pytest.mark.parametrize("code,reusable", [("device_fault", False), ("host_shortfall", True)])
def test_decode_selection_does_not_hide_other_faults(code: str, reusable: bool) -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    root = pure(torch.nn.Identity())
    owner.adopt("c", "vae", root, None, tiers.HOST)
    calls = []

    def probe(side: int) -> None:
        calls.append(side)
        raise memory.MemoryRefusal(code, "fault", reusable=reusable)

    with pytest.raises(memory.MemoryRefusal):
        root._cozy_select_decode_mode((8, 4), probe)
    assert calls == [8]


def test_measured_plan_stops_offloading_after_needed_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    root = torch.nn.Module()
    root.anchor = torch.nn.Parameter(torch.empty(0))
    root._no_split_modules = ["Linear"]
    root.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(4)])
    victim = owner.adopt("c", "victim", root, None, tiers.DEVICE)
    room = [0]
    moved = []
    monkeypatch.setattr(owner, "allocatable", lambda: room[0])

    def home(_part: tiers.Component, unit: tiers.Unit) -> None:
        moved.append(unit)
        room[0] += unit.nbytes
        unit.tier = tiers.HOST

    monkeypatch.setattr(owner, "_home", home)
    owner._offload(victim, target=8)
    assert len([unit for unit in moved if unit.nbytes]) == 1
    assert sum(unit.tier == tiers.DEVICE for unit in victim.part.blocks) == 3


def test_startup_reclaims_measured_context_without_requesting_extra_free_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from cozy_runtime.internal.executor import Executor

    owner = memory.Planner(torch, torch.device("cpu"))
    owner.overhead = 40
    owner.grant = 20
    calls = []
    executor = SimpleNamespace(host=SimpleNamespace(planner=owner))

    def room(free: int, grant: int, measured: bool) -> int:
        calls.append((free, grant, measured))
        owner.grant = grant
        return grant

    executor._ask_room = room
    monkeypatch.setattr(tiers, "trim", lambda _: None)
    Executor._startup_budget(cast(Executor, executor))
    assert calls == [(0, 40, True)]
    assert owner.grant == 40


@pytest.mark.parametrize("room", [3, 8])
def test_optional_cache_selects_once_before_the_request(
    monkeypatch: pytest.MonkeyPatch, room: int
) -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    held = owner.adopt("c", "model", pure(torch.nn.Identity()), None, tiers.HOST)
    monkeypatch.setattr(owner, "allocatable", lambda: room)
    calls = []

    def run(cache: bool) -> str:
        calls.append(cache)
        return "image"

    assert held.part.root._cozy_optional_cache(8, run) == "image"
    assert calls == [room == 8]


@pytest.mark.parametrize("requested", [False, True])
def test_explicit_cache_mode_is_not_changed_by_current_room(
    monkeypatch: pytest.MonkeyPatch, requested: bool
) -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    held = owner.adopt("c", "model", pure(torch.nn.Identity()), None, tiers.HOST)
    monkeypatch.setattr(owner, "allocatable", lambda: 0 if requested else 100)
    calls = []
    result = owner.optional_cache(held, 8, lambda mode: calls.append(mode), requested=requested)
    assert result is None
    assert calls == [requested]


@pytest.mark.parametrize(
    "fault",
    [
        memory.MemoryRefusal(memory.SHORTFALL, "reusable capacity refusal"),
        memory.MemoryRefusal(DEVICE_OOM, "reusable operation OOM"),
        memory.MemoryRefusal(DEVICE_OOM, "poisoned", reusable=False),
        memory.MemoryRefusal("host_shortfall", "host fault"),
        torch.OutOfMemoryError("outside managed operation"),
    ],
)
def test_optional_cache_never_restarts_the_request_after_a_fault(
    fault: BaseException,
) -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    held = owner.adopt("c", "model", pure(torch.nn.Identity()), None, tiers.HOST)
    calls = []

    def run(cache: bool) -> str:
        calls.append(cache)
        raise fault

    with pytest.raises(type(fault)):
        owner.optional_cache(held, 8, run)
    assert calls == [True]


def test_successful_decode_selection_releases_failed_probe_tensors_without_gc() -> None:
    import gc
    import weakref

    owner = memory.Planner(torch, torch.device("cpu"))
    root = pure(torch.nn.Identity())
    owner.adopt("c", "vae", root, None, tiers.HOST)
    retained = []

    def probe(side: int) -> None:
        if side == 8:
            latent = torch.zeros(4)
            retained.append(weakref.ref(latent))
            raise memory.MemoryRefusal(memory.SHORTFALL, "before body")

    enabled = gc.isenabled()
    gc.disable()
    try:
        assert root._cozy_select_decode_mode((8, 4), probe) == 4
        assert retained[0]() is None
    finally:
        if enabled:
            gc.enable()
        gc.collect()


def test_scope_minimum_reclaims_sibling_blocks_without_evicting_common(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    a = torch.nn.Module()
    a.anchor = torch.nn.Parameter(torch.empty(1))
    a._no_split_modules = ["Linear"]
    a.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(2)])
    b = torch.nn.Module()
    b.anchor = torch.nn.Parameter(torch.empty(2))
    first = owner.adopt("c", "first", a, None, tiers.DEVICE)
    second = owner.adopt("c", "last", b, None, tiers.HOST)
    assert first.part.nbytes == 20 and second.part.nbytes == 8
    owner.cuda = True  # Real tensor sizes; simulate only device accounting and transfers.

    def allocated() -> int:
        return sum(h.part.on(tiers.DEVICE) for h in owner.held.values())

    monkeypatch.setattr(owner, "_allocated", allocated)
    monkeypatch.setattr(owner, "_reserved", allocated)
    monkeypatch.setattr(owner, "allocatable", lambda: 24 - allocated())
    monkeypatch.setattr(owner, "_cap", lambda: None)
    monkeypatch.setattr(owner, "_home", lambda part, unit: setattr(unit, "tier", tiers.HOST))
    monkeypatch.setattr(
        owner, "_load_common", lambda held: setattr(held.part.common, "tier", tiers.DEVICE)
    )
    owner.enter("c", "render", ("first", "last"))
    assert first.part.common.tier == second.part.common.tier == tiers.DEVICE
    assert [unit.tier for unit in first.part.blocks] == [tiers.HOST, tiers.DEVICE]
    assert allocated() == 20


def test_replay_cannot_reclaim_the_only_scope_metadata_backing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    vae = pure(torch.nn.Linear(1, 1))
    sibling = torch.nn.Linear(1, 2)
    held = owner.adopt("c", "vae", vae, None, tiers.DEVICE)
    other = owner.adopt("c", "other", sibling, None, tiers.DEVICE)
    assert held.part.nbytes == 8 and other.part.nbytes == 16
    owner.scopes = [("c", frozenset(("vae", "other")))]
    owner.cuda = True

    def allocated() -> int:
        return sum(h.part.on(tiers.DEVICE) for h in owner.held.values())

    monkeypatch.setattr(owner, "_allocated", allocated)
    monkeypatch.setattr(owner, "_reserved", allocated)
    monkeypatch.setattr(owner, "allocatable", lambda: 32 - allocated())
    monkeypatch.setattr(owner, "_sample", lambda: None)
    monkeypatch.setattr(owner, "_cap", lambda: None)
    monkeypatch.setattr(owner, "_home", lambda part, unit: setattr(unit, "tier", tiers.HOST))
    monkeypatch.setattr(tiers, "trim", lambda *_: None)
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: None, raising=False)
    # Real CPU snapshots and retry body; CUDA facts and transfer are the only substitutes.
    arm, rollback = owner._arm, owner._rollback

    def cpu_arm(*args: Any) -> memory.Snapshot:
        owner.cuda = False
        try:
            return arm(*args)
        finally:
            owner.cuda = True

    def cpu_rollback(*args: Any) -> None:
        owner.cuda = False
        try:
            return rollback(*args)
        finally:
            owner.cuda = True

    monkeypatch.setattr(owner, "_arm", cpu_arm)
    monkeypatch.setattr(owner, "_rollback", cpu_rollback)
    calls = []

    def decode(value: Any) -> Any:
        calls.append(owner.allocatable())
        if len(calls) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 4 B")
        return value + 1

    with torch.no_grad(), pytest.raises(memory.MemoryRefusal, match="no smaller placement"):
        owner.run(held, "decode", decode, (torch.ones(1),), {})
    # There is no operation-only remainder: the sibling's real metadata backing
    # cannot disappear just because a replay wants more room.
    assert calls == [8]
    assert other.part.common.tier == tiers.DEVICE
    assert owner.room_asks == 0


@pytest.mark.parametrize("gained,success", [(2, True), (1, False)])
def test_replay_requires_uncovered_allocation_deficit_not_entire_buffer(
    monkeypatch: pytest.MonkeyPatch,
    gained: int,
    success: bool,
) -> None:
    owner = memory.Planner(torch, torch.device("cpu"), room=lambda *_: 100)
    held = owner.adopt("c", "vae", pure(torch.nn.Identity()), None, tiers.HOST)
    state = {"free": 10, "placements": 0}
    calls = []
    owner.cuda = True
    monkeypatch.setattr(owner, "_allocated", lambda: 0)
    monkeypatch.setattr(owner, "_reserved", lambda: 0)
    monkeypatch.setattr(owner, "allocatable", lambda: state["free"])
    monkeypatch.setattr(owner, "_sample", lambda: None)
    monkeypatch.setattr(owner, "_cap", lambda: None)
    monkeypatch.setattr(tiers, "trim", lambda *_: None)
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: None, raising=False)
    arm, rollback = owner._arm, owner._rollback

    def cpu_arm(*args: Any) -> memory.Snapshot:
        owner.cuda = False
        try:
            return arm(*args)
        finally:
            owner.cuda = True

    def cpu_rollback(*args: Any) -> None:
        owner.cuda = False
        try:
            return rollback(*args)
        finally:
            owner.cuda = True
            state["free"] = 10  # Failed transient activations are gone.

    def apply(*_: Any) -> None:
        state["placements"] += 1
        if state["placements"] > 1:
            state["free"] += gained

    monkeypatch.setattr(owner, "_arm", cpu_arm)
    monkeypatch.setattr(owner, "_rollback", cpu_rollback)
    monkeypatch.setattr(owner, "_apply", apply)

    def decode(value: Any) -> Any:
        calls.append(state["free"])
        if len(calls) == 1:
            state["free"] = 6  # Eight-byte allocation lacked two bytes at failure.
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 8 B")
        return value + 1

    with torch.no_grad():
        if success:
            result = owner.run(held, "decode", decode, (torch.ones(1),), {})
            assert torch.equal(result, torch.tensor([2.0]))
        else:
            with pytest.raises(memory.MemoryRefusal, match="shortfall needs 2 B"):
                owner.run(held, "decode", decode, (torch.ones(1),), {})
    assert len(calls) == (2 if success else 1)
    counts = owner.operations["c/vae.decode"]
    assert counts["oom_allocation_bytes"] == 8 and counts["oom_shortfall_bytes"] == 2
