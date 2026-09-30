"""Real registered CPU units, controlled allocator facts; never opens a GPU."""

from types import SimpleNamespace
from typing import Any

import pytest

from cozy_runtime.author import pure
from cozy_runtime.internal import memory, tiers

torch = pytest.importorskip("torch")


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> Any:
    state: dict[str, Any] = {
        "allocated": 100,
        "reserved": 100,
        "free": 0,
        "grant": 100,
        "homes": [],
        "trims": 0,
        "retained": set(),
        "release_limit": None,
    }
    cuda = SimpleNamespace(
        memory_allocated=lambda _: state["allocated"],
        memory_reserved=lambda _: state["reserved"],
        mem_get_info=lambda _: (state["free"], 100),
        set_per_process_memory_fraction=lambda *_: None,
    )

    class Torch:
        def __getattr__(self, name: str) -> Any:
            return cuda if name == "cuda" else getattr(torch, name)

    owner = memory.Planner(Torch(), torch.device("cpu"))
    owner.cuda, owner.device, owner.grant = True, torch.device("cuda:0"), 100
    actual_home = tiers.to_host

    def home(runtime: Any, unit: tiers.Unit) -> int:
        result = actual_home(runtime, unit)
        if unit.nbytes:
            state["homes"].append(unit.path)
            if unit.path not in state["retained"]:
                state["allocated"] -= unit.nbytes
        return result

    def trim(runtime: Any) -> None:
        state["trims"] += 1
        if state.get("fault"):
            raise RuntimeError("unproved device drain")
        released = state["reserved"] - state["allocated"]
        if state["release_limit"] is not None:
            released = min(released, state["release_limit"])
        state["reserved"] -= released
        state["free"] += released

    monkeypatch.setattr(tiers, "to_host", home)
    monkeypatch.setattr(tiers, "trim", trim)

    def component(name: str, sizes: list[int]) -> memory.Held:
        root = pure(torch.nn.Module())
        root.anchor = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        root._no_split_modules = ["Linear"]
        root.blocks = torch.nn.ModuleList([torch.nn.Linear(1, n // 4, bias=False) for n in sizes])
        for parameter in root.parameters():
            parameter.requires_grad_(False)
        owner.cuda = False
        held = owner.adopt("c", name, root, None, tiers.DEVICE)
        owner.cuda = True
        for unit in held.part.units:
            unit.path = name + "/" + (unit.path or "common")
        return held

    yield owner, state, component
    assert not torch.cuda.is_initialized()


def test_trim_proves_target_then_preserves_remaining_units(rig: Any) -> None:
    owner, state, component = rig
    held = component("m", [20, 20, 20])
    owner._offload(held, target=40)
    assert state["homes"] == ["m/blocks.0", "m/blocks.1"]
    assert state["trims"] == 1 and owner.allocatable() == 40
    assert held.part.blocks[2].tier == tiers.DEVICE
    assert not owner.dropped
    owner._cap()
    assert state["trims"] == 1  # no duplicate drain after already-proved trim


def test_actual_allocation_witness_accumulates_across_victims(rig: Any) -> None:
    owner, state, component = rig
    a, b, current = component("a", [20]), component("b", [20, 20]), component("current", [])
    owner._apply(current, memory.Plan(0, [a, b], needed=40))
    assert state["homes"] == ["a/common", "b/blocks.0"]
    assert b.part.blocks[1].tier == tiers.DEVICE
    assert state["trims"] == 1 and owner.allocatable() == 40


def test_retained_view_is_not_logical_release_credit(rig: Any) -> None:
    owner, state, component = rig
    held = component("m", [20, 20, 20])
    state["retained"] = {"m/blocks.0", "m/blocks.1", "m/blocks.2"}
    owner._offload(held, target=20)
    assert len(state["homes"]) == 3 and state["trims"] == 0
    assert owner.allocatable() == 0 and owner.dropped


@pytest.mark.parametrize("release_limit", [0, 10])
def test_fragmentation_requires_fresh_room_not_estimated_drop(rig: Any, release_limit: int) -> None:
    owner, state, component = rig
    held = component("m", [20, 20, 20])
    state["release_limit"] = release_limit
    owner._offload(held, target=40)
    assert len(state["homes"]) == 3
    assert owner.allocatable() < 40
    assert owner.dropped  # third drop happened after the unsuccessful trim


def test_external_free_improvement_can_stop_without_trim(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, state, component = rig
    held = component("m", [20, 20])
    state["grant"] = 200
    owner.grant = 200
    original = tiers.to_host

    def home(runtime: Any, unit: tiers.Unit) -> int:
        answer = original(runtime, unit)
        if unit.nbytes:
            state["free"] = 50
        return answer

    monkeypatch.setattr(tiers, "to_host", home)
    owner._offload(held, target=40)
    assert state["homes"] == ["m/blocks.0"] and state["trims"] == 0


def test_reallocation_consumes_release_witness(rig: Any) -> None:
    owner, state, _ = rig
    probe = owner._target_reclaim_probe(40)
    state["allocated"] = 80
    assert not probe()
    state["allocated"] = 95
    assert not probe()
    state["allocated"] = 70
    assert not probe()  # net30, not the sum45 of historical drops
    assert state["trims"] == 0
    state["allocated"] = 60
    assert probe() and state["trims"] == 1


@pytest.mark.parametrize("mode", ["unknown", "group", "group_phase"])
def test_unknown_and_group_policy_do_not_gain_target_trims(rig: Any, mode: str) -> None:
    owner, state, component = rig
    held = component("m", [20, 20, 20])
    owner.grouped = mode == "group"
    owner.group_phase = "execute" if mode == "group_phase" else ""
    owner._offload(held, target=None if mode == "unknown" else 20)
    assert len(state["homes"]) == 3 and state["trims"] == 0 and owner.dropped


def test_trim_failure_quarantines_committed_offload(rig: Any) -> None:
    owner, state, component = rig
    held = component("m", [20, 20])
    state["fault"] = True
    with pytest.raises(memory.MemoryRefusal) as error:
        owner._offload(held, target=20)
    assert error.value.code == "device_fault" and not error.value.reusable
    assert owner.unusable and owner.dropped
    assert held.part.blocks[0].tier == tiers.HOST and held.part.blocks[1].tier == tiers.DEVICE


def test_whole_store_stays_atomic_even_when_first_unit_would_cover_target(rig: Any) -> None:
    owner, state, component = rig
    held = component("m", [20, 20])
    held.part.store = SimpleNamespace()
    assert owner._whole_store(held.part)
    owner._offload(held, target=20)
    assert state["homes"] == ["m/blocks.0", "m/blocks.1"]
    assert state["trims"] == 1 and owner.allocatable() == 40
    assert all(unit.tier == tiers.HOST for unit in held.part.units)


def test_allocation_released_but_external_free_lost_is_not_admitted(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, state, component = rig
    held = component("m", [20, 20, 20])
    original = tiers.trim

    def trim(runtime: Any) -> None:
        original(runtime)
        state["free"] = 0  # another physical consumer took what the driver returned

    monkeypatch.setattr(tiers, "trim", trim)
    owner._offload(held, target=40)
    assert state["allocated"] == 40 and state["trims"] == 1
    assert owner.allocatable() == 0 and len(state["homes"]) == 3
