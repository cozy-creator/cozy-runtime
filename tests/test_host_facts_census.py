"""Actual registered CPU/meta tensors; host facts never initialize an accelerator."""

from types import SimpleNamespace
from typing import Any

import pytest

from cozy_runtime.internal import memory, tiers

torch = pytest.importorskip("torch")


def model() -> tuple[memory.Planner, memory.Held]:
    owner = memory.Planner(torch, torch.device("cpu"))
    root = torch.nn.Module()
    root.anchor = torch.nn.Parameter(torch.empty(0), requires_grad=False)
    root._no_split_modules = ["Linear"]
    root.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False) for _ in range(3)])
    for tensor in root.parameters():
        tensor.requires_grad_(False)
    held = owner.adopt("c", "m", root, None, tiers.HOST)
    return owner, held


def legacy_totals(owner: memory.Planner) -> tuple[int, int, int]:
    return (
        sum(owner._host_backings().values()),
        sum(owner._host_backings(cached=False).values()),
        sum(owner._host_backings(resident=False).values()),
    )


def actual_totals(owner: memory.Planner) -> tuple[int, int, int]:
    facts = owner.facts()
    return facts["host_bytes"], facts["host_resident_bytes"], facts["host_cached_bytes"]


def test_registered_units_overlap_deduplicate_and_do_not_add_classes() -> None:
    owner, held = model()
    a, b, c = held.part.blocks
    a.host[(id(a.module), "weight")] = a.module.weight.detach()
    b.tier = tiers.DEVICE
    b.host[(id(b.module), "weight")] = b.module.weight.detach()
    c.tier = tiers.DISK
    c.host[(id(c.module), "cached-alias")] = a.module.weight.detach()
    assert actual_totals(owner) == legacy_totals(owner) == (32, 16, 32)
    # A resident backing can also be cached; class subtotals are not additive.
    assert owner.facts()["disk_weight_bytes"] == 16
    held.hosted = True
    assert actual_totals(owner) == (32, 16, 32)  # host ownership includes hosted components
    assert owner.facts()["disk_weight_bytes"] == 0
    assert not torch.cuda.is_initialized()


def test_full_shared_storage_strides_empty_and_meta_are_preserved() -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    root = torch.nn.Module()
    storage = torch.arange(16, dtype=torch.float32)
    root.weight = torch.nn.Parameter(storage[2:7], requires_grad=False)
    root.register_buffer("alias", storage[::2])
    root.register_buffer("empty", torch.empty(0))
    root.register_buffer("parked", torch.empty(7, device="meta"))
    held = owner.adopt("c", "m", root, None, tiers.HOST)
    held.part.common.host[(id(root), "alias")] = storage[1:2]
    assert actual_totals(owner) == legacy_totals(owner) == (64, 64, 64)
    assert not torch.cuda.is_initialized()


def test_each_facts_call_recounts_fresh_backing_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, held = model()
    calls: list[tiers.Unit] = []
    original = tiers.Unit.slots

    def slots(unit: tiers.Unit) -> Any:
        calls.append(unit)
        return original(unit)

    monkeypatch.setattr(tiers.Unit, "slots", slots)
    expected = legacy_totals(owner)
    assert len(calls) == 2 * len(held.part.units)
    calls.clear()
    assert actual_totals(owner) == expected == (48, 48, 0)
    assert len(calls) == len(held.part.units)
    held.part.blocks[0].module.weight = torch.nn.Parameter(torch.ones(3, 3), requires_grad=False)
    assert actual_totals(owner) == legacy_totals(owner) == (68, 68, 0)
    assert not torch.cuda.is_initialized()


def test_non_cpu_cached_tensor_is_not_queried_or_charged() -> None:
    owner, held = model()

    class DeviceTensor:
        device = SimpleNamespace(type="cuda")

        def untyped_storage(self) -> Any:
            raise AssertionError("GPU backing must not be inspected")

    held.part.common.host[(0, "device")] = DeviceTensor()
    assert actual_totals(owner) == legacy_totals(owner) == (48, 48, 0)
    assert not torch.cuda.is_initialized()


def test_empty_owner_and_fresh_gpu_observation_order(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = memory.Planner(SimpleNamespace(), SimpleNamespace(type="cpu"))
    assert actual_totals(owner) == (0, 0, 0)
    events: list[str] = []
    owner.cuda = True

    def observe() -> bool:
        events.append("driver-observation")
        return False

    def reserved() -> int:
        events.append("reserved")
        return 50

    def allocated() -> int:
        events.append("allocated")
        return 30

    def room() -> int:
        events.append("room")
        return 70

    def host() -> tuple[int, int, int]:
        events.append("host")
        return 16, 12, 8

    monkeypatch.setattr(owner, "observe_process", observe)
    monkeypatch.setattr(owner, "_reserved", reserved)
    monkeypatch.setattr(owner, "_allocated", allocated)
    monkeypatch.setattr(owner, "allocatable", room)
    monkeypatch.setattr(owner, "_host_totals", host)
    facts = owner.facts()
    assert events == ["driver-observation", "reserved", "allocated", "room", "host"]
    assert facts["reserved_bytes"] == 50 and facts["allocated_bytes"] == 30
    assert facts["reserve_room_bytes"] == 70 and facts["reusable_cache_bytes"] == 20
    assert (facts["host_bytes"], facts["host_resident_bytes"], facts["host_cached_bytes"]) == (
        16,
        12,
        8,
    )
