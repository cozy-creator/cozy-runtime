"""Real CPU model contracts with simulated room, never a GPU timing claim."""

from __future__ import annotations

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cozy_runtime.author._retry import pure
from cozy_runtime.internal import memory, tiers
from test_memory_metadata_anchor import Block, Network, Store, adopted


def fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[memory.Planner, memory.Held, memory.Held]:
    root = pure(Network())
    root.marker = torch.nn.Linear(3, 3)
    owner, victim = adopted(root)
    for unit in victim.part.units:
        unit.tier = tiers.DEVICE  # CPU backing exercises metadata; no device transfers.
    owner.scopes = [("c", frozenset(("network",)))]
    target = pure(Network())
    target.blocks = torch.nn.ModuleList([Block() for _ in range(64)])
    held = owner.adopt(
        "next", "network", target, Store(target, "network"), tiers.HOST, forward_only=True
    )
    monkeypatch.setattr(owner, "allocatable", lambda: 0)
    return owner, victim, held


def test_many_alternatives_share_only_one_victim_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, victim, held = fixture(monkeypatch)
    before = owner.anchor_validation_calls
    plan = owner._fit(held, 0, [victim], floor=False)
    assert plan.keep == 0 and plan.evict == [victim]
    assert owner.anchor_validation_calls - before == 1
    # A separate decision cannot inherit the previous eligibility snapshot.
    owner._fit(held, 0, [victim], floor=False)
    assert owner.anchor_validation_calls - before == 2


def test_actual_room_callback_invalidates_quotes_before_plan_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, victim, held = fixture(monkeypatch)
    called = []

    def room(free: int, grant: int, measured: bool) -> None:
        called.append((free, grant, measured))
        # Legal when quoted, then incompatible backing is introduced during the callback.
        victim.part.root.other.weight = torch.nn.Parameter(victim.part.root.marker.weight)

    owner.room = room
    with pytest.raises(memory.MemoryRefusal, match="alias"):
        owner._fit(held, 0, [victim], floor=False)
    assert len(called) == 1
    assert owner.unusable
    assert all(unit.tier == tiers.DEVICE for unit in victim.part.units)


def test_cached_quote_cannot_authorize_a_later_mutated_victim_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, victim, held = fixture(monkeypatch)
    plan = owner._fit(held, 0, [victim], floor=False)
    victim.part.root.other.weight = torch.nn.Parameter(victim.part.root.marker.weight)
    with pytest.raises(memory.MemoryRefusal, match="alias"):
        owner._apply(held, plan)
    assert owner.unusable
    assert all(unit.tier == tiers.DEVICE for unit in victim.part.units)


def test_allocator_room_is_reread_during_candidate_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, victim, held = fixture(monkeypatch)
    room = iter([0, 0, 0, 1 << 30])
    monkeypatch.setattr(owner, "allocatable", lambda: next(room, 1 << 30))
    plan = owner._fit(held, 0, [victim], floor=False)
    assert plan.keep == len(held.part.blocks) - 1
    assert plan.evict == []


def test_whole_store_callback_keeps_uncached_victim_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, victim, held = fixture(monkeypatch)
    held.part.blocks = held.part.blocks[:1]
    held.part.unit_loads = False
    for unit in held.part.units:
        unit.tier = tiers.DISK
    calls = 0

    def transient(_name: str) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            victim.part.root.other.weight = torch.nn.Parameter(victim.part.root.marker.weight)
        return 0

    assert held.part.store is not None
    monkeypatch.setattr(held.part.store, "transient", transient)
    with pytest.raises(memory.MemoryRefusal, match="alias"):
        owner._fit(held, 0, [victim], floor=False)
    assert calls == 2


def test_pending_cast_dispatch_keeps_uncached_victim_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, victim, held = fixture(monkeypatch)
    unit = held.part.blocks[-1]
    unit.tier = tiers.DISK
    unit.pending = [tiers.DTypeConvert(lambda tensor: tensor.to(dtype=torch.float64))]

    class ChangeDuringMetaProbe(TorchDispatchMode):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.probes = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
            if func == torch.ops.aten.empty.memory_format:
                self.probes += 1
                if self.probes == 2:
                    victim.part.root.other.weight = torch.nn.Parameter(
                        victim.part.root.marker.weight
                    )
            return func(*args, **(kwargs or {}))

    with ChangeDuringMetaProbe() as mode, pytest.raises(memory.MemoryRefusal, match="alias"):
        owner._fit(held, 0, [victim], floor=False)
    assert mode.probes >= 2
