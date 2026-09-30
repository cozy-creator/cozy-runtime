"""Scope-entry transfers may retry; the authored method has not started. CPU/meta only."""

from typing import Any

import pytest

from cozy_runtime.author import Context, Model, pure, uses_components
from cozy_runtime.author._context import Device
from cozy_runtime.author._errors import DEVICE_OOM
from cozy_runtime.internal import memory, tiers
from cozy_runtime.internal.executor import _Scopes
from cozy_runtime.internal.warm import warm_generation

torch = pytest.importorskip("torch")


class Subject(Model[Any]):
    network: Any
    bodies: list[str]

    def warm(self, ctx: Context) -> None:
        self.bodies.append("warm prefix")
        torch.rand(1)
        self.render()
        self.bodies.append("warm suffix")

    @uses_components("network")
    def render(self) -> Any:
        self.bodies.append("authored body")
        return self.network.weight.sum() + torch.rand(2)


def subject(owner: memory.Planner, held: memory.Held) -> Subject:
    model = Subject.for_test(network=held.part.root, bodies=[])
    object.__setattr__(model, "_cozy_residency", _Scopes(owner, "construction"))
    return model


class Store:
    """Real CPU binding changes through the production disk-fill rollback path."""

    def __init__(
        self, owner: memory.Planner, *, always_fail: bool = False, whole: bool = False
    ) -> None:
        self.owner, self.always_fail, self.whole = owner, always_fail, whole
        self.part: tiers.Component | None = None
        self.calls = self.drops = 0

    def unit_loads(self, name: str, keys: frozenset[str]) -> bool:
        return not self.whole

    def load_schema(self, name: str, unit: tiers.Unit) -> tiers.RefillSchema:
        return tiers.RefillSchema(tuple(tensor.dtype for *_, tensor in unit.slots()))

    def transient(self, name: str) -> int:
        return 0

    def load(self, name: str) -> object:
        assert self.whole and self.part is not None
        self.load_unit(name, self.part.common)
        return self.part.root

    def drop(self, name: str) -> None:
        assert self.whole and self.part is not None
        self.drop_unit(self.part.common)

    def load_unit(self, name: str, unit: tiers.Unit) -> None:
        self.calls += 1
        for module, key, parameter, tensor in unit.slots():
            assert parameter
            module._parameters[key] = torch.nn.Parameter(torch.ones_like(tensor, device="cpu"))
        if self.calls == 1 or self.always_fail:
            self.owner.oom_asked = 8
            raise torch.OutOfMemoryError("scope fill allocation failed")

    def drop_unit(self, unit: tiers.Unit) -> None:
        self.drops += 1
        for module, key, parameter, tensor in unit.slots():
            assert parameter
            module._parameters[key] = torch.nn.Parameter(torch.empty_like(tensor, device="meta"))


def setup(
    monkeypatch: pytest.MonkeyPatch, *, gain: int, always_fail: bool = False, whole: bool = False
) -> tuple[memory.Planner, memory.Held, Store, list[tuple[int, int, bool]]]:
    owner = memory.Planner(torch, torch.device("cpu"))
    store = Store(owner, always_fail=always_fail, whole=whole)
    root = pure(torch.nn.Linear(2, 2, device="meta"))
    held = owner.adopt("construction", "network", root, store, tiers.DISK)
    store.part = held.part
    room = [128]
    requests: list[tuple[int, int, bool]] = []

    def ask(free: int, grant: int, measured: bool) -> int:
        requests.append((free, grant, measured))
        if len(requests) == 1:
            room[0] += gain
        return room[0]

    # Simulate only device accounting; tier loading, binding rollback and the scope
    # entry path are real. The launcher additionally denies accelerator device opens.
    owner.cuda = True
    owner.room = ask
    monkeypatch.setattr(owner, "allocatable", lambda: room[0])
    monkeypatch.setattr(owner, "_reserved", lambda: 0)
    monkeypatch.setattr(owner, "_allocated", lambda: 0)
    monkeypatch.setattr(owner, "_cap", lambda: None)
    monkeypatch.setattr(tiers, "trim", lambda _: None)
    return owner, held, store, requests


@pytest.mark.parametrize("whole", [False, True])
def test_scope_fill_oom_retries_only_transfer_after_measured_room(
    monkeypatch: pytest.MonkeyPatch,
    whole: bool,
) -> None:
    owner, held, store, requests = setup(monkeypatch, gain=16, whole=whole)
    rng = torch.get_rng_state().clone()
    expected = torch.Generator().set_state(rng)
    wanted = 4 + torch.rand(2, generator=expected)
    model = subject(owner, held)
    value = model.render()
    assert model.bodies == ["authored body"] and store.calls == 2 and store.drops == 1
    assert len(model.harness.calls) == 1 and not owner.scopes
    assert requests == [(136, 136, True)]
    assert held.part.common.tier == tiers.DEVICE
    assert all(torch.equal(t, torch.ones_like(t)) for t in held.part.root.parameters())
    assert torch.equal(value, wanted)
    assert torch.equal(torch.get_rng_state(), expected.get_state())
    counts = owner.operations["construction/network.scope:render"]
    assert counts["placement_ooms"] == 1 and counts["placement_retries"] == 1
    assert counts["placement_gain_bytes"] == 16


def test_scope_recovery_does_not_repeat_or_rewind_the_enclosing_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, held, store, _ = setup(monkeypatch, gain=16)
    model = subject(owner, held)
    expected = torch.Generator().set_state(torch.get_rng_state())
    torch.rand(1, generator=expected)
    torch.rand(2, generator=expected)
    warm_generation(model, device=Device(), cancel=lambda: False)
    assert store.calls == 2
    assert model.bodies == ["warm prefix", "authored body", "warm suffix"]
    assert torch.equal(torch.get_rng_state(), expected.get_state())


@pytest.mark.parametrize("gain", [0, 7])
def test_insufficient_gain_refuses_once_and_releases_scope_pins(
    monkeypatch: pytest.MonkeyPatch, gain: int
) -> None:
    owner, held, store, requests = setup(monkeypatch, gain=gain)
    model = subject(owner, held)
    with pytest.raises(memory.MemoryRefusal) as failed:
        model.render()
    assert failed.value.code == DEVICE_OOM
    assert store.calls == 1 and len(requests) == 1 and not owner.scopes
    assert not model.bodies and not model.harness.calls
    assert held.part.common.tier == tiers.DISK
    assert all(t.is_meta for t in held.part.root.parameters())


def test_a_second_oom_requires_another_real_gain(monkeypatch: pytest.MonkeyPatch) -> None:
    owner, _, store, requests = setup(monkeypatch, gain=16, always_fail=True)
    with pytest.raises(memory.MemoryRefusal):
        owner.enter("construction", "warm", ("network",))
    assert store.calls == 2 and store.drops == 2 and len(requests) == 2
    assert requests[1][0] == 152
    assert not owner.scopes


def test_observed_allocator_cache_release_can_suffice_without_a_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, held, store, requests = setup(monkeypatch, gain=0)
    room = [128]
    monkeypatch.setattr(owner, "allocatable", lambda: room[0])

    def trim(_: Any) -> None:
        room[0] += 8

    monkeypatch.setattr(tiers, "trim", trim)
    model = subject(owner, held)
    model.render()
    assert store.calls == 2 and not requests and model.bodies == ["authored body"]


def test_failed_device_cleanup_poison_prevents_later_scope_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, held, store, requests = setup(monkeypatch, gain=16)

    def broken(_: Any) -> None:
        raise RuntimeError("device synchronization failed")

    monkeypatch.setattr(tiers, "trim", broken)
    model = subject(owner, held)
    for _ in range(2):
        with pytest.raises(memory.MemoryRefusal) as failed:
            model.render()
        assert failed.value.code == "device_fault" and not failed.value.reusable
    assert owner.unusable and store.calls == 1 and not requests
    assert not owner.scopes and not model.bodies


@pytest.mark.parametrize("mode", ["grouped", "spare"])
def test_group_and_spare_keep_their_existing_admission_authority(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    owner, _, store, requests = setup(monkeypatch, gain=16)
    setattr(owner, mode, True)
    with pytest.raises(torch.OutOfMemoryError):
        owner.enter("construction", "warm", ("network",))
    assert store.calls == 1 and not requests and not owner.scopes


@pytest.mark.parametrize("failed_cache_growth", [0, 32])
def test_a_previous_unit_reusing_reserved_cache_is_not_recovery_progress(
    monkeypatch: pytest.MonkeyPatch,
    failed_cache_growth: int,
) -> None:
    owner, held, _, _ = setup(monkeypatch, gain=0)
    first = held.part.common
    first.tier = tiers.HOST
    second = tiers.Unit("second", torch.nn.Linear(1, 1), nbytes=8, tier=tiers.HOST)
    held.part.blocks = (second,)
    held.whole = True
    room = [128]
    monkeypatch.setattr(owner, "allocatable", lambda: room[0])
    monkeypatch.setattr(owner, "room", None)
    monkeypatch.setattr(tiers, "trim", lambda _: room.__setitem__(0, 128))
    calls = []

    def place(_: tiers.Component, unit: tiers.Unit) -> None:
        calls.append(unit)
        if unit is first:
            # This successful allocation fits an already charged reserved cache hole.
            # It does not lower allocatable or grant any additional physical capacity.
            first.tier = tiers.DEVICE
            return
        room[0] -= failed_cache_growth
        owner.oom_asked = 8
        raise torch.OutOfMemoryError("second unit failed atomically")

    monkeypatch.setattr(owner, "_load", place)
    with pytest.raises(memory.MemoryRefusal) as failed:
        owner.enter("construction", "warm", ("network",))
    assert calls == [first, second]
    assert failed.value.available == 0
    assert "gained 0 B" in str(failed.value)
    assert not owner.scopes


def test_non_oom_transfer_failure_is_not_replayed(monkeypatch: pytest.MonkeyPatch) -> None:
    owner, _, _, requests = setup(monkeypatch, gain=16)
    original = ValueError("invalid checkpoint")
    calls = []

    def fail(*_: Any) -> None:
        calls.append(1)
        raise original

    monkeypatch.setattr(owner, "_load", fail)
    with pytest.raises(ValueError) as caught:
        owner.enter("construction", "warm", ("network",))
    assert caught.value is original and calls == [1] and not requests and not owner.scopes
