"""CPU backing/transport cases for best-effort group memory reclamation."""

from types import SimpleNamespace

import pytest

from cozy_runtime.internal import memory, tiers
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.parallel import mirror

torch = pytest.importorskip("torch")


def test_grouped_unmovable_backing_is_not_promised_or_poisoned(monkeypatch) -> None:
    # CPU tensors stand in for device backing; a device-to-host copy needs new RAM.
    monkeypatch.setattr(tiers, "host_copy_bytes", lambda unit: unit.nbytes)
    owner = memory.Planner(torch, torch.device("cpu"), grouped=True)
    root = torch.nn.Linear(2, 2)
    held = owner.adopt("c", "weights", root, None, tiers.DEVICE)
    before = root.weight.detach().clone()
    assert owner._evictable(held) == 0
    answer = owner.vacate()
    assert answer["freed"] == {} and answer["held"] == {held.key: held.part.nbytes}
    assert torch.equal(root.weight, before) and not owner.unusable


def test_grouped_whole_store_can_drop_partly_hosted_component(monkeypatch) -> None:
    monkeypatch.setattr(tiers, "host_copy_bytes", lambda unit: unit.nbytes)
    owner = memory.Planner(torch, torch.device("cpu"), grouped=True)
    root = torch.nn.Module()
    root.anchor = torch.nn.Parameter(torch.ones(1))
    root._no_split_modules = ["Linear"]
    root.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)])
    held = owner.adopt("c", "weights", root, None, tiers.DEVICE)
    held.part.blocks[0].tier = tiers.HOST
    dropped = []

    def drop(name):
        dropped.append(name)
        root.to_empty(device="meta")

    held.part.store = SimpleNamespace(drop=drop)
    expected = held.part.on(tiers.DEVICE)
    assert owner._evictable(held) == expected
    answer = owner.vacate()
    assert answer["freed"] == {held.key: expected} and answer["held"] == {}
    assert dropped == [held.part.name]
    assert all(unit.tier == tiers.DISK for unit in held.part.units)


def test_leader_vacate_failure_drains_every_follower_reply() -> None:
    sent, read = [], []

    def fail():
        raise memory.MemoryRefusal("device_fault", "failed leader move", reusable=False)

    group = SimpleNamespace(
        send_one=lambda rank, command: sent.append(rank),
        reply_one=lambda rank, command: read.append(rank) or {"ok": True},
    )
    executor = SimpleNamespace(
        poisoned="", torch=torch, _device_ready=True, world=3, group=group,
        planner=SimpleNamespace(vacate=fail),
    )
    with pytest.raises(memory.MemoryRefusal, match="failed leader move"):
        Executor.vacate(executor)
    assert sent == read == [1, 2]


def test_hosted_reusable_refusal_preserves_the_group() -> None:
    group = SimpleNamespace(gpu=lambda rank: str(rank), broken="")
    with pytest.raises(memory.MemoryRefusal) as caught:
        mirror._accept(("c", "vae"), group, 1, {
            "ok": False, "code": "host_shortfall", "detail": "no host allocation",
            "reusable": True, "needed": 24, "available": 0,
        })
    assert caught.value.reusable and caught.value.shortfall["resource"] == "ram"
    assert group.broken == ""
