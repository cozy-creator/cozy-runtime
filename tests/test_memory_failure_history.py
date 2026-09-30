"""Repeated CPU managed calls with synthetic allocator facts, not GPU fit proof."""

from typing import Any

import pytest

from cozy_runtime.author import pure
from cozy_runtime.internal import memory, tiers

torch = pytest.importorskip("torch")


def test_smaller_replay_keeps_failure_until_richer_placement_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    root = pure(torch.nn.Module())
    root.anchor = torch.nn.Parameter(torch.empty(0))
    root._no_split_modules = ["Linear"]
    root.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)])
    held = owner.adopt("c", "transformer", root, None, tiers.HOST)
    assert [block.nbytes for block in held.part.blocks] == [8, 8]
    key = memory.OpKey(held.kind, "forward", (((1,), "torch.float32"),))
    owner.records[key] = memory.Record(ok=4)
    state = {"keep": 0, "working": 0, "budget": 32}
    placements: list[int] = []
    bodies: list[int] = []

    def allocated() -> int:
        return state["keep"] * 8 + state["working"]

    def apply(_: memory.Held, plan: memory.Plan) -> None:
        state["keep"] = plan.keep
        placements.append(plan.keep)
        for index, block in enumerate(held.part.blocks):
            block.tier = tiers.DEVICE if index < plan.keep else tiers.HOST

    def sample() -> None:
        if owner.stack:
            owner.stack[-1].working = max(owner.stack[-1].working, state["working"])

    def body(value: Any) -> Any:
        bodies.append(state["keep"])
        if state["keep"] == 2 and state["budget"] == 32:
            state["working"] = 16
            sample()
            state["working"] = 0
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 4 B")
        state["working"] = 4
        sample()
        state["working"] = 0
        return value + 1

    arm, rollback = owner._arm, owner._rollback

    def cpu_arm(target: memory.Held, leaves: list[object]) -> memory.Snapshot:
        owner.cuda = False
        try:
            return arm(target, leaves)
        finally:
            owner.cuda = True

    def cpu_rollback(target: memory.Held, snapshot: memory.Snapshot | None) -> None:
        owner.cuda = False
        try:
            rollback(target, snapshot)
        finally:
            owner.cuda = True

    # Keep real planning, OOM recovery, CPU state snapshots and module bodies.
    # Only accelerator measurement, transfer and CUDA RNG/cleanup are replaced.
    owner.cuda = True
    monkeypatch.setattr(owner, "_allocated", allocated)
    monkeypatch.setattr(owner, "_reserved", allocated)
    monkeypatch.setattr(owner, "allocatable", lambda: state["budget"] - allocated())
    monkeypatch.setattr(owner, "_apply", apply)
    monkeypatch.setattr(owner, "_sample", sample)
    monkeypatch.setattr(owner, "_cap", lambda: None)
    monkeypatch.setattr(owner, "_arm", cpu_arm)
    monkeypatch.setattr(owner, "_rollback", cpu_rollback)
    monkeypatch.setattr(tiers, "trim", lambda *_: None)
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: None, raising=False)
    with torch.no_grad():
        for _ in range(3):
            result = owner.run(held, "forward", body, (torch.ones(1),), {})
            assert torch.equal(result, torch.tensor([2.0]))
            assert owner.records[key] == memory.Record(ok=4, fail=20, fail_keep=2, fail_capacity=32)
        assert placements == bodies == [2, 0, 0, 0]
        assert len(owner.log) == 1
        # More room admits the originally failing placement again. Its actual
        # success retires the old failure without a permanent placement ceiling.
        state["budget"] = 64
        result = owner.run(held, "forward", body, (torch.ones(1),), {})
        assert torch.equal(result, torch.tensor([2.0]))
        assert placements == bodies == [2, 0, 0, 0, 2]
        assert owner.records[key] == memory.Record(ok=4)
        assert len(owner.log) == 1 and not owner.unusable


def test_multiple_failures_keep_bound_and_richest_failed_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    root = torch.nn.Module()
    held = owner.adopt("c", "model", root, None, tiers.HOST)
    key = memory.OpKey(held.kind, "forward", ())
    monkeypatch.setattr(tiers, "trim", lambda *_: None)
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: None, raising=False)
    monkeypatch.setattr(owner, "allocatable", lambda: 100)
    # Terminal recoveries also preserve what each real failure proved.
    for keep, working, asked, expected in (
        (7, 12, 8, memory.Record(fail=20, fail_keep=7, fail_capacity=100)),
        (3, 25, 5, memory.Record(fail=30, fail_keep=7, fail_capacity=100)),
        (9, 10, 2, memory.Record(fail=30, fail_keep=9, fail_capacity=100)),
    ):
        frame = memory.Frame(held, 0)
        frame.working = working
        with pytest.raises(memory.MemoryRefusal):
            owner._recover(
                held,
                key,
                memory.Plan(keep, []),
                frame,
                None,
                asked,
                "CUDA out of memory",
                request_room=False,
            )
        assert owner.records[key] == expected
