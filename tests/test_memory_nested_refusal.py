"""Real CPU module bodies under managed control flow, with CUDA measurement stubbed.

This proves refusal/rollback propagation, not physical placement or GPU recovery.
"""

import pytest

from cozy_runtime.internal import memory, tiers

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("nested", [False, True])
def test_preparation_refusal_cannot_hide_an_unrestored_parent(
    monkeypatch: pytest.MonkeyPatch, nested: bool,
) -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    child = torch.nn.Linear(2, 2)
    held_child = owner.adopt("generation", "child", child, None, tiers.HOST)

    class Parent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, value: object) -> object:
            self.calls += 1  # Unknown persistent state has no rollback contract.
            return owner.run(held_child, "forward", child, (value,), {})

    parent = Parent()
    held_parent = owner.adopt("generation", "parent", parent, None, tiers.HOST)
    owner.cuda = True  # Select managed control flow; no CUDA method is invoked below.
    monkeypatch.setattr(owner, "_sample", lambda: None)
    monkeypatch.setattr(owner, "_allocated", lambda: 0)
    monkeypatch.setattr(owner, "_apply", lambda *_: None)
    refusal = memory.MemoryRefusal("host_shortfall", "child preparation refused")

    def plan(held: memory.Held, *_: object) -> memory.Plan:
        if held is held_child:
            raise refusal
        return memory.Plan(0, [])

    monkeypatch.setattr(owner, "_plan", plan)
    target, module = (held_parent, parent) if nested else (held_child, child)
    with torch.no_grad(), pytest.raises(memory.MemoryRefusal) as caught:
        owner.run(target, "forward", module, (torch.ones(1, 2),), {})
    assert caught.value is refusal is owner.refusal
    assert refusal.reusable is (not nested)
    assert bool(owner.unusable) is nested
    assert parent.calls == int(nested)
    assert not owner.stack
    if nested:
        with pytest.raises(memory.MemoryRefusal) as again:
            owner.begin(None)
        assert again.value.code == "device_fault" and not again.value.reusable
