"""CPU accounting checks; synthetic transfers do not qualify GPU streaming."""

from typing import Any

import pytest

from cozy_runtime.internal import memory, tiers

torch = pytest.importorskip("torch")


def test_operation_traffic_separates_placement_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    root = torch.nn.Module()
    root.anchor = torch.nn.Parameter(torch.zeros(1))
    root._no_split_modules = ["Linear"]
    root.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
    held = owner.adopt("generation", "dit", root, None, tiers.DEVICE)
    owner.begin(None)
    # Exercise the managed path without CUDA calls or real device transfers.
    owner.cuda = True
    monkeypatch.setattr(owner, "_sample", lambda: None)
    monkeypatch.setattr(owner, "_allocated", lambda: 0)
    monkeypatch.setattr(owner, "_retryable", lambda *_: False)
    monkeypatch.setattr(owner, "_counters", lambda: {})
    keep = iter((2, 1, 0))
    monkeypatch.setattr(owner, "_plan", lambda *_: memory.Plan(next(keep), []))

    def placement(*_: Any) -> None:
        owner.loads += 1
        owner.offloads += 2
        owner.moved += 100

    def body() -> str:
        owner.loads += 3
        owner.offloads += 4
        owner.moved += 200
        return "ok"

    monkeypatch.setattr(owner, "_apply", placement)
    for _ in range(2):
        assert owner.run(held, "forward", body, (), {}) == "ok"
    with pytest.raises(ValueError, match="body failed"):
        def failed() -> None:
            body()
            raise ValueError("body failed")
        owner.run(held, "forward", failed, (), {})
    doc = owner.document("generation")
    assert doc["operations"] == {
        "generation/dit.forward": {
            "calls": 3, "total_blocks": 2, "min_keep_blocks": 0, "max_keep_blocks": 2,
            "placement_calls": 3, "placement_stages": 3, "placement_evictions": 6,
            "placement_moved_bytes": 300,
            "invoke_calls": 3, "invoke_stages": 9, "invoke_evictions": 12,
            "invoke_moved_bytes": 600,
        }
    }
    assert doc["attempt_moved_bytes"] == 900
    assert doc["attempt_stages"] == 12 and doc["attempt_evictions"] == 18
    owner.cuda, owner.unusable = False, ""
    owner.begin(None)
    assert owner.document("generation")["operations"] == {}
    assert doc["operations"]  # Already emitted snapshots survive the next attempt.


def test_nested_traffic_is_inclusive_even_on_failure() -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    outer: dict[str, int] = {}
    inner: dict[str, int] = {}
    with pytest.raises(ValueError), owner._traffic(outer, "invoke"):
        owner.moved += 10
        with owner._traffic(inner, "placement"):
            owner.moved += 20
            raise ValueError("placement failed")
    assert outer["invoke_moved_bytes"] == 30
    assert inner["placement_moved_bytes"] == 20
    assert outer["invoke_calls"] == inner["placement_calls"] == 1
