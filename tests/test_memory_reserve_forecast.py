"""Reserve observations are forecasts, never allocator cache credit or replay authority."""

from functools import partial
from types import MethodType
from typing import Any

import pytest

from cozy_runtime.author import pure
from cozy_runtime.internal import memory, tiers

torch = pytest.importorskip("torch")


def fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[memory.Planner, memory.Held, dict[str, int]]:
    class Net(torch.nn.Module):  # type: ignore[name-defined, misc]  # optional Torch base
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
            self._no_split_modules = ["Linear"]

        def forward(self, x: Any) -> Any:
            return self.blocks[1](self.blocks[0](x))

    owner = memory.Planner(torch, torch.device("cpu"))
    held = owner.adopt("c", "net", pure(Net()), None, tiers.DEVICE)
    state = {"allocated": 1000, "reserved": 1200, "ap": 1000, "rp": 1200, "room": 500}
    owner.cuda = True
    monkeypatch.setattr(owner, "observe_process", lambda: False)
    monkeypatch.setattr(owner, "_allocated", lambda: state["allocated"])
    monkeypatch.setattr(owner, "_reserved", lambda: state["reserved"])
    monkeypatch.setattr(owner, "allocatable", lambda: state["room"])

    def scalar(metric: str, field: str, fallback: str) -> int:
        return state["ap" if metric == "allocated_bytes" else "rp"]

    def reset(device: object) -> None:
        state["ap"], state["rp"] = state["allocated"], state["reserved"]

    monkeypatch.setattr(owner, "_allocator_scalar", scalar)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    return owner, held, state


def test_native_peaks_before_reset_are_independent_of_logical_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, held, state = fixture(monkeypatch)
    frame = memory.Frame(held, 1000, loaded=50, reserve_base=1200)
    owner.stack.append(frame)
    state.update(ap=1350, rp=1670)
    owner._sample()
    assert frame.working == 300
    assert frame.reserve_growth == 470  # Never subtract the 50 logical weight bytes from R.
    assert state["rp"] == 1200
    state.update(ap=1200, rp=1300)
    owner._sample()
    assert frame.reserve_growth == 470


def test_whole_candidate_uses_max_and_streamed_candidate_keeps_allocated_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, held, state = fixture(monkeypatch)
    state["room"] = 470
    with torch.no_grad():
        plan = owner._fit(held, 341, [], floor=False, reserve=461)
        assert plan.keep == 2 and plan.needed == 461  # Not 802.
        state["room"] = 385
        plan = owner._fit(held, 341, [], floor=False, reserve=461)
        assert plan.keep == 1 and plan.needed == owner._need(held.part, 1) + 341
        state["room"] = 500
        assert owner._fit(held, 480, [], floor=False, reserve=461).needed == 480


@pytest.mark.parametrize(
    "mode",
    [
        "grouped",
        "group_phase",
        "spare",
        "compiled",
        "hosted",
        "mutable",
        "pending",
        "nested",
        "grad",
    ],
)
def test_unsupported_execution_does_not_consume_reserve_forecast(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    owner, held, state = fixture(monkeypatch)
    key = memory.OpKey(held.kind, "forward", ())
    owner.records[key] = memory.Record(ok=341, reserve=461)
    state["room"] = 385
    monkeypatch.setattr(owner, "_reserved", lambda: 0)
    if mode == "grouped":
        owner.grouped = True
    elif mode == "group_phase":
        owner.group_phase = "execute"
    elif mode == "spare":
        owner.spare = True
    elif mode in ("compiled", "hosted"):
        setattr(held, mode, True)
    elif mode == "mutable":
        held.part.common.mutable = True
    elif mode == "pending":
        held.part.common.pending = [lambda x: x]
    elif mode == "nested":
        owner.stack.append(memory.Frame(held, 0))
    with torch.set_grad_enabled(mode == "grad"):
        assert not owner._reserve_eligible(held)
        assert owner._plan(held, key, True, False).keep == 2


def test_exact_key_only_and_no_retry_contract_leave_old_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, held, state = fixture(monkeypatch)
    key = memory.OpKey(held.kind, "forward", (((1, 2), "torch.float32"),))
    owner.records[key] = memory.Record(ok=341, reserve=461)
    state["room"] = 385
    monkeypatch.setattr(owner, "_reserved", lambda: 0)
    with torch.no_grad():
        assert owner._plan(held, key, True, False).keep == 1
        assert owner._plan(held, key, False, False).keep == 2
        other = memory.OpKey(held.kind, "forward", (((1, 1), "torch.float32"),))
        assert owner._plan(held, other, True, False).keep == 2


@pytest.mark.parametrize("event", ["move", "conversion", "nested", "failure", "clean"])
def test_real_wrapped_operation_records_only_successful_attributable_growth(
    monkeypatch: pytest.MonkeyPatch,
    event: str,
) -> None:
    owner, held, state = fixture(monkeypatch)
    original_arm = owner._arm

    def arm(target: memory.Held, leaves: list[object]) -> memory.Snapshot:
        owner.cuda = False
        try:
            return original_arm(target, leaves)
        finally:
            owner.cuda = True

    monkeypatch.setattr(owner, "_arm", arm)
    monkeypatch.setattr(owner, "_rollback", lambda *_: None)
    monkeypatch.setattr(owner, "_apply", lambda *_: None)
    monkeypatch.setattr(owner, "_plan", lambda *_: memory.Plan(2, []))
    child = owner.adopt("c", "child", pure(torch.nn.Identity()), None, tiers.DEVICE)
    child.hosted = True
    actual = held.part.root.forward

    def body(self: Any, x: Any) -> Any:
        out = actual(x)
        state.update(ap=1341, rp=1661)
        if event == "move":
            with owner._moving():
                pass
        elif event == "conversion":
            owner._converted(held)
        elif event == "nested":
            owner.run(child, "forward", child.part.root.forward, (x,), {})
        elif event == "failure":
            raise ValueError("authored failure")
        return out

    # Claimed class implementation is established before wrapping, not an unknown hook.
    held.part.root.forward = MethodType(pure(body), held.part.root)
    tiers.wrap(held.part.root, partial(owner.run, held))
    with torch.no_grad():
        if event == "failure":
            with pytest.raises(ValueError, match="authored failure"):
                held.part.root(torch.ones(1, 2))
            assert not owner.records
        else:
            held.part.root(torch.ones(1, 2))
            record = next(iter(owner.records.values()))
            assert record.ok == 341
            assert record.reserve == (461 if event == "clean" else 0)
    assert not owner.stack


def test_floor_does_not_trim_for_reserve_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    owner, held, state = fixture(monkeypatch)
    key = memory.OpKey(held.kind, "forward", ())
    owner.records[key] = memory.Record(ok=341, reserve=461)
    state["room"] = 385
    trims: list[object] = []
    monkeypatch.setattr(tiers, "trim", trims.append)
    with torch.no_grad():
        owner._plan(held, key, True, True)
    assert trims == []
