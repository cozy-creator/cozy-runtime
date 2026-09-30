"""Inexact shape forecasts cannot prove refusal; measured floors still can.

Real CPU module bodies use managed planning with CUDA measurements stubbed. This
checks admission/replay boundaries, not GPU fit or output qualification.
"""

from dataclasses import replace
from typing import Any

import pytest

from cozy_runtime.author import pure
from cozy_runtime.internal import memory, tiers

torch = pytest.importorskip("torch")


def forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[memory.Planner, memory.Held, memory.OpKey]:
    owner = memory.Planner(torch, torch.device("cpu"))
    root = pure(torch.nn.Identity())
    root.use_tiling = False
    root.tile_sample_min_size = 256
    held = owner.adopt("generation", "vae", root, None, tiers.DEVICE)
    warm = memory.OpKey(
        held.kind,
        "decode",
        (((1, 16, 1, 64, 64), "torch.bfloat16"),),
        mode=memory._mode(root),
    )
    key = replace(warm, tensors=(((1, 16, 1, 192, 192), "torch.bfloat16"),))
    owner.records[warm] = memory.Record(ok=280341504)
    monkeypatch.setattr(owner, "allocatable", lambda: 2025488704)
    return owner, held, key


def test_scaled_hint_selects_floor_then_runs_body_once(monkeypatch: pytest.MonkeyPatch) -> None:
    owner, held, key = forecast(monkeypatch)
    assert owner._estimate(key) == (2523073536, False)
    owner.grant = 2463891456
    mode = memory._mode(held.part.root)
    owner.cuda = True
    monkeypatch.setattr(owner, "_allocated", lambda: 0)
    monkeypatch.setattr(owner, "_reserved", lambda: 0)
    monkeypatch.setattr(owner, "_sample", lambda: None)
    monkeypatch.setattr(owner, "_cap", lambda: None)
    # No GPU RNG exists in this CPU fixture; retain the real CPU snapshot path.
    arm = owner._arm

    def cpu_snapshot(target: memory.Held, leaves: list[object]) -> memory.Snapshot:
        owner.cuda = False
        try:
            return arm(target, leaves)
        finally:
            owner.cuda = True

    monkeypatch.setattr(owner, "_arm", cpu_snapshot)
    calls: list[tuple[int, ...]] = []

    def decode(value: Any) -> object:
        calls.append(tuple(value.shape))
        return held.part.root(value)

    value = torch.ones((1, 16, 1, 192, 192), dtype=torch.bfloat16)
    with torch.no_grad():
        assert owner.run(held, "decode", decode, (value,), {}) is value
    assert len(calls) == 1
    assert owner.grant == 2463891456 and memory._mode(held.part.root) == mode
    assert owner._estimate(key) == (0, True)  # replaces the forecast with this call's observation
    assert len(owner.log) == 1 and "forecast exceeded room" in owner.log[0]
    assert not owner.unusable


@pytest.mark.parametrize("kind", ["exact", "failed", "spread"])
def test_measured_bounds_and_missing_retry_contract_still_refuse(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    owner, held, key = forecast(monkeypatch)
    if kind == "exact":
        owner.records[key] = memory.Record(ok=2523073536)
    elif kind == "failed":
        owner.records[key] = memory.Record(fail=2523073536)
    elif kind == "spread":
        owner.spare = True
    with pytest.raises(memory.MemoryRefusal) as caught:
        owner._plan(held, key, kind != "unknown_contract", False)
    if kind == "spread":
        assert caught.value.code == memory.SPARED
    else:
        assert caught.value.needed == 2523073536
        assert caught.value.available == 2025488704
    assert not owner.log


def test_failed_exact_key_floor_survives_larger_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, held, key = forecast(monkeypatch)
    owner.records[key] = memory.Record(fail=2000000000)
    plan = owner._plan(held, key, True, True)
    assert plan.keep == 0
    assert owner.records[key].fail == 2000000000
    monkeypatch.setattr(owner, "allocatable", lambda: 1999999999)
    with pytest.raises(memory.MemoryRefusal) as caught:
        owner._plan(held, key, True, True)
    assert caught.value.needed == 2000000000


def test_tiled_forecast_does_not_scale_with_image_area(monkeypatch: pytest.MonkeyPatch) -> None:
    owner, _held, key = forecast(monkeypatch)
    owner.records = {replace(k, mode=(("use_tiling", True),)): v for k, v in owner.records.items()}
    key = replace(key, mode=(("use_tiling", True),))
    assert owner._estimate(key) == (280341504, False)


def test_nonretryable_forecast_is_not_a_refusal_proof(monkeypatch: pytest.MonkeyPatch) -> None:
    owner, held, key = forecast(monkeypatch)
    assert owner._plan(held, key, False, False).keep == 0
    assert "forecast exceeded room" in owner.log[0]
