"""CPU accounting/refusal proof using authored allocator observations.

The real 3 GiB GPU request is the qualification gate; these tests do not allocate
on a GPU or claim native eviction/physical capacity proof.
"""

from types import SimpleNamespace

import pytest

from cozy_runtime.internal import weights as module
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.weights import INDIVISIBLE, ResidencyRefusal, Weights


MIB = 1 << 20


def account(monkeypatch, *, cached: int = 0, trim: bool = True):
    memory = {"driver_free_bytes": 200 * MIB, "reserved_bytes": (300 + cached) * MIB,
              "allocated_bytes": 300 * MIB}
    cap = 2951872512
    context = 213909504
    committed = cap + 40 * MIB - context - memory["reserved_bytes"]
    device = SimpleNamespace(committed=committed)
    asks, trimmed, unparked = [], [], []
    floor = committed - 100 * MIB
    own = SimpleNamespace(
        room=asks.append, cap=cap, host_only=False, torch=None, kind="cuda",
        floor_free=256 * MIB, activation={"denoise": 512 * MIB}, plane=None,
        index=0, context_bytes=context, floor=lambda: floor, drain=lambda: None,
        residencies=[SimpleNamespace(unpark=lambda: unparked.append(True))],
    )

    def release(*_):
        slack = memory["reserved_bytes"] - memory["allocated_bytes"]
        memory["reserved_bytes"] -= slack
        memory["driver_free_bytes"] += slack

    def shrink(need, protected_floor):
        trimmed.append((need, protected_floor))
        freed = min(need + module.weight_policy.MARGIN, device.committed - protected_floor) if trim else 0
        device.committed -= freed
        memory["driver_free_bytes"] += freed

    own.shrink = shrink
    monkeypatch.setattr(module.accel, "allocation", lambda *_: memory)
    monkeypatch.setattr(module.accel, "release_cached", release)
    monkeypatch.setattr(module.plane, "stats", lambda *_: SimpleNamespace(device=lambda *_: device))
    return own, memory, device, asks, trimmed, unparked


def test_cached_reservations_are_released_before_cap_credit_is_requested(monkeypatch):
    own, memory, device, asks, trimmed, unparked = account(monkeypatch, cached=100)
    Weights.room_before(own, "denoise")
    assert own.cap == 2951872512
    assert memory["reserved_bytes"] == memory["allocated_bytes"]
    assert own.context_bytes + memory["reserved_bytes"] + device.committed <= own.cap
    assert not trimmed and not unparked
    assert all(0 < n <= own.cap - own.context_bytes - memory["reserved_bytes"] - device.committed for n in asks)


def test_over_cap_maps_trim_at_stage_boundary_and_keep_active_floor(monkeypatch):
    own, memory, device, asks, trimmed, unparked = account(monkeypatch)
    original_floor = own.floor()
    Weights.room_before(own, "denoise")
    assert own.cap == 2951872512
    assert trimmed == [(40 * MIB, original_floor)]
    assert unparked == [True]
    assert device.committed >= original_floor
    assert own.context_bytes + memory["reserved_bytes"] + device.committed <= own.cap
    assert all(0 < n <= own.cap - own.context_bytes - memory["reserved_bytes"] - device.committed for n in asks)


def test_unreleasable_live_holdings_refuse_locally_without_negative_rpc(monkeypatch):
    own, _, _, asks, trimmed, _ = account(monkeypatch, trim=False)
    with pytest.raises(ResidencyRefusal) as failure:
        Weights.room_before(own, "denoise")
    assert failure.value.code == INDIVISIBLE
    assert "live device holdings" in str(failure.value)
    assert trimmed and asks == []
    assert own.cap == 2951872512


def test_negative_device_room_never_enters_the_durable_channel():
    emitted = []
    own = SimpleNamespace(_durable=lambda *args: emitted.append(args), weights=None)
    with pytest.raises(ResidencyRefusal) as failure:
        Executor._device_room(own, -41943040)
    assert failure.value.code == "invalid_device_room"
    assert emitted == []
