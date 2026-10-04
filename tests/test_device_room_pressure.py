"""A stage that finds its process over its cap (run 4267: 40 MiB over a 3 GiB pool's cap)
gives back its own bytes, then asks the machine for room: never a negative request.
Allocator figures are the run's; the GPU proof is a rental cell."""

from types import SimpleNamespace

from cozy_runtime.internal import weights as module
from cozy_runtime.internal.weights import Weights

MIB = 1 << 20
CAP = 2951872512


def account(monkeypatch, *, cached: int = 0, trim: bool = True):
    memory = {
        "driver_free_bytes": 200 * MIB,
        "reserved_bytes": (300 + cached) * MIB,
        "allocated_bytes": 300 * MIB,
    }
    context = 213909504
    device = SimpleNamespace(committed=CAP + 40 * MIB - context - memory["reserved_bytes"])
    floor = device.committed - 100 * MIB
    asks, trimmed, unparked = [], [], []
    own = SimpleNamespace(
        room=asks.append,
        cap=CAP,
        host_only=False,
        torch=None,
        kind="cuda",
        floor_free=256 * MIB,
        activation={"denoise": 512 * MIB},
        plane=None,
        index=0,
        context_bytes=context,
        floor=lambda: floor,
        drain=lambda: None,
        residencies=[SimpleNamespace(unpark=lambda: unparked.append(True))],
    )
    own._held = lambda: Weights._held(own)

    def release(*_):
        slack = memory["reserved_bytes"] - memory["allocated_bytes"]
        memory["reserved_bytes"] -= slack
        memory["driver_free_bytes"] += slack

    def shrink(need, protected_floor):
        trimmed.append((need, protected_floor))
        if trim:
            freed = min(need + module.weight_policy.MARGIN, device.committed - protected_floor)
            device.committed -= freed
            memory["driver_free_bytes"] += freed

    own.shrink = shrink
    monkeypatch.setattr(module.accel, "allocation", lambda *_: memory)
    monkeypatch.setattr(module.accel, "release_cached", release)
    monkeypatch.setattr(module.plane, "stats", lambda *_: SimpleNamespace(device=lambda *_: device))
    return own, memory, device, asks, trimmed, unparked


def held(own, memory, device) -> int:
    return own.context_bytes + memory["reserved_bytes"] + device.committed


def test_cached_segments_go_back_first_and_the_ask_stays_inside_the_cap(monkeypatch):
    own, memory, device, asks, trimmed, unparked = account(monkeypatch, cached=100)
    Weights.room_before(own, "denoise")
    assert memory["reserved_bytes"] == memory["allocated_bytes"]
    assert held(own, memory, device) <= own.cap
    assert not trimmed and not unparked
    assert all(0 < n <= own.cap - held(own, memory, device) for n in asks)


def test_idle_maps_go_back_down_to_the_open_stages_floor(monkeypatch):
    own, memory, device, asks, trimmed, unparked = account(monkeypatch)
    floor = own.floor()
    Weights.room_before(own, "denoise")
    assert trimmed == [(40 * MIB, floor)] and unparked == [True]
    assert device.committed >= floor
    assert held(own, memory, device) <= own.cap
    assert all(0 < n <= own.cap - held(own, memory, device) for n in asks)


def test_holdings_that_cannot_go_back_ask_the_machine_for_room(monkeypatch):
    own, memory, device, asks, trimmed, _ = account(monkeypatch, trim=False)
    Weights.room_before(own, "denoise")
    assert trimmed and held(own, memory, device) > own.cap
    assert asks == [512 * MIB + module.weight_policy.MARGIN]
