"""A stage over its process cap gives back its bytes before asking its owner for room.

Only accounting methods run here: allocator/driver observations are controlled CPU facts.
The actual GPU allocation and mapping behavior is qualified separately.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from types import ModuleType

import pytest

from cozy_runtime.internal import accel, plane, weight_policy
from cozy_runtime.internal.weights import WeightResidency, Weights

MIB = 1 << 20
CAP = 2951872512


@dataclass
class Account:
    weights: Weights
    memory: dict[str, int]
    committed: int
    asks: list[int]
    trimmed: list[tuple[int, int]]
    unparked: list[bool]
    residency: WeightResidency

    def held(self) -> int:
        return self.weights.context_bytes + self.memory["reserved_bytes"] + self.committed


def account(monkeypatch: pytest.MonkeyPatch, *, cached: int = 0, trim: bool = True) -> Account:
    memory = {
        "driver_free_bytes": 200 * MIB,
        "reserved_bytes": (300 + cached) * MIB,
        "allocated_bytes": 300 * MIB,
    }
    context = 213909504
    committed = CAP + 40 * MIB - context - memory["reserved_bytes"]
    floor = committed - 100 * MIB
    # The real methods operate on explicitly initialized accounting state. Construction
    # would measure a CUDA device, outside this CPU-only allocator-observation fixture.
    own = Weights.__new__(Weights)
    own.cap, own.host_only, own.kind = CAP, False, "cuda"
    own.torch = ModuleType("allocator_observations")
    own.plane = plane.open_plane(None)
    own.index, own.context_bytes, own.floor_free = 0, context, 256 * MIB
    own.activation, own.adaptive = {"denoise": 512 * MIB}, set()
    own.residencies = weakref.WeakSet()
    # Its unpark callback is the only residency I/O this accounting test observes.
    residency = WeightResidency.__new__(WeightResidency)
    own.residencies.add(residency)
    state = Account(own, memory, committed, [], [], [], residency)
    own.room = state.asks.append
    monkeypatch.setattr(own, "floor", lambda: floor)
    monkeypatch.setattr(own, "drain", lambda: None)
    monkeypatch.setattr(residency, "unpark", lambda: state.unparked.append(True))

    def release(*_: object) -> None:
        slack = memory["reserved_bytes"] - memory["allocated_bytes"]
        memory["reserved_bytes"] -= slack
        memory["driver_free_bytes"] += slack

    def shrink(need: int, keep: int = 0) -> int:
        state.trimmed.append((need, keep))
        freed = min(need + weight_policy.MARGIN, state.committed - keep) if trim else 0
        state.committed -= freed
        memory["driver_free_bytes"] += freed
        return freed

    def stats(_: object) -> plane.Stats:
        return plane.Stats(devices=(plane.DeviceStats(ordinal=0, committed=state.committed),))

    monkeypatch.setattr(own, "shrink", shrink)
    monkeypatch.setattr(accel, "allocation", lambda *_: memory)
    monkeypatch.setattr(accel, "release_cached", release)
    monkeypatch.setattr(plane, "stats", stats)
    return state


def test_cached_segments_go_back_first_and_the_ask_stays_inside_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = account(monkeypatch, cached=100)
    state.weights.room_before("denoise")
    assert state.memory["reserved_bytes"] == state.memory["allocated_bytes"]
    assert state.held() <= state.weights.cap
    assert not state.trimmed and not state.unparked
    assert all(0 < n <= state.weights.cap - state.held() for n in state.asks)


def test_maps_not_in_use_go_back_past_the_open_stages_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A planning floor counts regions no kernel reads now: only what is in use stays (the
    plane cannot unmap a leased region), never a floor."""
    state = account(monkeypatch)
    state.weights.room_before("denoise")
    assert state.trimmed == [(40 * MIB, 0)] and state.unparked == [True]
    assert state.held() <= state.weights.cap
    assert all(0 < n <= state.weights.cap - state.held() for n in state.asks)


def test_holdings_that_cannot_go_back_ask_the_machine_for_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = account(monkeypatch, trim=False)
    state.weights.room_before("denoise")
    assert state.trimmed and state.held() > state.weights.cap
    assert state.asks == [512 * MIB + weight_policy.MARGIN]
