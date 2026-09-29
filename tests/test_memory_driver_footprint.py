"""Exact driver/allocator observations, simulated on CPU, reduce subsequent caps."""

import os
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict, cast

import pytest

from cozy_runtime.internal import accel, executor, fill, memory, tiers
from cozy_runtime.internal.resolution import TensorResolution
from cozy_runtime.internal.seam import Channel


class Counters(TypedDict):
    used: int | None
    reserved: int
    allocated: int


def planner(monkeypatch: pytest.MonkeyPatch) -> tuple[memory.Planner, Counters, list[float]]:
    state: Counters = {"used": 500, "reserved": 400, "allocated": 300}
    fractions = []
    cuda = SimpleNamespace(
        set_per_process_memory_fraction=lambda value, device: fractions.append(value),
        max_memory_allocated=lambda device: state["allocated"],
        max_memory_reserved=lambda device: state["reserved"],
        reset_peak_memory_stats=lambda device: None,
    )
    owner = memory.Planner(
        SimpleNamespace(cuda=cuda), SimpleNamespace(type="cpu"), process_bytes=lambda: state["used"]
    )
    owner.cuda, owner.grant, owner.overhead = True, 500, 100
    monkeypatch.setattr(owner, "_reserved", lambda: state["reserved"])
    monkeypatch.setattr(owner, "_allocated", lambda: state["allocated"])
    monkeypatch.setattr(owner, "_driver", lambda: (800, 1000))
    return owner, state, fractions


def test_observed_external_growth_reduces_cap_and_exposes_peak_excess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, state, fractions = planner(monkeypatch)
    owner._cap()
    assert fractions[-1] == 0.4
    state["used"] = 560  # Same allocator, 60 more external bytes.
    owner._sample()
    assert owner.overhead == 160 and fractions[-1] == 0.34
    assert owner.peak_driver_process_bytes == 560
    assert owner.peak_driver_budget_excess_bytes == 60
    state["used"], state["reserved"] = 500, 340
    owner._sample()
    assert owner.driver_process_bytes == 500 and owner.overhead == 160
    assert owner.peak_driver_budget_excess_bytes == 60
    assert owner.facts()["peak_driver_process_bytes"] == 560


def test_unknown_or_racing_driver_read_does_not_fabricate_overhead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, state, _ = planner(monkeypatch)
    state["used"] = None
    assert not owner.observe_process()
    assert owner.driver_process_bytes == -1 and owner.overhead == 100

    def racing() -> int:
        state["reserved"] = 100
        return 500

    owner.process_bytes = racing
    state["reserved"] = 400
    assert not owner.observe_process()
    assert owner.overhead == 100
    assert owner.peak_driver_process_bytes == 500


def test_executor_queries_only_its_assigned_rank_without_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ours, theirs = socket.socketpair()
    owner = executor.Executor(Channel(ours), tmp_path, rank=1, world=2)
    owner.device_kind = "cuda"
    monkeypatch.setattr(executor, "_sealed_entries", lambda: ("GPU-a", "GPU-b"))
    calls = []

    def query(pids: set[int], kind: str, devices: tuple[str, ...], **kwargs: Any) -> Any:
        calls.append((pids, kind, devices, kwargs))
        return {"GPU-b": {os.getpid(): accel.ProcessMemory("present", 123)}}

    monkeypatch.setattr(accel, "process_memories_by_device", query)
    try:
        assert owner._process_bytes() == 123
        assert calls == [({os.getpid()}, "cuda", ("GPU-b",), {"allow_subprocess": False})]
        monkeypatch.setattr(
            accel,
            "process_memories_by_device",
            lambda *a, **k: {
                "GPU-b": {os.getpid(): accel.ProcessMemory("absent", 0)},
            },
        )
        assert owner._process_bytes() is None
    finally:
        ours.close()
        theirs.close()


def test_encoded_refill_uses_peak_relative_to_settled_native_bytes() -> None:
    torch = pytest.importorskip("torch")
    backend = fill.StreamingFillBackend.__new__(fill.StreamingFillBackend)
    cell = fill.ComponentMemory(
        destinations=64, native_destinations=64, native=16, native_resident=16
    )
    backend.component_memory = {"model": cell}
    backend.expected = {"model": frozenset({"model.weight"})}
    backend.resolved = {
        "model.weight": cast(TensorResolution, SimpleNamespace(route="encoded_gemm"))
    }
    root = torch.nn.Module()
    root.register_buffer("weight", torch.zeros(16, dtype=torch.uint8))
    owner = memory.Planner(torch, torch.device("cpu"))
    held = owner.adopt("c", "model", root, backend, tiers.DEVICE)
    root.weight = torch.empty((4, 4), dtype=torch.float32, device="meta")
    held.part.common.tier = tiers.DISK
    assert cell.settled == 16 and cell.peak == 80
    assert backend.transient("model") == 64
    assert owner._need(held.part, 0) == 80


def test_snapshot_rng_touches_only_assigned_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    owner = memory.Planner(torch, torch.device("cpu"))
    held = owner.adopt("c", "model", torch.nn.Identity(), None, tiers.HOST)
    owner.cuda, owner.device = True, torch.device("cuda:2")
    seen = []
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: (_ for _ in ()).throw(
            AssertionError("unassigned contexts must not be initialized")
        ),
    )

    def get_state(device: object) -> Any:
        seen.append(("get", device))
        return torch.get_rng_state()

    monkeypatch.setattr(torch.cuda, "get_rng_state", get_state)
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda state, device: seen.append(("set", device))
    )
    snapshot = owner._arm(held, [])
    owner._restore(snapshot)
    assert seen == [("get", torch.device("cuda:2")), ("set", torch.device("cuda:2"))]
