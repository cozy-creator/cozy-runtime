"""New GPU contexts use the real Worker's owned idle-memory admission.

The Worker, tenant processes and their Budget/Vacate exchange are real. NVML (the idle
tenant's device bytes, gone once it is cut) and the CUDA context allocation are hardware
seams, so this CPU proof does not claim a GPU OOM.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from cozy_runtime.author._executor_requests import Handler
from cozy_runtime.internal import accel, proctree
from cozy_runtime.internal.executor_commands import Budget, Command, Start, Vacate
from cozy_runtime.internal.executor_replies import PlaneFacts
from cozy_runtime.internal.worker import child, prespawn
from cozy_runtime.internal.worker import memory as memory_module
from cozy_runtime.internal.worker.session import HostedPlacement, Tenant
from cozy_runtime.protocol import worker_pb2 as pb
from test_gpu_scheduler import VIRTUAL, Machine
from test_memory_manager import TOTAL, GiB, needs_executor
from test_memory_manager import machine as machine

OOM = "GPU 1: cuDevicePrimaryCtxRetain failed: CUDA_ERROR_OUT_OF_MEMORY"
HELD = {0: 20 * GiB, 1: TOTAL - (225 << 20)}


def cut(machine: Machine) -> bool:
    return any(
        ("vacated" in event.step or "cut to 0 B" in event.step) and "executor startup" in event.step
        for event in machine.worker.activity
    )


def contexts(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> tuple[Tenant, Tenant, dict[int, int]]:
    """An idle tenant holding `HELD` on both GPUs, and a fresh one about to start there.
    Returns them and the idle tenant's device bytes, emptied when the Worker cuts it."""
    machine.activate_warm(monkeypatch)
    machine.serving("A", "h3", gpus=2)
    machine.await_granted("A")
    machine.tick()
    idle = next(iter(machine.worker.hosted.values()))
    machine.finish("A")
    machine.tick()
    worker = machine.worker
    previous = worker._tenant(idle.placement.placement_id)
    occupied = dict(HELD)
    monkeypatch.setattr(
        accel,
        "device_memory",
        lambda entry, kind: accel.DeviceMemory(
            "measured", TOTAL - occupied.get(VIRTUAL.split(",").index(entry), 0), TOTAL
        ),
    )
    document = pb.Placement()
    document.CopyFrom(idle.placement.document)
    document.placement_id = "fresh-context"
    placement = replace(idle.placement, document=document)
    supervision = worker._new_placement_supervision(placement.placement_id)
    fresh = HostedPlacement(placement, supervision, bindings=dict(idle.bindings))
    worker.hosted[placement.placement_id] = fresh
    assert worker._assign_lane(placement, fresh.bindings, supervision)
    current = worker._own_executor(worker._tenant(placement.placement_id))
    assert not isinstance(current, str)
    return previous, worker._tenant(placement.placement_id), occupied


def allocation(
    machine: Machine,
    tenant: Tenant,
    monkeypatch: pytest.MonkeyPatch,
    idle: Tenant,
    occupied: dict[int, int],
) -> tuple[list[Start], threading.Event]:
    executor = tenant.supervision.current
    holder = idle.supervision.current
    assert executor is not None and holder is not None
    original = child.Executor.call
    calls: list[Start] = []
    called = threading.Event()

    def allocate(
        current: child.Executor,
        command: Command,
        *,
        timeout: float | None = None,
        total_timeout: float | None = None,
        on_progress: Callable[[dict[str, object]], None] | None = None,
        on_request: Handler | None = None,
    ) -> dict[str, object]:
        if current is executor and isinstance(command, Start):
            calls.append(command)
            called.set()
            picture = tenant.lane.measure(machine.worker.memory.kind)
            if any(m.state != "measured" or m.free_bytes < GiB for m in picture.values()):
                return {"ok": False, "code": "group_unformed", "detail": OOM}
            return {"ok": True, "stages": [], "follower_pids": []}
        reply = original(
            current,
            command,
            timeout=timeout,
            total_timeout=total_timeout,
            on_progress=on_progress,
            on_request=on_request,
        )
        if current is holder and isinstance(command, Budget | Vacate) and reply.get("ok"):
            occupied.clear()
        return reply

    monkeypatch.setattr(child.Executor, "call", allocate)
    return calls, called


@needs_executor
@pytest.mark.parametrize("prespawned", [False, True])
def test_start_admits_the_full_group_before_allocating_a_new_context(
    machine: Machine,
    monkeypatch: pytest.MonkeyPatch,
    prespawned: bool,
) -> None:
    idle, fresh, occupied = contexts(machine, monkeypatch)
    if prespawned:
        fresh = replace(fresh, lane=prespawn._lane(machine.worker, (0, 1)))
    old = idle.supervision.current
    executor = fresh.supervision.current
    assert old is not None and executor is not None
    calls, _ = allocation(machine, fresh, monkeypatch, idle, occupied)
    bindings = dict(fresh.bindings)
    assert fresh.lane.measure(machine.worker.memory.kind)[1].free_bytes == 225 << 20
    # An attempt can already hold its GPUs' turn when its new context needs admission:
    # the turn is re-entered, never waited for.
    with machine.worker.stages.hold(fresh.lane.ordinals, "an attempt's turn"):
        assert machine.worker._start_executor(fresh, executor) == ""
    assert len(calls) == 1
    assert calls[0].devices == "64,65" and calls[0].sequence_parallel_degree == 2
    assert fresh.bindings == bindings and not fresh.failed_bindings
    assert executor.started and executor.alive() and old.alive()
    assert not occupied and cut(machine)
    assert all(
        m.free_bytes == TOTAL for m in fresh.lane.measure(machine.worker.memory.kind).values()
    )


@needs_executor
@pytest.mark.parametrize("prespawned", [False, True])
def test_start_waits_for_active_work_before_vacating_its_memory(
    machine: Machine,
    monkeypatch: pytest.MonkeyPatch,
    prespawned: bool,
) -> None:
    idle, fresh, occupied = contexts(machine, monkeypatch)
    if prespawned:
        fresh = replace(fresh, lane=prespawn._lane(machine.worker, (0, 1)))
    executor = fresh.supervision.current
    assert executor is not None
    calls, called = allocation(machine, fresh, monkeypatch, idle, occupied)
    active, leave = threading.Event(), threading.Event()
    stages = machine.worker.stages

    def executing() -> None:
        # Work that recovers from no out-of-memory holds the GPUs: a whole turn.
        with stages.hold(idle.lane.ordinals, "active work"):
            active.set()
            leave.wait()

    def waiting() -> bool:
        demands = stages.view()["demands"]
        return any("executor startup" in key for key in demands)

    result: list[str] = []
    holder = threading.Thread(target=executing)
    starter = threading.Thread(
        target=lambda: result.append(machine.worker._start_executor(fresh, executor))
    )
    holder.start()
    assert active.wait(10)
    try:
        starter.start()
        bound = time.monotonic() + 10  # a hang bound on a loaded box, not a budget
        while not waiting():
            assert time.monotonic() < bound, "startup never asked for its turn"
            time.sleep(0.01)
        assert not called.is_set() and not calls
        assert occupied == HELD
    finally:
        leave.set()
        holder.join(10)
        starter.join(10)
    assert not holder.is_alive() and not starter.is_alive()
    assert result == [""] and len(calls) == 1
    assert not occupied


@needs_executor
@pytest.mark.parametrize("readable", [True, False])
def test_unmanaged_or_unreadable_startup_oom_refuses_once_with_its_original_reason(
    machine: Machine,
    monkeypatch: pytest.MonkeyPatch,
    readable: bool,
) -> None:
    idle, fresh, occupied = contexts(machine, monkeypatch)
    executor = fresh.supervision.current
    assert executor is not None
    if readable:
        # The idle tenant's plane reports nothing mapped: the 225 MiB left is unmanaged.
        idle.row().ledger.plane = PlaneFacts(committed_bytes=0)
        idle.row().ledger.resident.clear()
    monkeypatch.setattr(
        accel,
        "device_memory",
        lambda *_: (
            accel.DeviceMemory("measured", 225 << 20, TOTAL)
            if readable
            else accel.DeviceMemory("unreadable")
        ),
    )
    calls, _ = allocation(machine, fresh, monkeypatch, idle, occupied)
    refusal = machine.worker._start_executor(fresh, executor)
    assert refusal == "group_unformed: " + OOM
    assert len(calls) == 1 and not executor.alive()
    assert all(value == "group_unformed: " + OOM for value in fresh.failed_bindings.values())
    assert not cut(machine)
    assert idle.supervision.current is not None and idle.supervision.current.alive()


def test_a_prespawn_lane_is_no_placement(machine: Machine) -> None:
    lane = prespawn._lane(machine.worker, (0, 1))
    assert lane.ordinals == (0, 1) and lane.devices == "64,65"
    # The temporary startup lane is not another placement or scheduler grant.
    assert not lane.placements and lane.lane_id not in machine.worker.lanes.composition


@needs_executor
def test_start_ends_an_idle_tenant_whose_executor_is_already_gone(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run 2567: the idle tenant on the cards was a group that had just broken. Cutting its
    budget wrote to its closed seam, and the next replica's activation failed on a bad file
    descriptor. A closed seam is an executor that is gone: it is ended and the start goes on."""
    idle, fresh, occupied = contexts(machine, monkeypatch)
    old, executor = idle.supervision.current, fresh.supervision.current
    assert old is not None and executor is not None
    calls, _ = allocation(machine, fresh, monkeypatch, idle, occupied)
    old.close()
    reclaim = machine.worker.memory.reclaim

    def ended(*args: Any) -> None:
        occupied.clear()  # the driver frees a dead process's memory
        reclaim(*args)

    monkeypatch.setattr(machine.worker.memory, "reclaim", ended)
    assert machine.worker._start_executor(fresh, executor) == ""
    assert len(calls) == 1 and not occupied


@needs_executor
def test_a_started_executor_reuses_its_context_without_evicting_idle_neighbors(
    machine: Machine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idle, fresh, occupied = contexts(machine, monkeypatch)
    executor = fresh.supervision.current
    assert executor is not None
    calls, _ = allocation(machine, fresh, monkeypatch, idle, occupied)
    executor.started = {"ok": True}
    assert machine.worker._prepare_generation(fresh) == ("", True)
    assert not calls and executor.alive()
    assert occupied == HELD and not cut(machine)


@needs_executor
def test_idle_tenants_give_pinned_memory_back_when_the_host_has_none_left(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The laptop under MemoryHigh 12 GiB (rebench, 2026-10-02): pinned tiers were only ever
    raised, each by half of what was free at its moment, and the cgroup sat over its limit
    with 10 GiB pinned. When the host has nothing left, an idle tenant pinning more than its
    share of the machine's pinned total is lowered to it (its device budget untouched) and
    the Worker's kept tiers are trimmed; a tenant in its turn is left alone."""
    idle, _fresh, _ = contexts(machine, monkeypatch)
    memory = machine.worker.memory
    row = idle.row()
    slot = next(slot for slot, _lane, found in memory.tenants() if found is row)
    row.ledger.weights_bytes = 8 * GiB
    held = PlaneFacts(budget_bytes=6 * GiB, pinned_budget_bytes=8 * GiB, pinned_bytes=8 * GiB)
    row.ledger.plane = held
    original = child.Executor.call
    sent: list[tuple[child.Executor, Budget]] = []

    def call(current: child.Executor, command: Command, **kwargs: Any) -> dict[str, object]:
        if isinstance(command, Budget):
            sent.append((current, command))
        return original(current, command, **kwargs)

    monkeypatch.setattr(child.Executor, "call", call)
    trimmed: list[bool] = []

    def trim() -> int:
        trimmed.append(True)
        return 0

    monkeypatch.setattr(memory, "trim", trim)
    # The cgroup is at its limit: nothing free, and 8 GiB of it is pinned.
    monkeypatch.setattr(memory_module, "read_host_memory", lambda: proctree.HostMemory(0, 8 * GiB))
    memory.active.add(slot)
    memory.shed()
    assert not sent and trimmed == [True], "a tenant in its turn keeps its tier"
    memory.active.discard(slot)
    memory.shed()
    assert [(e is idle.supervision.current, c.vram_bytes, c.pinned_bytes) for e, c in sent] == [
        (True, 6 * GiB, 4 * GiB)
    ]
