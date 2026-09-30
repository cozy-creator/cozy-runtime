"""New GPU contexts use the real Worker's owned idle-memory admission.

The Worker, tenant processes and Vacate exchange are real. NVML and the CUDA
context allocation are hardware seams, so this CPU proof does not claim a GPU OOM.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace

import pytest

from cozy_runtime.author._executor_requests import Handler
from cozy_runtime.internal import accel
from cozy_runtime.internal.executor_commands import Command, Start
from cozy_runtime.internal.worker import child, prespawn
from cozy_runtime.internal.worker.session import HostedPlacement, Tenant
from cozy_runtime.protocol import worker_pb2 as pb
from test_gpu_scheduler import Machine
from test_memory_manager import TOTAL, GiB, needs_executor
from test_memory_manager import machine as machine

OOM = "GPU 1: cuDevicePrimaryCtxRetain failed: CUDA_ERROR_OUT_OF_MEMORY"


def contexts(machine: Machine, monkeypatch: pytest.MonkeyPatch) -> tuple[Tenant, Tenant]:
    machine.activate_warm(monkeypatch)
    machine.serving("A", "h3", gpus=2)
    machine.await_granted("A")
    machine.tick()
    idle = next(iter(machine.worker.hosted.values()))
    machine.finish("A")
    machine.tick()
    worker = machine.worker
    previous = worker._tenant(idle.placement.placement_id)
    previous.row().held = {0: 20 * GiB, 1: TOTAL - (225 << 20)}
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
    return previous, worker._tenant(placement.placement_id)


def allocation(
    machine: Machine,
    tenant: Tenant,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[Start], threading.Event]:
    executor = tenant.supervision.current
    assert executor is not None
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
        return original(
            current,
            command,
            timeout=timeout,
            total_timeout=total_timeout,
            on_progress=on_progress,
            on_request=on_request,
        )

    monkeypatch.setattr(child.Executor, "call", allocate)
    return calls, called


@needs_executor
@pytest.mark.parametrize("prespawned", [False, True])
def test_start_admits_the_full_group_before_allocating_a_new_context(
    machine: Machine,
    monkeypatch: pytest.MonkeyPatch,
    prespawned: bool,
) -> None:
    idle, fresh = contexts(machine, monkeypatch)
    if prespawned:
        fresh = replace(fresh, lane=prespawn._lane(machine.worker, (0, 1)))
    old = idle.supervision.current
    executor = fresh.supervision.current
    assert old is not None and executor is not None
    calls, _ = allocation(machine, fresh, monkeypatch)
    bindings = dict(fresh.bindings)
    assert fresh.lane.measure(machine.worker.memory.kind)[1].free_bytes == 225 << 20
    # An attempt can already hold the lane when its new context needs admission:
    # the shared physical RLocks must be safely re-entered.
    with fresh.lane.device:
        assert machine.worker._start_executor(fresh, executor) == ""
    assert len(calls) == 1
    assert calls[0].devices == "0,1" and calls[0].sequence_parallel_degree == 2
    assert fresh.bindings == bindings and not fresh.failed_bindings
    assert executor.started and executor.alive() and old.alive()
    assert idle.row().held == {} and idle.row().emptied == {0, 1}
    assert all(
        m.free_bytes == TOTAL for m in fresh.lane.measure(machine.worker.memory.kind).values()
    )
    assert any(
        "vacated" in event.step and "executor startup" in event.step
        for event in machine.worker.activity
    )


@needs_executor
@pytest.mark.parametrize("prespawned", [False, True])
def test_start_waits_for_active_work_before_vacating_its_memory(
    machine: Machine,
    monkeypatch: pytest.MonkeyPatch,
    prespawned: bool,
) -> None:
    idle, fresh = contexts(machine, monkeypatch)
    if prespawned:
        fresh = replace(fresh, lane=prespawn._lane(machine.worker, (0, 1)))
    executor = fresh.supervision.current
    assert executor is not None
    calls, called = allocation(machine, fresh, monkeypatch)
    active, leave, waiting = threading.Event(), threading.Event(), threading.Event()
    lock = fresh.lane.device.locks[0]

    class ObservedLock:
        def acquire(self) -> bool:
            if threading.current_thread() is starter:
                waiting.set()
            return lock.acquire()

        def release(self) -> None:
            lock.release()

    monkeypatch.setattr(fresh.lane.device, "locks", (ObservedLock(), *fresh.lane.device.locks[1:]))

    def executing() -> None:
        with idle.lane.device:
            active.set()
            leave.wait()

    result: list[str] = []
    holder = threading.Thread(target=executing)
    starter = threading.Thread(
        target=lambda: result.append(machine.worker._start_executor(fresh, executor))
    )
    holder.start()
    assert active.wait(10)
    try:
        starter.start()
        assert waiting.wait(10), "startup never acquired the shared physical locks"
        assert not called.is_set() and not calls
        assert idle.row().held[1] == TOTAL - (225 << 20)
    finally:
        leave.set()
        holder.join(10)
        starter.join(10)
    assert not holder.is_alive() and not starter.is_alive()
    assert result == [""] and len(calls) == 1
    assert idle.row().held == {}


@needs_executor
@pytest.mark.parametrize("readable", [True, False])
def test_unmanaged_or_unreadable_startup_oom_refuses_once_with_its_original_reason(
    machine: Machine,
    monkeypatch: pytest.MonkeyPatch,
    readable: bool,
) -> None:
    idle, fresh = contexts(machine, monkeypatch)
    executor = fresh.supervision.current
    assert executor is not None
    if readable:
        idle.row().held.clear()
    monkeypatch.setattr(
        accel,
        "device_memory",
        lambda *_: (
            accel.DeviceMemory("measured", 225 << 20, TOTAL)
            if readable
            else accel.DeviceMemory("unreadable")
        ),
    )
    calls, _ = allocation(machine, fresh, monkeypatch)
    refusal = machine.worker._start_executor(fresh, executor)
    assert refusal == "group_unformed: " + OOM
    assert len(calls) == 1 and not executor.alive()
    assert all(value == "group_unformed: " + OOM for value in fresh.failed_bindings.values())
    assert not any("vacated" in event.step for event in machine.worker.activity)
    assert idle.supervision.current is not None and idle.supervision.current.alive()


def test_prespawn_admission_uses_the_same_ordered_physical_locks(machine: Machine) -> None:
    lane = prespawn._lane(machine.worker, (0, 1))
    physical = tuple(machine.worker.lanes.by_id[f"lane-{o}"].locks[0] for o in (0, 1))
    assert lane.locks == physical and lane.device.locks == physical
    # The temporary startup lane is not another placement or scheduler grant.
    assert not lane.placements and lane.lane_id not in machine.worker.lanes.composition


@needs_executor
def test_a_started_executor_reuses_its_context_without_evicting_idle_neighbors(
    machine: Machine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idle, fresh = contexts(machine, monkeypatch)
    executor = fresh.supervision.current
    assert executor is not None
    calls, _ = allocation(machine, fresh, monkeypatch)
    executor.started = {"ok": True}
    assert machine.worker._prepare_generation(fresh) == ("", True)
    assert not calls and executor.alive()
    assert idle.row().held == {0: 20 * GiB, 1: TOTAL - (225 << 20)}
