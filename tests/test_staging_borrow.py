"""Real CPU buffers and delayed simulated DMA; no device is opened."""

from __future__ import annotations

import gc
import socket
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from cozy_runtime.author._errors import RuntimeFailure
from cozy_runtime.internal import fill, tiers
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import Unload
from cozy_runtime.internal.seam import Channel
from test_host_transfer import Torch, unit
from test_host_transfer import device as device

torch = pytest.importorskip("torch")


def backend_for(
    device: Torch, *, pinned: bool = True, sizes: tuple[int, ...] = (40, 40)
) -> tuple[Any, set[Any]]:
    backend: Any = object.__new__(fill.StreamingFillBackend)
    backend.torch = device
    backend.state = "ready"
    backend._staging_lock = threading.Lock()
    backend._staging_busy = False
    backend._staging_borrow = None
    backend._staging_poisoned = ""
    backend._live_ring = None
    buffers = [torch.empty(size, dtype=torch.uint8, device="cpu") for size in sizes]
    if pinned:
        device.cuda.pinned.update(int(b.untyped_storage()._cdata) for b in buffers)
    backend._staging_ring = SimpleNamespace(
        buffers=buffers,
        allocated_bytes=sum(sizes),
        registered_bytes=sum(sizes),
        close=buffers.clear,
    )
    owners: set[Any] = set()
    backend.staging_started = lambda: owners.add(backend)
    backend.staging_drained = lambda: owners.discard(backend)
    backend.staging_released = lambda: None
    backend.enqueued = {"weight": object()}
    backend.leases = SimpleNamespace(release=lambda: None)
    return backend, owners


@pytest.mark.parametrize(
    "reason", ["missing", "pageable", "small", "live", "busy", "duplicate", "strided"]
)
def test_borrow_declines_without_allocation_or_custody(device: Torch, reason: str) -> None:
    backend, owners = backend_for(device, pinned=reason != "pageable")
    need = 66
    if reason == "missing":
        backend._staging_ring = None
    elif reason == "small":
        need = 81
    elif reason == "live":
        backend._live_ring = backend._staging_ring
    elif reason == "busy":
        backend._staging_busy = True
    elif reason == "duplicate":
        backend._staging_ring.buffers = [backend._staging_ring.buffers[0]] * 2
    elif reason == "strided":
        backend._staging_ring.buffers = [backend._staging_ring.buffers[0][::2]]
    assert backend.borrow_staging(need) is None
    assert not owners and device.allocations == 0


def test_lease_exact_geometry_and_stale_released_token_cannot_disarm_new_one(device: Torch) -> None:
    backend, owners = backend_for(device)
    first = backend.borrow_staging(66)
    assert first is not None and first.capacity == 80
    assert tuple(first.buffers) == tuple(backend._staging_ring.buffers)
    assert backend.borrow_staging(1) is None and owners == {backend}
    assert backend.release_staging() == 0
    first.release()
    assert not owners and not first.buffers
    second = backend.borrow_staging(80)
    assert second is not None
    first.release()
    assert backend._staging_borrow is second and owners == {backend}
    second.release()
    assert not owners
    assert backend.release_staging() == 80


def test_real_transfer_keeps_borrow_until_copy_completion(device: Torch) -> None:
    backend, owners = backend_for(device)
    device.cuda.auto = False
    owned: set[tiers.HostTransfer] = set()
    with torch.no_grad():
        transfer = tiers.HostTransfer.prepare(
            device,
            unit(),
            torch.device("cuda:0"),
            device.cuda.stream_value,
            acquire=backend.borrow_staging,
            device_capacity=66,
            valid=lambda: True,
            owned=owned,
        )
    assert transfer is not None and owners == {backend}
    result = []
    thread = threading.Thread(target=lambda: result.append(transfer.publish()))
    thread.start()
    try:
        assert transfer.copy_ready.waiting.wait(5)
        assert backend.release_staging() == 0 and backend.borrow_staging(1) is None
        assert owners == {backend} and transfer in owned
    finally:
        transfer.copy_ready.complete()
        thread.join(5)
    assert not thread.is_alive() and result == [True]
    assert not owners and backend._staging_borrow is None
    # Copy completion frees the pinned borrow; destination consumer custody is separate.
    assert transfer in owned
    transfer.finish()
    assert not owned


@pytest.mark.parametrize("action", ["fill", "close"])
def test_illegal_native_or_close_latches_poison_and_cannot_release_borrow(
    device: Torch, action: str
) -> None:
    backend, owners = backend_for(device)
    lease = backend.borrow_staging(66)
    assert lease is not None
    with pytest.raises(RuntimeFailure) as caught:
        backend._commit_enqueued() if action == "fill" else backend.close()
    assert caught.value.code == "device_fault" and backend.state == "poisoned"
    with pytest.raises(RuntimeFailure):
        lease.release()
    assert backend._staging_borrow is lease and lease.buffers and owners == {backend}
    assert backend.release_staging() == 0


def test_copy_drain_failure_retains_provider_and_transfer_custody(device: Torch) -> None:
    backend, owners = backend_for(device)
    owned: set[tiers.HostTransfer] = set()
    with torch.no_grad():
        transfer = tiers.HostTransfer.prepare(
            device,
            unit(),
            torch.device("cuda:0"),
            device.cuda.stream_value,
            acquire=backend.borrow_staging,
            device_capacity=66,
            valid=lambda: True,
            owned=owned,
        )
    assert transfer is not None
    device.cuda.fail_event = True
    with pytest.raises(RuntimeFailure):
        transfer.publish()
    assert backend._staging_borrow is not None and owners == {backend} and transfer in owned
    assert backend.release_staging() == 0


def test_old_drained_callback_cannot_race_new_borrow_quarantine(device: Torch) -> None:
    backend, owners = backend_for(device)
    lease = backend.borrow_staging(1)
    assert lease is not None
    entered = threading.Event()
    allow = threading.Event()
    attempting = threading.Event()
    acquired = threading.Event()

    def drained() -> None:
        entered.set()
        assert allow.wait(5)
        owners.discard(backend)

    backend.staging_drained = drained
    release = threading.Thread(target=lease.release)
    result = []

    def borrow() -> None:
        attempting.set()
        result.append(backend.borrow_staging(1))
        acquired.set()

    release.start()
    assert entered.wait(5)
    next_borrow = threading.Thread(target=borrow)
    next_borrow.start()
    try:
        assert attempting.wait(5) and not acquired.is_set()
    finally:
        allow.set()
        release.join(5)
        next_borrow.join(5)
    assert not release.is_alive() and not next_borrow.is_alive()
    assert result[0] is not None and owners == {backend}
    backend.staging_drained = lambda: owners.discard(backend)
    result[0].release()


def test_native_claim_precedes_work_and_does_not_hold_lock_during_work(device: Torch) -> None:
    backend, owners = backend_for(device)
    entered = threading.Event()
    allow = threading.Event()

    def work() -> None:
        entered.set()
        assert allow.wait(5)

    backend._commit_staging = work
    worker = threading.Thread(target=backend._commit_enqueued)
    worker.start()
    try:
        assert entered.wait(5)
        assert backend._staging_lock.acquire(blocking=False)
        backend._staging_lock.release()
        assert backend.borrow_staging(1) is None and backend.release_staging() == 0
    finally:
        allow.set()
        worker.join(5)
    assert not worker.is_alive() and not owners


def test_successful_construct_then_borrow_survives_unload_and_gc(
    device: Torch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ours, theirs = socket.socketpair()
    host = Executor(Channel(ours), tmp_path)
    child = host._construction_child("c")
    backend, _ = backend_for(device)
    child._own_fill_backend(backend)
    backend.staging_drained()  # successful construction drained, before the later borrow
    assert not host._staging_quarantine
    host.constructions["c"] = child
    monkeypatch.setattr(host, "_synchronize", lambda: None)
    lease = backend.borrow_staging(66)
    references = [weakref.ref(x) for x in (backend, lease, *lease.buffers)]
    try:
        with pytest.raises(RuntimeFailure):
            host.unload(Unload(construction="c"))
        assert "c" not in host.constructions
        del child, backend, lease
        gc.collect()
        assert host._staging_quarantine and all(ref() is not None for ref in references)
    finally:
        ours.close()
        theirs.close()


def test_construction_finalizer_cannot_disarm_outstanding_borrow(
    device: Torch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ours, theirs = socket.socketpair()
    host = Executor(Channel(ours), tmp_path)
    child = host._construction_child("c")
    backend, _ = backend_for(device)
    loans = []

    def body(*_: Any, **__: Any) -> dict[str, Any]:
        child._own_fill_backend(backend)
        loans.append(backend.borrow_staging(66))
        return {"ok": True}

    monkeypatch.setattr(child, "_prepare_first_body", body)
    try:
        assert child._prepare_first(cast(Any, None))["ok"]
        assert backend._live_ring is backend._staging_ring
        assert host._staging_quarantine[id(backend)] is backend
        loans[0].release()
        assert not host._staging_quarantine
    finally:
        ours.close()
        theirs.close()


def test_unreleased_foreign_token_fault_cannot_free_real_borrow(device: Torch) -> None:
    backend, owners = backend_for(device)
    active = backend.borrow_staging(66)
    assert active is not None
    stale = fill.StagingBorrow(backend, tuple(active.buffers))
    with pytest.raises(RuntimeFailure):
        stale.release()
    assert backend._staging_borrow is active and owners == {backend}
    with pytest.raises(RuntimeFailure):
        active.release()


@pytest.mark.parametrize("operation", ["prepare", "fill"])
def test_pre_dma_failure_does_not_leave_phantom_quarantine(device: Torch, operation: str) -> None:
    backend, owners = backend_for(device)
    backend._staging_ring = None
    original = RuntimeFailure("host staging not admitted", code="host_shortfall")

    def refuse() -> None:
        assert owners == {backend}
        raise original

    if operation == "prepare":
        backend._prepare_staging = refuse
    else:
        backend._commit_staging = refuse
    with pytest.raises(RuntimeFailure) as caught:
        backend.prepare_staging() if operation == "prepare" else backend._commit_enqueued()
    assert caught.value is original and not owners and not backend._staging_busy
