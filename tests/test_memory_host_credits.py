"""Real CPU backing copies under aggregate credits; device sources are CPU-backed markers.

No CUDA API is invoked. This checks admission and cleanup, not accelerator transfer ordering.
"""

from __future__ import annotations

import weakref
from typing import Any

import pytest

from cozy_runtime.author import _executor_requests as requests
from cozy_runtime.author._errors import RuntimeFailure, is_device_oom
from cozy_runtime.internal import memory, proctree, tiers

torch = pytest.importorskip("torch")


def planner(limit: int) -> tuple[memory.Planner, memory.Machine, list[tuple[int, bool]]]:
    machine = memory.Machine({}, lambda _: None, lambda _: None, host_available=lambda: limit)
    events: list[tuple[int, bool]] = []

    def room(token: str, nbytes: int, release: bool) -> bool:
        events.append((nbytes, release))
        if release:
            machine.release_host(token)
            return True
        return machine.reserve_host(token, nbytes)

    return memory.Planner(torch, torch.device("cpu"), host_room=room), machine, events


class DeviceSource:
    """An actual CPU tensor with a device marker; copying still allocates real CPU bytes."""

    device = torch.device("cuda")

    def __init__(self, value: Any, machine: memory.Machine, *, fail: bool = False) -> None:
        self.value, self.machine, self.fail = value, machine, fail
        self.copies: list[weakref.ReferenceType[Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.value, name)

    def detach(self) -> DeviceSource:
        return self


@pytest.fixture(autouse=True)
def observe_marker_copies(monkeypatch: Any) -> None:
    copy = tiers._copy_backing

    def observed(torch: Any, tensor: Any, device: Any) -> Any:
        if isinstance(tensor, DeviceSource):
            assert sum(tensor.machine.host_reserved.values()) >= tensor.untyped_storage().nbytes()
            if tensor.fail:
                raise MemoryError("injected host copy failure")
        storage = copy(torch, tensor, device)
        if isinstance(tensor, DeviceSource):
            tensor.copies.append(weakref.ref(storage))
        return storage

    monkeypatch.setattr(tiers, "_copy_backing", observed)


def component(broker: memory.Machine, *, fail: bool = False) -> tiers.Component:
    root = torch.nn.Module()
    root._buffers["first"] = DeviceSource(torch.arange(4), broker)
    root._buffers["second"] = DeviceSource(torch.arange(4), broker, fail=fail)
    return tiers.partition("weights", root, tiers.DEVICE)


def test_host_request_round_trip_is_additive() -> None:
    request = requests.HostRoom(reservation="copy-1", nbytes=32)
    assert requests.decode(requests.encode(request) | {"future_field": 7}) == request
    released = requests.decode(
        requests.encode(requests.HostRoom(reservation="copy-1", release=True))
    )
    assert isinstance(released, requests.HostRoom) and released.release


def test_actual_backing_copy_is_reserved_then_releases_pending() -> None:
    owner, broker, events = planner(64)
    part = component(broker)
    assert tiers.host_copy_bytes(part.common) == 64
    owner._home(part, part.common)
    assert part.common.tier == tiers.HOST
    assert all(isinstance(tensor, torch.Tensor) for tensor in part.root._buffers.values())
    assert events == [(64, False), (64, True)]
    assert not broker.host_reserved and owner._host_pending == 0


def test_failed_copy_retains_original_placement_and_releases_all_pending() -> None:
    owner, broker, events = planner(64)
    part = component(broker, fail=True)
    first, second = part.root._buffers.values()
    with pytest.raises(MemoryError) as failed:
        owner._home(part, part.common)
    assert "injected" in str(failed.value)
    assert list(part.root._buffers.values()) == [first, second]
    assert part.common.tier == tiers.DEVICE
    assert first.copies and first.copies[0]() is None  # Even with the traceback retained.
    assert not broker.host_reserved and owner._host_pending == 0
    assert events[-1] == (64, True)


def test_partial_clean_cache_never_bypasses_reservation() -> None:
    owner, broker, _ = planner(0)
    part = component(broker)
    part.common.readonly = True
    first = part.root._buffers["first"]
    key = (id(part.root), "first")
    part.common.host[key] = first.value.clone()
    part.common.seen[key] = first.data_ptr()
    assert tiers.host_copy_bytes(part.common) == 32
    with pytest.raises(memory.MemoryRefusal) as refused:
        owner._home(part, part.common)
    assert refused.value.code == "host_shortfall"
    assert refused.value.shortfall["resource"] == "ram"
    assert not first.copies and not broker.host_reserved


def test_declared_snapshot_reserves_unique_storage_and_restores_cpu_bytes() -> None:
    owner, broker, events = planner(32)
    root = torch.nn.Module()
    root.register_buffer("state", torch.arange(4))
    root.register_buffer("alias", root.state[1:])
    root._cozy_retry_tensors = lambda: (root.state, root.alias)
    held = owner.adopt("construction", "state", root, None, tiers.HOST)
    snapshot = owner._arm(held, [])
    assert events == [(32, False), (32, True)]
    assert len(snapshot.tensors) == 1 and not broker.host_reserved
    root.state.fill_(9)
    owner._restore(snapshot)
    assert torch.equal(root.state, torch.arange(4))


def test_snapshot_failure_releases_credit_and_drops_saved_backing() -> None:
    owner, broker, _ = planner(32)
    root = torch.nn.Module()
    root.register_buffer("state", torch.arange(4))
    root._cozy_retry_tensors = lambda: (root.state,)

    def fail() -> None:
        assert broker.host_reserved
        raise ValueError("save failed")

    root._cozy_retry_state = fail, lambda _: None
    held = owner.adopt("construction", "state", root, None, tiers.HOST)
    with pytest.raises(ValueError) as failed:
        owner._arm(held, [])
    assert not broker.host_reserved and owner._host_pending == 0
    trace = failed.value.__traceback__
    while trace:
        if trace.tb_frame.f_code.co_name == "_arm":
            assert trace.tb_frame.f_locals["tensors"] == []
        trace = trace.tb_next


@pytest.mark.parametrize("phase", ["prepare", "execute"])
def test_group_phase_never_sends_host_requests(phase: str) -> None:
    owner, broker, events = planner(1000)
    part = component(broker)
    owner.group_phase = phase
    with pytest.raises(memory.MemoryRefusal):
        owner._home(part, part.common)
    assert events == [] and not broker.host_reserved
    # A fully valid existing host cache needs neither a credit nor IPC.
    part.common.readonly = True
    for name, value in part.root._buffers.items():
        part.common.host[(id(part.root), name)] = value.value.clone()
        part.common.seen[(id(part.root), name)] = value.data_ptr()
    owner._home(part, part.common)
    assert part.common.tier == tiers.HOST and events == []


def test_cpu_dtype_conversion_requests_target_backing_bytes() -> None:
    owner, broker, events = planner(64)
    root = torch.nn.Module()
    root.register_buffer("state", torch.arange(8, dtype=torch.float32))
    part = tiers.partition("weights", root, tiers.HOST)
    tiers.intercept_apply(torch, part, lambda: None, host_copy=owner._host_copy)
    root.double()
    assert root.state.dtype == torch.float64
    assert events == [(64, False), (64, True)] and not broker.host_reserved
    owner.host_room = lambda *_: False
    with pytest.raises(RuntimeFailure) as refused:
        root.float()
    assert refused.value.code == "host_shortfall" and root.state.dtype == torch.float64


def test_unknown_standalone_host_capacity_refuses_fresh_backing(monkeypatch: Any) -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    monkeypatch.setattr(proctree, "available_host_bytes", lambda: -1)
    with owner._host_copy(1) as admitted:
        assert not admitted
    assert owner._host_pending == 0


def test_host_reclaim_preserves_the_cache_used_to_size_the_pending_copy() -> None:
    owner, broker, _ = planner(32)
    part = component(broker)
    held = owner.adopt("construction", "weights", part.root, None, tiers.DEVICE)
    part = held.part
    part.common.readonly = True
    first = part.root._buffers["first"]
    key = (id(part.root), "first")
    part.common.host[key] = first.value.clone()
    part.common.seen[key] = first.data_ptr()
    original = owner.host_room
    attempts = 0

    def initially_denied(token: str, nbytes: int, release: bool) -> bool:
        nonlocal attempts
        if not release:
            attempts += 1
            if attempts == 1:
                return False
        assert original is not None
        return original(token, nbytes, release)

    owner.host_room = initially_denied
    owner._home(part, part.common)
    assert attempts == 2 and not first.copies
    assert part.common.tier == tiers.HOST and not broker.host_reserved


def test_group_uses_legal_disk_fallback_without_host_ipc() -> None:
    owner, broker, events = planner(0)
    part = component(broker)
    part.common.readonly = True
    dropped: list[str] = []

    class Store:
        def drop_unit(self, unit: tiers.Unit) -> None:
            dropped.append(unit.path)
            for module, name, _, tensor in unit.slots():
                module._buffers[name] = torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")

    part.store = Store()  # type: ignore[assignment]
    part.unit_loads = True
    owner.group_phase = "execute"
    owner._home(part, part.common)
    assert dropped == [""] and part.common.tier == tiers.DISK
    assert events == [] and not broker.host_reserved


def test_group_process_cannot_use_standalone_host_admission_before_gating() -> None:
    owner, broker, events = planner(1000)
    owner.grouped = True
    part = component(broker)
    assert not owner.group_phase
    with pytest.raises(memory.MemoryRefusal):
        owner._home(part, part.common)
    assert events == [] and not broker.host_reserved


class CPUBackedDeviceTensor(torch.Tensor):  # type: ignore[name-defined, misc]
    """Exercise real tensor binding/copies with a device marker and CPU backing only."""

    @property
    def device(self) -> Any:
        return torch.device("cuda")


def test_tier_moves_preserve_tied_parameters_views_and_mixed_dtype_backing() -> None:
    owner, broker, events = planner(64)
    root = torch.nn.Module()
    backing = torch.arange(16, dtype=torch.float32).as_subclass(CPUBackedDeviceTensor)
    root.first = torch.nn.Parameter(backing[:8], requires_grad=False)
    root.second = torch.nn.Parameter(backing[4:12:2], requires_grad=False)
    root.register_buffer("bytes", backing.view(torch.uint8)[3:29])
    root.register_buffer("strided", backing.reshape(4, 4).t())
    original = backing.as_subclass(torch.Tensor).clone()
    part = tiers.partition("aliases", root, tiers.DEVICE)
    assert part.nbytes == 64 and tiers.host_copy_bytes(part.common) == 64
    owner._home(part, part.common)
    assert events == [(64, False), (64, True)] and not broker.host_reserved
    assert len({value.untyped_storage().data_ptr() for *_, value in part.common.slots()}) == 1
    assert root.second.storage_offset() == 4 and root.second.stride() == (2,)
    assert root.bytes.storage_offset() == 3 and root.strided.stride() == (1, 4)
    assert torch.equal(root.first, original[:8])
    assert torch.equal(root.second, original[4:12:2])
    assert torch.equal(root.bytes, original.view(torch.uint8)[3:29])
    # The device-entry wrapper uses the same real storage-copy and binding mechanism.
    # Reapply markers to the buffers to simulate the reverse transfer without CUDA.
    for name, value in root._buffers.items():
        root._buffers[name] = value.as_subclass(CPUBackedDeviceTensor)
    before = root.first.untyped_storage().data_ptr()
    assert tiers.to_device(torch, part.common, torch.device("cpu")) == 64
    assert root.first.untyped_storage().data_ptr() != before
    assert len({value.untyped_storage().data_ptr() for *_, value in part.common.slots()}) == 1
    assert torch.equal(root.second, original[4:12:2])


def test_partial_alias_cache_reserves_the_entire_backing() -> None:
    owner, broker, events = planner(32)
    root = torch.nn.Module()
    backing = torch.arange(4)
    root._buffers["first"] = DeviceSource(backing, broker)
    root._buffers["alias"] = DeviceSource(backing[1:], broker)
    part = tiers.partition("aliases", root, tiers.DEVICE)
    part.common.host[(id(root), "first")] = backing.clone()
    part.common.seen[(id(root), "first")] = backing.data_ptr()
    assert tiers.host_copy_bytes(part.common) == 32
    owner._home(part, part.common)
    assert events == [(32, False), (32, True)]
    assert root.first.untyped_storage().data_ptr() == root.alias.untyped_storage().data_ptr()


def test_failed_device_entry_releases_partial_copies_even_with_retained_traceback() -> None:
    _, broker, _ = planner(64)
    part = component(broker, fail=True)
    sources = list(part.root._buffers.values())
    broker.reserve_host("test-copy", 64)
    with pytest.raises(MemoryError) as failed:
        tiers.to_device(torch, part.common, torch.device("cpu"))
    assert part.common.tier == tiers.DEVICE
    assert list(part.root._buffers.values()) == sources
    assert sources[0].copies[0]() is None
    trace = failed.value.__traceback__
    while trace:
        if trace.tb_frame.f_code.co_name == "_move":
            assert trace.tb_frame.f_locals["moves"] == []
            assert trace.tb_frame.f_locals["backing"] is None
        trace = trace.tb_next
    broker.release_host("test-copy")


def test_dtype_conversion_refuses_alias_split_before_changing_other_units() -> None:
    owner, _, events = planner(1024)
    root = torch.nn.Module()
    root.register_buffer("unrelated", torch.ones(2))
    root.register_buffer("state", torch.arange(8, dtype=torch.float32))
    root.register_buffer("alias", root.state[1::2])
    part = tiers.partition("aliases", root, tiers.HOST)
    tiers.intercept_apply(torch, part, lambda: None, host_copy=owner._host_copy)
    before = tuple(tensor.data_ptr() for *_, tensor in part.common.slots())
    with pytest.raises(RuntimeError, match="aliased storage"):
        root.double()
    assert before == tuple(tensor.data_ptr() for *_, tensor in part.common.slots())
    assert root.unrelated.dtype == root.state.dtype == torch.float32
    assert not events


def test_measurements_never_cross_authored_modes_or_dtypes() -> None:
    from dataclasses import replace

    owner, _, _ = planner(1024)
    root = torch.nn.Module()
    root.use_tiling = True
    root.tile_sample_min_size = 256
    root.tile_overlap_factor = 0.25
    held = owner.adopt("c", "vae", root, None, tiers.HOST)
    mode = memory._mode(root)
    key = memory.OpKey("vae", "decode", (((1, 4, 128, 128), "float32"),), mode=mode)
    owner.records[key] = memory.Record(ok=100)
    assert owner._estimate(key) == (100, True)
    assert owner._estimate(replace(key, mode=(("use_tiling", False),))) == (None, False)
    assert owner._estimate(replace(key, tensors=(((1, 4, 128, 128), "float16"),))) == (None, False)
    owner._plan(held, key, True, False)
    owner._plan(held, key, True, True)
    assert memory._mode(root) == mode


def test_only_device_allocation_failures_are_classified_as_oom() -> None:
    unrelated = type("OutOfMemoryError", (RuntimeError,), {})
    assert not is_device_oom(unrelated("host store is full"))
    assert not is_device_oom(MemoryError("host memory exhausted"))
    assert not is_device_oom(torch.OutOfMemoryError("DefaultCPUAllocator: can't allocate memory"))
    assert is_device_oom(torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 MiB"))
    assert is_device_oom(RuntimeError("CUDA error: out of memory"))
    assert not is_device_oom(RuntimeError("CUDA error: illegal memory access"))


def test_alias_backing_cannot_be_discarded_for_slot_only_store_refill() -> None:
    owner, broker, _ = planner(0)
    root = torch.nn.Module()
    backing = torch.arange(4)
    root._buffers["first"] = DeviceSource(backing, broker)
    root._buffers["alias"] = DeviceSource(backing[1:], broker)
    part = tiers.partition("aliases", root, tiers.DEVICE)

    class Store:
        def drop_unit(self, unit: tiers.Unit) -> None:
            raise AssertionError("alias geometry must be refused before any store mutation")

    part.store = Store()  # type: ignore[assignment]
    part.unit_loads = True
    with pytest.raises(memory.MemoryRefusal) as refused:
        owner._home(part, part.common)
    assert refused.value.code == "host_shortfall"
    assert part.common.tier == tiers.DEVICE and not broker.host_reserved
    with pytest.raises(RuntimeFailure, match="aliased backing"):
        tiers.to_disk(torch, part, part.common)


@pytest.mark.parametrize("whole", [False, True])
def test_failed_post_fill_conversion_restores_disk_placement_and_conversion_queue(
    whole: bool,
) -> None:
    root = torch.nn.Module()
    root.register_buffer("first", torch.empty(4, device="meta"))
    root.register_buffer("second", torch.empty(4, device="meta"))
    part = tiers.partition("weights", root, tiers.DISK)
    allocated: list[weakref.ReferenceType[Any]] = []
    dropped: list[str] = []
    conversions = 0

    def convert(tensor: Any) -> Any:
        nonlocal conversions
        conversions += 1
        if conversions == 2:
            raise ValueError("conversion failed after first slot")
        return tensor.double()

    class Store:
        def load(self, name: str) -> None:
            assert name == "weights"
            self.load_unit(name, part.common)

        def load_unit(self, name: str, unit: tiers.Unit) -> None:
            for module, name, _, tensor in unit.slots():
                value = torch.ones(tensor.shape, dtype=tensor.dtype)
                allocated.append(weakref.ref(value))
                module._buffers[name] = value

        def drop(self, name: str) -> None:
            assert whole and name == "weights"
            dropped.append("whole")
            self._drop(part.common)

        def drop_unit(self, unit: tiers.Unit) -> None:
            assert not whole
            dropped.append("unit")
            self._drop(unit)

        def _drop(self, unit: tiers.Unit) -> None:
            for module, name, _, tensor in unit.slots():
                module._buffers[name] = torch.empty(
                    tensor.shape, dtype=torch.float32, device="meta"
                )

    part.store = Store()  # type: ignore[assignment]
    part.unit_loads = not whole
    part.common.pending = [convert]
    with pytest.raises(ValueError, match="conversion failed") as failed:
        tiers.from_disk(torch, part, None if whole else part.common)
    assert failed.value.__traceback__ is not None
    assert all(value() is None for value in allocated)
    assert all(
        tensor.is_meta and tensor.dtype == torch.float32 for *_, tensor in part.common.slots()
    )
    assert part.common.tier == tiers.DISK and part.common.pending == [convert]
    assert part.common.applied == [] and not part.moving
    assert dropped == ["whole" if whole else "unit"]
    # A second preparation starts from the same checkpoint and applies the queue once.
    tiers.from_disk(torch, part, None if whole else part.common)
    assert part.common.tier == tiers.DEVICE and part.common.pending == []
    assert all(tensor.dtype == torch.float64 for *_, tensor in part.common.slots())


def test_failed_placement_rollback_poison_survives_next_attempt() -> None:
    owner, _, _ = planner(0)
    with pytest.raises(RuntimeFailure), owner._moving():
        raise RuntimeFailure("placement rollback failed", code="device_fault")
    with pytest.raises(memory.MemoryRefusal) as refused:
        owner.begin(None)
    assert refused.value.code == "device_fault" and not refused.value.reusable


def test_device_oom_classifier_is_shared_and_excludes_cpu() -> None:
    from cozy_runtime.author._errors import classify, is_device_oom

    assert vars(memory)["is_oom"] is is_device_oom
    for exc in (
        torch.OutOfMemoryError("DefaultCPUAllocator: can't allocate memory"),
        MemoryError("host memory exhausted"),
        RuntimeError("CUDA error: illegal memory access"),
    ):
        assert not is_device_oom(exc)
        assert classify(exc).code != "device_out_of_memory"
    for exc in (
        torch.OutOfMemoryError("CUDA out of memory"),
        RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"),
        RuntimeError("CUDNN_STATUS_ALLOC_FAILED"),
    ):
        assert is_device_oom(exc)
        assert classify(exc).code == "device_out_of_memory"
