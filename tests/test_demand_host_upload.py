"""Exercise the real Planner demand path with CPU storage and delayed CUDA commands."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from cozy_runtime.author._errors import RuntimeFailure
from cozy_runtime.internal import memory, tiers
from test_host_transfer import Lease, Torch, unit
from test_host_transfer import device as fake_cuda  # noqa: F401
from test_staging_borrow import backend_for

torch = pytest.importorskip("torch")


@pytest.fixture
def demand(fake_cuda: Torch, monkeypatch: pytest.MonkeyPatch) -> Any:  # noqa: F811
    device = fake_cuda
    owner = memory.Planner(device, torch.device("cpu"))
    held = owner.adopt("construction", "model", unit().module, None, tiers.HOST)
    owner.cuda, owner.device = True, torch.device("cuda:0")
    state: dict[str, Any] = {"leases": [], "deny": False, "fallback": []}

    class Store:
        def borrow_staging(self, size: int) -> Lease | None:
            if state["deny"]:
                return None
            buffer = torch.empty(size, dtype=torch.uint8)
            device.cuda.pinned.add(int(buffer.untyped_storage()._cdata))
            lease = Lease((buffer,), size, device.cuda)
            state["leases"].append(lease)
            return lease

    held.part.store = Store()  # type: ignore[assignment]
    monkeypatch.setattr(owner, "_allocated", lambda: 0)

    def ordinary(torch: Any, item: tiers.Unit, target: Any) -> int:
        device.cuda.trace.append("ordinary-copy")
        state["fallback"].append((item, target))
        item.tier = tiers.DEVICE
        return item.nbytes

    monkeypatch.setattr(tiers, "to_device", ordinary)
    yield owner, held, held.part.common, state, device
    assert not torch.cuda.is_initialized()


def load(demand: Any) -> None:
    owner, held, item, _, _ = demand
    with torch.no_grad():
        owner._load(held.part, item)


def test_demand_preserves_values_aliases_parameter_and_host_cache(demand: Any) -> None:
    owner, _, item, state, device = demand
    root = item.module
    identity = id(root.weight)
    storage = root.weight.untyped_storage().data_ptr()
    values = {name: tensor.clone() for name, tensor in root.state_dict().items()}
    load(demand)
    assert item.tier == tiers.DEVICE and id(root.weight) == identity
    assert all(torch.equal(root.state_dict()[name], value) for name, value in values.items())
    assert root.weight.stride() == (1, 4) and root.weight.storage_offset() == 1
    assert root.weight.untyped_storage().data_ptr() == root.mixed.untyped_storage().data_ptr()
    assert item.host[(id(root), "weight")].untyped_storage().data_ptr() == storage
    assert owner.loads == 1 and owner.moved == item.nbytes == 66
    assert owner.demand_upload["uploaded_bytes"] == 66
    assert not state["fallback"] and not owner._host_transfers
    assert state["leases"][0].releases == 1
    assert device.cuda.trace.index("event-wait") < device.cuda.trace.index("release-staging")
    assert "device-drain" in device.cuda.trace
    load(demand)  # Already resident: no repeated transfer or traffic accounting.
    assert owner.loads == 1 and owner.moved == 66


@pytest.mark.parametrize(
    "exclusion", ["ring", "group", "phase", "compiled", "mutable", "cast", "grad", "shard"]
)
def test_excluded_or_insufficient_staging_keeps_ordinary_path(demand: Any, exclusion: str) -> None:
    owner, held, item, state, device = demand
    if exclusion == "ring":
        state["deny"] = True
    elif exclusion == "group":
        owner.grouped = True
    elif exclusion == "phase":
        owner.group_phase = "decode"
    elif exclusion == "compiled":
        held.compiled = True
    elif exclusion == "mutable":
        item.mutable = True
    elif exclusion == "cast":
        item.pending.append(lambda tensor: tensor)
    elif exclusion == "shard":
        owner.sharded = lambda key: True
    with torch.set_grad_enabled(exclusion == "grad"):
        owner._load(held.part, item)
    assert state["fallback"] == [(item, owner.device)]
    assert owner.loads == 1 and owner.moved == 66
    assert owner.demand_upload["fallback_bytes"] == 66
    assert not device.cuda.events and not owner._host_transfers


def test_delayed_copy_cannot_return_to_authored_consumer(
    demand: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, _, item, state, device = demand
    device.cuda.auto = False
    entered = threading.Event()
    done = threading.Event()
    available = threading.Event()
    original = device.cuda.Event

    def create_event() -> Any:
        event = original()
        available.set()
        return event

    monkeypatch.setattr(device.cuda, "Event", create_event)

    def invoke() -> None:
        entered.set()
        load(demand)
        done.set()  # The first point an authored consumer could run.

    thread = threading.Thread(target=invoke)
    thread.start()
    assert entered.wait(5)
    try:
        # Wait for the prepared object through its test event, not by timing GPU work.
        assert available.wait(5)
        event = device.cuda.events[0]
        assert event.waiting.wait(5)
        assert not done.is_set() and item.tier == tiers.HOST
        assert owner._host_transfers and state["leases"][0].releases == 0
    finally:
        device.cuda.events[0].complete()
        thread.join(5)
    assert done.is_set() and not thread.is_alive() and not owner._host_transfers


def test_allocation_oom_propagates_to_normal_recovery(
    demand: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, _, item, state, device = demand
    original = device.empty
    failure = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB")

    def allocate(*args: Any, **kwargs: Any) -> Any:
        if torch.device(kwargs.get("device", "cpu")).type == "cuda" and args[0] != 0:
            raise failure
        return original(*args, **kwargs)

    monkeypatch.setattr(device, "empty", allocate)
    with pytest.raises(torch.OutOfMemoryError) as raised:
        load(demand)
    assert raised.value is failure and item.tier == tiers.HOST
    assert not owner.unusable and not owner._host_transfers and not state["fallback"]
    assert owner.loads == owner.moved == 0 and state["leases"][0].releases == 1


def test_enqueue_failure_drains_before_releasing_or_reusing_host(demand: Any) -> None:
    owner, _, item, state, device = demand
    device.cuda.enqueue_failure = 2
    with pytest.raises(RuntimeError, match="copy enqueue failed"):
        load(demand)
    assert item.tier == tiers.HOST and not owner._host_transfers and not state["fallback"]
    assert state["leases"][0].releases == 1
    assert device.cuda.trace.index("stream-drain") < device.cuda.trace.index("release-staging")


def test_published_drain_fault_keeps_bindings_and_custody_until_reap(demand: Any) -> None:
    owner, _, item, state, device = demand
    device.cuda.fail_device = True
    with pytest.raises(RuntimeFailure) as raised:
        load(demand)
    assert raised.value.code == "device_fault" and owner.unusable
    assert item.tier == tiers.DEVICE and len(owner._host_transfers) == 1
    transfer = next(iter(owner._host_transfers))
    assert transfer.state == "poisoned" and transfer.destinations and transfer.originals
    assert not state["fallback"] and owner.loads == 0
    with pytest.raises(memory.MemoryRefusal, match="HOST transfer"):
        owner.begin(None)


def test_copy_fault_keeps_host_and_dma_custody(demand: Any) -> None:
    owner, _, item, state, device = demand
    device.cuda.fail_event = True
    with pytest.raises(RuntimeFailure) as raised:
        load(demand)
    assert raised.value.code == "device_fault" and owner.unusable
    assert item.tier == tiers.HOST and len(owner._host_transfers) == 1
    assert state["leases"][0].releases == 0 and not state["fallback"]


@pytest.mark.parametrize("sizes", [(40, 40), (40,)])
def test_real_fill_provider_reuses_only_existing_ring(
    demand: Any, monkeypatch: pytest.MonkeyPatch, sizes: tuple[int, ...]
) -> None:
    owner, held, item, state, device = demand
    backend, owners = backend_for(device, sizes=sizes)
    held.part.store = backend
    ring = backend._staging_ring
    identities = tuple(buffer.untyped_storage().data_ptr() for buffer in ring.buffers)
    original = device.empty

    def allocate(*args: Any, **kwargs: Any) -> Any:
        # Only zero-byte CPU views may be constructed; staging must already exist.
        if torch.device(kwargs.get("device", "cpu")).type == "cpu":
            assert args[0] == 0
        return original(*args, **kwargs)

    monkeypatch.setattr(device, "empty", allocate)
    load(demand)
    assert tuple(buffer.untyped_storage().data_ptr() for buffer in ring.buffers) == identities
    assert backend._staging_ring is ring and backend._staging_borrow is None
    assert backend._live_ring is None and not owners and not owner._host_transfers
    assert owner.loads == 1 and owner.moved == item.nbytes
    assert bool(state["fallback"]) == (sum(sizes) < 66)
    assert backend.staging_memory() == {"allocated_bytes": sum(sizes), "pinned_bytes": sum(sizes)}


def test_binding_failure_rolls_back_through_planner(
    demand: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, _, item, state, _ = demand
    originals = {
        name: tensor.untyped_storage().data_ptr()
        for name, tensor in item.module.state_dict().items()
    }
    bind = tiers._bind
    calls = 0

    def fail_once(*args: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("binding interrupted")
        bind(*args)

    monkeypatch.setattr(tiers, "_bind", fail_once)
    with pytest.raises(RuntimeError, match="binding interrupted"):
        load(demand)
    assert item.tier == tiers.HOST and not item.host and not item.seen
    assert originals == {
        name: tensor.untyped_storage().data_ptr()
        for name, tensor in item.module.state_dict().items()
    }
    assert not owner._host_transfers and not owner.unusable and not state["fallback"]
    assert state["leases"][0].releases == 1 and owner.loads == owner.moved == 0


def test_invalidated_source_drains_then_falls_back(
    demand: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, _, item, state, device = demand
    publish = tiers.HostTransfer.publish

    def invalidate(transfer: tiers.HostTransfer) -> bool:
        item.readonly = False
        return publish(transfer)

    monkeypatch.setattr(tiers.HostTransfer, "publish", invalidate)
    load(demand)
    assert len(state["fallback"]) == 1 and owner.loads == 1 and owner.moved == 66
    assert not owner._host_transfers and state["leases"][0].releases == 1
    assert device.cuda.trace.index("release-staging") < device.cuda.trace.index("ordinary-copy")
    assert "uploaded_units" not in owner.demand_upload


def test_zero_byte_unit_does_not_join_a_cuda_stream(demand: Any) -> None:
    owner, held, _, state, device = demand
    owner.cuda = False
    empty = owner.adopt("construction", "empty", torch.nn.Module(), None, tiers.HOST)
    empty.part.store = held.part.store
    owner.cuda = True
    with torch.no_grad():
        owner._load(empty.part, empty.part.common)
    assert empty.part.common.nbytes == 0
    assert device.cuda.trace == ["ordinary-copy"] and len(state["fallback"]) == 1
