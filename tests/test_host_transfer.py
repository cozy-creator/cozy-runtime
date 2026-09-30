"""Real CPU storage geometry, simulated CUDA events; never initializes an accelerator."""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any

import pytest

from cozy_runtime.author._errors import RuntimeFailure
from cozy_runtime.internal import tiers

torch = pytest.importorskip("torch")


class Event:
    def __init__(self, cuda: Any) -> None:
        self.cuda = cuda
        self.done = threading.Event()
        self.waiting = threading.Event()
        self.commands: list[Any] = []

    def record(self, stream: Any) -> None:
        self.commands, stream.commands = stream.commands, []
        self.cuda.trace.append("event-record")
        if self.cuda.auto:
            self.complete()

    def complete(self) -> None:
        for command in self.commands:
            command()
        self.commands.clear()
        self.done.set()

    def synchronize(self) -> None:
        self.cuda.trace.append("event-wait")
        self.waiting.set()
        if self.cuda.fail_event:
            raise RuntimeError("copy event failed")
        assert self.done.wait(5), "test did not complete simulated copy"


class Stream:
    def __init__(self, cuda: Any) -> None:
        self.cuda = cuda
        self.device = torch.device("cuda:0")
        self.commands: list[Any] = []

    def synchronize(self) -> None:
        self.cuda.trace.append("stream-drain")
        if self.cuda.fail_stream:
            raise RuntimeError("copy drain failed")
        for command in self.commands:
            command()
        self.commands.clear()
        for event in self.cuda.events:
            event.complete()


class Cuda:
    def __init__(self) -> None:
        self.trace: list[str] = []
        self.events: list[Event] = []
        self.gpu: set[int] = set()
        self.pinned: set[int] = set()
        self.auto = True
        self.fail_event = self.fail_stream = self.fail_device = False
        self.enqueue_failure = 0
        self.enqueues = 0
        self.consumer_waiting = threading.Event()
        self.consumer_done = threading.Event()
        self.consumer_done.set()
        self.stream_value = Stream(self)

    def stream(self, stream: Stream) -> Any:
        assert stream is self.stream_value
        return contextlib.nullcontext()

    def Event(self) -> Event:
        event = Event(self)
        self.events.append(event)
        return event

    def current_stream(self, device: Any) -> Stream:
        return self.stream_value

    def synchronize(self, device: Any) -> None:
        self.trace.append("device-drain")
        self.consumer_waiting.set()
        if self.fail_device:
            raise RuntimeError("consumer drain failed")
        assert self.consumer_done.wait(5), "test did not complete simulated consumer"
        self.stream_value.synchronize()


class Torch:
    def __init__(self, cuda: Cuda) -> None:
        self.cuda = cuda
        self.allocations = 0
        self.fail_allocate = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(torch, name)

    def empty(self, *shape: Any, **kwargs: Any) -> Any:
        device = torch.device(kwargs.get("device", "cpu"))
        if device.type == "cuda":
            self.allocations += 1
            self.cuda.trace.append("allocate-device")
            if self.allocations == self.fail_allocate:
                raise RuntimeError("destination allocation failed")
            kwargs["device"] = "cpu"
        tensor = torch.empty(*shape, **kwargs)
        if device.type == "cuda" and tensor.numel():
            self.cuda.gpu.add(int(tensor.untyped_storage()._cdata))
        return tensor


@dataclass
class Lease:
    buffers: tuple[Any, ...]
    capacity: int
    cuda: Cuda
    releases: int = 0

    def release(self) -> None:
        self.releases += 1
        assert self.releases == 1
        self.cuda.trace.append("release-staging")


@pytest.fixture
def device(monkeypatch: pytest.MonkeyPatch) -> Iterator[Torch]:
    cuda = Cuda()
    original = torch.Tensor.copy_

    def copy(target: Any, source: Any, non_blocking: bool = False) -> Any:
        if int(target.untyped_storage()._cdata) not in cuda.gpu:
            return original(target, source, non_blocking=non_blocking)
        assert non_blocking
        cuda.enqueues += 1
        cuda.trace.append("enqueue-copy")
        if cuda.enqueues == cuda.enqueue_failure:
            raise RuntimeError("copy enqueue failed")
        cuda.stream_value.commands.append(lambda: original(target, source))
        return target

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    monkeypatch.setattr(
        torch.Tensor,
        "is_pinned",
        lambda tensor: int(tensor.untyped_storage()._cdata) in cuda.pinned,
    )
    yield Torch(cuda)
    assert not torch.cuda.is_initialized()


def unit() -> tiers.Unit:
    root = torch.nn.Module()
    backing = torch.arange(16, dtype=torch.float32)
    root.register_parameter(
        "weight", torch.nn.Parameter(backing.view(4, 4).t()[1:3], requires_grad=False)
    )
    root.register_buffer("mixed", backing.view(torch.int32)[::2])
    root.register_buffer("scalar", torch.tensor(2.5, dtype=torch.float16))
    root.register_buffer("empty", torch.empty(0, 3))
    result = tiers.Unit("", root, tier=tiers.HOST)
    result.nbytes = 66
    return result


def prepare(
    device: Torch,
    item: tiers.Unit,
    *,
    sizes: tuple[int, ...] | None = None,
    valid: Any = lambda: True,
    capacity: int = 1000,
    pinned: bool = True,
) -> tuple[tiers.HostTransfer | None, set[tiers.HostTransfer], list[Lease]]:
    owned: set[tiers.HostTransfer] = set()
    leases = []

    def acquire(nbytes: int) -> Lease:
        buffers = tuple(
            torch.empty(size, dtype=torch.uint8, device="cpu") for size in (sizes or (nbytes,))
        )
        if pinned:
            device.cuda.pinned.update(int(b.untyped_storage()._cdata) for b in buffers)
        lease = Lease(buffers, sum(b.numel() for b in buffers), device.cuda)
        leases.append(lease)
        return lease

    with torch.no_grad():
        result = tiers.HostTransfer.prepare(
            device,
            item,
            torch.device("cuda:0"),
            device.cuda.stream_value,
            acquire=acquire,
            device_capacity=capacity,
            valid=valid,
            owned=owned,
        )
    return result, owned, leases


def test_real_values_independent_backings_and_alias_geometry(device: Torch) -> None:
    item = unit()
    root = item.module
    original_id = id(root.weight)
    values = {name: tensor.clone() for name, tensor in root.state_dict().items()}
    transfer, owned, leases = prepare(device, item, sizes=(7, 13, 46))
    assert transfer is not None and transfer.source_bytes == 66
    assert item.tier == tiers.HOST and leases[0].releases == 0
    assert transfer.publish()
    assert id(root.weight) == original_id
    assert root.weight.stride() == (1, 4) and root.weight.storage_offset() == 1
    assert root.mixed.stride() == (2,)
    assert root.weight.untyped_storage().data_ptr() == root.mixed.untyped_storage().data_ptr()
    assert root.scalar.untyped_storage().data_ptr() != root.weight.untyped_storage().data_ptr()
    assert root.scalar.shape == () and root.empty.shape == (0, 3)
    assert all(torch.equal(root.state_dict()[name], value) for name, value in values.items())
    assert len(transfer.destinations) == 3  # shared64B, independent2B, empty backing
    assert leases[0].releases == 1 and transfer in owned
    transfer.finish()
    assert not owned and transfer.state == "retired"
    transfer.finish()
    assert leases[0].releases == 1


def test_delayed_copy_cannot_publish_or_return_pinned_slots(device: Torch) -> None:
    device.cuda.auto = False
    item = unit()
    transfer, owned, leases = prepare(device, item)
    assert transfer is not None
    published = []
    thread = threading.Thread(target=lambda: published.append(transfer.publish()))
    thread.start()
    try:
        assert transfer.copy_ready.waiting.wait(5)
        assert not published and item.tier == tiers.HOST
        assert transfer in owned and leases[0].releases == 0
    finally:
        transfer.copy_ready.complete()
        thread.join(5)
    assert published == [True] and not thread.is_alive()
    transfer.finish()


def test_consumer_must_drain_device_before_retirement(device: Torch) -> None:
    transfer, owned, leases = prepare(device, unit())
    assert transfer is not None and transfer.publish()
    device.cuda.consumer_done.clear()
    thread = threading.Thread(target=transfer.finish)
    thread.start()
    try:
        assert device.cuda.consumer_waiting.wait(5)
        assert transfer in owned and transfer.destinations
        assert leases[0].releases == 1  # copy done; consumer still owns destinations
    finally:
        device.cuda.consumer_done.set()
        thread.join(5)
    assert not thread.is_alive() and not owned
    assert "device-drain" in device.cuda.trace


@pytest.mark.parametrize("change", ["generation", "mutable", "readonly", "inplace", "new-slot"])
def test_stale_source_or_generation_is_not_published(device: Torch, change: str) -> None:
    current = [True]
    item = unit()
    transfer, owned, leases = prepare(device, item, valid=lambda: current[0])
    assert transfer is not None
    if change == "generation":
        current[0] = False
    elif change == "mutable":
        item.mutable = True
    elif change == "readonly":
        item.readonly = False
    elif change == "new-slot":
        item.module.register_buffer("new", torch.ones(1))
    else:
        item.module.scalar.add_(1)
    assert not transfer.publish()
    assert item.tier == tiers.HOST and not owned and leases[0].releases == 1


@pytest.mark.parametrize(
    "capacity,pinned,sizes", [(65, True, None), (1000, False, None), (1000, True, (65,))]
)
def test_missing_exact_capacity_or_pin_declines_before_cuda(
    device: Torch, capacity: int, pinned: bool, sizes: tuple[int, ...] | None
) -> None:
    transfer, owned, leases = prepare(device, unit(), capacity=capacity, pinned=pinned, sizes=sizes)
    assert transfer is None and not owned and device.allocations == 0
    assert all(lease.releases == 1 for lease in leases)


def test_zero_bytes_needs_no_host_slab(device: Torch) -> None:
    root = torch.nn.Module()
    root.register_buffer("empty", torch.empty(0))
    transfer, owned, leases = prepare(device, tiers.Unit("", root, tier=tiers.HOST), capacity=0)
    assert transfer is not None and not leases and transfer.source_bytes == 0
    assert transfer.publish()
    transfer.finish()
    assert not owned


def test_failed_atomic_bind_restores_originals(
    device: Torch, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = unit()
    before = {name: tensor.clone() for name, tensor in item.module.state_dict().items()}
    transfer, owned, leases = prepare(device, item)
    assert transfer is not None
    original = tiers._bind
    calls = 0

    def bind(*args: Any) -> None:
        nonlocal calls
        calls += 1
        original(*args)
        if calls == 2:
            raise ValueError("binding failed after replacement")

    monkeypatch.setattr(tiers, "_bind", bind)
    with pytest.raises(ValueError, match="binding failed"):
        transfer.publish()
    assert item.tier == tiers.HOST and not item.host and not item.seen
    assert all(torch.equal(item.module.state_dict()[name], value) for name, value in before.items())
    assert not owned and leases[0].releases == 1


def test_failed_consumer_drain_retains_poisoned_custody(device: Torch) -> None:
    transfer, owned, _ = prepare(device, unit())
    assert transfer is not None and transfer.publish()
    device.cuda.fail_device = True
    with pytest.raises(RuntimeFailure) as failure:
        transfer.finish()
    assert failure.value.code == "device_fault"
    assert transfer.state == "poisoned" and transfer in owned and transfer.destinations
    count = device.cuda.trace.count("device-drain")
    with pytest.raises(RuntimeFailure):
        transfer.abort()
    assert device.cuda.trace.count("device-drain") == count


@pytest.mark.parametrize("poison", [False, True])
def test_enqueue_failure_drains_or_retains_every_owned_resource(
    device: Torch, poison: bool
) -> None:
    item = unit()
    owned: set[tiers.HostTransfer] = set()
    leases = []

    def acquire(nbytes: int) -> Lease:
        buffer = torch.empty(nbytes, dtype=torch.uint8)
        device.cuda.pinned.add(int(buffer.untyped_storage()._cdata))
        lease = Lease((buffer,), nbytes, device.cuda)
        leases.append(lease)
        return lease

    device.cuda.enqueue_failure = 2
    device.cuda.fail_stream = poison
    with torch.no_grad(), pytest.raises(RuntimeError if not poison else RuntimeFailure):
        tiers.HostTransfer.prepare(
            device,
            item,
            torch.device("cuda:0"),
            device.cuda.stream_value,
            acquire=acquire,
            device_capacity=1000,
            valid=lambda: True,
            owned=owned,
        )
    assert item.tier == tiers.HOST and not item.host
    if poison:
        [transfer] = owned
        assert transfer.state == "poisoned" and transfer.staging is leases[0]
        assert transfer.destinations and transfer.originals and leases[0].releases == 0
    else:
        assert not owned and leases[0].releases == 1
        assert device.cuda.trace.index("stream-drain") < device.cuda.trace.index("release-staging")


def test_inference_source_and_buffer_gradient_metadata_survive(device: Torch) -> None:
    with torch.inference_mode():
        item = unit()
    transfer, owned, _ = prepare(device, item)
    assert transfer is not None and transfer.publish()
    transfer.finish()
    assert not owned
    root = torch.nn.Module()
    root.register_buffer("value", torch.ones(2, requires_grad=True))
    transfer, _, _ = prepare(device, tiers.Unit("", root, tier=tiers.HOST))
    assert transfer is not None and transfer.publish()
    assert root.value.requires_grad
    assert transfer.unit.host[(id(root), "value")].requires_grad
    transfer.finish()


def test_destination_failure_preserves_host_and_releases_after_drain(device: Torch) -> None:
    device.fail_allocate = 2
    item = unit()
    owned: set[tiers.HostTransfer] = set()
    lease = Lease((torch.empty(66, dtype=torch.uint8),), 66, device.cuda)
    device.cuda.pinned.add(int(lease.buffers[0].untyped_storage()._cdata))
    with torch.no_grad(), pytest.raises(RuntimeError, match="destination allocation"):
        tiers.HostTransfer.prepare(
            device,
            item,
            torch.device("cuda:0"),
            device.cuda.stream_value,
            acquire=lambda _: lease,
            device_capacity=66,
            valid=lambda: True,
            owned=owned,
        )
    assert item.tier == tiers.HOST and not owned and lease.releases == 1
    assert device.cuda.trace.index("stream-drain") < device.cuda.trace.index("release-staging")


def test_copy_join_failure_does_not_return_the_staging_lease(device: Torch) -> None:
    transfer, owned, leases = prepare(device, unit())
    assert transfer is not None
    device.cuda.fail_event = True
    with pytest.raises(RuntimeFailure) as failed:
        transfer.publish()
    assert failed.value.code == "device_fault"
    assert transfer in owned and transfer.state == "poisoned"
    assert transfer.staging is leases[0] and leases[0].releases == 0


def test_mutable_or_gradient_operation_declines_without_calling_provider(device: Torch) -> None:
    item = unit()
    item.mutable = True
    transfer, owned, leases = prepare(device, item)
    assert transfer is None and not owned and not leases
    item.mutable = False
    with torch.enable_grad():
        transfer = tiers.HostTransfer.prepare(
            device,
            item,
            torch.device("cuda:0"),
            device.cuda.stream_value,
            acquire=lambda _: pytest.fail("no staging for gradient operation"),
            device_capacity=66,
            valid=lambda: True,
            owned=owned,
        )
    assert transfer is None and not owned and device.allocations == 0


def test_wrong_copy_stream_device_declines_before_borrow(device: Torch) -> None:
    device.cuda.stream_value.device = torch.device("cuda:1")
    transfer, owned, leases = prepare(device, unit())
    assert transfer is None and not owned and not leases and not device.cuda.trace


def test_overlapping_staging_slots_cannot_corrupt_sources(device: Torch) -> None:
    buffer = torch.empty(66, dtype=torch.uint8)
    device.cuda.pinned.add(int(buffer.untyped_storage()._cdata))
    lease = Lease((buffer, buffer), 132, device.cuda)
    owned: set[tiers.HostTransfer] = set()
    with torch.no_grad():
        transfer = tiers.HostTransfer.prepare(
            device,
            unit(),
            torch.device("cuda:0"),
            device.cuda.stream_value,
            acquire=lambda _: lease,
            device_capacity=66,
            valid=lambda: True,
            owned=owned,
        )
    assert transfer is None and not owned and lease.releases == 1 and device.allocations == 0


@pytest.mark.parametrize("kind", ["inference", "data"])
def test_immutable_authority_must_revoke_even_when_versions_cannot_observe_write(
    device: Torch, kind: str
) -> None:
    with torch.inference_mode(kind == "inference"):
        item = unit()
    version = tiers._transfer_version(item.module.scalar)
    immutable = [True]
    transfer, owned, leases = prepare(device, item, valid=lambda: immutable[0])
    assert transfer is not None
    # The provider revokes custody BEFORE allowing a writer. A version check alone
    # cannot prove immutability for either of these supported Torch behaviors.
    immutable[0] = False
    with torch.inference_mode():
        item.module.scalar.data.add_(1)
    assert tiers._transfer_version(item.module.scalar) == version
    assert not transfer.publish()
    assert item.module.scalar.item() == 3.5 and item.tier == tiers.HOST
    assert not owned and leases[0].releases == 1
