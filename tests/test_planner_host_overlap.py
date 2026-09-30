"""Planner hooks with real CPU tensors and simulated CUDA; no accelerator opens."""

from __future__ import annotations

from typing import Any

import pytest

from cozy_runtime.author import pure
from cozy_runtime.author._errors import RuntimeFailure
from cozy_runtime.internal import memory, tiers
from test_host_transfer import Lease, Torch, device as fake_cuda  # CPU-only delayed-event fixture

torch = pytest.importorskip("torch")


@pytest.fixture
def policy(fake_cuda: Torch, monkeypatch: pytest.MonkeyPatch) -> Any:  # noqa: F811
    device = fake_cuda
    owner = memory.Planner(device, torch.device("cpu"))
    root = pure(torch.nn.Module())
    root.anchor = torch.nn.Parameter(torch.empty(0), requires_grad=False)
    root._no_split_modules = ["Linear"]
    root.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False) for _ in range(3)])
    for tensor in root.parameters():
        tensor.requires_grad_(False)
    held = owner.adopt("construction", "model", root, None, tiers.HOST, forward_only=True)
    owner.cuda, owner.device = True, torch.device("cuda:0")
    key = memory.OpKey(held.kind, "forward", ())
    owner.records[key] = memory.Record(ok=32)
    state: dict[str, Any] = {"room": 1000, "allocated": 0, "peak": 0, "leases": []}

    class Store:
        def borrow_staging(self, size: int) -> Lease | None:
            if state.get("deny"):
                return None
            buffer = torch.empty(size, dtype=torch.uint8)
            device.cuda.pinned.add(int(buffer.untyped_storage()._cdata))
            lease = Lease((buffer,), size, device.cuda)
            state["leases"].append(lease)
            return lease

    held.part.store = Store()  # type: ignore[assignment]
    held.overlap_key, held.overlap_order = key, held.part.blocks
    held.overlap_contract = owner._overlap_contract(held)
    monkeypatch.setattr(owner, "allocatable", lambda: state["room"])
    monkeypatch.setattr(owner, "_allocated", lambda: state["allocated"])
    monkeypatch.setattr(owner, "_cap", lambda: None)
    monkeypatch.setattr(device.cuda, "Stream", lambda **_: device.cuda.stream_value, raising=False)
    monkeypatch.setattr(device.cuda, "max_memory_allocated", lambda _: state["peak"], raising=False)
    monkeypatch.setattr(device.cuda, "max_memory_reserved", lambda _: state["peak"], raising=False)
    monkeypatch.setattr(device.cuda, "reset_peak_memory_stats", lambda _: None, raising=False)

    def load(part: tiers.Component, unit: tiers.Unit) -> None:
        owner._overlap_retire(pending_only=True)
        device.cuda.trace.append("ordinary-load:" + unit.path)
        unit.tier = tiers.DEVICE

    def home(part: tiers.Component, unit: tiers.Unit) -> None:
        device.cuda.trace.append("home:" + unit.path)
        unit.tier = tiers.HOST

    monkeypatch.setattr(owner, "_load", load)
    monkeypatch.setattr(owner, "_home", home)
    frame = memory.Frame(held, 0)
    with torch.no_grad():
        owner._overlap_frame(frame, key, True)
    owner.stack.append(frame)
    yield owner, held, frame, key, state, device
    if not owner.unusable:
        owner._overlap_retire()
    assert not torch.cuda.is_initialized()


def test_actual_pre_and_post_hooks_publish_once_and_drain_before_home(policy: Any) -> None:
    owner, held, frame, _, state, device = policy
    tiers.install(held.part, owner._enter_unit, owner._leave_unit)
    value = torch.ones(1, 2)
    with torch.no_grad():
        value = held.part.blocks[0].module(value)
        assert frame.overlap_pending is not None
        assert frame.overlap_pending.unit is held.part.blocks[1]
        value = held.part.blocks[1].module(value)
        value = held.part.blocks[2].module(value)
    assert value.shape == (1, 2)
    assert owner.overlap["hits"] == 2
    assert owner.overlap["cpu_staged_bytes"] == owner.overlap["h2d_submitted_bytes"] == 32
    assert owner.moved == 32 and owner.loads == 2
    assert frame.loaded == 0 and not owner._host_transfers
    assert all(lease.releases == 1 for lease in state["leases"])
    trace = device.cuda.trace
    for index in range(3):
        home = trace.index(f"home:blocks.{index}")
        assert "device-drain" in trace[:home]


@pytest.mark.parametrize(
    "change", ["capacity", "staging", "unknown", "group", "grad", "cast", "mutable", "disk"]
)
def test_conservative_declines_without_optional_cuda(policy: Any, change: str) -> None:
    owner, held, frame, key, state, device = policy
    if change == "capacity":
        state["room"] = 47  # next16 + known workspace32 does not fit
    elif change == "staging":
        state["deny"] = True
    elif change == "unknown":
        owner.records[key].ok = -1
    elif change == "group":
        owner.grouped = True
    elif change == "cast":
        held.part.blocks[1].pending.append(lambda x: x)
    elif change == "mutable":
        held.part.blocks[1].mutable = True
    elif change == "disk":
        held.part.blocks[1].tier = tiers.DISK
    if change in ("unknown", "group", "cast", "mutable"):
        owner.stack.clear()
        frame = memory.Frame(held, 0)
        with torch.no_grad():
            owner._overlap_frame(frame, key, True)
        owner.stack.append(frame)
    with torch.set_grad_enabled(change == "grad"):
        owner._enter_unit(held.part, held.part.blocks[0])
    assert not owner._host_transfers
    assert "allocate-device" not in device.cuda.trace


def test_wrong_hint_drains_before_ordinary_demand(policy: Any) -> None:
    owner, held, frame, _, _, device = policy
    with torch.no_grad():
        owner._enter_unit(held.part, held.part.blocks[0])
        owner._leave_unit(held.part, held.part.blocks[0])
        owner._enter_unit(held.part, held.part.blocks[2])
    assert frame.overlap_pending is None
    assert not frame.overlap_enabled
    assert owner.overlap["wasted_bytes"] == 16
    trace = device.cuda.trace
    assert trace.index("release-staging") < trace.index("ordinary-load:blocks.2")


def test_nested_callee_discards_hint_but_keeps_published_consumer(policy: Any) -> None:
    owner, held, frame, _, _, _ = policy
    with torch.no_grad():
        owner._enter_unit(held.part, held.part.blocks[0])
        owner._leave_unit(held.part, held.part.blocks[0])
        owner._enter_unit(held.part, held.part.blocks[1])
        published = frame.overlap_consumers[held.part.blocks[1]]
        owner._overlap_retire(pending_only=True)
        assert published in owner._host_transfers and published.state == "published"
        assert frame.overlap_pending is None
        owner._leave_unit(held.part, held.part.blocks[1])
        assert not owner._host_transfers


def test_pending_destination_is_not_working_and_publish_counts_once(policy: Any) -> None:
    owner, held, frame, _, state, _ = policy
    with torch.no_grad():
        owner._enter_unit(held.part, held.part.blocks[0])
        state["peak"] = 16 + 16 + 7  # current, pending, activation
        owner._sample()
        assert frame.working == 7
        owner._leave_unit(held.part, held.part.blocks[0])
        state["peak"] = 16 + 7
        owner._enter_unit(held.part, held.part.blocks[1])
        assert frame.loaded == 16
        state["peak"] = 16 + 16 + 9
        owner._sample()
        assert frame.working == 9


def test_failed_completion_poison_retains_custody(policy: Any) -> None:
    owner, held, frame, _, _, device = policy
    with torch.no_grad():
        owner._enter_unit(held.part, held.part.blocks[0])
        device.cuda.fail_stream = True
        with pytest.raises(RuntimeFailure, match="drain failed"):
            owner._overlap_retire(pending_only=True)
    assert owner.unusable and owner._host_transfers
    assert frame.overlap_pending.state == "poisoned"


def test_history_requires_success_and_exact_operation_and_clears_at_scope_exit(policy: Any) -> None:
    owner, held, frame, key, _, _ = policy
    frame.overlap_visits = list(held.part.blocks)
    owner._overlap_learn(frame, key)
    other = memory.OpKey(held.kind, "forward", (), flags=(True,))
    owner.stack.clear()
    new = memory.Frame(held, 0)
    with torch.no_grad():
        owner._overlap_frame(new, other, True)
    assert not new.overlap_enabled
    owner.exit()
    assert held.overlap_key is None and not held.overlap_order


def test_real_non_oom_failure_is_not_silently_declined(policy: Any) -> None:
    owner, held, _, _, _, device = policy
    device.fail_allocate = 1
    with torch.no_grad(), pytest.raises(RuntimeError, match="destination allocation failed"):
        owner._enter_unit(held.part, held.part.blocks[0])
    assert not owner._host_transfers and not owner.unusable
    assert owner.overlap["cpu_staged_bytes"] == 16
    assert owner.overlap["h2d_submitted_bytes"] == 0


def test_real_operation_boundary_learns_only_success_then_uses_hooks(
    policy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner: memory.Planner
    owner, held, _, key, _, _ = policy
    owner.stack.clear()
    held.overlap_key, held.overlap_order = None, ()
    tiers.install(held.part, owner._enter_unit, owner._leave_unit)
    monkeypatch.setattr(owner, "_plan", lambda *_: memory.Plan(0, []))
    monkeypatch.setattr(owner, "_apply", lambda *_: None)
    arm, rollback = owner._arm, owner._rollback

    def cpu_arm(target: memory.Held, leaves: list[object]) -> memory.Snapshot:
        owner.cuda = False
        try:
            return arm(target, leaves)
        finally:
            owner.cuda = True

    def cpu_rollback(target: memory.Held, snapshot: memory.Snapshot | None) -> None:
        owner.cuda = False
        try:
            rollback(target, snapshot)
        finally:
            owner.cuda = True

    monkeypatch.setattr(owner, "_arm", cpu_arm)
    monkeypatch.setattr(owner, "_rollback", cpu_rollback)

    def body() -> Any:
        value = torch.ones(1, 2)
        for unit in held.part.blocks:
            value = unit.module(value)
        return value

    with torch.no_grad():
        first = owner.run(held, "forward", body, (), {})
        assert held.overlap_key == key
        assert held.overlap_order == held.part.blocks
        assert not owner.overlap
        second = owner.run(held, "forward", body, (), {})
        assert torch.equal(first, second)
        assert owner.overlap["hits"] == 2

        def broken() -> None:
            body()
            raise ValueError("authored failure")

        with pytest.raises(ValueError, match="authored failure"):
            owner.run(held, "forward", broken, (), {})
        assert held.overlap_key is None and not held.overlap_order
        assert not owner.stack and not owner._host_transfers


def test_optional_oom_requires_real_capacity_gain_not_cleaned_cache(
    policy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, held, frame, _, state, device = policy
    calls = 0
    real = tiers.HostTransfer.prepare

    def fail(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        state["room"] = 1100  # automatic drain returned the failed destination
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 16 B")

    monkeypatch.setattr(tiers.HostTransfer, "prepare", fail)
    with torch.no_grad():
        owner._enter_unit(held.part, held.part.blocks[0])
        assert owner._overlap_oom_room == 1116
        owner._leave_unit(held.part, held.part.blocks[0])
        frame.overlap_visits.clear()
        owner._enter_unit(held.part, held.part.blocks[0])
        assert calls == 1  # cleanup gain cannot immediately retry
        state["room"] = 1115
        owner._leave_unit(held.part, held.part.blocks[0])
        frame.overlap_visits.clear()
        owner._enter_unit(held.part, held.part.blocks[0])
        assert calls == 1
        monkeypatch.setattr(tiers.HostTransfer, "prepare", real)
        state["room"] = 1116
        owner._leave_unit(held.part, held.part.blocks[0])
        frame.overlap_visits.clear()
        owner._enter_unit(held.part, held.part.blocks[0])
        assert frame.overlap_pending is not None
        assert "enqueue-copy" in device.cuda.trace


def test_changed_hook_contract_invalidates_previous_order(policy: Any) -> None:
    owner, held, _, key, _, _ = policy
    owner.stack.clear()
    held.part.root.register_forward_hook(lambda *_: None)
    frame = memory.Frame(held, 0)
    with torch.no_grad():
        owner._overlap_frame(frame, key, True)
    assert not frame.overlap_enabled


def test_synchronous_current_drain_fault_quarantines_pending_borrow(policy: Any) -> None:
    owner, held, frame, _, state, device = policy
    with torch.no_grad():
        owner._enter_unit(held.part, held.part.blocks[0])
        device.cuda.fail_device = True
        with pytest.raises(RuntimeFailure, match="consumer drain"):
            owner._leave_unit(held.part, held.part.blocks[0])
    assert owner.unusable and frame.overlap_pending in owner._host_transfers
    assert frame.overlap_pending.state == "poisoned"
    assert state["leases"][0].releases == 0


def test_publication_waits_for_copy_before_releasing_borrow(policy: Any) -> None:
    import threading

    owner, held, frame, _, state, device = policy
    device.cuda.auto = False
    with torch.no_grad():
        owner._enter_unit(held.part, held.part.blocks[0])
    pending = frame.overlap_pending
    failures: list[BaseException] = []

    def demand() -> None:
        try:
            with torch.no_grad():
                owner._enter_unit(held.part, held.part.blocks[1])
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=demand)
    thread.start()
    assert pending.copy_ready.waiting.wait(2)
    assert pending.state == "prepared" and state["leases"][0].releases == 0
    assert held.part.blocks[1].tier == tiers.HOST
    device.cuda.auto = True
    pending.copy_ready.complete()
    thread.join(2)
    assert not thread.is_alive() and not failures
    assert pending.state == "published" and state["leases"][0].releases == 1
    with torch.no_grad():
        owner._leave_unit(held.part, held.part.blocks[1])


def test_scope_close_keeps_outer_published_consumer(policy: Any) -> None:
    owner, held, frame, _, _, _ = policy
    with torch.no_grad():
        owner._enter_unit(held.part, held.part.blocks[0])
        owner._leave_unit(held.part, held.part.blocks[0])
        owner._enter_unit(held.part, held.part.blocks[1])
        published = frame.overlap_consumers[held.part.blocks[1]]
        owner.exit()
        assert published in owner._host_transfers and published.state == "published"
        assert frame.overlap_pending is None and not frame.overlap_learnable
        owner._leave_unit(held.part, held.part.blocks[1])
        assert not owner._host_transfers


def test_unknown_oom_request_cannot_retry_on_one_byte_room_gain(
    policy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, held, frame, _, state, _ = policy
    calls = 0

    def fail(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(tiers.HostTransfer, "prepare", fail)
    with torch.no_grad():
        owner._enter_unit(held.part, held.part.blocks[0])
        assert owner._overlap_oom_unknown
        state["room"] += 1
        owner._leave_unit(held.part, held.part.blocks[0])
        frame.overlap_visits.clear()
        owner._enter_unit(held.part, held.part.blocks[0])
        assert calls == 1


def test_recursive_published_block_keeps_weights_until_outer_posthook(policy: Any) -> None:
    owner, held, frame, _, _, _ = policy
    block = held.part.blocks[1]
    old_forward = block.module.forward
    recursing = False
    observed: list[str] = []

    def forward(value: Any) -> Any:
        nonlocal recursing
        if not recursing:
            recursing = True
            value = block.module(value)
            recursing = False
            observed.append(block.tier)
            assert frame.overlap_consumers[block].state == "published"
        return old_forward(value)

    block.module.forward = forward
    tiers.install(held.part, owner._enter_unit, owner._leave_unit)
    with torch.no_grad():
        held.part.blocks[0].module(torch.ones(1, 2))
        block.module(torch.ones(1, 2))
    assert observed == [tiers.DEVICE]
    assert block.tier == tiers.HOST and frame.loaded == 0
    assert not owner._host_transfers and not frame.overlap_learnable


def test_nested_whole_scope_retires_shared_backend_borrow_before_fill(
    policy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_staging_borrow import backend_for

    owner, held, frame, _, _, device = policy
    backend, custody = backend_for(device, sizes=(16,))
    held.part.store = backend
    root = pure(torch.nn.Module())
    root.weight = torch.nn.Parameter(torch.empty(2, device="meta"), requires_grad=False)
    root._cozy_read_whole = True
    owner.cuda = False
    sibling = owner.adopt("construction", "whole", root, None, tiers.DISK)
    owner.cuda = True
    sibling.part.store = held.part.store
    sibling.part.store.transient = lambda _: 0
    fills: list[str] = []

    def refill(torch_arg: Any, part: tiers.Component, unit: tiers.Unit | None) -> None:
        assert part is sibling.part and unit is None
        assert frame.overlap_pending is None
        assert backend._staging_borrow is None and not custody
        with backend._staging_use():
            assert backend._staging_busy
            fills.append(part.name)
            part.common.tier = tiers.DEVICE

    monkeypatch.setattr(tiers, "from_disk", refill)
    with torch.no_grad():
        owner._enter_unit(held.part, held.part.blocks[0])
        owner._leave_unit(held.part, held.part.blocks[0])
        assert frame.overlap_pending is not None
        owner.enter("construction", "nested", ("whole",))
    assert fills == ["whole"] and not owner.unusable
    assert not frame.overlap_enabled and not frame.overlap_learnable
