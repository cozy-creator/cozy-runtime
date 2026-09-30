"""Host promises and native ring lifetimes; CUDA is simulated, never initialized."""

from __future__ import annotations

import contextlib
import threading
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from cozy_runtime.author._errors import RuntimeFailure
from cozy_runtime.internal import fill, memory
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.parallel.group import RankGroup
from cozy_runtime.internal.parallel.plan import GroupPlan, GroupRefusal


torch = pytest.importorskip("torch")


class Stream:
    def __init__(self) -> None:
        self.syncs = 0

    def wait_stream(self, other: Any) -> None:
        pass

    def synchronize(self) -> None:
        self.syncs += 1


def fake_torch(fail_at: int = 0) -> Any:
    calls = []
    stream = Stream()

    def empty(size: int, **kwargs: Any) -> Any:
        calls.append(size)
        if len(calls) == fail_at:
            raise RuntimeError("unrelated CUDA driver fault")
        return torch.empty(size, dtype=torch.uint8, device="cpu")

    return SimpleNamespace(
        empty=empty,
        uint8=torch.uint8,
        calls=calls,
        cuda=SimpleNamespace(Stream=lambda **_: stream, current_stream=lambda _: stream),
    )


def policy(limit: int, grouped: bool = False) -> tuple[memory.Planner, memory.Machine]:
    machine = memory.Machine({}, lambda _: None, lambda _: None, host_available=lambda: limit)

    def reserve(token: str, size: int, release: bool) -> bool:
        if release:
            machine.release_host(token)
            return True
        return machine.reserve_host(token, size)

    owner = memory.Planner(torch, torch.device("cpu"), grouped=grouped, host_room=reserve)
    return owner, machine


def test_smaller_admitted_ring_never_expands_and_releases_after_drain() -> None:
    owner, machine = policy(16)
    fake = fake_torch()
    with owner.staging_host(16, 4) as slots:
        assert slots == 1 and sum(machine.host_reserved.values()) == 16
        ring = fill.StagingRing(fake, "cpu", window_bytes=16, slots=slots, inflight=8)
        assert len(ring.buffers) == 1 and ring.keep == 0
        assert ring.allocated_bytes == 16 and ring.registered_bytes == 0
        event = SimpleNamespace(query=lambda: False, synchronize=lambda: released.append("DMA"))
        released: list[str] = []
        batch = SimpleNamespace(release=lambda: released.append("slot"))
        ring.hold(event, batch)
        assert not released
        ring.drain(ring.keep)
        assert released == ["DMA", "slot"] and not ring.inflight
        assert machine.host_reserved
        ring.close()
    assert not machine.host_reserved and owner._host_pending == 0


@pytest.mark.parametrize("failure", ["constructor", "body", "drain"])
def test_exception_releases_only_proven_drained_reservation(failure: str) -> None:
    owner, machine = policy(32)
    with pytest.raises((RuntimeError, RuntimeFailure)):
        with owner.staging_host(16, 2) as slots:
            if failure == "constructor":
                fake = fake_torch(fail_at=2)
                fill.StagingRing(fake, "cpu", window_bytes=16, slots=slots, inflight=1)
            elif failure == "body":
                raise RuntimeError("native read refused before DMA")
            else:
                raise RuntimeFailure("staging drain failed", code="device_fault")
    assert bool(machine.host_reserved) == (failure == "drain")
    assert bool(owner.unusable) == (failure == "drain")
    assert owner._host_pending == (32 if failure == "drain" else 0)


def test_group_credit_is_local_transient_and_does_not_reuse_other_host_promises() -> None:
    owner, machine = policy(100, grouped=True)
    with pytest.raises(memory.MemoryRefusal):
        with owner.staging_host(16, 4):
            pytest.fail("unadmitted allocation")
    assert owner.host_needed() == 16
    with owner.host_credit(48):
        with owner._host_copy(16) as ok:
            assert ok
        with owner.staging_host(16, 4) as slots:
            assert slots == 1 and owner._group_host_credit == 16
        assert owner._group_host_credit == 16
        with owner.staging_host(16, 4, owner="other") as slots:
            assert slots == 1
        assert owner._group_host_credit == 0
    assert not machine.host_reserved


def test_staging_vector_preserves_old_peer_full_geometry() -> None:
    sizes = memory.Planner.staging_vector(((16, 4, 1), (16, 4, 4)), lambda n: n <= 80)
    assert sizes == (16, 64)
    assert not memory.Planner.staging_vector(((16, 4, 1), (16, 4, 4)), lambda n: n < 80)


PLAN = GroupPlan(2, ("0", "1"), "sha256:" + "a" * 64, 1000).document()
FOLLOWER = """
import json,socket,sys
from cozy_runtime.internal.parallel.group import offer_plan
from cozy_runtime.internal.seam import Channel
fd=int(sys.argv[1]);mode=sys.argv[2];plan=json.loads(sys.argv[3])
channel=Channel(socket.socket(fileno=fd))
if mode=='bad_offer':
    plan['plan_digest']='sha256:'+'b'*64
credit=offer_plan(channel,rank=1,model='base',plan=plan,staging=(16,4))
channel.send({'reply':'filled','ok':mode!='fill_failure','credit':credit})
channel.recv()
"""


@pytest.mark.parametrize("mode", ["ready", "refused", "bad_offer", "fill_failure"])
def test_construction_agreement_admits_before_real_cpu_follower_fill(
    tmp_path: Path, mode: str
) -> None:
    def launch(argv: list[str], fd: int) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [sys.executable, "-c", FOLLOWER, str(fd), mode, json.dumps(PLAN)],
            pass_fds=(fd,),
            env=os.environ.copy(),
        )

    group = RankGroup(degree=2, backend="cpu:gloo", root=str(tmp_path), launch=launch)
    machine = memory.Machine({}, lambda _: None, lambda _: None, host_available=lambda: 32)
    calls = []

    def admit(demands: tuple[tuple[int, int, int], ...]) -> tuple[int, ...]:
        calls.append(demands)
        return memory.Planner.staging_vector(
            demands, lambda size: mode != "refused" and machine.reserve_host("construction", size)
        )

    try:
        group.spawn()
        if mode in ("refused", "bad_offer"):
            with pytest.raises(GroupRefusal):
                group.agree_plan(model="base", plan=PLAN, staging=(16, 4), reserve_staging=admit)
            assert not machine.host_reserved
            assert bool(calls) == (mode == "refused")
        else:
            assert (
                group.agree_plan(model="base", plan=PLAN, staging=(16, 4), reserve_staging=admit)
                == 16
            )
            assert sum(machine.host_reserved.values()) == 32
            [reply] = group.collect("filled")
            assert reply["credit"] == 16 and reply["ok"] == (mode == "ready")
    finally:
        group.close()
        assert all(f.exit_status() is not None for f in group.followers)
        machine.release_host("construction")


@pytest.mark.parametrize("failure", ["leader_abort", "drain_failed", "reap_unproved", "healthy"])
def test_construction_wrapper_releases_after_proven_followers_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import socket
    from cozy_runtime.internal.seam import Channel

    ours, theirs = socket.socketpair()
    executor = Executor(Channel(ours), tmp_path, world=2)
    owner, machine = policy(32, grouped=True)
    executor.planner = owner
    events = []
    follower = SimpleNamespace(exit_status=lambda: None if failure == "reap_unproved" else 0)
    executor.group = cast(
        RankGroup,
        SimpleNamespace(broken=False, followers=[follower], close=lambda: events.append("reap")),
    )

    def body(*args: Any, **kwargs: Any) -> dict[str, Any]:
        executor._construction_staging_tokens.append(("token", 32))
        assert machine.reserve_host("token", 32)
        if failure == "drain_failed":
            owner.unusable = "drain failed"
        if failure == "healthy":
            executor._construction_staging_drained = True
            return {"ok": True}
        raise ValueError("construction failed")

    def durable(request: Any, kind: Any) -> Any:
        assert request.release
        events.append("release")
        machine.release_host(request.reservation)
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(executor, "_prepare_first_body", body)
    monkeypatch.setattr(executor, "_durable", durable)
    try:
        if failure == "healthy":
            assert executor._prepare_first(cast(Any, None)) == {"ok": True}
        else:
            with pytest.raises(ValueError, match="construction failed"):
                executor._prepare_first(cast(Any, None))
        assert bool(machine.host_reserved) == (failure in ("drain_failed", "reap_unproved"))
        if failure == "leader_abort":
            assert events == ["reap", "release"]
        if failure == "healthy":
            assert events == ["release"]
    finally:
        ours.close()
        theirs.close()


def test_native_one_slot_stream_waits_for_delayed_dma_before_releasing(tmp_path: Path) -> None:
    from test_weights_resume_native import _committed

    store, _, _, receipt = _committed(tmp_path)
    checkpoint = fill.Checkpoint(str(store.root), receipt.artifact.manifest.digest)
    rows = checkpoint.rows("model")
    plan = checkpoint.read_plan([("model", row.name) for row in rows], fill.WINDOW_BYTES, ["model"])
    fake = fake_torch()
    owner, machine = policy(fill.slot_bytes(fill.WINDOW_BYTES))
    events = []
    with owner.staging_host(fill.slot_bytes(fill.WINDOW_BYTES), 16) as slots:
        ring = fill.StagingRing(
            fake, "cpu", window_bytes=fill.slot_bytes(fill.WINDOW_BYTES), slots=slots, inflight=8
        )

        def consume(batch: Any) -> None:
            assert machine.host_reserved
            events.append(("read", batch.nbytes))
            ring.hold(
                SimpleNamespace(query=lambda: False, synchronize=lambda: events.append(("dma", 0))),
                batch,
            )
            ring.drain(0)
            assert not ring.inflight

        with checkpoint.acquire() as lease:
            lease.stream(plan, ring.views, consume, readers=1)
        ring.close()
    assert events[0] == ("read", 2048) and events[1] == ("dma", 0)
    assert not machine.host_reserved


@pytest.mark.parametrize("fail_close", [False, True])
def test_real_fill_boundary_holds_credit_through_close(
    monkeypatch: pytest.MonkeyPatch, fail_close: bool
) -> None:
    owner, machine = policy(16)
    backend: Any = object.__new__(fill.StreamingFillBackend)
    backend.enqueued = {"x": object()}
    backend.staging_host = owner.staging_host
    backend.slot_bytes = 16
    backend.slots = 4
    backend.inflight = 1
    backend.fence = True
    backend.torch = SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda _: Stream()))
    backend.device = "cpu"
    backend.kind = "cpu"
    backend.stats = {}
    backend.decoded = {}
    backend.leaf_sites = {}
    backend.forensics = None
    backend.store_read = contextlib.nullcontext
    backend._live_ring = None
    backend._staging_lock = threading.Lock()
    backend._staging_busy = False
    backend._staging_borrow = None
    backend._staging_poisoned = ""
    backend._staging_ring = None
    backend.staging_started = backend.staging_drained = lambda: None
    events = []

    class Ring:
        waits = 0
        registered_bytes = 16
        allocated_bytes = 16
        in_flight_peak = 1
        window_bytes = 16

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.stream = Stream()
            assert kwargs["slots"] == 1 and machine.host_reserved

        def track(self, t: Any) -> None:
            pass

        def finish(self) -> None:
            assert machine.host_reserved or backend._staging_ring is self
            events.append("finish")
            if fail_close:
                raise RuntimeError("DMA drain not proved")

        def close(self) -> None:
            assert machine.host_reserved
            events.append("close")
            if fail_close:
                raise RuntimeError("DMA drain not proved")

    monkeypatch.setattr(fill, "StagingRing", Ring)
    from cozy_runtime.internal import accel

    monkeypatch.setattr(accel, "synchronize", lambda *_: None)
    monkeypatch.setattr(backend, "_stream", lambda ring: 1)
    monkeypatch.setattr(backend, "_reconcile_leaves", lambda: None)
    monkeypatch.setattr(backend, "_mark", lambda *_, **__: None)
    if fail_close:
        with pytest.raises(RuntimeFailure, match="staging drain failed"):
            backend._commit_enqueued()
        assert backend._live_ring is not None and machine.host_reserved and owner.unusable
    else:
        backend._commit_enqueued()
        assert backend._live_ring is None and not machine.host_reserved
    assert events == ["finish", "finish"]
    if not fail_close:
        ring = backend._staging_ring
        ring.waits = 9
        ring.in_flight_peak = 7
        backend.enqueued = {"again": object()}

        def unexpected(*_: Any) -> Any:
            pytest.fail("already backed ring requested a second host reservation")

        backend.staging_host = unexpected
        backend._commit_enqueued()
        assert backend._staging_ring is ring and events == ["finish"] * 4
        assert backend.stats["waits"] == 0 and backend.pinned["in_flight_peak_bytes"] == 0


def test_first_construction_failed_drain_survives_child_and_registry_gc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gc
    import socket
    import weakref
    from cozy_runtime.internal.seam import Channel

    ours, theirs = socket.socketpair()
    host = Executor(Channel(ours), tmp_path)
    child = host._construction_child("construction")
    child_ref = weakref.ref(child)
    refs: list[Any] = []

    class Owned:
        pass

    def body(*_: Any, **__: Any) -> dict[str, Any]:
        backend: Any = Owned()
        backend._live_ring = Owned()
        backend.leases = Owned()
        current = child_ref()
        assert current is not None
        current._own_fill_backend(backend)
        refs.extend(weakref.ref(x) for x in (backend, backend._live_ring, backend.leases))
        raise RuntimeFailure("staging drain failed", code="device_fault")

    monkeypatch.setattr(child, "_prepare_first_body", body)
    try:
        with pytest.raises(RuntimeFailure):
            child._prepare_first(cast(Any, None))
        monkeypatch.undo()
        del child
        gc.collect()
        assert len(host._staging_quarantine) == 1 and all(ref() is not None for ref in refs)
    finally:
        ours.close()
        theirs.close()


def test_operation_preparation_vector_funds_only_minimum_unbacked_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket
    from cozy_runtime.author._executor_requests import Answer
    from cozy_runtime.internal.parallel.mirror import preparation_status
    from cozy_runtime.internal.seam import Channel

    owners = [policy(0, grouped=True)[0] for _ in range(2)]
    for owner in owners:
        owner.staging_unbacked["actual-backend"] = 16
    statuses = [
        preparation_status(owner, 7, rank, owner._host_refusal(16))
        for rank, owner in enumerate(owners)
    ]
    assert [s.host_needed for s in statuses] == [16, 16]
    machine = memory.Machine({}, lambda _: None, lambda _: None, host_available=lambda: 32)
    ours, theirs = socket.socketpair()
    leader = Executor(Channel(ours), tmp_path, world=2)
    leader.group = cast(RankGroup, SimpleNamespace(broken=False, operation_sequence=7))

    def durable(request: Any, kind: Any) -> Answer:
        if request.release:
            machine.release_host(request.reservation)
            return Answer(ok=True)
        return Answer(ok=machine.reserve_host(request.reservation, request.nbytes))

    monkeypatch.setattr(leader, "_durable", durable)
    try:
        with leader._group_host_room(statuses) as grants:
            assert grants == (16, 16)
            for owner, credit in zip(owners, grants, strict=True):
                with owner.host_credit(credit):
                    with owner.staging_host(16, 16, owner="actual-backend") as slots:
                        assert slots == 1
                    assert owner.host_remaining() == 0 and owner.host_needed() == 0
        assert not machine.host_reserved
    finally:
        ours.close()
        theirs.close()


def test_idle_staging_release_reports_logical_bytes_without_physical_credit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, machine = policy(16)
    backend: Any = object.__new__(fill.StreamingFillBackend)
    backend._live_ring = None
    backend._staging_lock = threading.Lock()
    backend._staging_busy = False
    backend._staging_borrow = None
    backend._staging_poisoned = ""
    backend.staging_started = backend.staging_drained = lambda: None
    backend._staging_ring = fill.StagingRing(
        fake_torch(), "cpu", window_bytes=16, slots=1, inflight=0
    )
    backend.staging_released = lambda: owner.staging_unbacked.__setitem__("backend", 16)
    monkeypatch.setattr(owner, "_staging_stores", lambda: (backend,))
    assert owner.host_memory()["staging_host_bytes"] == 16
    assert owner.host_memory()["staging_pinned_bytes"] == 0
    # Host physical room is deliberately unchanged: logical release is not a grant.
    assert owner.reclaim_host(1) == 16
    assert owner.host_memory()["staging_host_bytes"] == 0 and owner.staging_unbacked == {
        "backend": 16
    }
    assert not machine.reserve_host("too-big", 17)


@pytest.mark.parametrize("fault", [False, True])
def test_store_gate_retained_only_when_device_drain_is_unproved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: bool
) -> None:
    import socket
    from cozy_runtime.author._executor_requests import Answer
    from cozy_runtime.internal.seam import Channel

    ours, theirs = socket.socketpair()
    executor = Executor(Channel(ours), tmp_path)
    executor.worker_memory = frozenset({"store_read_gate"})
    requests = []

    def durable(request: Any, kind: Any) -> Answer:
        requests.append(request)
        return Answer(ok=True)

    monkeypatch.setattr(executor, "_durable", durable)
    try:
        with pytest.raises(RuntimeFailure):
            with executor._store_read():
                raise RuntimeFailure(
                    "fill failure", code="device_fault" if fault else "missing_object"
                )
        assert len(requests) == (1 if fault else 2)
        assert bool(executor.poisoned) == fault
    finally:
        ours.close()
        theirs.close()


def test_real_group_fill_drain_releases_construction_before_warm_host_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket
    from cozy_runtime.author._executor_requests import Answer
    from cozy_runtime.internal.parallel.group import OpReady
    from cozy_runtime.internal.seam import Channel

    def launch(argv: list[str], fd: int) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [sys.executable, "-c", FOLLOWER, str(fd), "ready", json.dumps(PLAN)],
            pass_fds=(fd,),
            env=os.environ.copy(),
        )

    group = RankGroup(degree=2, backend="cpu:gloo", root=str(tmp_path), launch=launch)
    ours, theirs = socket.socketpair()
    leader = Executor(Channel(ours), tmp_path, world=2)
    leader.group = group
    owner, machine = policy(32, grouped=True)
    leader.planner = owner
    leader._construction_staging_tokens = []
    leader._previous_construction_staging_credit = None

    def durable(request: Any, kind: Any) -> Answer:
        if request.release:
            machine.release_host(request.reservation)
            return Answer(ok=True)
        return Answer(ok=machine.reserve_host(request.reservation, request.nbytes))

    monkeypatch.setattr(leader, "_durable", durable)
    try:
        group.spawn()
        owner.construction_staging_credit = group.agree_plan(
            model="base",
            plan=PLAN,
            staging=(16, 4),
            reserve_staging=leader._reserve_construction_staging,
        )
        assert owner.construction_staging_credit == 16 and sum(machine.host_reserved.values()) == 32
        [reply] = group.collect("filled")
        assert reply["ok"]
        statuses = [
            OpReady(operation=group.operation_sequence, rank=rank, ok=False, host_needed=16)
            for rank in range(2)
        ]
        with leader._group_host_room(statuses) as credits:
            assert not credits  # redundant construction promise would deny the warm round
        leader._release_construction_staging()
        assert owner.__dict__["construction_staging_credit"] is None and not machine.host_reserved
        with leader._group_host_room(statuses) as credits:
            assert credits == (16, 16)
        assert not machine.host_reserved
    finally:
        group.close()
        ours.close()
        theirs.close()


def test_repeated_preallocation_fit_refusal_removes_only_drained_phantom_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket
    from cozy_runtime.internal.seam import Channel

    ours, theirs = socket.socketpair()
    host = Executor(Channel(ours), tmp_path)
    owner, _ = policy(32)
    host.planner = owner
    try:
        for number in range(3):
            child = host._construction_child(str(number))
            child._group_model_key = f"model-{number}"

            def body(*_: Any, selected: Executor = child, **__: Any) -> dict[str, Any]:
                key = selected._group_model_key
                owner.staging_unbacked[key] = 16
                selected._own_fill_backend(SimpleNamespace(_live_ring=None, close=lambda: None))
                return {"ok": False, "code": "model_fit_refused"}

            monkeypatch.setattr(child, "_prepare_first_body", body)
            assert not child._prepare_first(cast(Any, None))["ok"]
            assert owner.host_needed() == 0 and not host._staging_quarantine
    finally:
        ours.close()
        theirs.close()


def test_parked_group_scope_before_operation_reuses_prepaid_actual_ring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cozy_runtime.internal import accel
    from test_scope_placement_recovery import setup, subject

    owner, held, store, _ = setup(monkeypatch, gain=0)
    owner.grouped = True
    owner.construction_staging_credit = 16
    backend: Any = object.__new__(fill.StreamingFillBackend)
    backend.torch = fake_torch()
    backend.device = "cpu"
    backend.kind = "cpu"
    backend.slot_bytes = 16
    backend.slots = 16
    backend.inflight = 8
    backend.fence = True
    backend._staging_ring = backend._live_ring = None
    backend._staging_lock = threading.Lock()
    backend._staging_busy = False
    backend._staging_borrow = None
    backend._staging_poisoned = ""
    backend.staging_started = backend.staging_drained = lambda: None
    backend.staging_host = owner.staging_host
    backend.prepare_staging()
    ring = backend._staging_ring
    assert len(backend.torch.calls) == 1 and len(ring.buffers) == 1
    assert held.part.common.tier == "disk"  # no weight read happened during ring preparation
    owner.construction_staging_credit = None  # all-rank construction promise released
    assert owner._group_host_credit is None
    backend.store_read = contextlib.nullcontext
    backend.stats = {}
    backend.decoded = {}
    backend.leaf_sites = {}
    backend.forensics = None
    monkeypatch.setattr(accel, "synchronize", lambda *_: None)
    monkeypatch.setattr(backend, "_stream", lambda _: 0)
    monkeypatch.setattr(backend, "_reconcile_leaves", lambda: None)
    monkeypatch.setattr(backend, "_mark", lambda *_, **__: None)

    def unfunded(*_: Any, **__: Any) -> Any:
        pytest.fail("scope before mirrored readiness requested another allocation")

    backend.staging_host = unfunded
    load = store.load_unit
    store.calls = 1  # helper normally injects failure on its first load; this arm tests success

    def fill_unit(name: str, unit: Any) -> None:
        backend.enqueued = {"weight": SimpleNamespace(record_stream=lambda _: None)}
        backend._commit_enqueued()
        load(name, unit)

    monkeypatch.setattr(store, "load_unit", fill_unit)
    model = subject(owner, held)
    model.render()
    assert model.bodies == ["authored body"] and held.part.common.tier == "device"
    assert backend._staging_ring is ring and len(backend.torch.calls) == 1
    ring.close()


def test_successful_construction_later_failed_refill_survives_unload_and_gc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gc
    import socket
    import weakref
    from cozy_runtime.internal import accel
    from cozy_runtime.internal.executor_commands import Unload
    from cozy_runtime.internal.seam import Channel

    ours, theirs = socket.socketpair()
    host = Executor(Channel(ours), tmp_path)
    child = host._construction_child("c")
    backend: Any = object.__new__(fill.StreamingFillBackend)
    backend.enqueued = {"first": object()}
    backend.staging_host = lambda q, s: contextlib.nullcontext(1)
    backend.slot_bytes = 16
    backend.slots = 1
    backend.inflight = 0
    backend.fence = True
    backend.torch = SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda _: Stream()))
    backend.device = "cpu"
    backend.kind = "cpu"
    backend.stats = {}
    backend.decoded = {}
    backend.leaf_sites = {}
    backend.forensics = None
    backend.store_read = contextlib.nullcontext
    backend._live_ring = backend._staging_ring = None
    backend._staging_lock = threading.Lock()
    backend._staging_busy = False
    backend._staging_borrow = None
    backend._staging_poisoned = ""

    class Lease:
        def release(self) -> None:
            pytest.fail("unproved DMA lease released")

    backend.leases = Lease()

    class Ring:
        allocated_bytes = registered_bytes = window_bytes = 16
        in_flight_peak = waits = 0

        def __init__(self, *_: Any, **__: Any) -> None:
            self.stream = Stream()
            self.fail = False

        def track(self, tensor: Any) -> None:
            pass

        def finish(self) -> None:
            if self.fail:
                raise RuntimeError("delayed DMA fault")

    monkeypatch.setattr(fill, "StagingRing", Ring)
    monkeypatch.setattr(accel, "synchronize", lambda *_: None)
    backend._stream = lambda ring: 1
    backend._reconcile_leaves = lambda: None
    backend._mark = lambda *_, **__: None
    child._own_fill_backend(backend)
    host.constructions["c"] = child
    monkeypatch.setattr(host, "_synchronize", lambda: None)
    try:
        backend._commit_enqueued()
        assert not host._staging_quarantine
        backend._staging_ring.fail = True
        backend.enqueued = {"refill": object()}
        with pytest.raises(RuntimeFailure, match="staging drain failed"):
            backend._commit_enqueued()
        references = [weakref.ref(value) for value in (backend, backend._live_ring, backend.leases)]
        with pytest.raises(RuntimeFailure, match="staging drain is unproved"):
            host.unload(Unload(construction="c"))
        assert "c" not in host.constructions
        del child, backend
        gc.collect()
        assert len(host._staging_quarantine) == 1 and all(
            reference() is not None for reference in references
        )
    finally:
        ours.close()
        theirs.close()
