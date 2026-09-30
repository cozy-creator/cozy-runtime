"""Foreground ACK waits for native optional readers; tokens belong to executor generations."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from cozy_runtime.internal import fill, memory
from test_weights_resume_native import _committed


def policy() -> memory.Machine:
    result = memory.Machine({}, lambda _: None, lambda _: None, host_available=lambda: 1 << 30)
    result.optional_host_available = lambda: 1 << 30
    return result


def test_foreground_ack_waits_for_real_native_stream_drain(tmp_path: Path) -> None:
    store, _, _, receipt = _committed(tmp_path)
    checkpoint = fill.Checkpoint(str(store.root), receipt.artifact.manifest.digest)
    plan = checkpoint.read_plan([("model", "weight")], fill.WINDOW_BYTES, ("model",))
    machine = policy()
    in_batch, end_batch, acknowledged, drained = (threading.Event() for _ in range(4))
    errors = []

    def guarded(work: Callable[[], None]) -> None:
        try:
            work()
        except BaseException as exc:
            errors.append(exc)

    def background() -> None:
        with machine.optional_store_read() as admitted:
            assert admitted

            def delivered(batch: fill.Batch) -> None:
                in_batch.set()
                assert end_batch.wait(5)
                batch.release()

            with checkpoint.acquire() as lease:
                lease.stream(
                    plan, [bytearray(fill.slot_bytes(fill.WINDOW_BYTES))], delivered, readers=1
                )
            drained.set()

    def demand() -> None:
        assert machine.begin_store_read("executor:1:2:slot:read")
        assert drained.is_set()  # ACK cannot outlive a callback/native reader.
        acknowledged.set()

    reader = threading.Thread(target=lambda: guarded(background))
    foreground = threading.Thread(target=lambda: guarded(demand))
    reader.start()
    try:
        assert in_batch.wait(5), errors
        foreground.start()
        with machine._store_condition:
            assert machine._store_condition.wait_for(lambda: bool(machine._store_demands), 5)
        assert not machine.optional_store_wanted() and not acknowledged.is_set()
        with machine.optional_store_read() as admitted:
            assert not admitted
    finally:
        end_batch.set()
        reader.join(5)
        if foreground.ident is not None:
            foreground.join(5)
    assert not reader.is_alive() and not foreground.is_alive() and not errors
    assert acknowledged.is_set()
    machine.end_store_read("executor:1:2:slot:read")
    with machine.optional_store_read() as admitted:
        assert admitted


def test_failed_optional_reader_and_confirmed_retirement_release_gate() -> None:
    machine = policy()
    with (
        pytest.raises(RuntimeError, match="native failure"),
        machine.optional_store_read() as admitted,
    ):
        assert admitted
        raise RuntimeError("native failure")
    assert machine.begin_store_read("executor:1:2:slot:old")
    assert machine.begin_store_read("executor:1:3:slot:new")
    machine.release_store_prefix("executor:1:2:")
    assert (
        not machine.optional_store_wanted()
    )  # replacement's token survives old-generation cleanup
    machine.end_store_read("executor:1:3:slot:new")
    with machine.optional_store_read() as admitted:
        assert admitted


def test_optional_credit_cannot_spend_reclaimable_demand_headroom() -> None:
    machine = policy()
    machine.optional_host_available = lambda: 10
    assert machine.reserve_host("demand", 4)
    assert machine.prefetch_allowance(3) == 3
    assert not machine.reserve_host("optional", 7, optional=True)
    assert machine.reserve_host("optional", 6, optional=True)
    assert machine.prefetch_allowance(0) == 0
    machine.release_host("optional")
    machine.optional_host_available = lambda: -1
    assert machine.prefetch_allowance(0) == 0
    assert not machine.reserve_host("optional", 1, optional=True)
    assert machine.reserve_host("ordinary", 100)  # unchanged ordinary allocation credit


@pytest.mark.parametrize(
    ("physical", "current", "expected"), [(40, 950, 40 * 1024), (100, 999, 1024), (100, 1001, 0)]
)
def test_optional_headroom_uses_memfree_and_ancestor_limits_without_cache_credit(
    monkeypatch: pytest.MonkeyPatch,
    physical: int,
    current: int,
    expected: int,
) -> None:
    from cozy_runtime.internal import proctree

    files = {
        "/proc/meminfo": f"MemAvailable: 5000 kB\nMemFree: {physical} kB\n",
        "/proc/self/cgroup": "0::/parent/child\n",
        "/sys/fs/cgroup/parent/child/memory.max": "max",
        "/sys/fs/cgroup/parent/child/memory.high": "max",
        "/sys/fs/cgroup/parent/memory.max": str(2000 * 1024),
        "/sys/fs/cgroup/parent/memory.high": str(1000 * 1024),
        "/sys/fs/cgroup/parent/memory.current": str(current * 1024),
        "/sys/fs/cgroup/parent/memory.stat": f"inactive_file {900 * 1024}\n",
    }
    monkeypatch.setattr(
        Path, "is_file", lambda path: str(path) in files or str(path).endswith("cgroup.controllers")
    )

    def read(path: Path) -> str:
        if str(path) not in files:
            raise FileNotFoundError(path)
        return files[str(path)]

    monkeypatch.setattr(Path, "read_text", read)
    assert proctree.optional_host_bytes() == expected
    assert proctree.available_host_bytes() > expected
    del files["/sys/fs/cgroup/parent/memory.current"]
    assert proctree.optional_host_bytes() == -1


def test_retiring_waiting_generation_returns_no_permission_to_read() -> None:
    machine = policy()
    finished = threading.Event()
    answers = []

    def demand() -> None:
        answers.append(machine.begin_store_read("executor:1:2:slot:read"))
        finished.set()

    with machine.optional_store_read() as admitted:
        assert admitted
        thread = threading.Thread(target=demand)
        thread.start()
        try:
            with machine._store_condition:
                assert machine._store_condition.wait_for(lambda: bool(machine._store_demands), 5)
            machine.release_store_prefix("executor:1:2:")
            assert finished.wait(5)
            assert answers == [False]
        finally:
            thread.join(5)
    assert machine.optional_store_wanted()


def test_nested_phase_tokens_are_independently_owned_and_empty_cleanup_refuses() -> None:
    machine = policy()
    with machine.store_demand("prepare:placement"):
        with machine.store_demand("prepare:placement"):
            assert len(machine._store_demands) == 2
        assert not machine.optional_store_wanted()
        with pytest.raises(ValueError, match="explicit generation"):
            machine.release_store_prefix("")
    assert machine.optional_store_wanted()


def test_retired_phase_never_enters_body(monkeypatch: pytest.MonkeyPatch) -> None:
    from cozy_runtime.author._errors import RuntimeFailure

    machine = policy()
    monkeypatch.setattr(machine, "begin_store_read", lambda _: False)
    with pytest.raises(RuntimeFailure, match="retired"), machine.store_demand("phase"):
        pytest.fail("retired demand was allowed to execute")


def test_optional_allocation_never_reclaims_an_idle_tenant() -> None:
    from types import SimpleNamespace
    from typing import Any, cast

    machine = policy()
    machine.optional_host_available = lambda: 0

    def reclaim(_nbytes: int) -> None:
        pytest.fail("optional read reclaimed useful backing")

    victim = cast(Any, SimpleNamespace(slot="idle", state=lambda: "idle", reclaim_host=reclaim))
    assert not machine.reserve_host("optional", 1, optional=True, tenants=[victim])


@pytest.mark.parametrize(
    ("world", "supported", "calls"), [(1, True, 2), (1, False, 0), (2, True, 0)]
)
def test_executor_store_read_is_single_rank_negotiated_and_ends_on_failure(
    world: int,
    supported: bool,
    calls: int,
) -> None:
    from types import SimpleNamespace
    from typing import Any, cast

    from cozy_runtime.author._executor_requests import Answer, StoreRead, decode, encode
    from cozy_runtime.internal.executor import Executor

    requests = []

    def durable(request: StoreRead, _: object) -> Answer:
        requests.append(request)
        assert decode(encode(request)) == request
        return Answer(ok=True)

    owner = cast(
        Any,
        SimpleNamespace(
            world=world,
            host=SimpleNamespace(worker_memory={"store_read_gate"} if supported else set()),
            _durable=durable,
        ),
    )
    with pytest.raises(ValueError, match="fill failed"), Executor._store_read(owner):
        raise ValueError("fill failed")
    assert len(requests) == calls
    if calls:
        assert requests[0].reservation == requests[1].reservation
        assert not requests[0].release and requests[1].release


def test_failed_phase_keeps_generation_gate_until_confirmed_reap() -> None:
    from types import SimpleNamespace
    from typing import Any, cast

    from cozy_runtime.internal.worker.session import Worker

    machine = policy()
    current = SimpleNamespace(pid=123, epoch=4, poisoned="", alive=lambda: True)
    supervision = SimpleNamespace(current=current)
    worker = cast(Any, SimpleNamespace(memory=machine))
    with Worker._store_phase(worker, "enter", cast(Any, supervision)):
        current.poisoned = "group still has readers"
    assert not machine.optional_store_wanted()
    machine.release_store_prefix("executor:123:4:")
    supervision.current = None
    assert machine.optional_store_wanted()
    # Reclaim raced before the phase exited: no stale token may be recreated afterward.
    supervision.current = current
    with Worker._store_phase(worker, "enter", cast(Any, supervision)):
        supervision.current = None
        machine.release_store_prefix("executor:123:4:")
    assert machine.optional_store_wanted()


def test_native_lease_acquisition_is_gated_but_cached_lease_needs_no_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import Iterator
    from contextlib import contextmanager

    store, _, _, receipt = _committed(tmp_path)
    checkpoint = fill.Checkpoint(str(store.root), receipt.artifact.manifest.digest)
    machine = policy()
    scopes = []

    @contextmanager
    def reading() -> Iterator[None]:
        with machine.store_demand("acquisition"):
            scopes.append("begin")
            yield
            scopes.append("end")

    acquire = checkpoint.acquire

    def guarded_acquire() -> fill.ReadLease:
        assert not machine.optional_store_wanted()
        return acquire()

    monkeypatch.setattr(checkpoint, "acquire", guarded_acquire)
    leases = fill.ReadLeases({"model": checkpoint}, store_read=reading)
    try:
        first = leases.acquire("model")
        assert leases.acquire("model") is first
        assert scopes == ["begin", "end"] and machine.optional_store_wanted()
        leases.release()
        assert leases.acquire("model").live
        assert scopes == ["begin", "end", "begin", "end"]
    finally:
        leases.release()


def test_job_store_read_uses_generation_room_broker_not_child_calls() -> None:
    from types import SimpleNamespace
    from typing import Any, cast

    from cozy_runtime.author._executor_requests import Answer, StoreRead
    from cozy_runtime.internal.worker.attempts import AttemptEngine

    seen = []

    def room(_attempt: object) -> Callable[[StoreRead], Answer]:
        def answer(request: StoreRead) -> Answer:
            seen.append(request)
            return Answer(ok=True)

        return answer

    engine = cast(Any, SimpleNamespace(room=room, calls=None))
    for release in (False, True):
        request = StoreRead(reservation="read", release=release)
        reply = AttemptEngine.durable_exchange(engine, cast(Any, object()), request)
        assert isinstance(reply, Answer) and reply.ok
    assert [request.release for request in seen] == [False, True]
    engine.room = None
    refused = AttemptEngine.durable_exchange(
        engine, cast(Any, object()), StoreRead(reservation="read")
    )
    assert isinstance(refused, Answer) and not refused.ok


@pytest.mark.parametrize("initial", ["old", "none-healthy", "none-poisoned"])
def test_phase_exception_cannot_pin_a_healthy_successor(initial: str) -> None:
    from types import SimpleNamespace
    from typing import Any, cast

    from cozy_runtime.internal.worker.session import Worker

    machine = policy()
    old = SimpleNamespace(pid=123, epoch=4, poisoned="", alive=lambda: True)
    new = SimpleNamespace(pid=456, epoch=5, poisoned="", alive=lambda: True)
    supervision = SimpleNamespace(current=old if initial == "old" else None)
    worker = cast(Any, SimpleNamespace(memory=machine))
    with (
        pytest.raises(ValueError, match="old phase failed"),
        Worker._store_phase(worker, "prepare", cast(Any, supervision)),
    ):
        # Confirmed old-generation retirement and replacement happen before the old
        # phase unwinds. Its exception is not evidence against the healthy successor.
        supervision.current = None
        machine.release_store_prefix("executor:123:4:")
        supervision.current = new
        if initial == "none-poisoned":
            new.poisoned = "new generation still has readers"
        raise ValueError("old phase failed")
    with machine.optional_store_read() as admitted:
        assert admitted == (initial != "none-poisoned")
    if initial == "none-poisoned":
        machine.release_store_prefix("executor:456:5:")
    assert not machine._store_demands
