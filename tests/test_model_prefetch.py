"""Known-next preparation does not require a future inference payload or GPU admission."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, cast

import msgspec
import pytest

from cozy_runtime import author
from cozy_runtime.author._calls import _Broker, _CallType, _current
from cozy_runtime.internal.worker import activity
from cozy_runtime.internal.worker.preparations import Preparations
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.protocol import worker_pb2 as pb
from durable_seam import wire


class Payload(msgspec.Struct):
    image: str


class Result(msgspec.Struct):
    value: str


def next_model(*, image: str) -> None:
    raise AssertionError("a hint invoked inference")


def test_hint_uses_locked_callable_without_reserving_a_call_or_future_payload() -> None:
    received: list[tuple[str, dict[str, Any]]] = []

    def exchange(kind: str, body: dict[str, Any]) -> dict[str, Any]:
        received.append((kind, body))
        return {"ok": True}

    binding = _CallType("sha256:" + "a" * 64, __name__, "next_model", Payload, Result)
    broker = _Broker("parent", {(__name__, "next_model"): binding}, wire(exchange))
    broker.bind(author.Context("parent", time.monotonic() + 60))
    token = _current.set(broker)
    try:
        author.prefetch(next_model)
        assert received == [
            (
                "model_prefetch",
                {
                    "module": __name__,
                    "export": "next_model",
                    "payload": "{}",
                },
            )
        ]
        assert broker.calls == {} and broker.next_index == 0
        with pytest.raises(author.CapabilityError, match="locked callable"):
            author.prefetch(lambda: None)
        with pytest.raises(author.ConformanceError, match="ModelArtifact"):
            author.prefetch(next_model, models=cast(Any, {"model": "unresolved/alias"}))
        broker.finish()
    finally:
        _current.reset(token)


@pytest.mark.parametrize("code", ["unknown_durable_request", "model_prefetch_unsupported"])
def test_old_worker_can_decline_optional_hint(code: str) -> None:
    binding = _CallType("sha256:" + "a" * 64, __name__, "next_model", Payload, Result)
    broker = _Broker(
        "parent", {(__name__, "next_model"): binding}, wire(lambda *_: {"ok": False, "code": code})
    )
    broker.bind(author.Context("parent", time.monotonic() + 60))
    token = _current.set(broker)
    try:
        author.prefetch(next_model)
        broker.finish()
    finally:
        _current.reset(token)


def test_background_preparation_overlaps_current_work_and_demand_joins() -> None:
    preparing, finish, current_work = threading.Event(), threading.Event(), threading.Event()
    current_work.set()
    queue = Preparations[bytes]()
    count = 0

    def prepare(cancelled: Callable[[], bool]) -> bytes:
        nonlocal count
        count += 1
        assert current_work.is_set() and not cancelled()
        preparing.set()
        assert finish.wait(5), "test did not release model preparation"
        return b"verified model binding"

    try:
        hinted = queue.request("exact-B", ("parent", 1), prepare, speculative=True)
        assert preparing.wait(5)
        demanded = queue.request("exact-B", ("parent", 1), prepare, speculative=False)
        assert demanded is hinted and count == 1
        finish.set()
        assert demanded is not None and demanded.result(5) == b"verified model binding"
        assert current_work.is_set() and not queue.started()
        assert queue.request("exact-B", ("second-parent", 1), prepare, speculative=False) is hinted
    finally:
        finish.set()
        queue.close()


def test_optional_failure_can_retry_after_repair_without_a_new_parent() -> None:
    queue = Preparations[bytes]()

    def failed(_: Callable[[], bool]) -> bytes:
        raise OSError("model transfer unavailable")

    try:
        future = queue.request("B", "parent", failed, speculative=True)
        assert future is not None
        with pytest.raises(OSError, match="model transfer unavailable"):
            future.result(5)
        demand = queue.request("A", "parent", lambda _: b"A result", speculative=False)
        assert demand is not None and demand.result(5) == b"A result"
        repaired = queue.request("B", "parent", lambda _: b"B repaired", speculative=False)
        assert repaired is not None and repaired is not future
        assert repaired.result(5) == b"B repaired"
    finally:
        queue.close()


def test_speculation_reserves_demand_capacity_and_promotes_queued_work() -> None:
    queue = Preparations[str]()
    started, release = threading.Event(), threading.Event()
    queued_started = threading.Event()

    def slow(_: Callable[[], bool]) -> str:
        started.set()
        assert release.wait(5)
        return "first"

    def queued(_: Callable[[], bool]) -> str:
        queued_started.set()
        return "next"

    try:
        first = queue.request("slow-B", "parent", slow, speculative=True)
        assert started.wait(5)
        next_ = queue.request("next-C", "parent", queued, speculative=True)
        assert not queued_started.is_set()
        promoted = queue.request("next-C", "parent", queued, speculative=False)
        assert promoted is next_ and promoted is not None
        assert promoted.result(5) == "next" and not cast(Future[str], first).done()
    finally:
        release.set()
        queue.close()


def idle_worker(queue: Preparations[bytes]) -> Worker:
    """A worker doing nothing but its preparations."""
    return cast(
        Worker,
        SimpleNamespace(
            phase=pb.WorkerPhase.WORKER_PHASE_ONLINE,
            supervisor=SimpleNamespace(keys=lambda _prefix: []),
            machine_calls=SimpleNamespace(serving=SimpleNamespace(preparations=queue)),
            preparation_lock=threading.Lock(),
            placement=None,
            hosted={},
        ),
    )


def test_a_queued_hint_holds_nothing_and_work_under_way_is_named() -> None:
    """A hint still waiting for a worker is no work: it never keeps the machine from idling.
    The preparation under way is named as what holds the machine."""
    queue = Preparations[bytes]()
    go = threading.Event()
    worker = idle_worker(queue)

    def prepare(_cancelled: Callable[[], bool]) -> bytes:
        go.wait(5)
        return b""

    try:
        demanded = queue.request("model-a", "caller", prepare, speculative=False)
        queue.request("model-c", "caller", prepare, speculative=False)
        queue.request("model-b", "parent", prepare, speculative=True)
        deadline = time.monotonic() + 5
        while len(queue.started()) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        # Both workers prepare demanded models; the hint waits, and holds nothing.
        assert "model-b" in queue.entries
        assert activity.holding(worker) == ["preparation:model-a", "preparation:model-c"]
        go.set()
        assert demanded is not None and demanded.result(5) == b""
    finally:
        go.set()
        queue.close()


def test_last_interest_cancels_native_work_and_activity_lasts_until_it_stops() -> None:
    queue = Preparations[bytes]()
    started, native_cancel, stop_finished = (threading.Event() for _ in range(3))
    canceled = False

    def transfer(cancelled: Callable[[], bool]) -> bytes:
        nonlocal canceled
        started.set()
        assert native_cancel.wait(5)
        canceled = cancelled()
        assert stop_finished.wait(5)
        return b"partial native bytes retained"

    worker = idle_worker(queue)
    try:
        first = queue.request(
            "B", "parent-one", transfer, speculative=True, cancel=native_cancel.set
        )
        queue.request("B", "parent-two", transfer, speculative=False)
        assert started.wait(5) and activity.active_work(worker)
        queue.retain(lambda parent: parent == "parent-two")
        assert not native_cancel.is_set()
        queue.retain(lambda _: False)
        assert native_cancel.is_set() and activity.active_work(worker)
        stop_finished.set()
        assert first is not None and first.result(5) == b"partial native bytes retained"
        assert canceled and not activity.active_work(worker)
        queue.retain(lambda _: False)
        assert not queue.entries
    finally:
        native_cancel.set()
        stop_finished.set()
        queue.close()


def test_canceled_queued_hint_never_starts_and_demand_without_hint_still_runs() -> None:
    queue = Preparations[str]()
    started, release = threading.Event(), threading.Event()

    def slow(_: Callable[[], bool]) -> str:
        started.set()
        assert release.wait(5)
        return "kept"

    def forbidden(_: Callable[[], bool]) -> str:
        raise AssertionError("unused queued hint started")

    try:
        queue.request("B", "kept", slow, speculative=True)
        assert started.wait(5)
        dropped = queue.request("C", "gone", forbidden, speculative=True)
        queue.retain(lambda parent: parent == "kept")
        assert dropped is not None and dropped.cancelled()
        demanded = queue.request("D", "demand", lambda _: "ordinary", speculative=False)
        assert demanded is not None and demanded.result(5) == "ordinary"
    finally:
        release.set()
        queue.close()


def test_demand_before_callee_installation_holds_hints_until_submission() -> None:
    queue = Preparations[str]()
    computing = threading.Event()

    def hint(_: Callable[[], bool]) -> str:
        assert computing.is_set(), "hint ran while a real call still needed preparation"
        return "next model"

    try:
        with queue.demand("parent"):
            with queue.demand("parent"):
                pending = queue.request("next", "parent", hint, speculative=True)
                assert not queue.started()
            # One parent's concurrent call must not release its sibling's demand.
            assert not queue.started()
            ready = queue.request("current", "parent", lambda _: "ready", speculative=False)
            assert ready is not None and ready.result(5) == "ready"
            assert pending is not None and not pending.done()
            computing.set()  # the actual child can now be submitted
        assert pending.result(5) == "next model"
    finally:
        computing.set()
        queue.close()


def test_running_hint_yields_and_promoted_demand_keeps_its_shared_future() -> None:
    queue = Preparations[str]()
    started, cancel, unwound = (threading.Event() for _ in range(3))
    release = threading.Event()
    attempts = 0

    def prepare(cancelled: Callable[[], bool]) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            started.set()
            assert cancel.wait(5)
            assert cancelled()
            assert release.wait(5)
            unwound.set()  # native transfer and temporary holds have finished
            raise InterruptedError("speculative transfer paused")
        assert unwound.is_set() and not cancelled()
        return "verified bytes retained across pause"

    try:
        shared = queue.request("next", "hint-parent", prepare, speculative=True, cancel=cancel.set)
        assert started.wait(5)
        with queue.demand("real-parent"):
            assert cancel.is_set() and shared is not None and not shared.done()
            ordinary = queue.request("current", "real-parent", lambda _: "ready", speculative=False)
            assert ordinary is not None and ordinary.result(5) == "ready"
            # Another real call needs the paused selection. It joins the SAME future;
            # its restart must wait for the old transfer to relinquish its writer.
            promoted = queue.request("next", "second-parent", prepare, speculative=False)
            assert promoted is shared and attempts == 1
            release.set()
            assert promoted.result(5) == "verified bytes retained across pause"
            assert attempts == 2
    finally:
        cancel.set()
        release.set()
        queue.close()


def test_another_demand_never_cancels_preparation_with_a_real_interest() -> None:
    queue = Preparations[str]()
    started, finish, canceled = (threading.Event() for _ in range(3))

    def prepare(cancelled: Callable[[], bool]) -> str:
        started.set()
        assert finish.wait(5)
        assert not cancelled()
        return "shared"

    try:
        hint = queue.request("same", "hint", prepare, speculative=True, cancel=canceled.set)
        assert started.wait(5)
        actual = queue.request("same", "actual", prepare, speculative=False)
        assert actual is hint
        with queue.demand("other"):
            queue.retain(lambda parent: parent != "hint")
            assert not canceled.is_set()
            finish.set()
            assert actual is not None and actual.result(5) == "shared"
    finally:
        finish.set()
        queue.close()


def test_demanded_preparation_finishes_before_a_queued_hint_starts() -> None:
    queue = Preparations[str]()
    started, finish = threading.Event(), threading.Event()

    def prepare(_: Callable[[], bool]) -> str:
        started.set()
        assert finish.wait(5)
        return "ready"

    try:
        actual = queue.request("current", "actual", prepare, speculative=False)
        assert started.wait(5)
        hint = queue.request("next", "hint", lambda _: "next", speculative=True)
        assert queue.started() == ["current"]
        assert hint is not None and not hint.done()
        finish.set()
        assert actual is not None and actual.result(5) == "ready"
        assert hint is not None and hint.result(5) == "next"
    finally:
        finish.set()
        queue.close()


def test_parent_fencing_releases_its_early_demand_lease() -> None:
    queue = Preparations[str]()
    try:
        with queue.demand("gone"):
            hint = queue.request("next", "kept", lambda _: "next", speculative=True)
            assert hint is not None and not queue.started()
            queue.retain(lambda parent: parent != "gone")
            assert hint.result(5) == "next"
        assert not queue.demands
    finally:
        queue.close()


def test_abandoning_a_paused_hint_does_not_restart_it() -> None:
    queue = Preparations[str]()
    started, cancel, finish = (threading.Event() for _ in range(3))
    attempts = 0

    def prepare(_: Callable[[], bool]) -> str:
        nonlocal attempts
        attempts += 1
        started.set()
        assert cancel.wait(5) and finish.wait(5)
        raise InterruptedError("no remaining consumers")

    try:
        hint = queue.request("next", "hint", prepare, speculative=True, cancel=cancel.set)
        assert started.wait(5)
        with queue.demand("actual"):
            assert cancel.is_set()
            queue.retain(lambda interest: interest != "hint")
            finish.set()
            assert hint is not None
            with pytest.raises(InterruptedError, match="no remaining"):
                hint.result(5)
        assert attempts == 1
    finally:
        cancel.set()
        finish.set()
        queue.close()
