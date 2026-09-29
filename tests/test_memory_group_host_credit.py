"""Group host promises cross a real control socket; copies spend finite rank credit."""

import socket
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn, cast

import pytest

from cozy_runtime.author._executor_requests import Answer, HostRoom
from cozy_runtime.internal import memory, tiers
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.parallel.group import OpReady, RankGroup
from cozy_runtime.internal.seam import Channel

torch = pytest.importorskip("torch")


def test_rank_credit_counts_alias_backings_and_cannot_allocate_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = memory.Planner(torch, torch.device("cpu"), grouped=True)
    root = torch.nn.Module()
    backing = torch.arange(4, dtype=torch.float32)
    root.a = torch.nn.Parameter(backing[:2])
    root.b = torch.nn.Parameter(backing[2:])
    held = owner.adopt("c", "model", root, None, tiers.DEVICE)
    # CPU tensors supply real alias geometry; only the device-copy requirement is simulated.
    monkeypatch.setattr(tiers, "host_copy_bytes", lambda unit: unit.nbytes)
    assert owner.host_needed() == 16
    assert not owner._movable(held.part, held.part.common)
    with owner.host_credit(16):
        assert owner._movable(held.part, held.part.common)
        owner._home(held.part, held.part.common)
        assert held.part.common.tier == tiers.HOST
        assert owner.host_needed() == 0
        with owner._host_copy(1) as admitted:
            assert not admitted
        with owner._host_copy(0) as admitted:
            assert admitted
    assert owner._group_host_credit is None


@pytest.mark.parametrize("available", [31, 32])
def test_worker_reserves_aggregate_before_distributing_and_releases_after_use(
    tmp_path: Path, available: int
) -> None:
    ours, theirs = socket.socketpair()
    leader = Executor(Channel(ours), tmp_path, world=2)
    leader.group = cast(RankGroup, SimpleNamespace(broken=False, operation_sequence=7))
    worker = Channel(theirs)
    policy = memory.Machine({}, lambda _: None, lambda _: None, host_available=lambda: available)
    requests = []
    error = []

    def serve() -> None:
        try:
            for _ in range(2 if available == 32 else 1):
                request = worker.recv()
                assert request is not None
                requests.append(request)
                assert request["kind"] == "host_room"
                token = request["reservation"]
                if request.get("release"):
                    policy.release_host(token)
                    ok = True
                else:
                    ok = policy.reserve_host(token, request["nbytes"])
                worker.send({"event": "answer", "seq": request["seq"], "ok": ok})
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        statuses = [OpReady(operation=7, rank=r, ok=False, host_needed=16) for r in range(2)]
        with leader._group_host_room(statuses) as credits:
            assert credits == ((16, 16) if available == 32 else ())
            assert sum(policy.host_reserved.values()) == (32 if credits else 0)
        thread.join()
        assert not error and not policy.host_reserved
        assert requests[0]["reservation"] == "group-host-7"
        if available == 32:
            assert requests[1]["release"]
    finally:
        ours.close()
        theirs.close()


def test_broken_followers_are_reaped_before_promise_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ours, theirs = socket.socketpair()
    leader = Executor(Channel(ours), tmp_path, world=2)
    events = []
    leader.group = cast(
        RankGroup,
        SimpleNamespace(broken=True, operation_sequence=1, close=lambda: events.append("reaped")),
    )

    def durable(request: HostRoom, _: object) -> Answer:
        events.append("release" if request.release else "reserve")
        return Answer(ok=True)

    monkeypatch.setattr(leader, "_durable", durable)
    try:
        with leader._group_host_room(
            [OpReady(operation=1, rank=r, ok=False, host_needed=8) for r in range(2)]
        ):
            events.append("operation")
        assert events == ["reserve", "operation", "reaped", "release"]
    finally:
        ours.close()
        theirs.close()


def test_host_credit_remaining_survives_abort_without_being_reissued() -> None:
    owner = memory.Planner(torch, torch.device("cpu"), grouped=True)
    with pytest.raises(ValueError), owner.host_credit(32):
        with owner._host_copy(16) as admitted:
            assert admitted
        raise ValueError("preparation aborted")
    assert owner.host_remaining() == 16
    with owner.host_credit(owner.host_remaining()):
        with owner._host_copy(17) as admitted:
            assert not admitted
        with owner._host_copy(16) as admitted:
            assert admitted
    assert owner.host_remaining() == 0


def test_failed_reap_retains_reservation_for_worker_generation_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ours, theirs = socket.socketpair()
    leader = Executor(Channel(ours), tmp_path, world=2)
    events = []

    def close() -> NoReturn:
        raise OSError("reap not proven")

    leader.group = cast(RankGroup, SimpleNamespace(broken=True, operation_sequence=1, close=close))

    def durable(request: HostRoom, _: object) -> Answer:
        events.append("release" if request.release else "reserve")
        return Answer(ok=True)

    monkeypatch.setattr(leader, "_durable", durable)
    try:
        with (
            pytest.raises(OSError),
            leader._group_host_room(
                [OpReady(operation=1, rank=r, ok=False, host_needed=8) for r in range(2)]
            ),
        ):
            pass
        assert events == ["reserve"]
    finally:
        ours.close()
        theirs.close()
