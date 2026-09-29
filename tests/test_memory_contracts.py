"""Memory authorization and rank facts across real codec and control-channel boundaries.

These CPU checks do not qualify collective execution or physical GPU budget enforcement.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import contextlib
from dataclasses import replace
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.author._executor_requests import Answer, DeviceRoom, Request, Room
from cozy_runtime.internal import executor_commands, executor_replies
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import Activate, Invoke, Load, Start
from cozy_runtime.internal.memory import MemoryRefusal
from cozy_runtime.internal.parallel.group import RankGroup
from cozy_runtime.internal.seam import (
    EXECUTOR_MEMORY_CAPABILITIES,
    EXECUTOR_PROTOCOL_REVISION,
    Channel,
    SeamError,
)
from cozy_runtime.internal.worker.child import (
    ExecutorProtocolMismatch,
    require_executor_protocol,
    respond,
)
from cozy_runtime.internal.worker.lanes import LaneSet
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.session import Worker


def test_absent_zero_and_rank_authorizations_survive_the_codec() -> None:
    absent = Invoke(request_id="r", entrypoint="run", spool="/unused", deadline_s=None)
    frame = executor_commands.encode(absent)
    assert "memory_grant_bytes" not in frame
    legacy = executor_commands.decode(frame)
    assert isinstance(legacy, Invoke) and legacy.memory_grant_bytes is msgspec.UNSET
    zero = msgspec.structs.replace(absent, memory_grant_bytes=0, memory_grants=(0, 37))
    restored = executor_commands.decode(executor_commands.encode(zero))
    assert isinstance(restored, Invoke)
    assert restored.memory_grant_bytes == 0 and restored.memory_grants == (0, 37)
    for command in (
        Start(authorized_device_limit_bytes=11, memory_grants=(11, 23)),
        Load(construction="c", authorized_device_limit_bytes=11, memory_grants=(11, 23)),
        Activate(construction="c", authorized_device_limit_bytes=11, memory_grants=(11, 23)),
    ):
        assert executor_commands.decode(executor_commands.encode(command)) == command
    # A worker predating the additive vector still authorizes through its scalar field.
    old_room = msgspec.convert({"ok": True, "authorized_device_limit_bytes": 19}, Room)
    assert old_room.memory_grants == () and old_room.authorized_device_limit_bytes == 19
    assert executor_commands.decode({"cmd": "start", "future": True}) == Start()


def test_grants_are_validated_before_replacing_group_authority(tmp_path: Path) -> None:
    ours, theirs = socket.socketpair()
    channel = Channel(ours)
    try:
        executor = Executor(channel, tmp_path, world=2)
        executor.group = RankGroup(degree=2, backend="cpu:gloo", root=str(tmp_path))
        executor._grant(7, (7, 19))
        assert executor.group.memory_grants == (7, 19)
        executor._grant(msgspec.UNSET)
        assert executor.group.memory_grants == (7, 19)
        for grant, ranks in ((-1, ()), (0, (8, -1)), (0, (8,))):
            with pytest.raises(MemoryRefusal) as refused:
                executor._grant(grant, ranks)
            assert refused.value.code == "device_shortfall"
            assert executor.group.memory_grants == (7, 19)
        executor._grant(0)
        assert executor.group.memory_grants == (0, 0)
    finally:
        channel.close()
        theirs.close()


def test_unsupported_host_credit_protocol_refuses_before_allocating(tmp_path: Path) -> None:
    ours, theirs = socket.socketpair()
    executor = Executor(Channel(ours), tmp_path)
    worker = Channel(theirs)

    def answer() -> None:
        try:
            request = worker.recv()
            assert request is not None and request["kind"] == "host_room"
            worker.send(
                {
                    "event": "answer",
                    "seq": request["seq"],
                    "ok": False,
                    "code": "unknown_durable_request",
                    "detail": "legacy worker",
                }
            )
        finally:
            worker.close()

    reader = threading.Thread(target=answer)
    reader.start()
    try:
        with pytest.raises(Exception, match="update the machine Runtime"):
            executor._ask_host_room("copy-1", 64)
        assert executor._exchange == 1
    finally:
        executor.channel.close()
        reader.join()


def test_follower_memory_is_not_copied_from_the_leader() -> None:
    lane = LaneSet.from_envelope("a,b,c", worker_pid=os.getpid()).group((1, 2))
    row = lane.row("tenant")
    Worker._observe_memory(
        lane, row, {"rank_memory": {"0": {"reserved_bytes": 10, "overhead_bytes": 2}}}
    )
    assert row.held == {1: 12} and row.overhead == {1: 2}
    Worker._observe_memory(
        lane,
        row,
        {
            "memory": {"reserved_bytes": 10, "overhead_bytes": 2},
            "rank_memory": {"1": {"reserved_bytes": 70, "overhead_bytes": 5}},
        },
    )
    assert row.held == {1: 12, 2: 75}
    # A rank-selective vacate must not report the idle leader's allocator as the follower's.
    Worker._observe_memory(
        lane,
        row,
        {
            "memory": {"reserved_bytes": 10, "overhead_bytes": 2},
            "rank_memory": {"1": {"reserved_bytes": 3, "overhead_bytes": 5}},
        },
        (2,),
    )
    assert row.held == {1: 12, 2: 8}
    # Missing follower facts retain the prior conservative report, never a guessed zero.
    Worker._observe_memory(lane, row, {"reserved_bytes": 0}, (2,))
    assert row.held == {1: 12, 2: 8}
    # A full snapshot replaces every rank. An omitted rank is unknown rather than
    # the previous observation, while rank-selective vacate above keeps untouched ranks.
    Worker._observe_memory(lane, row, {"rank_memory": {"1": {"reserved_bytes": 4}}})
    assert row.held == {2: 4}


def test_old_namespace_pid_is_translated_for_driver_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cozy_runtime.internal import proctree

    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path: "Name:\texecutor\nNSpid:\t420 42\n"
        if str(path) == "/proc/42/status"
        else (_ for _ in ()).throw(AssertionError(path)),
    )
    assert proctree.host_pid(42) == 420


def test_process_facts_do_not_survive_its_generation() -> None:
    row = LaneSet.from_envelope("a", worker_pid=os.getpid()).lanes[0].row("tenant")
    row = replace(row, memory_generation=(123, 4))
    row.held[0], row.overhead[0], row.peak[("run", "shape")] = 30, 5, {0: 60}
    row.sync_memory_generation()
    assert row.memory_generation is None and not row.held and not row.overhead and not row.peak


def test_current_memory_contract_is_required_before_device_initialization() -> None:
    hello = {
        "executor_protocol_revision": EXECUTOR_PROTOCOL_REVISION,
        "native_interfaces": {},
        "memory": sorted(EXECUTOR_MEMORY_CAPABILITIES),
    }
    require_executor_protocol(hello)
    for missing in EXECUTOR_MEMORY_CAPABILITIES:
        with pytest.raises(ExecutorProtocolMismatch, match="current Runtime memory contract"):
            require_executor_protocol(
                {
                    **hello,
                    "memory": [v for v in sorted(EXECUTOR_MEMORY_CAPABILITIES) if v != missing],
                }
            )


def test_additive_rank_facts_do_not_change_the_attempt_outcome() -> None:
    frame = {
        "ok": True,
        "memory": {"reserved_bytes": 3},
        "rank_memory": {"0": {"reserved_bytes": 3}, "1": {"reserved_bytes": 17}},
    }
    reply = executor_replies.decode(frame, executor_replies.AttemptReply)
    assert reply.ok and reply.rank_memory["1"]["reserved_bytes"] == 17
    # Unreadable additive telemetry costs itself, not a successfully returned operation.
    unreadable = executor_replies.decode(
        {**frame, "rank_memory": {"1": {"reserved_bytes": "unreadable"}}},
        executor_replies.AttemptReply,
    )
    assert unreadable.ok and unreadable.memory == reply.memory and unreadable.rank_memory == {}
    partial = executor_replies.decode(
        {
            **frame,
            "memory": {"reserved_bytes": 3, "future": {"unfamiliar": True}},
            "rank_memory": {
                "1": {"reserved_bytes": 17, "overhead_bytes": "unknown", "flag": True},
                "2": "unreadable",
            },
        },
        executor_replies.AttemptReply,
    )
    assert partial.ok and partial.memory == {"reserved_bytes": 3}
    assert partial.rank_memory == {"1": {"reserved_bytes": 17}}


def test_device_room_preserves_independent_counters_and_the_credential_boundary() -> None:
    seen: list[Request] = []

    def handle(request: Request) -> Answer:
        seen.append(request)
        return Answer(ok=True)

    frame = {
        "kind": "device_room",
        "seq": 4,
        "free_bytes": 100,
        "rank_memory": {"1": {"reserved_bytes": 17, "future": {"unfamiliar": True}}},
    }
    answer, descriptor = respond(frame, handle)
    assert answer["ok"] and descriptor is None
    assert isinstance(seen[0], DeviceRoom)
    assert not seen[0].measured
    assert seen[0].rank_memory == {"1": {"reserved_bytes": 17}}
    with pytest.raises(SeamError, match="credential_at_seam"):
        respond({**frame, "rank_memory": {"1": {"future": {"token": "forbidden"}}}}, handle)
    assert len(seen) == 1


def test_ledger_reports_planner_room_without_counting_cache_as_allocatable() -> None:
    ledger = Ledger()
    ledger.observe_probe(
        {
            "allocator_bytes": 20,
            "reserved_bytes": 90,
            "device_free_bytes": 40,
            "memory": {
                "reserve_room_bytes": -3,
                "reusable_cache_bytes": 70,
                "overhead_bytes": 5,
                "future": {"unfamiliar": True},
            },
        }
    )
    counters = {counter.name: counter for counter in ledger.counters()}
    assert ledger.allocatable == -3 and ledger.out_of_allocator == 5
    assert counters["device_allocatable"].kind == "measured"
    assert counters["device_reusable_cache"].bytes == 70
    ledger.observe_probe({"allocator_bytes": 20, "reserved_bytes": 90, "device_free_bytes": 40})
    assert ledger.allocatable == -1 and ledger.out_of_allocator == -1
    assert {counter.name: counter for counter in ledger.counters()}[
        "device_allocatable"
    ].kind == "unreadable"
    ledger.device_unreadable("executor exited")
    assert ledger.device_reserved == ledger.device_process == -1


_REPORTER = """
import socket, sys
from cozy_runtime.internal.seam import Channel
fd, rank = map(int, sys.argv[1:])
channel = Channel(socket.socket(fileno=fd))
while (command := channel.recv()) is not None:
    if command['cmd'] == 'shutdown':
        break
    channel.send({'reply': command['cmd'], 'ok': True,
                  'memory': {'reserved_bytes': rank * 17, 'overhead_bytes': rank}})
"""


def test_rank_control_replies_retain_each_process_report(tmp_path: Path) -> None:
    def launch(argv: list[str], fd: int) -> subprocess.Popen[bytes]:
        rank = argv[argv.index("--rank") + 1]
        return subprocess.Popen(
            [sys.executable, "-c", _REPORTER, str(fd), rank],
            pass_fds=(fd,),
        )

    group = RankGroup(degree=3, backend="cpu:gloo", root=str(tmp_path), launch=launch)
    try:
        group.spawn()
        replies = group.call({"cmd": "probe"})
        assert len(replies) == 2
        assert group.rank_memory == {
            "1": {"reserved_bytes": 17, "overhead_bytes": 1},
            "2": {"reserved_bytes": 34, "overhead_bytes": 2},
        }
    finally:
        group.close()


def test_old_worker_refuses_start_before_package_or_cuda_initialization(tmp_path: Path) -> None:
    ours, theirs = socket.socketpair()
    executor = Executor(Channel(ours), tmp_path)
    try:
        # No readable package or torch is needed: capability refusal comes first.
        result = executor.start(Start(package_interface="/does/not/exist"))
        assert result["code"] == "worker_memory_upgrade_required"
        assert executor.torch is None and not executor.poisoned
    finally:
        ours.close()
        theirs.close()


def test_residency_restore_precedes_fresh_attempt_grant() -> None:
    from types import SimpleNamespace

    class Ledger:
        evicted = {"weights": 1}

        def observe_restore(self, _reply: object, _why: str) -> None:
            pass

        def observe_devices(self, _facts: object) -> None:
            pass

    class Row:
        peak: dict[tuple[str, str], int] = {}
        supervision = None
        ledger = Ledger()

        def executor(self) -> object:
            return executor

    class Lane:
        ordinals = (0,)
        lane_id = "lane"
        rows = {"placement": Row()}

        def row(self, _slot: str) -> Row:
            return self.rows["placement"]

        def touch(self, _slot: str) -> None:
            pass

        def measure(self, _kind: str) -> dict[int, object]:
            return {}

        def device_facts(self, _facts: object) -> dict[int, object]:
            return {}

    events: list[str] = []

    class ExecutorStub:
        hello = {"memory": ()}

        @contextlib.contextmanager
        def watched(self, _name: str):
            yield

        def call(self, _command: object, **_kwargs: object) -> dict[str, object]:
            events.append("restore")
            return {"ok": True, "residency": {}}

    executor = ExecutorStub()
    lane = Lane()
    declared = SimpleNamespace(weightless=False, entrypoint="generate")
    slot = SimpleNamespace(bindings={"binding": declared}, lane_id="lane")
    attempt = SimpleNamespace(
        spec={"serving": {"entrypoint_binding_digest": "binding"}},
        placement_id="placement",
        prepared=None,
        memory_grants=None,
        memory_grant=None,
    )
    worker = object.__new__(Worker)
    worker.accepted = SimpleNamespace(mode="serving")
    worker.lanes = SimpleNamespace(by_id={"lane": lane})

    class Memory:
        @staticmethod
        def executor_placement(_capabilities: object) -> str:
            return "all_resident"

        def attempt_grants(self, _me: object, _tenants: object, _peak: object, authorize):
            events.append("grant")
            return authorize()

    worker.memory = Memory()
    worker._tenants = lambda _ordinals: []
    worker._grants = lambda _lane, _slot: (17,)
    worker._room = lambda _lane, _slot: None
    worker._observe_memory = lambda *_args: None

    Worker._arbitrate_residency(worker, attempt, slot)
    assert events == ["restore", "grant"]
    assert attempt.memory_grants == (17,)
