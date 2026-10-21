"""A request's memory on every rank: real CPU memories (varena's host pages) and executor
sockets. Each rank opens its own request beside rank 0's, `ctx.plan` replaces its plan on every
rank, and each rank's record (`measurements.memory.varena`) comes back, failed or not."""

from __future__ import annotations

import functools
import os
import socket
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import msgspec
import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.author import App, Context  # noqa: E402
from cozy_runtime.author._invoke import Invocation, prepare, run_prepared  # noqa: E402
from cozy_runtime.internal import execution, spawn  # noqa: E402
from cozy_runtime.internal.executor import Executor, serve  # noqa: E402
from cozy_runtime.internal.executor_replies import RankMemory  # noqa: E402
from cozy_runtime.internal.parallel.group import RankGroup  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402
from execution_models import memory  # noqa: E402


class Empty(msgspec.Struct):
    pass


def placed(executor: Executor, store: Path) -> Executor:
    """`executor` with a memory over host pages and the execution layer."""
    executor.memory, executor.execution, executor.store = memory(), execution, store
    return executor


@contextmanager
def follower(tmp_path: Path) -> Iterator[tuple[Executor, Channel]]:
    left, right = socket.socketpair()
    worker = placed(Executor(Channel(left), tmp_path, rank=1, world=2), tmp_path / "store")
    peer = Channel(right)
    worker.constructions["construction"] = worker._construction_child("construction")
    thread = threading.Thread(target=serve, args=(worker, worker.channel))
    thread.start()
    try:
        yield worker, peer
    finally:
        peer.send({"cmd": "shutdown"})
        assert peer.recv() == {"reply": "shutdown", "ok": True}
        thread.join(5)
        assert not thread.is_alive()
        left.close()
        right.close()


def command(
    peer: Channel, phase: str, request_id: str = "request", steps: tuple[tuple[str, ...], ...] = ()
) -> RankMemory:
    peer.send(
        {
            "cmd": "request_memory",
            "construction": "construction",
            "request_id": request_id,
            "phase": phase,
            "steps": steps,
        }
    )
    return msgspec.convert(peer.recv(), RankMemory, strict=True)


def test_a_followers_request_begins_plans_and_ends_with_its_record(tmp_path: Path) -> None:
    with follower(tmp_path) as (worker, peer):
        for request_id in ("first", "second"):
            assert command(peer, "begin", request_id).ok
            assert list(worker.requests) == [request_id]
            assert command(peer, "begin", "another").code == "request_memory_busy"
            assert command(peer, "plan", request_id, (("old",),)).ok
            assert command(peer, "plan", request_id, (("right", "left"), ("last",))).ok
            assert command(peer, "end", "another").code == "request_memory_absent"
            reply = command(peer, "end", request_id)
            assert reply.ok and reply.rank == 1 and reply.request_id == request_id
            assert reply.memory is not None and reply.memory.gpu == -1  # a CPU names no GPU
            assert reply.memory.stats["request"] == request_id
            assert reply.memory.stats["leftover"] == 0
            assert not worker.requests and worker._rank_request is None


def test_a_failed_mirrored_command_still_closes_its_request(tmp_path: Path) -> None:
    with follower(tmp_path) as (worker, peer):
        assert command(peer, "begin").ok
        peer.send({"cmd": "run", "component": ["missing", "component"]})
        reply = peer.recv()
        assert reply is not None and reply["ok"] is False
        assert command(peer, "end").ok
        assert not worker.requests and worker._rank_request is None
        assert command(peer, "begin", "next").ok
        assert command(peer, "end", "next").ok


def test_a_weightless_follower_reports_no_record(tmp_path: Path) -> None:
    with follower(tmp_path) as (worker, peer):
        worker.memory = None
        assert command(peer, "begin").ok
        assert command(peer, "plan", steps=(("component",),)).ok
        ended = command(peer, "end")
        assert ended.ok and ended.memory is None and worker._rank_request is None


def test_one_rank_plans_its_own_request_and_records_it(tmp_path: Path) -> None:
    left, right = socket.socketpair()
    executor = placed(Executor(Channel(left), tmp_path), tmp_path / "store")
    try:
        with executor._request("single") as rows, executor._ranks_request("single") as ranks:
            executor._plan_request("single", ["old"])
            executor._plan_request("single", ["local"])
            assert list(executor.requests) == ["single"]
        assert not ranks and not executor.requests
        assert len(rows) == 1 and rows[0].stats["request"] == "single"
    finally:
        left.close()
        right.close()


def test_a_group_plans_every_rank_and_closes_every_request_after_a_failure(
    tmp_path: Path,
) -> None:
    fixture = Path(__file__).parent / "testdata/request_memory_follower.py"

    def launch(argv: list[str], fd: int) -> spawn.Child:
        return spawn.spawn_follower(
            python=sys.executable,
            module_argv=[str(fixture), *argv[2:]],
            inherit_fd=fd,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )

    group = RankGroup(degree=2, backend="cpu:gloo", root=str(tmp_path), launch=launch)
    left, right = socket.socketpair()
    leader = Executor(Channel(left), tmp_path, world=2, construction="construction")
    placed(leader, tmp_path / "store").group = group
    app = App()

    @app.entrypoint(accelerator=False)
    def planned(ctx: Context, payload: Empty) -> Empty:
        ctx.plan("old")
        ctx.plan({"right", "left"}, "last")
        raise ValueError("request failure")

    prepared = prepare(app.get("planned"), {})
    group.spawn()
    try:
        for request_id in ("first", "after-failure"):
            with leader._request(request_id) as own, leader._ranks_request(request_id) as ranks:
                _, outcome, _ = run_prepared(
                    prepared,
                    Invocation(
                        request_id,
                        tmp_path / request_id,
                        time.monotonic() + 30,
                        plan=functools.partial(leader._plan_request, request_id),
                    ),
                )
                assert outcome.terminal == "failed" and "request failure" in outcome.message
            assert [row.stats["request"] for row in (*own, *ranks)] == [request_id] * 2
            assert not leader.requests
    finally:
        group.close()
        left.close()
        right.close()
