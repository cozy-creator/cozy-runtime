"""Actual CPU Memory requests and executor sockets; synthetic GPU facts test identity conversion."""

from __future__ import annotations

import dataclasses
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

from tensorfs.varena.torch import Copy, Eviction, Memory, Step  # noqa: E402

from cozy_runtime.author import App, Context  # noqa: E402
from cozy_runtime.author._invoke import Invocation, prepare, run_prepared  # noqa: E402
from cozy_runtime.internal import executor_commands, request_memory, spawn  # noqa: E402
from cozy_runtime.internal.executor import Executor, serve  # noqa: E402
from cozy_runtime.internal.executor_replies import Memory as ReplyMemory  # noqa: E402
from cozy_runtime.internal.executor_replies import RankMemory  # noqa: E402
from cozy_runtime.internal.parallel.group import RankGroup  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402


class Empty(msgspec.Struct):
    pass


@contextmanager
def follower(tmp_path: Path) -> Iterator[tuple[Executor, Channel]]:
    left, right = socket.socketpair()
    worker, peer = Executor(Channel(left), tmp_path, rank=1, world=2), Channel(right)
    worker.memory = Memory.open(torch.device("cpu"))
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


def test_answer_path_preserves_keep_weights_and_request_boundaries(tmp_path: Path) -> None:
    with follower(tmp_path) as (worker, peer):
        peer.send(
            executor_commands.encode(executor_commands.KeepWeights(construction="construction"))
        )
        assert peer.recv() == {"reply": "keep_weights", "ok": True}
        for request_id in ("first", "second"):
            assert command(peer, "begin", request_id).ok
            assert worker.memory is not None and len(worker.memory.active) == 1
            peer.send(executor_commands.encode(executor_commands.KeepWeights(construction="construction")))
            assert peer.recv() == {
                "reply": "keep_weights", "ok": False, "code": "request_memory_busy"
            }
            assert command(peer, "plan", request_id, (("old",),)).ok
            assert command(peer, "plan", request_id, (("right", "left"), ("last",))).ok
            assert not command(peer, "end", "another-request").ok
            assert len(worker.memory.active) == 1
            reply = command(peer, "end", request_id)
            assert reply.ok and reply.rank == 1 and reply.request_id == request_id
            assert reply.memory is not None and reply.memory.plan == [["left", "right"], ["last"]]
            assert not reply.memory.leftover_by_gpu, "CPU requests cannot supply GPU evidence"
            assert (
                not worker.memory.active
                and not worker.memory.plans
                and worker._rank_request is None
            )
            peer.send(executor_commands.encode(executor_commands.KeepWeights(construction="construction")))
            assert peer.recv() == {"reply": "keep_weights", "ok": True}


def test_failed_mirrored_command_still_closes_its_request(tmp_path: Path) -> None:
    with follower(tmp_path) as (worker, peer):
        assert command(peer, "begin").ok
        peer.send({"cmd": "run", "component": ["missing", "component"]})
        reply = peer.recv()
        assert reply is not None and reply["ok"] is False
        assert command(peer, "end").ok
        assert (
            worker.memory is not None and not worker.memory.active and worker._rank_request is None
        )
        peer.send(executor_commands.encode(executor_commands.KeepWeights(construction="construction")))
        assert peer.recv() == {"reply": "keep_weights", "ok": True}
        assert command(peer, "begin", "next").ok
        assert command(peer, "end", "next").ok


def test_weightless_follower_request_returns_no_gpu_evidence(tmp_path: Path) -> None:
    with follower(tmp_path) as (worker, peer):
        worker.memory = None
        assert command(peer, "begin").ok
        assert command(peer, "plan", steps=(("component",),)).ok
        ended = command(peer, "end")
        assert ended.ok and ended.memory is None and worker._rank_request is None


def test_single_rank_keeps_its_local_request_plan_and_reply(tmp_path: Path) -> None:
    left, right = socket.socketpair()
    executor = Executor(Channel(left), tmp_path)
    executor.memory = Memory.open(torch.device("cpu"))
    try:
        with executor._request("single") as request, executor._ranks_request("single") as ranks:
            executor._plan_request("single", ["old"])
            executor._plan_request("single", ["local"])
            assert torch.arange(4).sum().item() == 6
        assert request is not None and request.stats is not None and not ranks
        reply = msgspec.convert(executor._memory(request.stats, ranks), ReplyMemory, strict=True)
        assert reply.plan == [["local"]] and reply.leftover == 0 and not reply.leftover_by_gpu
        assert not executor.memory.active and not executor.memory.plans
    finally:
        left.close()
        right.close()


def test_real_cpu_context_stats_map_logical_two_gpu_facts_to_lane_two_three() -> None:
    memory = Memory.open(torch.device("cpu"))
    with memory.request("cpu") as request:
        request.plan(["transformer"])
        value = torch.arange(8).sum()
        assert value.item() == 28
    assert request.stats is not None
    source = request.stats
    records = []
    for ordinal in (0, 1):
        step = Step("transformer", ordinal, 1, 10, 10, ordinal + 1, 0, 0, 0)
        copies = tuple(
            Copy(
                kind=kind,
                gpu=gpu,
                bytes=2,
                seconds=1,
                gbps=2e-9,
                ceiling=1,
                pct=0,
                link="",
                busy_s=0,
                pin_s=0,
                staged=0,
                cached=0,
            )
            for kind, gpu in (("h2d", ordinal), ("disk", -1))
        )
        facts = dataclasses.replace(
            source,
            steps=(step,),
            leftover=((ordinal, ordinal + 2),),
            copied=ordinal + 1,
            retries=ordinal,
            copies=copies,
            evictions=(Eviction("transformer", ordinal, 2, "room", 4),),
        )
        measured = request_memory.measured(facts, [2, 3])
        assert measured is not None
        records.append(measured)
    result = request_memory.combined(records)
    assert result is not None
    assert result.leftover_by_gpu == [(2, 2), (3, 3)] and result.leftover == 5
    assert [row.gpu for row in result.rows] == [2, 3]
    assert [row.gpu for row in result.evictions] == [2, 3]
    assert [row.gpu for row in result.copies] == [2, -1, 3, -1]
    assert result.steps == {"transformer": [3]} and result.paged_in == 3 and result.retries == 1
    assert result.plan == [["transformer"]]
    with pytest.raises(ValueError, match="duplicate physical"):
        request_memory.combined([records[0], records[0]])
    with pytest.raises(ValueError, match="different request plans"):
        request_memory.combined([records[0], msgspec.structs.replace(records[1], plan=[])])
    with pytest.raises(ValueError, match="unmapped CUDA"):
        request_memory.measured(facts, [])


def test_group_forwards_plans_and_closes_every_request_after_failure(tmp_path: Path) -> None:
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
    leader.memory = Memory.open(torch.device("cpu"))
    leader.group = group
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
            with (
                leader._request(request_id) as request,
                leader._ranks_request(request_id) as replies,
            ):
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
            assert request is not None and request.stats is not None
            assert len(replies) == 1 and replies[0].plan == [["left", "right"], ["last"]]
            assert not leader.memory.active and not leader.memory.plans
            own = request_memory.measured(request.stats, [])
            assert own is not None and request_memory.combined([own, *replies]) is not None
        kept = group.call(
            executor_commands.encode(executor_commands.KeepWeights(construction="construction"))
        )
        assert all(reply.get("ok") is True for reply in kept)
    finally:
        group.close()
        left.close()
        right.close()
