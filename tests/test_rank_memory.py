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
from typing import Any

import msgspec
import pytest

torch = pytest.importorskip("torch")

from tensorfs.varena.torch import Copy, Eviction, Memory, Step  # noqa: E402

from cozy_runtime.author import App, Context  # noqa: E402
from cozy_runtime.author._errors import classify  # noqa: E402
from cozy_runtime.author._invoke import Invocation, prepare, run_prepared  # noqa: E402
from cozy_runtime.internal import request_memory, spawn  # noqa: E402
from cozy_runtime.internal.executor import Executor, serve  # noqa: E402
from cozy_runtime.internal.executor_commands import RequestMemory  # noqa: E402
from cozy_runtime.internal.executor_replies import Copy as ReplyCopy  # noqa: E402
from cozy_runtime.internal.executor_replies import Memory as ReplyMemory  # noqa: E402
from cozy_runtime.internal.executor_replies import RankMemory  # noqa: E402
from cozy_runtime.internal.parallel.group import RankGroup  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402


class Empty(msgspec.Struct):
    pass


def test_unmeasured_copy_ceiling_stays_null_on_the_wire() -> None:
    copy = ReplyCopy(kind="disk", gpu=-1, bytes=16, seconds=0.1, gbps=0.0, ceiling=0.0, pct=None)
    memory = ReplyMemory(copies=[copy])
    decoded = msgspec.json.decode(msgspec.json.encode(memory), type=ReplyMemory)
    assert decoded.copies[0].pct is None
    combined = request_memory.combined([decoded])
    assert combined is not None and combined.copies[0].pct is None


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
    peer: Channel, phase: str, request_id: str = "request", steps: tuple[tuple[str, ...], ...] = (),
    *, failed: bool = False,
) -> RankMemory:
    peer.send(
        {
            "cmd": "request_memory",
            "construction": "construction",
            "request_id": request_id,
            "phase": phase,
            "steps": steps,
            "failed": failed,
        }
    )
    return msgspec.convert(peer.recv(), RankMemory, strict=True)


def test_answer_path_preserves_request_boundaries(tmp_path: Path) -> None:
    with follower(tmp_path) as (worker, peer):
        for request_id in ("first", "second"):
            assert command(peer, "begin", request_id).ok
            assert worker.memory is not None and len(worker.memory.active) == 1
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
                and not worker.memory_requests
                and worker._rank_request is None
            )


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
        assert command(peer, "begin", "next").ok
        assert command(peer, "end", "next").ok


def test_weightless_follower_request_returns_no_gpu_evidence(tmp_path: Path) -> None:
    with follower(tmp_path) as (worker, peer):
        worker.memory = None
        assert command(peer, "begin").ok
        assert command(peer, "plan", steps=(("component",),)).ok
        ended = command(peer, "end")
        assert ended.ok and ended.memory is None and worker._rank_request is None


def test_failed_end_marks_actual_follower_request_and_next_request_stays_healthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorfs.varena.torch import Request

    marked: list[Request] = []
    mark_failed = Request.mark_failed

    def observe(request: Request) -> None:
        assert request in request.memory.active
        marked.append(request)
        mark_failed(request)

    monkeypatch.setattr(Request, "mark_failed", observe)
    with follower(tmp_path) as (worker, peer):
        assert command(peer, "begin", "failed").ok
        assert command(peer, "end", "failed", failed=True).ok
        assert [request.id for request in marked] == ["failed"]
        assert marked[0].stats is not None and not marked[0].cleanup_errors
        assert command(peer, "begin", "healthy").ok
        assert command(peer, "end", "healthy").ok
        assert [request.id for request in marked] == ["failed"]
        assert worker.memory is not None and not worker.memory.active
        assert not worker.memory_requests and worker._rank_request is None


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
        assert not executor.memory.active and not executor.memory_requests
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
        proof_fields: dict[str, Any] = {}
        if hasattr(source, "proofs"):
            proof_fields["proofs"] = (
                {
                    "request": "cpu",
                    "gpu": ordinal,
                    "version": 1,
                    "complete": True,
                    "isolated": True,
                    "capacity": 16,
                    "pages": [{"page": 1, "content": "fingerprint/0", "payload": 3, "mapped": 8}],
                    "initial": [1],
                    "invalid": [],
                },
            )
        facts = dataclasses.replace(
            source,
            steps=(step,),
            leftover=((ordinal, ordinal + 2),),
            copied=ordinal + 1,
            retries=ordinal,
            copies=copies,
            evictions=(Eviction("transformer", ordinal, 2, "room", 4),),
            **proof_fields,
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
    if hasattr(source, "proofs"):
        assert [row.gpu for row in result.proofs] == [2, 3]
        assert result.proofs[0].pages[0].payload == 3
        assert result.proofs[0].pages[0].mapped == 8
    else:
        assert not result.proofs, "older TensorFS cannot provide proof evidence"
    assert result.steps == {"transformer": [3]} and result.paged_in == 3 and result.retries == 1
    assert result.plan == [["transformer"]]
    with pytest.raises(ValueError, match="duplicate physical"):
        request_memory.combined([records[0], records[0]])
    with pytest.raises(ValueError, match="different request plans"):
        request_memory.combined([records[0], msgspec.structs.replace(records[1], plan=[])])
    with pytest.raises(ValueError, match="unmapped CUDA"):
        request_memory.measured(facts, [])


@pytest.mark.parametrize("terminal", ["failed", "canceled"])
def test_group_forwards_plans_and_closes_every_request_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: str,
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
    leader.memory = Memory.open(torch.device("cpu"))
    leader.group = group
    commands: list[RequestMemory] = []
    send = leader._rank_memory_command
    cleanup_failure: ValueError | None = None

    def observe(command: RequestMemory) -> list[RankMemory]:
        commands.append(command)
        replies = send(command)
        if command.phase == "end" and cleanup_failure is not None:
            raise cleanup_failure
        return replies

    monkeypatch.setattr(leader, "_rank_memory_command", observe)
    app = App()
    cancelled = [False]

    @app.entrypoint(accelerator=False)
    def planned(ctx: Context, payload: Empty) -> Empty:
        ctx.plan("old")
        ctx.plan({"right", "left"}, "last")
        if terminal == "canceled":
            cancelled[0] = True
            ctx.raise_if_cancelled()
        raise ValueError("request failure")

    prepared = prepare(app.get("planned"), {})
    outcome = classify(None)

    def outcome_failed() -> bool:
        return outcome.terminal != "succeeded"

    group.spawn()
    try:
        for request_id in ("first", "after-failure"):
            outcome = classify(None)
            cancelled[0] = False
            with (
                leader._request(request_id) as request,
                leader._ranks_request(request_id, failed=outcome_failed) as replies,
            ):
                _, outcome, _ = run_prepared(
                    prepared,
                    Invocation(
                        request_id,
                        tmp_path / request_id,
                        time.monotonic() + 30,
                        plan=functools.partial(leader._plan_request, request_id),
                        cancel=lambda: cancelled[0],
                    ),
                )
                assert outcome.terminal == terminal
                if terminal == "failed":
                    assert "request failure" in outcome.message
                assert request is not None
                request.mark_failed()
            assert request is not None and request.stats is not None
            assert len(replies) == 1 and replies[0].plan == [["left", "right"], ["last"]]
            assert not leader.memory.active and not leader.memory_requests
            own = request_memory.measured(request.stats, [])
            assert own is not None and request_memory.combined([own, *replies]) is not None
            assert commands[-1].phase == "end" and commands[-1].failed

        original = RuntimeError("body failure must stay primary")
        cleanup_failure = ValueError("rank cleanup transport failed after closing")
        with pytest.raises(RuntimeError) as caught:
            with leader._request("escaping"), leader._ranks_request("escaping"):
                raise original
        assert caught.value is original
        assert original.__notes__ == [
            "rank memory cleanup failed: rank cleanup transport failed after closing"
        ]
        assert commands[-1].phase == "end" and commands[-1].failed
        assert not leader.memory.active and not leader.memory_requests
    finally:
        group.close()
        left.close()
        right.close()
