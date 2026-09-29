"""Real rank control sockets agree the plan without a CUDA collective; where each rank's
weights live is that rank's own."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal.parallel.group import RankGroup, offer_plan
from cozy_runtime.internal.parallel.plan import GpuDivergence, GroupPlan, GroupRefusal
from cozy_runtime.internal.seam import Channel

PLAN = GroupPlan(2, ("0", "1"), "sha256:" + "a" * 64, 98091614208).document()

_FOLLOWER = """
import json, socket, sys
from cozy_runtime.internal.parallel.group import offer_plan
from cozy_runtime.internal.seam import Channel
fd, rank, degree = map(int, sys.argv[1:4])
mode = sys.argv[4]
plan = json.loads(sys.argv[5])
channel = Channel(socket.socket(fileno=fd))
for model in ('base', 'overlay'):
    if mode == 'early_failure' and rank == 1:
        channel.send({'reply': 'load', 'ok': False, 'code': 'device_shortfall'})
        channel.recv()
        break
    if mode == 'wrong_model' and rank == 1:
        channel.send({'reply': 'plan_offer', 'rank': rank, 'model': 'wrong', 'plan': plan})
        channel.recv()
        break
    mine = dict(plan)
    if mode == 'divergent' and rank == degree - 1:
        mine['plan_digest'] = 'sha256:' + 'b' * 64
    offer_plan(channel, rank=rank, model=model, plan=mine)
    channel.send({'reply': 'filled', 'model': model})
channel.recv()
"""


@pytest.mark.parametrize("degree", [2, 4])
@pytest.mark.parametrize("mode", ["agree", "early_failure", "wrong_model", "divergent"])
def test_plan_agreement_over_real_follower_processes(
    tmp_path: Path, degree: int, mode: str
) -> None:
    def launch(argv: list[str], fd: int) -> subprocess.Popen[bytes]:
        rank = int(argv[argv.index("--rank") + 1])
        return subprocess.Popen(
            [sys.executable, "-c", _FOLLOWER, str(fd), str(rank), str(degree), mode, json.dumps(PLAN)],
            pass_fds=(fd,),
            env=os.environ.copy(),
        )

    group = RankGroup(degree=degree, backend="cpu:gloo", root=str(tmp_path), launch=launch)
    try:
        group.spawn()
        if mode != "agree":
            with pytest.raises(GroupRefusal) as refused:
                group.agree_plan(model="base", plan=PLAN)
            if mode == "early_failure":
                assert "device_shortfall" in str(refused.value)
            if mode == "divergent":
                assert isinstance(refused.value, GpuDivergence)
                assert "plan_digest" in str(refused.value)
            return
        for model in ("base", "overlay"):
            group.agree_plan(model=model, plan=PLAN)
            assert [reply["model"] for reply in group.collect("filled")] == [model] * (degree - 1)
    finally:
        group.close()
    assert all(follower.exit_status() is not None for follower in group.followers)


@pytest.mark.parametrize(
    "verdict",
    [
        {"cmd": "plan_agreed", "model": "overlay"},
        {"cmd": "load", "model": "base"},
        {"cmd": "plan_agreed"},
    ],
)
def test_a_follower_fills_only_on_rank_0s_agreement_for_its_own_model(
    verdict: dict[str, object],
) -> None:
    ours, theirs = socket.socketpair()
    follower, leader = Channel(ours), Channel(theirs)
    try:
        leader.send(verdict)
        with pytest.raises(GpuDivergence, match="^gpu_divergence: GPU 1 disagrees .*plan"):
            offer_plan(follower, rank=1, model="base", plan=PLAN)
        offered = leader.recv()
        assert offered is not None and offered["plan"] == PLAN and offered["rank"] == 1
    finally:
        follower.close()
        leader.close()


_JOINER = """
import socket, sys
import torch
from cozy_runtime.internal.parallel.group import RankSpec, join_rank
from cozy_runtime.internal.seam import Channel
fd, rank, degree = map(int, sys.argv[1:4])
channel = Channel(socket.socket(fileno=fd))
join = channel.recv()
_, agree = join_rank(torch, RankSpec(rank, degree, int(join["port"]), "gloo"), {})
channel.send({"reply": "join", "ok": True})
channel.send({"reply": "agreed", "worst": agree(rank * 1000)})
channel.recv()
"""


@pytest.mark.parametrize("degree", [2, 4])
def test_every_gpu_agrees_on_the_largest_shortfall_without_a_device(
    tmp_path: Path, degree: int
) -> None:
    """N4: the memory agreement runs over gloo beside the device communicator, so a card
    that is out of memory never has to allocate to say so."""
    torch = pytest.importorskip("torch")

    def launch(argv: list[str], fd: int) -> subprocess.Popen[bytes]:
        rank = int(argv[argv.index("--rank") + 1])
        return subprocess.Popen(
            [sys.executable, "-c", _JOINER, str(fd), str(rank), str(degree)],
            pass_fds=(fd,),
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )

    group = RankGroup(degree=degree, backend="gloo", root=str(tmp_path), launch=launch)
    try:
        group.spawn()
        group.form(torch, {})
        assert group.agree is not None
        assert group.agree(0) == (degree - 1) * 1000
        assert [reply["worst"] for reply in group.collect("agreed")] == [
            (degree - 1) * 1000
        ] * (degree - 1)
    finally:
        group.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
