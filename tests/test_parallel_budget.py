"""Real rank control sockets agree admission without starting a CUDA collective."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal.parallel.group import RankGroup, receive_resident_budget
from cozy_runtime.internal.parallel.plan import GroupRefusal, RankDivergence
from cozy_runtime.internal.seam import Channel

_FOLLOWER = """
import socket, sys
from cozy_runtime.internal.parallel.group import receive_resident_budget
from cozy_runtime.internal.seam import Channel
fd, rank, degree = map(int, sys.argv[1:4])
mode = sys.argv[4]
channel = Channel(socket.socket(fileno=fd))
for model in ('base', 'overlay'):
    if mode == 'early_failure' and rank == 1:
        channel.send({'reply': 'prepare', 'ok': False, 'code': 'device_shortfall'})
        channel.recv()
        break
    if mode == 'wrong_model' and rank == 1:
        channel.send({'reply': 'resident_budget', 'rank': rank, 'model': 'wrong', 'ceiling': 1})
        channel.recv()
        break
    offset = 100 if model == 'overlay' else 0
    ceiling = 98091614208 - rank * 33554432 - offset
    if mode == 'zero' and rank == degree - 1:
        ceiling = 0
    actual = receive_resident_budget(channel, rank=rank, model=model, ceiling=ceiling)
    expected = 0 if mode == 'zero' else 98091614208 - (degree-1) * 33554432 - offset
    assert actual == expected, (actual, expected)
    channel.send({'reply': 'filled', 'ceiling': actual, 'model': model})
channel.recv()
"""


@pytest.mark.parametrize("degree", [2, 4])
@pytest.mark.parametrize("mode", ["unequal", "zero", "early_failure", "wrong_model"])
def test_budget_agreement_over_real_follower_processes(
    tmp_path: Path, degree: int, mode: str
) -> None:
    def launch(argv: list[str], fd: int) -> subprocess.Popen[bytes]:
        rank = int(argv[argv.index("--rank") + 1])
        return subprocess.Popen(
            [sys.executable, "-c", _FOLLOWER, str(fd), str(rank), str(degree), mode],
            pass_fds=(fd,),
            env=os.environ.copy(),
        )

    group = RankGroup(degree=degree, backend="cpu:gloo", root=str(tmp_path), launch=launch)
    try:
        group.spawn()
        if mode in ("early_failure", "wrong_model"):
            with pytest.raises(GroupRefusal):
                group.agree_resident_budget(model="base", ceiling=98091614208)
            return
        for model, offset in [("base", 0), ("overlay", 100)]:
            agreed = group.agree_resident_budget(model=model, ceiling=98091614208 - offset)
            expected = 0 if mode == "zero" else 98091614208 - (degree - 1) * 33554432 - offset
            assert agreed == expected
            replies = group.collect("filled")
            assert all(
                reply["ceiling"] == expected and reply["model"] == model for reply in replies
            )
    finally:
        group.close()
    assert all(follower.exit_status() is not None for follower in group.followers)


@pytest.mark.parametrize(
    "decision",
    [
        {"cmd": "resident_budget", "model": "base", "ceiling": 101},
        {"cmd": "resident_budget", "model": "base", "ceiling": -1},
        {"cmd": "resident_budget", "model": "base", "ceiling": True},
        {"cmd": "resident_budget", "model": "overlay", "ceiling": 50},
        {"cmd": "prepare", "model": "base", "ceiling": 50},
        {"cmd": "resident_budget", "model": "base", "ceiling": 50, "extra": 1},
    ],
)
def test_follower_refuses_invalid_or_larger_decision(decision: dict[str, object]) -> None:
    ours, theirs = socket.socketpair()
    follower, leader = Channel(ours), Channel(theirs)
    try:
        leader.send(decision)
        with pytest.raises(RankDivergence, match="invalid group residence ceiling"):
            receive_resident_budget(follower, rank=1, model="base", ceiling=100)
        offered = leader.recv()
        assert offered is not None and offered["ceiling"] == 100
    finally:
        follower.close()
        leader.close()
