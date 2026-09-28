"""Real rank control sockets agree plan, parking and ceiling without a CUDA collective."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal.parallel.group import RankGroup, receive_residence
from cozy_runtime.internal.parallel.plan import GpuDivergence, GroupPlan, GroupRefusal
from cozy_runtime.internal.seam import Channel

PLAN = GroupPlan(2, ("0", "1"), "sha256:" + "a" * 64, 98091614208).document()

_FOLLOWER = """
import json, socket, sys
from cozy_runtime.internal.parallel.group import receive_residence
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
        channel.send(
            {'reply': 'resident_budget', 'rank': rank, 'model': 'wrong', 'plan': plan,
             'required': 1}
        )
        channel.recv()
        break
    mine = dict(plan)
    if mode == 'divergent' and rank == degree - 1:
        mine['plan_digest'] = 'sha256:' + 'b' * 64
    offset = 100 if model == 'overlay' else 0
    ceiling = 98091614208 - rank * 33554432 - offset
    if mode == 'zero' and rank == degree - 1:
        ceiling = 0
    parked = []
    actual = receive_residence(
        channel, rank=rank, model=model, plan=mine, required=1000 + rank,
        park=parked.extend, ceiling=lambda: ceiling,
    )
    expected = 0 if mode == 'zero' else 98091614208 - (degree-1) * 33554432 - offset
    assert actual == expected, (actual, expected)
    channel.send({'reply': 'filled', 'ceiling': actual, 'model': model, 'parked': parked})
channel.recv()
"""


@pytest.mark.parametrize("degree", [2, 4])
@pytest.mark.parametrize("mode", ["unequal", "zero", "early_failure", "wrong_model", "divergent"])
def test_residence_agreement_over_real_follower_processes(
    tmp_path: Path, degree: int, mode: str
) -> None:
    def launch(argv: list[str], fd: int) -> subprocess.Popen[bytes]:
        rank = int(argv[argv.index("--rank") + 1])
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                _FOLLOWER,
                str(fd),
                str(rank),
                str(degree),
                mode,
                json.dumps(PLAN),
            ],
            pass_fds=(fd,),
            env=os.environ.copy(),
        )

    group = RankGroup(degree=degree, backend="cpu:gloo", root=str(tmp_path), launch=launch)
    needs: list[int] = []

    def park(need: int) -> list[str]:
        needs.append(need)
        return ["sha256:old"]

    try:
        group.spawn()
        if mode in ("early_failure", "wrong_model", "divergent"):
            with pytest.raises(GroupRefusal) as refused:
                group.agree_residence(
                    model="base", plan=PLAN, required=1000, park=park, ceiling=lambda: 1
                )
            if mode == "early_failure":
                assert "device_shortfall" in str(refused.value)
            if mode == "divergent":
                assert isinstance(refused.value, GpuDivergence)
                assert "plan_digest" in str(refused.value)
            assert not needs, "nothing parks before every plan agrees"
            return
        for model, offset in [("base", 0), ("overlay", 100)]:

            def ceiling(offset: int = offset) -> int:
                return 98091614208 - offset

            agreed, parked = group.agree_residence(
                model=model, plan=PLAN, required=1000, park=park, ceiling=ceiling
            )
            expected = 0 if mode == "zero" else 98091614208 - (degree - 1) * 33554432 - offset
            assert agreed == expected and parked == ["sha256:old"]
            # rank 0 parks for the LARGEST need any rank reported
            assert needs[-1] == 1000 + degree - 1
            replies = group.collect("filled")
            assert all(
                reply["ceiling"] == expected
                and reply["model"] == model
                and reply["parked"] == ["sha256:old"]
                for reply in replies
            )
    finally:
        group.close()
    assert all(follower.exit_status() is not None for follower in group.followers)


@pytest.mark.parametrize(
    ("park", "ceiling", "match"),
    [
        ({"cmd": "resident_budget", "model": "base", "park": "x"}, None, "parking"),
        ({"cmd": "resident_budget", "model": "overlay", "park": []}, None, "parking"),
        ({"cmd": "load", "model": "base", "park": []}, None, "parking"),
        (None, {"cmd": "resident_ceiling", "model": "base", "ceiling": 101}, "ceiling"),
        (None, {"cmd": "resident_ceiling", "model": "base", "ceiling": -1}, "ceiling"),
        (None, {"cmd": "resident_ceiling", "model": "base", "ceiling": True}, "ceiling"),
        (None, {"cmd": "resident_ceiling", "model": "overlay", "ceiling": 50}, "ceiling"),
    ],
)
def test_follower_refuses_invalid_or_larger_decision(
    park: dict[str, object] | None, ceiling: dict[str, object] | None, match: str
) -> None:
    ours, theirs = socket.socketpair()
    follower, leader = Channel(ours), Channel(theirs)
    try:
        leader.send(park or {"cmd": "resident_budget", "model": "base", "park": []})
        if ceiling is not None:
            leader.send(ceiling)
        refusal = f"^gpu_divergence: GPU 1 disagrees .*invalid group .*{match}"
        with pytest.raises(GpuDivergence, match=refusal):
            receive_residence(
                follower,
                rank=1,
                model="base",
                plan=PLAN,
                required=7,
                park=lambda keys: None,
                ceiling=lambda: 100,
            )
        offered = leader.recv()
        assert offered is not None and offered["required"] == 7 and offered["plan"] == PLAN
        if ceiling is not None:
            measured = leader.recv()
            assert measured is not None and measured["ceiling"] == 100
    finally:
        follower.close()
        leader.close()
