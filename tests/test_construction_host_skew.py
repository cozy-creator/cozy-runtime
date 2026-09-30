"""One leader's construction vector supports old/new Worker capabilities, CPU only."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from cozy_runtime.author._errors import RuntimeFailure
from cozy_runtime.author._executor_requests import Answer
from cozy_runtime.internal import memory, proctree
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.parallel.group import RankGroup
from cozy_runtime.internal.parallel.plan import GroupPlan, GroupRefusal
from cozy_runtime.internal.seam import Channel

torch = pytest.importorskip("torch")
PLAN = GroupPlan(2, ("0", "1"), "sha256:" + "a" * 64, 1000).document()
FOLLOWER = """
import json,socket,sys
from cozy_runtime.internal.parallel.group import offer_plan
from cozy_runtime.internal.seam import Channel
channel=Channel(socket.socket(fileno=int(sys.argv[1])))
credit=offer_plan(channel,rank=1,model='metadata',plan=json.loads(sys.argv[2]),staging=(16,16))
backing=bytearray(credit)
channel.send({'reply':'filled','ok':True,'credit':credit,'materialized':len(backing)})
channel.recv()
"""


@pytest.mark.parametrize("capable", [False, True])
@pytest.mark.parametrize(
    "outcome", ["success", "refusal", "leader_abort", "drain_failure", "unproved_reap"]
)
def test_construct_metadata_only_with_one_coherent_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capable: bool, outcome: str
) -> None:
    available = 31 if outcome == "refusal" else 32
    monkeypatch.setattr(proctree, "available_host_bytes", lambda: available)
    owner = memory.Planner(torch, torch.device("cpu"), grouped=True)
    machine = memory.Machine({}, lambda _: None, lambda _: None, host_available=lambda: available)
    left, right = socket.socketpair()
    leader = Executor(Channel(left), tmp_path, world=2)
    leader.planner = owner
    leader.worker_memory = frozenset({"host_room"}) if capable else frozenset()
    requests = []
    events = []

    def durable(request: Any, kind: Any) -> Answer:
        assert capable, "old Worker must not receive unsupported HostRoom"
        requests.append(request)
        if request.release:
            events.append("release")
            machine.release_host(request.reservation)
            return Answer(ok=True)
        return Answer(ok=machine.reserve_host(request.reservation, request.nbytes))

    monkeypatch.setattr(leader, "_durable", durable)

    def launch(argv: list[str], fd: int) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [sys.executable, "-c", FOLLOWER, str(fd), json.dumps(PLAN)],
            pass_fds=(fd,),
            env=os.environ.copy(),
        )

    group = RankGroup(degree=2, backend="cpu:gloo", root=str(tmp_path), launch=launch)
    leader.group = group
    actual_close = group.close

    def close() -> None:
        if outcome != "unproved_reap":
            actual_close()
        events.append("reap")

    monkeypatch.setattr(group, "close", close)
    original_release = owner.release_local_staging

    def release(nbytes: int) -> None:
        events.append("release")
        original_release(nbytes)

    monkeypatch.setattr(owner, "release_local_staging", release)

    def body(*_: Any, **__: Any) -> dict[str, Any]:
        try:
            credit = group.agree_plan(
                model="metadata",
                plan=PLAN,
                staging=(16, 16),
                reserve_staging=leader._reserve_construction_staging,
            )
        except GroupRefusal as exc:
            return leader._group_refusal(exc)
        owner.construction_staging_credit = credit
        assert credit == 16
        assert (sum(machine.host_reserved.values()) if capable else owner._host_pending) == 32
        backing = bytearray(credit)
        if outcome in ("leader_abort", "unproved_reap"):
            raise ValueError("leader aborted")
        if outcome == "drain_failure":
            owner.unusable = "drain failed"
            raise RuntimeFailure("drain failed", code="device_fault")
        [reply] = group.collect("filled")
        assert reply["materialized"] == len(backing) == 16
        leader._construction_staging_drained = True
        leader._release_construction_staging()
        assert owner.__dict__["construction_staging_credit"] is None
        return {"ok": True, "metadata_only": True}

    monkeypatch.setattr(leader, "_prepare_first_body", body)
    try:
        group.spawn()
        if outcome in ("leader_abort", "unproved_reap"):
            with pytest.raises(ValueError):
                leader._prepare_first(cast(Any, None))
        elif outcome == "drain_failure":
            with pytest.raises(RuntimeFailure):
                leader._prepare_first(cast(Any, None))
        else:
            assert bool(leader._prepare_first(cast(Any, None))["ok"]) == (outcome == "success")
        retained = sum(machine.host_reserved.values()) if capable else owner._host_pending
        assert retained == (32 if outcome in ("drain_failure", "unproved_reap") else 0)
        assert not requests if not capable else requests
        if outcome == "leader_abort":
            assert events == ["reap", "release"]
        if outcome in ("drain_failure", "unproved_reap"):
            assert "release" not in events
    finally:
        actual_close()
        left.close()
        right.close()


def test_local_vector_accounts_existing_pending_and_validates_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proctree, "available_host_bytes", lambda: 48)
    owner = memory.Planner(torch, torch.device("cpu"), grouped=True)
    owner._host_pending = 16
    credits, charged = owner.reserve_local_staging(((16, 16, 1), (16, 16, 1)))
    assert credits == (16, 16) and charged == 32 and owner._host_pending == 48
    assert owner.reserve_local_staging(((16, 16, 1), (16, 16, 1))) == ((), 0)
    owner.release_local_staging(charged)
    assert owner._host_pending == 16
    with pytest.raises(ValueError):
        owner.release_local_staging(17)
    assert owner._host_pending == 16
