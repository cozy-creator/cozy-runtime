"""Real CPU control peers prove preparation precedes commit and abort is reusable."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from cozy_runtime.internal.parallel.group import OpReady, PreparedGroup, RankGroup
from cozy_runtime.internal.parallel.plan import GroupRefusal

_PEER = """
import os, socket, sys, threading
from pathlib import Path
import msgspec
from cozy_runtime.internal.parallel.group import OpReady, PreparedAbort, PreparedPeer
from cozy_runtime.internal.seam import Channel
fd, rank = map(int, sys.argv[1:3])
mode, directory = sys.argv[3:]
sock = socket.socket(fileno=fd)
channel = Channel(sock)
while (command := channel.recv()) is not None:
    if command['cmd'] == 'shutdown':
        break
    operation = command['operation']
    if mode == 'death' and rank == 1:
        os._exit(4)
    failure = mode in ('shortfall', 'input') and rank == 1 and operation == 1
    if mode == 'host_credit':
        failure = not command.get('host_grants')
        if not failure:
            assert command['host_grants'][rank] == 16
    detail = ''
    if failure and mode == 'input':
        try:
            Path(directory, 'missing-input').read_bytes()
        except OSError as exc:
            detail = str(exc)
    status = OpReady(operation=operation, rank=rank, ok=not failure,
        code='device_shortfall' if mode == 'shortfall' else 'group_preparation_failed',
        detail=detail, needed=20 if failure else 0, available=10 if failure else 0,
        host_needed=16 if mode == 'host_credit' else 0)
    if mode == 'host_credit':
        status = msgspec.structs.replace(status, code='host_shortfall' if failure else '')
    if mode == 'wrong_sequence' and rank == 1:
        status = msgspec.structs.replace(status, operation=operation + 1)
    if mode == 'partial_commit' and rank == 2:
        channel.send({'reply': 'op_ready', **msgspec.to_builtins(status)})
        sock.shutdown(socket.SHUT_RD)
        threading.Event().wait()  # killed by existing broken-group teardown
    peer = PreparedPeer(channel, status.operation, rank)
    try:
        peer.ready(status)
    except PreparedAbort:
        channel.send({'reply': 'run', 'ok': True, 'operation': operation, 'aborted': True})
        continue
    # The observable authored operation is impossible before the commit decision.
    with Path(directory, f'entered-{rank}').open('a') as output:
        output.write(str(operation) + '\\n')
    channel.send({'reply': 'run', 'ok': True, 'operation': operation})
"""


def _group(tmp_path: Path, mode: str, degree: int = 3) -> RankGroup:
    def launch(argv: list[str], fd: int) -> subprocess.Popen[bytes]:
        rank = argv[argv.index("--rank") + 1]
        return subprocess.Popen(
            [sys.executable, "-c", _PEER, str(fd), rank, mode, str(tmp_path)],
            pass_fds=(fd,),
        )

    group = RankGroup(degree=degree, backend="cpu:gloo", root=str(tmp_path), launch=launch)
    group.spawn()
    return group


@pytest.mark.parametrize("mode", ["shortfall", "input"])
def test_one_rank_refuses_before_any_body_and_the_group_can_run_again(
    tmp_path: Path, mode: str
) -> None:
    group = _group(tmp_path, mode)
    try:
        with group.hold(range(1, group.degree)):
            first = PreparedGroup(group, {"cmd": "run"})
            assert not first.ready(OpReady(operation=first.operation, rank=0, ok=True))
            assert not list(tmp_path.glob("entered-*"))
            first.abort()
            assert first.aborted and not group.broken
            second = PreparedGroup(group, {"cmd": "run"})
            assert second.ready(OpReady(operation=second.operation, rank=0, ok=True))
            assert all(reply["ok"] for reply in group.collect("run"))
            assert [path.read_text() for path in sorted(tmp_path.glob("entered-*"))] == [
                "2\n",
                "2\n",
            ]
    finally:
        group.close()


def test_a_leader_failure_before_its_gate_aborts_waiting_followers(tmp_path: Path) -> None:
    group = _group(tmp_path, "ready")
    try:
        transaction = PreparedGroup(group, {"cmd": "run"})
        transaction.abort()
        assert transaction.aborted and not group.broken
        assert not list(tmp_path.glob("entered-*"))
    finally:
        group.close()


@pytest.mark.parametrize("mode", ["death", "wrong_sequence", "partial_commit"])
def test_lost_or_divergent_rank_breaks_the_group(tmp_path: Path, mode: str) -> None:
    group = _group(tmp_path, mode)
    try:
        transaction = PreparedGroup(group, {"cmd": "run"})
        with pytest.raises(GroupRefusal):
            transaction.ready(OpReady(operation=transaction.operation, rank=0, ok=True))
        assert group.broken
        if mode == "partial_commit":
            assert transaction.committed, "a partial commit must never take reusable-abort recovery"
        else:
            assert not transaction.committed and not list(tmp_path.glob("entered-*"))
    finally:
        group.close()
    assert all(follower.exit_status() is not None for follower in group.followers)


@pytest.mark.skipif(
    not os.environ.get("COZY_CPU_TEST_PYTHON"), reason="CPU Torch interpreter not selected"
)
def test_actual_executor_and_gloo_prepare_before_model_hooks(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "testdata" / "group_preparation.py"
    done = subprocess.run(
        [os.environ["COZY_CPU_TEST_PYTHON"], str(fixture), "--root", str(tmp_path)],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=90,  # fixture deadlock bound, not product policy
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_mirrored_host_abort_reserves_once_and_commits_only_with_rank_credits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from cozy_runtime.author._executor_requests import Answer
    from cozy_runtime.internal import memory, tiers
    from cozy_runtime.internal.executor import Executor
    from cozy_runtime.internal.parallel.mirror import mirror_component

    torch = pytest.importorskip("torch")
    group = _group(tmp_path, "host_credit")
    group.memory_capabilities = {rank: ("prepare_commit", "group_host_credit") for rank in (1, 2)}
    group.memory_grants = (100, 100, 100)
    policy = memory.Machine({}, lambda _: None, lambda _: None, host_available=lambda: 48)
    events = []

    def durable(request: Any, _: Any) -> Any:
        events.append("release" if request.release else "reserve")
        if request.release:
            # All run replies must have been consumed before credits return.
            assert all(follower.replies.empty() for follower in group.followers)
            policy.release_host(request.reservation)
            return Answer(ok=True)
        return Answer(ok=policy.reserve_host(request.reservation, request.nbytes))

    executor = SimpleNamespace(rank=0, world=3, group=group, _durable=durable)
    owner = memory.Planner(torch, torch.device("cpu"), grouped=True)
    root = torch.nn.Identity()
    owner.adopt("generation", "model", root, None, tiers.HOST)
    monkeypatch.setattr(owner, "host_needed", lambda: 16)
    model = SimpleNamespace(_cozy_active=("denoise", ("model",)))
    entered = []
    root.register_forward_hook(lambda *_: entered.append("leader"))
    mirror_component(
        root,
        name=("generation", "model"),
        prepared={("generation", "model"): (model, root)},
        group=group,
        spool=lambda: tmp_path,
        planner=owner,
        host_room=lambda statuses: Executor._group_host_room(cast(Executor, executor), statuses),
    )
    try:
        value = torch.ones(1)
        assert root(value) is value
        assert group.operation_sequence == 2
        assert entered == ["leader"]
        assert events == ["reserve", "release"] and not policy.host_reserved
        assert [p.read_text() for p in sorted(tmp_path.glob("entered-*"))] == ["2\n", "2\n"]
    finally:
        group.close()
