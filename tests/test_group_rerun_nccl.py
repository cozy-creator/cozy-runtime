"""A GPU group runs a call again when one GPU runs out of memory, over NCCL on two GPUs.

Real follower processes, a real NCCL communicator. GPU 1 runs out of memory BEFORE its
collective, so GPU 0 waits inside it: only aborting the communicator (`RankGroup.interrupt`)
lets GPU 0 return, and the group forms again in the same processes. Every GPU puts back what
the failed attempt changed (its call journal) and the call runs again from its inputs; the
output equals a control's. With the tiny H3 DiT the failure is in its second transformer
block, between Ulysses all-to-alls.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal import executor as executor_module  # noqa: E402
from cozy_runtime.internal import spawn  # noqa: E402
from cozy_runtime.internal.executor import Executor  # noqa: E402
from cozy_runtime.internal.parallel import links  # noqa: E402
from cozy_runtime.internal.parallel.group import RankGroup  # noqa: E402
from cozy_runtime.internal.parallel.mirror import mirror_component  # noqa: E402
from cozy_runtime.internal.parallel.plan import GroupRefusal  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402
from testdata.parallel_calls import collective_model, h3_batch, h3_model, prepared  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or not torch.distributed.is_nccl_available(),
    reason="two CUDA GPUs and NCCL",
)
#: a group lane's seal (pgw#929): NVLS off, peer memory only over NVLink
SEAL = {"NCCL_NVLS_ENABLE": "0", "NCCL_P2P_LEVEL": "NVL"}


def group_of_two(tmp_path: Path, *flags: str) -> RankGroup:
    fixture = Path(__file__).parent / "testdata" / "parallel_calls.py"

    def launch(module_argv: list[str], fd: int) -> spawn.Child:
        return spawn.spawn_follower(
            python=sys.executable,
            module_argv=[str(fixture), *module_argv[2:], "--cuda", *flags],
            inherit_fd=fd,
            env={**os.environ, **SEAL, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )

    return RankGroup(degree=2, backend="nccl", root=str(tmp_path), launch=launch)


def leader_of(group: RankGroup, tmp_path: Path, monkeypatch: Any) -> tuple[Any, Any]:
    for name, value in SEAL.items():
        monkeypatch.setitem(executor_module._SEALED, name, value)
        monkeypatch.setenv(name, value)
    torch.cuda.set_device(0)
    ours, theirs = socket.socketpair()
    leader: Any = Executor(Channel(ours), tmp_path, world=2)
    leader.device_kind = "cuda"
    leader.group, leader.torch = group, torch
    leader._group_model_key = "fixture"
    return leader, (ours, theirs)


def test_a_gpu_out_of_memory_before_its_collective_reruns_the_call(
    tmp_path: Path, monkeypatch: Any
) -> None:
    group = group_of_two(tmp_path)
    leader, sockets = leader_of(group, tmp_path, monkeypatch)
    model: Any = collective_model()
    roots = {"block": model.block, "overlay": model.overlay, "spare": model.spare}
    for root in roots.values():
        root.to("cuda:0")
    try:
        group.spawn()
        first = leader.pg = group.form(torch, SEAL)
        mirror_component(
            model.block,
            name=("fixture", "block"),
            prepared={("fixture", key): (model, root) for key, root in roots.items()},
            group=group,
            spool=lambda: tmp_path / "attempt",
            rerun=leader._group_rerun,
            journaled=leader._call_journal,
        )
        pids = group.pids()
        # Each GPU: (2 + its number + the calls it counted), summed.
        assert float(model.counted()) == 7.0
        # GPU 1 out after its collective: GPU 0 finished its part, both put it back.
        assert float(model.counted(after=1)) == 9.0
        # GPU 1 out BEFORE its collective: GPU 0 waits in it until the abort.
        assert float(model.counted(before=1)) == 11.0
        assert float(model.block.seen) == 3.0, "GPU 0 counted each call once"
        assert group.pids() == pids and not group.broken and group.pg is not first
        assert float(model.counted()) == 13.0, "the group formed again serves on"
    finally:
        group.close()
        for end in sockets:
            end.close()


def test_the_h3_dit_reruns_between_its_all_to_alls_and_matches_a_control(
    tmp_path: Path, monkeypatch: Any
) -> None:
    pytest.importorskip("diffusers")
    group = group_of_two(tmp_path, "--h3", "--oom-at", "2")
    leader, sockets = leader_of(group, tmp_path, monkeypatch)
    model: Any = h3_model()
    model.dit.to("cuda:0")
    model.text_encoder.to("cuda:0")
    batch = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in h3_batch().items()}
    with torch.no_grad():
        alone = model.dit(**batch)
    leader.backend = prepared({"dit": model.dit, "text_encoder": model.text_encoder})
    try:
        group.spawn()
        leader.pg = group.form(torch, SEAL)
        assert leader._install_group(torch, model) is None
        leader._attempt_spool = tmp_path / "attempt"
        control = model.infer(batch)  # GPU 1's second transformer block runs its first time
        rerun = model.infer(batch)  # ... and runs out of memory on its second: the call reruns
        for a, b, c in zip(control, rerun, alone, strict=True):
            assert torch.equal(a, b), "the rerun's output is the control's"
            torch.testing.assert_close(a, c, rtol=1e-4, atol=1e-5)
        assert not group.broken
    finally:
        group.close()
        for end in sockets:
            end.close()


def test_the_h3_dit_forms_over_its_measured_route_and_matches_one_gpu(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The group measures its two GPUs' link in a child process, forms NCCL at the level that
    measurement allows and, where the copy is direct, times NCCL against the copy engines on
    both GPUs (the follower through its own `join`). The tiny H3 DiT's Ulysses output equals
    one GPU's whichever route won, and the execution record says which and why."""
    pytest.importorskip("diffusers")
    group = group_of_two(tmp_path, "--h3")
    leader, sockets = leader_of(group, tmp_path, monkeypatch)
    model: Any = h3_model()
    model.dit.to("cuda:0")
    model.text_encoder.to("cuda:0")
    batch = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in h3_batch().items()}
    with torch.no_grad():
        alone = model.dit(**batch)
    leader.backend = prepared({"dit": model.dit, "text_encoder": model.text_encoder})
    measured = links.measured(tmp_path / "links", "0,1", 2)
    leader.transport = links.Transport(2, [0, 1], "nccl", measured.nccl(), measured)
    try:
        group.spawn()
        leader.pg = group.form(
            torch,
            SEAL,
            nccl=leader.transport.nccl,
            formed=leader._calibrated if measured.copy_engines else None,
        )
        assert leader._install_group(torch, model) is None
        leader._attempt_spool = tmp_path / "attempt"
        for a, c in zip(model.infer(batch), alone, strict=True):
            torch.testing.assert_close(a, c, rtol=1e-4, atol=1e-5)
        record = leader.transport.document()
        print(json.dumps(record))
        assert [(p["a"], p["b"]) for p in record["pairs"]] == [(0, 1), (1, 0)] and record["reason"]
        if measured.copy_engines:
            assert record["calibration"]["copy_engine_exact"] == 1.0, record
            assert record["route"] == (
                "copy-engine"
                if record["calibration"]["copy_engine_gbps"] > record["calibration"]["nccl_gbps"]
                else "nccl"
            )
        assert not group.broken
    finally:
        group.close()
        for end in sockets:
            end.close()


def test_a_follower_killed_inside_a_collective_fails_the_call_on_gpu_0(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """With NCCL's own watchdog off (`abortable`), a dead peer is ours to notice: GPU 1 is
    SIGKILLed while GPU 0 waits in their collective. Its seam closes, the group aborts GPU 0's
    communicator and the call fails, naming the GPU, in about the time the kill took. No
    clock decides it, and nothing reruns: a dead process has nothing to run again."""
    group = group_of_two(tmp_path)
    leader, sockets = leader_of(group, tmp_path, monkeypatch)
    model: Any = collective_model()
    roots = {"block": model.block, "overlay": model.overlay, "spare": model.spare}
    for root in roots.values():
        root.to("cuda:0")
    try:
        group.spawn()
        leader.pg = group.form(torch, SEAL)
        mirror_component(
            model.block,
            name=("fixture", "block"),
            prepared={("fixture", key): (model, root) for key, root in roots.items()},
            group=group,
            spool=lambda: tmp_path / "attempt",
            rerun=leader._group_rerun,
            journaled=leader._call_journal,
        )
        assert float(model.counted()) == 7.0
        follower = group.pids()[0]  # the followers only: GPU 1
        killed: list[float] = []

        def kill() -> None:
            time.sleep(3)  # GPU 1 sleeps 600 s before its collective; GPU 0 waits in it
            os.kill(follower, signal.SIGKILL)
            killed.append(time.monotonic())

        threading.Thread(target=kill, daemon=True).start()
        with pytest.raises(GroupRefusal) as failed:
            model.stalled(600.0)
        took = time.monotonic() - killed[0]
        print(f"GPU 0 returned {took:.2f} s after the kill: {failed.value}")
        assert took < 30, took
        assert group.broken and "GPU 1" in str(failed.value)
        # GPU 0's stream is not left behind a collective that cannot complete.
        drained = threading.Event()

        def drain() -> None:
            torch.cuda.synchronize(0)
            drained.set()

        threading.Thread(target=drain, daemon=True).start()
        assert drained.wait(30), "GPU 0's stream still waits on the dead GPU's collective"
    finally:
        group.close()
        for end in sockets:
            end.close()


def test_a_follower_stuck_in_a_collective_is_found_by_its_lack_of_progress(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """With NCCL's watchdog off, a LIVE peer stuck in a collective is ours to notice too: GPU 0
    never joins the collective GPU 1 waits in. GPU 1 spins (it burns CPU), so its CPU time is
    no evidence; the call is judged by its own pace (`RankGroup.watched`): GPU 0's device
    progress, the bytes GPU 1 moves and its answer. Nothing moves, the group breaks and the
    call fails naming GPU 1, after the noise floor (this first call showed no longer pause).
    A rescue kills GPU 1 after 240 s if that never happens, and the test fails."""
    group = group_of_two(tmp_path)
    leader, sockets = leader_of(group, tmp_path, monkeypatch)
    model: Any = collective_model()
    roots = {"block": model.block, "overlay": model.overlay, "spare": model.spare}
    for root in roots.values():
        root.to("cuda:0")
    try:
        group.spawn()
        leader.pg = group.form(torch, SEAL)
        mirror_component(
            model.block,
            name=("fixture", "block"),
            prepared={("fixture", key): (model, root) for key, root in roots.items()},
            group=group,
            spool=lambda: tmp_path / "attempt",
            rerun=leader._group_rerun,
            journaled=leader._call_journal,
        )
        follower = group.pids()[0]
        rescue = threading.Timer(240, lambda: os.kill(follower, signal.SIGKILL))
        rescue.start()
        started = time.monotonic()
        try:
            with pytest.raises(GroupRefusal) as failed:
                model.stuck()
        finally:
            rescue.cancel()
        took = time.monotonic() - started
        print(f"GPU 0 returned after {took:.1f} s: {failed.value}")
        assert "GPU 1 wedged in 'block'" in str(failed.value), str(failed.value)
        assert took < 120, took
    finally:
        group.close()
        for end in sockets:
            end.close()
