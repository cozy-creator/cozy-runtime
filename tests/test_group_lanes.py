"""Group placement bookkeeping and sealed executor lifecycle regressions.

Actual sharded component forwards and native residency are exercised separately
in test_parallel_calls.py. These tests do not establish multi-GPU H3 performance.

The gloo cases open the NVIDIA device nodes on a host that has a driver, even with
`CUDA_VISIBLE_DEVICES=""`: `torch.distributed` asks torch for its accelerator, and a CUDA
build of torch answers by counting devices. Beside a GPU benchmark, run this file only while
holding the GPU lock, or where `/dev/nvidia*` is hidden.
"""

from __future__ import annotations

import dataclasses
import os
import signal
import sys
from pathlib import Path
from typing import Any

import msgspec
import pytest

from cozy_runtime.author._signature import ModelBinding
from cozy_runtime.internal import budget_cell, package_interface, spawn
from cozy_runtime.internal import executor as executor_module
from cozy_runtime.internal.config import seal_snapshot
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.parallel import cp, wire
from cozy_runtime.internal.parallel.group import RankGroup
from cozy_runtime.internal.parallel.plan import GpuDivergence, GroupPlan, GroupRefusal

PACKAGE = Path(__file__).resolve().parent.parent / "examples" / "marco-polo"
GiB = 1 << 30


def _torch_here() -> str:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        return f"torch must be importable in the test interpreter for the gloo arms: {exc}"
    return ""


NO_TORCH = _torch_here()
needs_torch = pytest.mark.skipif(bool(NO_TORCH), reason=NO_TORCH or "")


# --------------------------------------------------------------------- bookkeeping


# ------------------------------------------------------------------------ the opt-in


def _slot_document(slot: ModelBinding) -> dict[str, object]:
    return package_interface.model_slot(
        slot.path,
        slot.class_key,
        slot.encoded_leaves,
        slot.fusion,
        slot.components,
        slot.sequence_parallel,
    )


def test_the_authors_degrees_reach_the_interface_and_its_fixed_point() -> None:
    """`@sequence_parallel(degrees=...)` on the Model class -> `ModelBinding.sequence_parallel`
    -> the interface slot `sequence_parallel: {degrees}` -> `DeclaredBinding` degrees.
    The decorator refuses malformed degrees at BUILD; the interface's closed scan refuses
    a malformed slot; a class that says nothing writes no key."""
    from cozy_runtime.author import ConformanceError, Model, sequence_parallel
    from cozy_runtime.author._signature import ModelBinding as AuthorSlot

    @sequence_parallel(degrees=(2, 4))
    class Sharded(Model[object]):
        pass

    class Plain(Model[object]):
        pass

    assert Sharded.__sequence_parallel__ == (2, 4) and Plain.__sequence_parallel__ == ()
    for bad in ((), (1,), (4, 2), (2, 2), (True,), (2.0,)):
        with pytest.raises(ConformanceError) as refused:
            sequence_parallel(degrees=bad)  # type: ignore[arg-type]
        assert refused.value.code == "sequence_parallel_degrees"
    with pytest.raises(ConformanceError):
        sequence_parallel(degrees=(2,))(object)  # type: ignore[type-var]
    slot = AuthorSlot(
        path="serve.models.m",
        param="m",
        model_class=Sharded,
        components={},
        sequence_parallel=Sharded.__sequence_parallel__,
    )
    document = _slot_document(slot)
    assert document["sequence_parallel"] == {"degrees": [2, 4]}
    assert "sequence_parallel" not in _slot_document(
        dataclasses.replace(slot, model_class=Plain, sequence_parallel=())
    )
    # the real PackageInterface of the marco-polo example carries no key (nothing declared)
    body = package_interface.build(discover(PACKAGE))
    assert not any(
        "sequence_parallel" in model
        for surface in body.get("entrypoints", [])
        for model in surface.get("models", [])
    )
    # the closed scan over REAL interface bytes: a well-formed slot reads back, a
    # malformed one refuses by path, an unknown key inside the slot is ignored
    import copy

    def with_slot(slot: dict[str, Any]) -> dict[str, Any]:
        mutated = copy.deepcopy(body)
        entry = next(row for row in mutated["entrypoints"] if row["name"] == "marco")
        entry["models"] = [
            {
                "path": "marco.models.m",
                "class": "Sharded",
                "component_use": {},
                **slot,
            }
        ]
        return mutated

    for degrees, ok in (([2, 4], True), ([1], False), ([4, 2], False), ([], False), ("2", False)):
        raw = package_interface.canonical_bytes(
            with_slot({"sequence_parallel": {"degrees": degrees}})
        )
        if ok:
            parsed = package_interface.read_bytes(raw)
            assert parsed["entrypoints"][0]["models"][0]["sequence_parallel"] == {
                "degrees": degrees
            }
            continue
        with pytest.raises(package_interface.StalePackageInterface) as malformed:
            package_interface.read_bytes(raw)
        assert "sequence_parallel.degrees" in str(malformed.value), (degrees, malformed.value)
    with pytest.raises(package_interface.StalePackageInterface):
        package_interface.read_bytes(
            package_interface.canonical_bytes(with_slot({"sequence_parallel": {"x": 1}}))
        )
    tolerated = package_interface.read_bytes(
        package_interface.canonical_bytes(
            with_slot({"sequence_parallel": {"degrees": [2], "x": 1}})
        )
    )
    assert tolerated["entrypoints"][0]["models"][0]["sequence_parallel"]["degrees"] == [2]


# ---------------------------------------------------------------------- gloo (torch)


def _launcher(env: dict[str, str]) -> Any:
    """The real launch shape under a chosen environment: the trampoline's `inherit`
    backend (parent check, pdeathsig, no_new_privs, the executor's OOM score) exec'ing
    the executor program with its rank, world, seam fd and leader."""

    def launch(module_argv: list[str], fd: int) -> spawn.Child:
        return spawn.spawn_follower(
            python=sys.executable, module_argv=module_argv, env=env, inherit_fd=fd
        )

    return launch


@needs_torch
def test_a_gloo_group_forms_on_a_kernel_chosen_port_and_agrees_or_refuses(
    tmp_path: Path,
) -> None:
    """THIS process is rank 0. Real follower executor processes are spawned by the real
    launch shape (the executor program, told its rank, world, seam fd and leader), dial
    back over their `socketpair` seam under exactly rank 0's seal, join a TCPStore on a
    port the kernel chose, pass the formation barrier, and refuse `gpu_divergence` on a
    broadcast plan whose hardware half is not theirs, naming each GPU as nvidia-smi does:
    sealed to cards 4-6, rank 1 is GPU 5. gloo stands in for NCCL: the rendezvous, the
    barrier, the plan and the seam are backend-agnostic."""
    import torch

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "4,5,6"}
    sealed = seal_snapshot(env)
    group = RankGroup(
        degree=3,
        backend="gloo",
        python=sys.executable,
        root=str(tmp_path),
        sample_seconds=0.2,
        launch=_launcher(env),
        devices=("4", "5", "6"),
    )
    try:
        group.spawn()
        assert [(f.rank, f.gpu) for f in group.followers] == [(1, "GPU 5"), (2, "GPU 6")]
        assert all(f.process.poll() is None for f in group.followers)
        # the followers are DIRECT CHILDREN of rank 0 (they inherit its cgroup and die
        # with it): pdeathsig was armed before exec
        for follower in group.followers:
            stat = Path(f"/proc/{follower.pid}/stat").read_text().rsplit(") ", 1)[1].split()
            assert int(stat[1]) == os.getpid(), "a follower's parent is rank 0"
        # each follower answers as its own rank of this world, under this seal
        group.dial_back(sealed)
        # Probe the actual followers before Torch is loaded: rank identity is a
        # process fact, available even for a weightless executor or failed prepare.
        for follower, probe in zip(group.followers, group.call({"cmd": "probe"}), strict=True):
            (process,) = probe["processes"]
            assert process["rank"] == follower.rank and process["pid"] == follower.pid
            assert process["started_ticks"] > 0
        assert group.port == 0, "no store exists before formation"
        group.form(torch, {"NCCL_NVLS_ENABLE": "0"})
        assert group.port > 1024, group.port
        assert group.formed
        assert torch.distributed.get_world_size() == 3 and torch.distributed.get_rank() == 0
        # the followers answered `join` only after their own barrier: the group is whole
        command = {
            "cmd": "load",
            "construction": "fixture",
            "devices": "4,5,6",
            "sequence_parallel_degree": 3,
            "authorized_device_limit_bytes": 1,
            "binding": {
                "application": "x:app",
                "package_interface": "/nowhere",
                "model_class": "",
            },
            "budgets": {},
        }
        # the hardware half disagrees: refused BEFORE torch is imported on the follower
        divergent = {"degree": 3, "devices": ["0", "1", "7"]}
        replies = group.call({**command, "group": divergent})
        assert [r["code"] for r in replies] == ["gpu_divergence"] * 2, replies
        assert replies[0]["detail"].startswith("GPU 5 is sealed to ['4', '5', '6']"), replies
        assert "GPU 4 broadcast 3 GPUs over ['0', '1', '7']" in replies[1]["detail"], replies
        assert replies[0]["rank"] == 1 and replies[1]["rank"] == 2
        # a follower that refused is poisoned: a second command says so
        replies = group.call({"cmd": "probe"})
        assert all(r["poisoned"] == "GPU divergence" for r in replies), replies
        # an out-of-vocabulary command on a follower refuses; rank 0's plane is closed
        replies = group.call({"cmd": "invoke"})
        assert [r["code"] for r in replies] == ["follower_command_unsupported"] * 2
    finally:
        group.close()
    assert all(f.process.poll() is not None for f in group.followers)
    assert not torch.distributed.is_initialized()


@needs_torch
def test_the_plan_agrees_field_by_field_or_refuses() -> None:
    """A follower ASSERTS the broadcast plan; the first field that differs is named, with
    the GPU that disagrees, and no GPU ever adapts."""
    plan = GroupPlan(2, ("2", "3"), "sha256:a", 1024)
    plan.assert_agrees(plan, rank=1)
    for field_name, other in (
        ("plan_digest", msgspec.structs.replace(plan, plan_digest="sha256:b")),
        (
            "authorized_device_limit_bytes",
            msgspec.structs.replace(plan, authorized_device_limit_bytes=1),
        ),
        ("degree", msgspec.structs.replace(plan, degree=4)),
    ):
        with pytest.raises(GpuDivergence) as refused:
            plan.assert_agrees(other, rank=1)
        assert refused.value.code == "gpu_divergence"
        assert refused.value.field_name == field_name and refused.value.gpu == "GPU 3"
        assert refused.value.message.startswith(f"GPU 3 disagrees on {field_name!r}: GPU 2 decided")
        assert "no GPU adapts locally" in refused.value.message


@needs_torch
def test_each_follower_takes_its_own_budget_cell_with_its_join(tmp_path: Path) -> None:
    """`RANK_CELLS`: rank 0 hands every follower its own budget cell right after `join`, so
    the owner's floor watchdog writes each GPU's cap where that GPU's process reads it. Real
    follower executors; the cell is the memfd each one maps, and nobody else's."""
    import torch

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "4,5,6"}
    group = RankGroup(
        degree=3,
        backend="gloo",
        python=sys.executable,
        root=str(tmp_path),
        sample_seconds=0.2,
        launch=_launcher(env),
        devices=("4", "5", "6"),
    )
    cells = [budget_cell.Cell.create() for _ in range(2)]
    try:
        group.spawn()
        group.dial_back(seal_snapshot(env))
        group.form(torch, {"NCCL_NVLS_ENABLE": "0"}, [fd for _, fd in cells])
        assert group.formed
        for (_cell, fd), follower in zip(cells, group.followers, strict=True):
            mine = os.fstat(fd).st_ino
            maps = Path(f"/proc/{follower.pid}/maps").read_text().splitlines()
            held = {int(line.split()[4]) for line in maps if "memfd:cozy-budget-cell" in line}
            assert held == {mine}, (follower.gpu, held, mine)
        # the followers still answer, and the cells are the owner's to write
        assert [r["ok"] for r in group.call({"cmd": "probe"})] == [True, True]
        assert cells[0][0].ask(-2 - (1 << 30)) == 1
    finally:
        group.close()
        for cell, fd in cells:
            cell.close()
            os.close(fd)


def test_a_rank_takes_its_own_cap_or_the_one_cap() -> None:
    assert executor_module._rank_cap((7, 8, 9), 1, 5) == 8
    assert executor_module._rank_cap((7,), 2, 5) == 5
    assert executor_module._rank_cap((), 0, msgspec.UNSET) is msgspec.UNSET


@needs_torch
def test_a_follower_death_breaks_the_group_loudly(tmp_path: Path) -> None:
    """A follower killed -9 is observed through its seam the moment it dies; the group
    latches `group_broken` naming its GPU and exit status, every later command
    refuses, and teardown reaps what is left. Never a park on a collective."""
    import torch

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"}
    group = RankGroup(
        degree=2,
        backend="gloo",
        python=sys.executable,
        root=str(tmp_path),
        sample_seconds=0.2,
        launch=_launcher(env),
        devices=("0", "1"),
    )
    try:
        group.spawn()
        group.dial_back(seal_snapshot(env))
        group.form(torch, {"NCCL_NVLS_ENABLE": "0"})
        victim = group.followers[0]
        victim.process.send_signal(signal.SIGKILL)
        victim.process.wait()
        with pytest.raises(GroupRefusal) as refused:
            group.call({"cmd": "probe"})
        assert refused.value.code == "group_broken"
        assert "GPU 1" in refused.value.message and "-9" in refused.value.message, refused.value
        assert group.broken and "GPU 1" in group.broken
        with pytest.raises(GroupRefusal):
            group.check_alive()
    finally:
        group.close()


@needs_torch
def test_a_follower_that_never_dials_back_is_judged_by_its_own_meter(tmp_path: Path) -> None:
    """No clock: a follower that never answers `hello` is condemned only once its own
    progress meter has been flat for longer than the floor (`liveness.Pace`), and the
    refusal names the measurement."""
    env = {**os.environ}

    def sleeper(module_argv: list[str], fd: int) -> spawn.Child:
        del module_argv
        return spawn.spawn_follower(
            python=sys.executable,
            module_argv=["-c", "import time; time.sleep(600)"],
            env=env,
            inherit_fd=fd,
        )

    group = RankGroup(
        degree=2,
        backend="gloo",
        python=sys.executable,
        root=str(tmp_path),
        sample_seconds=0.2,
        launch=sleeper,
    )
    try:
        group.spawn()
        with pytest.raises(GroupRefusal) as refused:
            group.dial_back(seal_snapshot(env))
        assert refused.value.code == "group_broken"
        assert "wedged during 'hello'" in refused.value.message
        assert "no measurable progress for" in refused.value.message, refused.value.message
    finally:
        group.close()


# ------------------------------------------------------------------ the wire and gate


@needs_torch
def test_the_rank_wire_is_closed(tmp_path: Path) -> None:
    """Scalars, lists, tuples, str-keyed dicts, tensors (through the spool) and generators
    cross; generators and callbacks stay on the leader."""
    import torch

    spool = wire.TensorSpool(tmp_path / "ranks")
    tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    arguments = [wire.marshal(a, spool) for a in (tensor, 1.5, "x", None, (1, 2))]
    assert (tmp_path / "ranks" / "tensor-0.raw").stat().st_size == 12
    assert arguments[0] == {
        "__sp__": "tensor",
        "v": "tensor-0.raw",
        "dtype": "bfloat16",
        "shape": [2, 3],
    }
    steps = wire.marshal([1, 2], spool)
    run = wire.RunCommand(
        ("m", "dit"), str(tmp_path), "ranks", (), False, True, {}, None, arguments, {"steps": steps}
    )
    command = msgspec.to_builtins(run)
    assert command["cmd"] == "run"
    args, kwargs = wire.run_call(command, spool, device=torch.device("cpu"))
    assert torch.equal(args[0], tensor) and args[1:] == (1.5, "x", None, (1, 2))
    assert kwargs["steps"] == [1, 2]
    empty = wire.marshal(torch.empty(0, 3), spool)
    assert torch.equal(wire.unmarshal(empty, spool, device="cpu"), torch.empty(0, 3))
    for bad in (lambda: None, {1: 2}, {"__sp__": 1}, object(), torch.Generator()):
        with pytest.raises(wire.UncrossableArgument) as refused:
            wire.marshal(bad, spool)
        assert refused.value.code == "uncrossable_argument"
    with pytest.raises(wire.UncrossableArgument):
        wire.unmarshal(
            {"__sp__": "tensor", "v": "../etc/passwd", "dtype": "float32", "shape": []},
            spool,
            device="cpu",
        )
    spool.clear()
    assert not any((tmp_path / "ranks").iterdir())


@needs_torch
def test_a_sharded_forward_outside_the_gate_refuses() -> None:
    import torch

    component = torch.nn.Identity()
    cp._install_gate_guard(component, "dit")
    with pytest.raises(cp.UngatedShardedForward):
        component(torch.ones(1))
    with cp.gated_call():
        torch.testing.assert_close(component(torch.ones(1)), torch.ones(1))
    assert not cp.in_gated_call()
    cp.refuse_unless_heads_divisible(heads=16, degree=4)
    cp.refuse_unless_heads_divisible(heads=0, degree=1)
    with pytest.raises(cp.ContextParallelUnavailable) as heads:
        cp.refuse_unless_heads_divisible(heads=6, degree=4)
    assert (
        heads.value.code == "context_parallel_unavailable" and "head count 6" in heads.value.message
    )
    # H3's 56 heads divide every declared degree; its packed sequence divides none of them,
    # and `ulysses_anything` is why that is no longer a refusal.
    cp.refuse_unless_heads_divisible(heads=56, degree=8)
    with pytest.raises(cp.ContextParallelUnavailable) as ring:
        cp._GroupMesh(cp.CpComms(pg=None, rank=0, device=None), 2, 0)["ring"].get_group()
    assert "Ulysses" in ring.value.message
    with pytest.raises(cp.ContextParallelUnavailable) as unsharded:
        cp.install_context_parallel(object(), degree=2, comms=None)
    assert "no declared component carries a `_cp_plan`" in unsharded.value.message
    assert cp.install_context_parallel(object(), degree=1, comms=None) == ()
