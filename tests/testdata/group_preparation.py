"""CPU/Gloo qualification of the actual executor, module adapters and ready/commit path."""

from __future__ import annotations

import argparse
import functools
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import msgspec
import torch

from cozy_runtime.author import Model, uses_components
from cozy_runtime.author._errors import classify
from cozy_runtime.internal import executor_commands, memory, tiers
from cozy_runtime.internal.executor import Executor, _capture_seal
from cozy_runtime.internal.parallel import mirror
from cozy_runtime.internal.parallel.group import OpReady, PreparedGroup, RankGroup
from cozy_runtime.internal.seam import Channel


class Collective(torch.nn.Module):  # type: ignore[misc]
    def __init__(self, rank: int, root: Path) -> None:
        super().__init__()
        self.rank, self.root = rank, root
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.register_forward_pre_hook(self.entered)

    def entered(self, module: Any, args: Any) -> None:
        with (self.root / f"entered-{self.rank}").open("a") as stream:
            stream.write("hook\n")

    def forward(self, value: Any) -> Any:
        if self.rank == 1 and value.item() < 0:
            raise RuntimeError("authored failure after collective commit")
        result = value.clone() + self.rank
        torch.distributed.all_reduce(result)
        return result


class Tiny(Model[object]):
    block: Any

    @uses_components("block")
    def compute(self, value: Any) -> Any:
        return self.block(value)


def make(channel: Channel, root: Path, rank: int) -> tuple[Executor, Tiny]:
    executor = Executor(channel, root, rank=rank, world=2)
    executor.torch = torch
    executor.device_kind = "cpu"  # type: ignore[assignment]  # the real Gloo qualification lane
    executor.ready = True
    planner = executor.planner = memory.Planner(
        torch, torch.device("cpu"), sharded=lambda key: True
    )
    model = Tiny.for_test(block=Collective(rank, root))
    held = planner.adopt("m", "block", model.block, None, tiers.DEVICE)
    # The eager adapter is the same one CUDA adoption installs; this fixture runs it on CPU.
    tiers.wrap(model.block, functools.partial(planner.run, held))
    executor.group_components[("m", "block")] = (model, model.block)
    executor.group_sharded.add(("m", "block"))
    return executor, model


def follower(fd: int, root: Path) -> None:
    channel = Channel(socket.socket(fileno=fd))
    executor, _model = make(channel, root, 1)
    while (command := channel.recv()) is not None:
        name = command["cmd"]
        if name == "shutdown":
            break
        try:
            reply = (
                executor.run(command)
                if name == "run"
                else executor.handle(executor_commands.decode(command))
            )
        except Exception as exc:
            reply = {"ok": False, "code": classify(exc).code, "detail": str(exc)}
        channel.send({"reply": name, **reply})


def leader(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    ours, unused = socket.socketpair()
    channel = Channel(ours)
    executor, model = make(channel, root, 0)

    def launch(argv: list[str], fd: int) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [sys.executable, __file__, "--follower", str(fd), "--root", str(root)],
            pass_fds=(fd,),
        )

    group = executor.group = RankGroup(degree=2, backend="cpu:gloo", root=str(root), launch=launch)
    try:
        group.spawn()
        sealed = executor.hello()["sealed"]
        group.dial_back(sealed)
        executor.pg = group.form(torch, sealed)
        prepared = executor.group_components
        mirror.mirror_component(
            model.block,
            name=("m", "block"),
            prepared=prepared,
            group=group,
            spool=lambda: root,
            planner=executor.planner,
        )
        # Scope entry may re-install adapters. The group router must remain outermost.
        assert executor.planner is not None
        router = model.block._call_impl
        tiers.wrap(
            model.block, functools.partial(executor.planner.run, executor.planner.held["m/block"])
        )
        assert model.block._call_impl is router
        with model._cozy_scope("compute", ("block",)), torch.no_grad():
            command, payloads = mirror._command(
                ("m", "block"),
                prepared,
                group,
                lambda: root,
                (torch.ones(1),),
                {},
                lambda: False,
            )
            payloads.clear()  # a real missing tensor payload, before the follower's body
            with group.hold((1,)):
                transaction = PreparedGroup(group, msgspec.to_builtins(command))
                assert not transaction.ready(
                    OpReady(operation=transaction.operation, rank=0, ok=True)
                )
                assert not list(root.glob("entered-*"))
                transaction.abort()
            assert not group.broken
        with torch.no_grad():
            assert torch.equal(model.compute(torch.tensor([2.0])), torch.tensor([5.0]))
            assert torch.equal(model.compute(torch.tensor([3.0])), torch.tensor([7.0]))
        assert [(root / f"entered-{rank}").read_text() for rank in (0, 1)] == ["hook\nhook\n"] * 2
        assert not group.broken and not executor.planner.stack and not executor.planner._gates
        # Once committed, a rank error tears down the whole group. The leader's blocked
        # Gloo collective must unwind, and the authored operation must not be retried.
        try:
            with torch.no_grad():
                model.compute(torch.tensor([-1.0]))
        except Exception:
            assert group.verdict()
        else:
            raise AssertionError("a committed rank failure returned a result")
        assert [(root / f"entered-{rank}").read_text() for rank in (0, 1)] == [
            "hook\nhook\nhook\n"
        ] * 2
        assert not executor.planner.stack and not executor.planner._gates
    finally:
        group.close()
        channel.close()
        unused.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--follower", type=int)
    options = parser.parse_args()
    _capture_seal()
    if options.follower is None:
        leader(options.root)
    else:
        follower(options.follower, options.root)
