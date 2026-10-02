"""Real Gloo ranks exercising request spools under the executor UID.

Model filling and H3 numerics are separate gates. This fixture runs the actual
component scope, mirror, tensor transport, and follower execution code.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

import msgspec
import torch

from cozy_runtime.author import Model, uses_components
from cozy_runtime.internal import spawn
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import Join
from cozy_runtime.internal.parallel.group import RankGroup
from cozy_runtime.internal.parallel.mirror import mirror_component
from cozy_runtime.internal.seam import Channel


class Collective(torch.nn.Module):  # type: ignore[misc]
    def forward(self, value: Any) -> Any:
        result = value.clone()
        torch.distributed.all_reduce(result)
        return result


class CollectiveModel(Model[object]):
    block: Any

    @uses_components("block")
    def compute(self) -> None:
        with torch.no_grad():
            value = torch.tensor([1.0, 3.0, 7.0])
            result = self.block(value)
        torch.testing.assert_close(result, value * torch.distributed.get_world_size())


def follower(args: argparse.Namespace) -> None:
    channel = Channel(socket.socket(fileno=args.rank_fd))
    executor = Executor(channel, args.root, rank=args.rank, world=args.world)
    executor.device_kind = "cpu"  # type: ignore[assignment]  # CPU/Gloo hardware boundary.
    executor.torch = torch
    model = CollectiveModel.for_test(block=Collective())
    executor.group_components = {("fixture", "block"): (model, model.block)}
    executor.group_sharded.add(("fixture", "block"))
    executor.ready = True
    while (command := channel.recv()) is not None:
        name = command["cmd"]
        if name == "shutdown":
            return
        if name == "join":
            reply = executor.join(msgspec.convert(command, Join))
        elif name == "run":
            reply = executor.run(command)
        elif name == "probe":
            reply = {"ok": True, "uid": os.getuid(), "gid": os.getgid(), "groups": os.getgroups()}
        else:
            raise AssertionError(name)
        channel.send({**reply, "reply": name})


def leader(args: argparse.Namespace) -> None:
    assert os.getuid() == os.getgid() == 65533
    assert os.getgroups() == []
    # The worker's records must remain inaccessible while the named request paths work.
    for operation in (
        lambda: list(args.root.iterdir()),
        lambda: (args.root / "private").read_bytes(),
        lambda: (args.root / "ungranted").write_bytes(b"x"),
    ):
        try:
            operation()
        except PermissionError:
            pass
        else:
            raise AssertionError("executor can access the worker's private state")

    def launch(module_argv: list[str], fd: int) -> spawn.Child:
        return spawn.spawn_follower(
            python=sys.executable,
            module_argv=[str(Path(__file__).resolve()), *module_argv[2:]],
            inherit_fd=fd,
            env=dict(os.environ),
        )

    group = RankGroup(degree=args.world, backend="cpu:gloo", root=str(args.root), launch=launch)
    ours, theirs = socket.socketpair()
    executor = Executor(Channel(ours), args.root, world=args.world)
    executor.device_kind = "cpu"  # type: ignore[assignment]  # CPU/Gloo hardware boundary.
    executor.torch = torch
    executor._group_model_key = "fixture"
    model = CollectiveModel.for_test(block=Collective())
    try:
        group.spawn()
        group.form(torch, {})
        for row in group.call({"cmd": "probe"}):
            assert row["uid"] == row["gid"] == 65533 and row["groups"] == []
        mirror_component(
            model.block,
            name=("fixture", "block"),
            prepared={("fixture", "block"): (model, model.block)},
            group=group,
            spool=lambda: executor._attempt_spool,
        )
        for index in range(2):
            executor._attempt_spool = args.requests / str(index)
            model.compute()
            assert not list(executor._attempt_spool.rglob("tensor-*.raw"))
        print(json.dumps({"uid": os.getuid(), "degree": args.world, "requests": 2}))
    finally:
        group.close()
        ours.close()
        theirs.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--rank-fd", type=int)
    parser.add_argument("--leader")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.rank:
        follower(args)
    else:
        leader(args)


if __name__ == "__main__":
    main()
