"""Rank 0 of a gloo group in a fresh process, imports done and no device: its followers fork."""

import os
import socket
import sys
from pathlib import Path

import torch

from cozy_runtime.internal import executor, zygote
from cozy_runtime.internal.config import seal_snapshot
from cozy_runtime.internal.parallel.group import RankGroup
from cozy_runtime.internal.seam import Channel

executor._SEALED.update(seal_snapshot())
root = Path(sys.argv[1])
assert zygote.unsafe(torch) == "", zygote.unsafe(torch)
ours, theirs = socket.socketpair()
leader = executor.Executor(Channel(ours), root, world=3)
leader.imported = True
group = RankGroup(
    degree=3,
    backend="gloo",
    python=sys.executable,
    root=str(root),
    sample_seconds=0.2,
    launch=leader._follower,
    devices=("4", "5", "6"),
)
leader.group = group
try:
    group.spawn()
    own = Path("/proc/self/cmdline").read_bytes()
    for follower in group.followers:
        assert Path(f"/proc/{follower.pid}/cmdline").read_bytes() == own, "spawned, not forked"
        stat = Path(f"/proc/{follower.pid}/stat").read_text().rsplit(") ", 1)[1].split()
        assert int(stat[1]) == os.getpid() and int(stat[2]) == os.getpgid(0), stat[:3]
    group.dial_back(seal_snapshot())
    for follower, probe in zip(group.followers, group.call({"cmd": "probe"}), strict=True):
        (process,) = probe["processes"]
        assert process["rank"] == follower.rank and process["pid"] == follower.pid, process
    group.form(torch, {"NCCL_NVLS_ENABLE": "0"})
    world = torch.distributed.get_world_size()
finally:
    group.close()
assert all(f.process.poll() is not None for f in group.followers)
print(f"forked {len(group.followers)} followers, formed {world}")
