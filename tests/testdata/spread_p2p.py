"""A spread result over two gloo ranks: rank 1 sends what `Executor.run` sends, rank 0 reads it.

Rank 1 sends an fp8 tensor, a bool tensor, then answers a tuple result the way a follower
does: its header check fails before any byte moves, so it sends `send_nothing`. Rank 0
writes what it read to `<root>/rank-0.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

from cozy_runtime.internal.parallel import wire

SENT = (
    torch.linspace(-3, 3, 24).reshape(2, 3, 4).to(torch.float8_e4m3fn),
    torch.tensor([[True, False, True]]),
)


def rank(index: int, root: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{root}/store", rank=index, world_size=2)
    device = torch.device("cpu")
    if index == 1:
        for tensor in SENT:
            wire.send_tensor(tensor, wire.p2p_head(tensor), 0, None)
        try:
            wire.p2p_head((SENT[1], SENT[1]))
        except wire.UncrossableArgument:
            wire.send_nothing(0, None, device, failed=True)
    else:
        got = [wire.recv_tensor(1, None, device) for _ in range(len(SENT) + 1)]
        report = {
            "equal": [
                g is not None
                and g.dtype == s.dtype
                and bool(torch.equal(g.view(torch.uint8), s.view(torch.uint8)))
                for g, s in zip(got, SENT, strict=False)
            ],
            "tuple": got[-1],
        }
        (Path(root) / "rank-0.json").write_text(json.dumps(report))
    dist.destroy_process_group()


if __name__ == "__main__":
    spawn(rank, args=(sys.argv[1],), nprocs=2)
