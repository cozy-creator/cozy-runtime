"""Plant the defect the Ulysses wrapper exists to prevent: attention over the rank's own
sequence shard only, which raises nothing and has the right shape. The real qualification
ranks run on gloo and must each report the kernel not equivalent to one process."""

from __future__ import annotations

import sys
from typing import Any

from torch.multiprocessing.spawn import spawn

from cozy_runtime.internal import attention_qualify, attention_ulysses


def planted(rank: int, degree: int, root: str) -> None:
    attention_ulysses.exchange = lambda q, k, v, handle, compute: compute(q, k, v)
    attention_qualify._rank(rank, degree, root, "cpu", ["sdpa"], (1, 67, 4, 128))


if __name__ == "__main__":
    spawn_ranks: Any = spawn
    spawn_ranks(planted, args=(2, sys.argv[1]), nprocs=2, join=True)
