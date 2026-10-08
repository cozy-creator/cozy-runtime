"""Changing, skewed Q/K/V partitions and padded heads over real Gloo exchanges."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

from cozy_runtime.internal import attention_ulysses, attention_upstream


def rank_main(rank: int, root: Path) -> None:
    torch.set_num_threads(1)
    degree = 4
    dist.init_process_group(
        "cpu:gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=degree
    )
    try:
        reports = []
        for q_sizes, kv_sizes in [([1, 2, 3, 25], [28, 3, 2, 4]), ([7, 5, 4, 2], [1, 8, 2, 6])]:
            seed = torch.Generator().manual_seed(382)
            # Batch >1 and ten heads exercise the same upstream head-padding helpers
            # as the reference; Q and KV have different global AND local lengths.
            tensors = [
                torch.randn(2, sum(sizes), 10, 16, generator=seed)
                for sizes in (q_sizes, kv_sizes, kv_sizes)
            ]
            shards = [
                tensor.split(sizes, dim=1)[rank].contiguous()
                for tensor, sizes in zip(tensors, (q_sizes, kv_sizes, kv_sizes), strict=True)
            ]
            before = [tensor.clone() for tensor in shards]
            q, k, v = shards
            with torch.no_grad():
                got = attention_ulysses.exchange_shared(
                    q, k, v, dist.group.WORLD, attention_upstream.sdpa
                )
                old = attention_ulysses.exchange(q, k, v, dist.group.WORLD, attention_upstream.sdpa)
                full = attention_upstream.sdpa(*tensors).split(q_sizes, dim=1)[rank]
            reports.append(
                {
                    "q_sizes": q_sizes,
                    "kv_sizes": kv_sizes,
                    "equal_to_original_exchange": bool(torch.equal(got, old)),
                    "equal_to_unsharded": bool(torch.equal(got, full)),
                    "inputs_unchanged": all(
                        torch.equal(a, b) for a, b in zip(shards, before, strict=True)
                    ),
                }
            )
        (root / f"rank-{rank}.json").write_text(json.dumps(reports))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    start: Any = spawn
    start(rank_main, args=(Path(sys.argv[1]),), nprocs=4, join=True)
