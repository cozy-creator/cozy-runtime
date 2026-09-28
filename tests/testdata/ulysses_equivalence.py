"""Degree-N Ulysses against degree 1 for SDPA on gloo ranks: one JSON report per rank.

`--plant` replaces the exchange with the rank's own shard only, the silent-corruption shape the
wrapper exists to prevent: right shape, no error, wrong attention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from diffusers.models._modeling_parallel import ContextParallelConfig, ParallelConfig
from diffusers.models.attention_dispatch import _AttentionBackendRegistry
from torch.multiprocessing.spawn import spawn

from cozy_runtime.internal import attention, attention_ulysses
from cozy_runtime.internal.encoding import DeviceFacts
from cozy_runtime.internal.parallel.cp import _GroupMesh


def rank_main(rank: int, degree: int, root: Path, shape: tuple[int, ...], plant: bool) -> None:
    torch.set_num_threads(1)
    if plant:
        attention_ulysses.exchange = lambda query, key, value, handle, compute: compute(
            query, key, value
        )
    dist.init_process_group(
        "cpu:gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=degree
    )
    try:
        config = ParallelConfig(
            context_parallel_config=ContextParallelConfig(
                ulysses_degree=degree, ulysses_anything=True
            )
        )
        config.setup(
            rank,
            degree,
            torch.device("cpu"),
            mesh=cast(Any, _GroupMesh(dist.group.WORLD, degree, rank)),
        )
        ready = attention.pinned("sdpa", DeviceFacts("cpu", "CPU", 0, "", "", 0))[0]
        backend = _AttentionBackendRegistry._backends[ready.member]
        seed = torch.Generator().manual_seed(0x51D3)
        q, k, v = (torch.randn(shape, generator=seed) for _ in range(3))
        shards = [t.tensor_split(degree, dim=1)[rank].contiguous() for t in (q, k, v)]
        with torch.no_grad(), attention_ulysses.observing() as seen:
            got = backend(query=shards[0], key=shards[1], value=shards[2], _parallel_config=config)
            one = backend(query=q, key=k, value=v).tensor_split(degree, dim=1)[rank]
        report = {
            "rank": rank,
            "impl": attention.implementations(seen),
            "byte_equal": bool(torch.equal(got.contiguous(), one.contiguous())),
            "rel_l2_to_degree1": float((got.double() - one.double()).norm() / one.double().norm()),
        }
        (root / f"rank-{rank}.json").write_text(json.dumps(report))
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--rows", type=int, default=1031)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    shape = (1, args.rows, args.heads, 128)
    start: Any = spawn
    start(
        rank_main, args=(args.degree, args.root, shape, args.plant), nprocs=args.degree, join=True
    )


if __name__ == "__main__":
    main()
