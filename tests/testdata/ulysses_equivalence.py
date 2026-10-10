"""Degree-N Ulysses against degree 1 for SDPA on gloo ranks: one JSON report per rank.

`--plant` replaces the exchange with the rank's own shard only, the silent-corruption shape the
wrapper exists to prevent: right shape, no error, wrong attention. `--cuda` runs the ranks on
NCCL, one GPU each, and also through the copy-engine exchange, which must give the NCCL bytes.
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
from torch.distributed import _symmetric_memory as symm
from torch.multiprocessing.spawn import spawn
from torch.utils._python_dispatch import TorchDispatchMode

from cozy_runtime.internal import attention, attention_ulysses
from cozy_runtime.internal.encoding import DeviceFacts
from cozy_runtime.internal.parallel import copy_engine
from cozy_runtime.internal.parallel.cp import CpComms, _GroupMesh


class Writes(TorchDispatchMode):
    """Counts mutable ops whose written tensor lies in any of `spans` (device addresses)."""

    def __init__(self, spans: list[tuple[int, int]]) -> None:
        super().__init__()
        self.spans, self.seen = spans, 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        if func._schema.is_mutable:
            for a in args:
                if isinstance(a, torch.Tensor):
                    self.seen += any(lo <= a.data_ptr() < hi for lo, hi in self.spans)
        return func(*args, **(kwargs or {}))


def rank_main(
    rank: int, degree: int, root: Path, shape: tuple[int, ...], plant: bool, cuda: bool
) -> None:
    torch.set_num_threads(1)
    if plant:
        attention_ulysses.exchange = lambda query, key, value, handle, compute: compute(
            query, key, value
        )
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if cuda else "cpu:gloo",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=degree,
        **({"device_id": device} if cuda else {}),
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
            device,
            mesh=cast(Any, _GroupMesh(CpComms(dist.group.WORLD, rank, device), degree, rank)),
        )
        facts = DeviceFacts("cpu", "CPU", 0, "", "", 0)
        if cuda:
            major, minor = torch.cuda.get_device_capability(device)
            name = torch.cuda.get_device_name(device)
            facts = DeviceFacts("cuda", name, major * 10 + minor, "", "", rank)
        ready = attention.pinned("sdpa", facts)[0]
        backend = _AttentionBackendRegistry._backends[ready.member]
        seed = torch.Generator().manual_seed(0x51D3)
        q, k, v = (torch.randn(shape, generator=seed).to(device) for _ in range(3))
        if cuda:
            q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))
        shards = [t.tensor_split(degree, dim=1)[rank].contiguous() for t in (q, k, v)]
        with torch.no_grad(), attention_ulysses.observing() as seen:
            got = backend(query=shards[0], key=shards[1], value=shards[2], _parallel_config=config)
            one = backend(query=q, key=k, value=v).tensor_split(degree, dim=1)[rank]
            engines: Any = None
            if cuda:
                _, calibration = copy_engine.calibrate(torch, dist.group.WORLD, rank, degree)
                world = dist.group.WORLD
                engines = copy_engine.CopyEngineExchange(torch, symm, world, rank, degree)
                copy_engine.install(engines, dist.group.WORLD)
                engined = backend(
                    query=shards[0], key=shards[1], value=shards[2], _parallel_config=config
                )
                # A group GPU runs its blocks under the plane's per-op recovery, which clones
                # every mutable op's target: no peer copy may reach Python dispatch.
                spans = [(t.data_ptr(), t.data_ptr() + t.numel()) for t in engines.peers]
                with Writes(spans) as writes:
                    backend(
                        query=shards[0], key=shards[1], value=shards[2], _parallel_config=config
                    )
        report = {
            "rank": rank,
            "impl": attention.implementations(seen),
            "byte_equal": bool(torch.equal(got.contiguous(), one.contiguous())),
            "rel_l2_to_degree1": float((got.double() - one.double()).norm() / one.double().norm()),
        }
        if engines is not None:
            report["copy_engine_equal_nccl"] = bool(torch.equal(engined, got))
            report["calibration"] = calibration
            report["dispatched_peer_writes"] = writes.seen
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
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    shape = (1, args.rows, args.heads, 128)
    start: Any = spawn
    start(
        rank_main,
        args=(args.degree, args.root, shape, args.plant, args.cuda),
        nprocs=args.degree,
        join=True,
    )


if __name__ == "__main__":
    main()
