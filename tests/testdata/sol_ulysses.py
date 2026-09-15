"""Real Diffusers exchanges; CPU qualifies dense policy, CUDA additionally runs upstream Sol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from diffusers.models._modeling_parallel import ContextParallelConfig, ParallelConfig
from torch.multiprocessing.spawn import spawn

from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope
from cozy_runtime.internal import attention_sol
from cozy_runtime.internal.parallel.cp import _GroupMesh


def dense(*, query: Any, key: Any, value: Any, scale: float | None) -> Any:
    return torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale=scale
    ).transpose(1, 2)


def rank_main(rank: int, degree: int, root: Path, device_kind: str) -> None:
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if device_kind == "cuda" else "cpu")
    if device_kind == "cuda":
        torch.cuda.set_device(device)
        assert torch.cuda.get_device_capability(device) == (9, 0)
    dist.init_process_group(
        "nccl" if device_kind == "cuda" else "gloo",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=degree,
    )
    try:
        cp = ContextParallelConfig(ulysses_degree=degree, ulysses_anything=True)
        config = ParallelConfig(context_parallel_config=cp)
        config.setup(
            rank, degree, device, mesh=cast(Any, _GroupMesh(dist.group.WORLD, degree, rank))
        )
        group = attention_sol._parallel_group(config)
        attention_sol.bind_dense(dense, "test-native-dense")
        results = []
        for length in (1029, 1032) if device_kind == "cuda" else (29, 32):
            generator = torch.Generator(device=device).manual_seed(87)
            tensors = [
                torch.randn(
                    1,
                    length,
                    8,
                    128,
                    generator=generator,
                    device=device,
                    dtype=torch.bfloat16 if device_kind == "cuda" else torch.float32,
                )
                for _ in range(3)
            ]
            # Poisoned trailing rows may straddle the final shard; never normalize over them.
            for tensor in tensors:
                tensor[:, -3:] = float("nan")
            layouts = [
                AttentionLayout(length - 3, 9, 0, 4),
                AttentionLayout(length - 3, 9, 7, 4, ("blocks.0",)),
            ]
            if device_kind == "cuda":
                layouts.append(AttentionLayout(length - 3, 9, 7, 4))
            for layout in layouts:
                site = attention_sol.Site("dit", "blocks.0.attn")
                shards = [
                    tensor.tensor_split(degree, dim=1)[rank].contiguous() for tensor in tensors
                ]
                before = [shard.clone() for shard in shards]
                token = attention_sol._SITE.set(site)
                try:
                    with (
                        torch.no_grad(),
                        attention_sol.observing() as counts,
                        attention_scope(layout),
                    ):
                        if device_kind == "cuda":
                            q, k, v = shards
                            actual = attention_sol.sol_attention(q, k, v, _parallel_config=config)
                            q, k, v = tensors
                            reference = attention_sol.sol_attention(q, k, v)
                        else:
                            q, k, v = shards
                            actual = attention_sol._execute_parallel(
                                q, k, v, None, layout, site, group
                            )
                            q, k, v = tensors
                            reference = attention_sol._execute(q, k, v, None, layout, site)
                finally:
                    attention_sol._SITE.reset(token)
                expected = reference.tensor_split(degree, dim=1)[rank]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for original, shard in zip(before, shards, strict=True):
                    assert torch.equal(original.view(torch.uint8), shard.view(torch.uint8))
                assert torch.isfinite(actual).all()
                results.append(
                    {
                        "length": length,
                        "local_length": shards[0].shape[1],
                        "step": layout.step,
                        "dense_paths": layout.dense_paths,
                        "counts": counts,
                        "equal": True,
                    }
                )
        (root / f"rank-{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, choices=(2, 4), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    spawn_ranks: Any = spawn
    spawn_ranks(
        rank_main, args=(args.degree, args.root, args.device), nprocs=args.degree, join=True
    )
    ranks = [
        json.loads((args.root / f"rank-{rank}.json").read_text()) for rank in range(args.degree)
    ]
    report = {"degree": args.degree, "device": args.device, "cases": ranks[0], "ranks": ranks}
    (args.root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"degree": args.degree, "device": args.device, "cases": len(ranks[0])}))


if __name__ == "__main__":
    main()
