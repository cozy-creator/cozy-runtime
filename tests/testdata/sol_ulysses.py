"""Real Diffusers exchanges; CPU qualifies dense policy, CUDA additionally runs upstream Sol."""

from __future__ import annotations

import argparse
import json
import time
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from diffusers.models._modeling_parallel import ContextParallelConfig, ParallelConfig
from torch.multiprocessing.spawn import spawn

from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope
from cozy_runtime.internal import attention, attention_sol, attention_ulysses, kernel_cache
from cozy_runtime.internal.parallel.cp import _GroupMesh


def dense(*, query: Any, key: Any, value: Any, scale: float | None) -> Any:
    return torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale=scale
    ).transpose(1, 2)


def structured(tensors: list[Any]) -> None:
    """Correlated K/V within each block cannot be recovered from zero block centroids."""
    q, k, v = tensors
    for tensor in tensors:
        tensor.zero_()
    signs = (1 - 2 * (torch.arange(q.shape[1], device=q.device) % 2)).to(q.dtype)
    for head in range(q.shape[2]):
        q[0, :, head, 2 * head] = 4 + head / 4
        q[0, :, head, 2 * head + 1] = 8 * signs
        k[0, :64, head, 2 * head] = 2
        k[0, :, head, 2 * head + 1] = 8 * signs
        v[0, :, head, 0] = signs * (head + 1) / 8
    for tensor in tensors:
        tensor[:, -3:] = float("nan")


def sparse_control(tensors: list[Any], output: Any, degree: int, live: int) -> dict[str, Any]:
    """Use upstream preparation and all-keys mode to establish genuinely omitted work."""
    prepare = import_module("sol_attn.preprocess").prepare
    q, k, v = [tensor[:, :live].contiguous() for tensor in tensors]
    full = prepare(q, k, v, tau=1.0, scale=128**-0.5, thresh_type="exact")
    split = []
    heads_per_rank = q.shape[2] // degree
    for start in range(0, q.shape[2], heads_per_rank):
        operands = [
            tensor[:, :, start : start + heads_per_rank].contiguous() for tensor in (q, k, v)
        ]
        split.append(prepare(*operands, tau=1.0, scale=128**-0.5, thresh_type="exact"))
    equality = {
        name: torch.equal(
            value.contiguous().view(torch.uint8),
            torch.cat([part[index] for part in split], dim=2).contiguous().view(torch.uint8),
        )
        for index, (name, value) in enumerate(zip(("kc", "vc", "threshold"), full, strict=True))
    }
    kc, _vc, threshold = full
    # Reviewed upstream predicate: colmean > threshold, local distance <=1, or sink.
    # Query block8 and KV block3 are neither local nor within the nine-token sink.
    colmean = (q[:, 512:576].float().mean(dim=1) * kc[:, 3].float()).sum(dim=-1)
    colmean *= 128**-0.5 * 1.4426950408889634
    omitted = bool((colmean < threshold[:, 8]).all())
    while (loaded := attention_sol._ready(live, q.device, 0)) is None:
        time.sleep(0.2)  # the builder process compiles it
    all_keys = attention_sol._native(q, k, v, None, live, loaded)
    numerator = (output[:, 512:576].float() - all_keys[:, 512:576].float()).norm()
    denominator = all_keys[:, 512:576].float().norm().clamp_min(1e-30)
    return {
        "upstream_preparation_equal_across_head_partitions": equality,
        "query_block": 8,
        "kv_block": 3,
        "kv_block_width": 64,
        "route_column_mean": colmean.tolist(),
        "route_threshold": threshold[:, 8].tolist(),
        "route_omitted_by_reviewed_upstream_predicate": omitted,
        "sparse_vs_native_all_keys_relative_l2": float((numerator / denominator).item()),
    }


def rank_main(rank: int, degree: int, root: Path, device_kind: str, heads: int) -> None:
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if device_kind == "cuda" else "cpu")
    if device_kind == "cuda":
        torch.cuda.set_device(device)
        kernel_cache.configure(kernel_cache.Store(root / "kernels" / "u0"))
    dist.init_process_group(
        "nccl" if device_kind == "cuda" else "cpu:gloo",
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
        group = attention_ulysses.group(config)
        registry = import_module("diffusers.models.attention_dispatch")._AttentionBackendRegistry
        backend = registry._backends[attention._member(attention.BY_NAME["sol-attn"])]
        attention_sol.bind_dense(dense, "test-native-dense")
        results = []
        for length in (1029, 1032) if device_kind == "cuda" else (29, 32):
            generator = torch.Generator(device=device).manual_seed(87)
            tensors = [
                torch.randn(
                    1,
                    length,
                    heads,
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
                ("dense_step", AttentionLayout(length - 3, 9, 0, 4)),
                ("dense_path", AttentionLayout(length - 3, 9, 7, 4, ("blocks.0",))),
            ]
            if device_kind == "cuda":
                layouts.extend(
                    [
                        ("random_sparse", AttentionLayout(length - 3, 9, 7, 4)),
                        ("structured_sparse", AttentionLayout(length - 3, 9, 7, 4)),
                    ]
                )
            for case, layout in layouts:
                if case == "structured_sparse":
                    structured(tensors)
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
                            actual = backend(query=q, key=k, value=v, _parallel_config=config)
                            q, k, v = tensors
                            reference = backend(query=q, key=k, value=v)
                        else:
                            q, k, v = shards
                            actual = attention_ulysses.exchange(
                                q,
                                k,
                                v,
                                group,
                                lambda a, b, c, at=layout, where=site: attention_sol._execute(
                                    a, b, c, None, at, where
                                ),
                            )
                            q, k, v = tensors
                            reference = attention_sol._execute(q, k, v, None, layout, site)
                finally:
                    attention_sol._SITE.reset(token)
                control = None
                if case == "structured_sparse":
                    control = sparse_control(tensors, reference, degree, layout.live_tokens)
                    (root / f"rank-{rank}-length-{length}-sparse-control.json").write_text(
                        json.dumps(control, indent=2) + "\n"
                    )
                    assert all(
                        control["upstream_preparation_equal_across_head_partitions"].values()
                    ), control
                    assert control["route_omitted_by_reviewed_upstream_predicate"], control
                    assert control["sparse_vs_native_all_keys_relative_l2"] > 1e-3, control
                expected = reference.tensor_split(degree, dim=1)[rank]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert torch.equal(
                    actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
                )
                for original, shard in zip(before, shards, strict=True):
                    assert torch.equal(original.view(torch.uint8), shard.view(torch.uint8))
                assert torch.isfinite(actual).all()
                results.append(
                    {
                        "case": case,
                        "sparse_control": control,
                        "length": length,
                        "local_length": shards[0].shape[1],
                        "heads": heads,
                        "heads_per_rank": heads // degree,
                        "step": layout.step,
                        "dense_paths": layout.dense_paths,
                        "combined_reference_and_parallel_calls": counts,
                        "equal": True,
                        "output_bytes_equal": True,
                    }
                )
        if device_kind == "cpu":
            assert not torch.cuda.is_initialized(), "CPU Gloo proof initialized CUDA"
        (root / f"rank-{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--heads", type=int, choices=(8, 56), default=8)
    args = parser.parse_args()
    if args.degree < 2 or args.heads % args.degree:
        parser.error(f"Ulysses degree {args.degree} must be >= 2 and divide {args.heads} heads")
    args.root.mkdir(parents=True, exist_ok=True)
    spawn_ranks: Any = spawn
    spawn_ranks(
        rank_main,
        args=(args.degree, args.root, args.device, args.heads),
        nprocs=args.degree,
        join=True,
    )
    ranks = [
        json.loads((args.root / f"rank-{rank}.json").read_text()) for rank in range(args.degree)
    ]
    report = {
        "degree": args.degree,
        "device": args.device,
        "heads": args.heads,
        "cases": ranks[0],
        "ranks": ranks,
    }
    (args.root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "degree": args.degree,
                "device": args.device,
                "heads": args.heads,
                "cases": len(ranks[0]),
            }
        )
    )


if __name__ == "__main__":
    main()
