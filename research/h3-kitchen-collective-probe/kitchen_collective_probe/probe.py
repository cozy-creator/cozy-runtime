"""Existing Kitchen producer over raw QKV, full heads versus real two-rank Ulysses."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import os
import time
from pathlib import Path

from .metrics import compared, operand_layout

TOKENS, HEADS, DIM, DEGREE, CHUNK, PREFIX = 109225, 56, 128, 2, 4096, 257


def token_bounds(rank, tokens=TOKENS, degree=DEGREE):
    width, remainder = divmod(tokens, degree)
    start = rank * width + min(rank, remainder)
    return start, start + width + (rank < remainder)


def chunks(torch, values):
    """Raw [1,T,H,D] -> global-row [M,3*H*D], restarting at global row0 per call."""
    for start in range(0, values[0].shape[1], CHUNK):
        yield torch.cat(
            [x[:, start : start + CHUNK].flatten(2, 3) for x in values], dim=-1
        ).squeeze(0)


def exact(metric):
    return metric["bytes_equal"] and metric["nonfinite_elements"] == 0


def nhd_metric(torch, actual, reference):
    return compared(torch, actual[0].transpose(0, 1), reference[0].transpose(0, 1))


def checked_identities(rows):
    if len(rows) != DEGREE or sorted(x["rank"] for x in rows) != list(range(DEGREE)):
        raise RuntimeError("Two ranks are required")
    if len({x["uuid"] for x in rows}) != DEGREE or len({x["pid"] for x in rows}) != DEGREE:
        raise RuntimeError("Two distinct assigned devices and executors are required")
    if any(
        x["world_size"] != DEGREE
        or x["sm"] != [12, 0]
        or x["runtime"] != "0.23.5"
        or x["tensorfs"] != "0.6.1"
        or x["group_global_ranks"] != [0, 1]
        for x in rows
    ):
        raise RuntimeError("This fixture requires its public two-SM120 cohort")


def timed(torch, dist, handle, call):
    dist.barrier(group=handle)
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    result = call()
    stop.record()
    stop.synchronize()
    return result, {
        "cuda_ms": float(start.elapsed_time(stop)),
        "wall_ms": (time.perf_counter() - wall) * 1000,
    }


def run(seed, handle):
    import torch
    import torch.distributed as dist
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

    from cozy_runtime.internal import accel, attention, attention_producer, attention_ulysses
    from cozy_runtime.internal.encoding import DeviceFacts

    rank, degree = dist.get_rank(handle), dist.get_world_size(handle)
    identity = {
        "rank": rank,
        "world_size": degree,
        "pid": os.getpid(),
        "uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
        "sm": list(torch.cuda.get_device_capability()),
        "runtime": importlib.metadata.version("cozy-runtime"),
        "tensorfs": importlib.metadata.version("tensorfs"),
        "group_global_ranks": dist.get_process_group_ranks(handle),
    }
    identities = [None] * degree
    dist.all_gather_object(identities, identity, group=handle)
    checked_identities(identities)
    device = DeviceFacts(
        "cuda",
        torch.cuda.get_device_name(),
        120,
        accel.driver_version(),
        "raw Kitchen producer correctness",
        index=torch.cuda.current_device(),
    )
    selection = attention.pinned("kitchen-sol", device)[0]
    kitchen = importlib.import_module("comfy_kitchen")
    backend = importlib.import_module("comfy_kitchen.backends.cuda")
    if importlib.metadata.version("comfy-kitchen") != "0.2.37":
        raise RuntimeError("This replay requires reviewed Kitchen0.2.37")
    free, total = torch.cuda.mem_get_info()
    available = free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    if available < 28 << 30:
        raise RuntimeError("Full shape requires28GiB free/reusable memory; no downsizing")
    torch.cuda.reset_peak_memory_stats()
    begun = time.perf_counter()
    start, stop = token_bounds(rank)
    hs, he = rank * (HEADS // degree), (rank + 1) * (HEADS // degree)
    with torch.inference_mode(), attention_producer.request_scope() as state:
        # Actual H3 rotary producer, synthetic but fixed global positions. Its outputs
        # are deliberately not CP-sharded: the caller has already restored global rows.
        index = torch.arange(TOKENS, device="cuda")
        positions = torch.stack((index // (84 * 48), (index // 84) % 48, index % 84), dim=-1)
        rotary = MiniMaxH3RotaryPosEmbed().cuda()(positions)
        freqs = attention_producer.rope_presentation(rotary, TOKENS, state)
        norm = (
            torch.linspace(0.875, 1.125, DIM, device="cuda").bfloat16(),
            torch.linspace(1.125, 0.875, DIM, device="cuda").bfloat16(),
        )
        del index, positions

        def producer(values, prior):
            return kitchen.sol_attn_chunked(
                lambda: chunks(torch, values),
                TOKENS,
                values[0].shape[2],
                freqs,
                norm,
                kmean=prior[0],
                vscale=prior[1],
                tau=1.0,
                scale=DIM**-0.5,
                rope_eps=1e-6,
                sink_blocks=[0, (PREFIX + 63) // 64],
                sink_q=[0, (PREFIX + 63) // 64],
                token_aug=0,
            )

        prior_full = (None, None)
        prior_rank = (None, None)
        steps = []
        for step in range(2):
            packed = torch.randn(
                1,
                TOKENS,
                3 * HEADS,
                DIM,
                device="cuda",
                dtype=torch.bfloat16,
                generator=torch.Generator(device="cuda").manual_seed(seed + step),
            )
            full = packed.chunk(3, dim=2)
            local = tuple(x[:, start:stop] for x in full)
            transport = {}

            def identity_call(q, k, v, *, transport=transport, full=full):
                transport["callback_layouts"] = [operand_layout(x) for x in (q, k, v)]
                transport["qkv"] = {
                    name: nhd_metric(torch, x, ref[:, :, hs:he])
                    for name, x, ref in zip(("q", "k", "v"), (q, k, v), full, strict=True)
                }
                return q.clone()

            copied = attention_ulysses.exchange(*local, handle, identity_call)
            transport["inverse"] = nhd_metric(torch, copied, local[0])
            del copied
            transport_ok = all(exact(x) for x in transport["qkv"].values()) and exact(
                transport["inverse"]
            )
            agreed = torch.tensor([int(transport_ok)], device="cuda")
            dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=handle)
            if not agreed.item():
                diagnostics = [None] * degree
                dist.all_gather_object(
                    diagnostics, {**identity, "step": step, "transport": transport}, group=handle
                )
                return {
                    "passed": False,
                    "attention_skipped": "raw transport mismatch",
                    "ranks": diagnostics,
                }

            reference = producer(full, prior_full)
            repeated = producer(full, prior_full)
            full_repeat = {
                "output": nhd_metric(torch, repeated[0], reference[0]),
                "kmean": compared(torch, repeated[1], reference[1]),
                "vscale": compared(torch, repeated[2], reference[2]),
            }
            del repeated
            prior_comparison = (
                {}
                if step == 0
                else {
                    "kmean": compared(torch, prior_rank[0], prior_full[0][hs:he]),
                    "vscale": compared(torch, prior_rank[1], prior_full[1][hs:he]),
                }
            )
            returned = {}

            def compute(q, k, v, *, prior=prior_rank, returned=returned, reference=reference):
                output, km, vs = producer((q, k, v), prior)
                returned["kmean"], returned["vscale"] = km, vs
                returned["head_output_vs_full"] = nhd_metric(
                    torch, output, reference[0][:, :, hs:he]
                )
                return output

            output = attention_ulysses.exchange(*local, handle, compute)
            first_stats = (returned["kmean"], returned["vscale"])
            head_output = returned["head_output_vs_full"]
            output_metric = nhd_metric(torch, output, reference[0][:, start:stop])
            # Prefix is global before redistribution. Compare its rows inside each
            # rank's token shard, not the first PREFIX rows of every local shard.
            prefix_stop = max(start, min(stop, PREFIX))
            prefix_metric = (
                None
                if prefix_stop == start
                else nhd_metric(
                    torch, output[:, : prefix_stop - start], reference[0][:, start:prefix_stop]
                )
            )
            suffix_start = min(stop, max(start, PREFIX))
            suffix_metric = (
                None
                if suffix_start == stop
                else nhd_metric(
                    torch, output[:, suffix_start - start :], reference[0][:, suffix_start:stop]
                )
            )
            repeat = attention_ulysses.exchange(*local, handle, compute)
            repeats = {
                "output": nhd_metric(torch, repeat, output),
                "kmean": compared(torch, returned["kmean"], first_stats[0]),
                "vscale": compared(torch, returned["vscale"], first_stats[1]),
            }
            del output, repeat
            stats = {
                "kmean": compared(torch, first_stats[0], reference[1][hs:he]),
                "vscale": compared(torch, first_stats[1], reference[2][hs:he]),
            }
            times = []
            if step == 1:
                # Identical prior inputs on every repeat; timing outputs never
                # replace the cached statistics of the simulated step.
                for rep in range(3):
                    calls = [
                        (
                            "full_reference",
                            lambda full=full, prior=prior_full: producer(full, prior),
                        ),
                        (
                            "two_rank_exchange_and_producer",
                            lambda local=local, prior=prior_rank: attention_ulysses.exchange(
                                *local, handle, lambda q, k, v: producer((q, k, v), prior)[0]
                            ),
                        ),
                    ]
                    for name, call in calls if rep % 2 == 0 else list(reversed(calls)):
                        value, measurement = timed(torch, dist, handle, call)
                        times.append({"kind": name, "repeat": rep, **measurement})
                        del value
            regenerated = torch.randn(
                packed.shape,
                device="cuda",
                dtype=torch.bfloat16,
                generator=torch.Generator(device="cuda").manual_seed(seed + step),
            )
            custody = compared(torch, packed[0], regenerated[0])
            del regenerated
            metrics = [
                custody,
                output_metric,
                head_output,
                *stats.values(),
                *repeats.values(),
                *full_repeat.values(),
                *prior_comparison.values(),
            ]
            if prefix_metric is not None:
                metrics.append(prefix_metric)
            if suffix_metric is not None:
                metrics.append(suffix_metric)
            steps.append(
                {
                    "step": step,
                    "statistics": "bootstrap" if step == 0 else "previous-step",
                    "seed": seed + step,
                    "transport": transport,
                    "full_repeat": full_repeat,
                    "prior_stats_vs_full_slice": prior_comparison,
                    "returned_stats_vs_full_slice": stats,
                    "head_output_vs_full": head_output,
                    "inverse_output_vs_full_token_shard": output_metric,
                    "protected_prefix_local_rows": [start, prefix_stop],
                    "protected_prefix_output": prefix_metric,
                    "suffix_global_rows": [suffix_start, stop],
                    "suffix_output": suffix_metric,
                    "same_prior_repeat": repeats,
                    "input_unchanged": custody,
                    "timings": times,
                    "passed": transport_ok and all(exact(x) for x in metrics),
                }
            )
            prior_full, prior_rank = reference[1:], first_stats
            if step == 1:
                del calls, call
            del reference, returned, full, local, packed, compute, identity_call
        report = {
            **identity,
            "token_range": [start, stop],
            "head_range": [hs, he],
            "steps": steps,
            "prefix_tokens": PREFIX,
            "sink_query_key_blocks": [0, (PREFIX + 63) // 64],
            "global_chunk_rows": CHUNK,
            "global_rotary_shape": list(freqs.shape),
            "rms_weight_shapes": [list(x.shape) for x in norm],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "total_device_bytes": total,
            "elapsed_seconds": time.perf_counter() - begun,
            "kitchen_version": importlib.metadata.version("comfy-kitchen"),
            "kernel_artifact": selection.artifact,
            "native_extension": {
                "path": str(backend._C.__file__),
                "sha256": hashlib.sha256(Path(backend._C.__file__).read_bytes()).hexdigest(),
            },
            "source_sha256": {
                m.__name__: hashlib.sha256(Path(inspect.getfile(m)).read_bytes()).hexdigest()
                for m in (attention_ulysses, attention_producer, kitchen, backend)
            },
        }
        rows = [None] * degree
        dist.all_gather_object(rows, report, group=handle)
        return {
            "seed": seed,
            "shape": [1, TOKENS, HEADS, DIM],
            "ranks": rows,
            "passed": all(all(s["passed"] for s in r["steps"]) for r in rows),
            "limits": [
                "Synthetic raw QKV, existing fused Kitchen producer and actual two-rank Ulysses.",
                "No H3 projection, LoRA, video, quality or full-pipeline identity claim.",
                "Full56 reference runs per rank; distributed timing includes exchange and compute.",
                "Global RoPE supplied explicitly; production retention is not implemented.",
                "Three warmed samples check kernel regression only.",
            ],
        }
