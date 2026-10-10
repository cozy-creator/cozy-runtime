"""Real rowwise FP8 modules and Turbo hooks at H3's unequal sequence boundaries.

Synthetic deterministic weights exercise actual serving arithmetic. The tiny admitted
checkpoint supplies the Runtime group; this does not qualify checkpoint loading or baking.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import os
import time
from pathlib import Path
from types import SimpleNamespace

from .metrics import compared, operand_layout

TOKENS, HEADS, DIM, DEGREE = 109225, 56, 128, 2
INPUT_WIDTH, OUTPUT_WIDTH, LORA_RANK = 5376, HEADS * DIM, 64


def token_bounds(rank, tokens=TOKENS, degree=DEGREE):
    width, remainder = divmod(tokens, degree)
    start = rank * width + min(rank, remainder)
    return start, start + width + (rank < remainder)


def exact(metric):
    return metric["bytes_equal"] and metric["nonfinite_elements"] == 0


def quantizer_route(leaves):
    if not hasattr(leaves._triton_rowwise_kernel, "_cached"):
        return "not initialized"
    return "triton" if leaves._triton_rowwise_kernel._cached is not None else "eager fallback"


def projections(torch, state, seed):
    from cozy_runtime.internal.encoding.leaves import RowwiseNativeLeaf, quantize_activation_rowwise
    from cozy_runtime.models.minimax_h3.turbo import LoRAFactors, _LoRAHook

    generator = torch.Generator(device="cuda").manual_seed(seed)
    factors_by_path = {}
    owner = SimpleNamespace(_arming=None)
    if state != "inactive":
        owner._arming = SimpleNamespace(
            overlay=SimpleNamespace(get_submodule=factors_by_path.__getitem__)
        )
    sites = {}
    for name in ("to_q", "to_k", "to_v", "to_out"):
        incoming, outgoing = (
            (OUTPUT_WIDTH, INPUT_WIDTH) if name == "to_out" else (INPUT_WIDTH, OUTPUT_WIDTH)
        )
        factors = LoRAFactors(incoming, outgoing, LORA_RANK, 0.25).to("cuda")
        factors.lora_down.normal_(std=0.01, generator=generator)
        factors.lora_up.normal_(std=0.01, generator=generator)
        factors_by_path[name] = factors
        weight = torch.randn(
            outgoing, incoming, dtype=torch.bfloat16, device="cuda", generator=generator
        ).mul_(0.01)
        if state == "baked":
            weight.add_((factors.lora_up @ factors.lora_down) * factors.scale)
        data, scale = quantize_activation_rowwise(torch, weight)
        original = torch.nn.Linear(
            incoming, outgoing, bias=False, dtype=torch.bfloat16, device="meta"
        )
        leaf = RowwiseNativeLeaf("fp8-rowwise/1").leaf(
            torch, {"data": data, "scale": scale}, original, torch.bfloat16
        )
        hook = _LoRAHook(owner, name)
        hook._cozy_baked = state == "baked"
        leaf.register_forward_hook(hook)
        sites[name] = leaf
        del weight, original
    return SimpleNamespace(**sites, heads=HEADS, head_dim=DIM), owner


def materialized(torch, attn, hidden):
    return torch.cat([getattr(attn, n)(hidden) for n in ("to_q", "to_k", "to_v")], -1)


def materialized_projections(torch, attn, hidden):
    del torch
    return tuple(getattr(attn, n)(hidden) for n in ("to_q", "to_k", "to_v"))


def chunked(torch, attn, hidden):
    from cozy_runtime.internal import attention_producer

    counts = {}
    result = torch.cat(
        list(
            attention_producer.qkv_chunks(attn, hidden, hidden.shape[1], shared=True, counts=counts)
        ),
        0,
    ).unsqueeze(0)
    return result, counts


def prepared_qkv(torch, attn, hidden):
    del torch
    from cozy_runtime.internal import attention_producer

    counts = {}
    result = attention_producer._project_qkv(attn, hidden, hidden.shape[1], counts)
    return result, counts


def timed(torch, dist, handle, call):
    dist.barrier(group=handle)
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    value = call()
    stop.record()
    stop.synchronize()
    del value
    return {
        "wall_ms": (time.perf_counter() - wall) * 1000,
        "cuda_ms": float(start.elapsed_time(stop)),
    }


def run(seed, handle):
    import torch
    import torch.distributed as dist

    from cozy_runtime.internal import attention_producer, attention_ulysses
    from cozy_runtime.internal.encoding import leaves
    from cozy_runtime.models.minimax_h3 import turbo

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
    if (
        degree != DEGREE
        or sorted(x["rank"] for x in identities) != [0, 1]
        or len({x["uuid"] for x in identities}) != DEGREE
        or len({x["pid"] for x in identities}) != DEGREE
        or any(x["sm"] != [12, 0] or x["group_global_ranks"] != [0, 1] for x in identities)
    ):
        raise RuntimeError("This projection fixture requires two distinct granted SM120 ranks")
    free, total = torch.cuda.mem_get_info()
    if free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated() < 28 << 30:
        raise RuntimeError("Full H3 geometry needs28GiB headroom; no automatic downsizing")
    torch.cuda.reset_peak_memory_stats()
    start, stop = token_bounds(rank)
    hs, he = rank * (HEADS // degree), (rank + 1) * (HEADS // degree)
    results = []
    with torch.inference_mode():
        hidden = torch.randn(
            1,
            TOKENS,
            INPUT_WIDTH,
            dtype=torch.bfloat16,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        local_hidden = hidden[:, start:stop]
        for state in ("inactive", "baked", "unbaked"):
            quantizer_before = quantizer_route(leaves)
            attn, owner = projections(torch, state, seed + 100)
            reference, full_counts = chunked(torch, attn, hidden)
            whole = materialized(torch, attn, hidden)
            whole_vs_chunks = compared(torch, whole[0], reference[0])
            del whole
            local_output, local_counts = chunked(torch, attn, local_hidden)
            local_vs_full = compared(torch, local_output[0], reference[0, start:stop])
            repeated, _ = chunked(torch, attn, local_hidden)
            repeat_metric = compared(torch, repeated[0], local_output[0])
            del repeated
            local_whole = materialized(torch, attn, local_hidden)
            local_whole_metric = compared(torch, local_whole[0], reference[0, start:stop])
            del local_whole
            full_qkv = tuple(x.unflatten(-1, (HEADS, DIM)) for x in reference.chunk(3, -1))
            local_qkv = tuple(x.unflatten(-1, (HEADS, DIM)) for x in local_output.chunk(3, -1))
            prepared, prepared_counts = prepared_qkv(torch, attn, local_hidden)
            assembly_metrics = {
                name: compared(torch, value[0], original[0])
                for name, value, original in zip(("q", "k", "v"), prepared, local_qkv, strict=True)
            }
            transport = {}

            def identity_call(q, k, v, *, transport=transport, full_qkv=full_qkv):
                transport["layouts"] = [operand_layout(x) for x in (q, k, v)]
                transport["exchanged_projection_vs_full"] = {
                    name: compared(torch, x[0], ref[0, :, hs:he])
                    for name, x, ref in zip(("q", "k", "v"), (q, k, v), full_qkv, strict=True)
                }
                return q.clone()

            restored = attention_ulysses.exchange(*prepared, handle, identity_call)
            transport["inverse"] = compared(torch, restored[0], local_qkv[0][0])
            del restored, identity_call, full_qkv, local_qkv, local_output, reference, prepared
            out_input = torch.randn(
                1,
                TOKENS,
                OUTPUT_WIDTH,
                dtype=torch.bfloat16,
                device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(seed + 200),
            )
            out_reference = attn.to_out(out_input)
            out_local = attn.to_out(out_input[:, start:stop])
            out_metric = compared(torch, out_local[0], out_reference[0, start:stop])
            out_repeat = attn.to_out(out_input[:, start:stop])
            out_repeat_metric = compared(torch, out_repeat[0], out_local[0])
            del out_reference, out_local, out_repeat
            out_regenerated = torch.randn(
                out_input.shape,
                dtype=out_input.dtype,
                device=out_input.device,
                generator=torch.Generator(device="cuda").manual_seed(seed + 200),
            )
            out_custody = compared(torch, out_input[0], out_regenerated[0])
            del out_input, out_regenerated
            times = []
            # Current separate projection outputs versus proposed CP QKV materialization.
            # The single-GPU producer streams chunks instead; this is a CP packing control.
            # Norm, rotary, attention, transfer, FFN, VAE and all video work are excluded.
            for rep in range(3):
                kinds = ("local_separate_projections", "local_cp_projection_assembly")
                for kind in kinds if rep % 2 == 0 else reversed(kinds):
                    fn = (
                        materialized_projections
                        if kind == "local_separate_projections"
                        else prepared_qkv
                    )
                    times.append(
                        {
                            "kind": kind,
                            "repetition": rep,
                            **timed(
                                torch,
                                dist,
                                handle,
                                lambda fn=fn, attn=attn: fn(torch, attn, local_hidden),
                            ),
                        }
                    )
            metrics = [
                whole_vs_chunks,
                local_vs_full,
                repeat_metric,
                local_whole_metric,
                transport["inverse"],
                *transport["exchanged_projection_vs_full"].values(),
                *assembly_metrics.values(),
                out_metric,
                out_repeat_metric,
                out_custody,
            ]
            results.append(
                {
                    "lora_state": state,
                    "lora_rank": LORA_RANK,
                    "whole_vs_full_chunks": whole_vs_chunks,
                    "local_chunks_vs_full_slice": local_vs_full,
                    "local_materialized_vs_full_slice": local_whole_metric,
                    "local_repeat": repeat_metric,
                    "output_projection_vs_full_slice": out_metric,
                    "output_projection_repeat": out_repeat_metric,
                    "output_projection_input_unchanged": out_custody,
                    "quantizer_before": quantizer_before,
                    "quantizer_after": quantizer_route(leaves),
                    "transport": transport,
                    "full_chunk_counts": full_counts,
                    "local_chunk_counts": local_counts,
                    "cp_assembly_counts": prepared_counts,
                    "cp_assembly_vs_local_chunks": assembly_metrics,
                    "timings": times,
                    "passed": all(exact(m) for m in metrics),
                }
            )
            del attn, owner
        regenerated = torch.randn(
            hidden.shape,
            dtype=hidden.dtype,
            device=hidden.device,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        unchanged = compared(torch, hidden[0], regenerated[0])
        report = {
            **identity,
            "token_range": [start, stop],
            "head_range": [hs, he],
            "results": results,
            "input_unchanged": unchanged,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "total_device_bytes": total,
            "source_sha256": {
                m.__name__: hashlib.sha256(Path(inspect.getfile(m)).read_bytes()).hexdigest()
                for m in (attention_producer, attention_ulysses, leaves, turbo)
            },
        }
        reports = [None] * degree
        dist.all_gather_object(reports, report, group=handle)
        return {
            "seed": seed,
            "input_shape": [1, TOKENS, INPUT_WIDTH],
            "qkv_projection_shape": [OUTPUT_WIDTH, INPUT_WIDTH],
            "ranks": reports,
            "passed": all(
                exact(r["input_unchanged"]) and all(s["passed"] for s in r["results"])
                for r in reports
            ),
            "limits": [
                "Synthetic weights; actual FP8 native leaves and real Turbo hooks.",
                "Does not qualify TensorFS LoRA baking, full Transformer or video determinism.",
                "Three warm projection timings, not full-generation performance.",
                "Exchanged-projection comparison includes upstream arithmetic; "
                "inverse identity is the copy control.",
            ],
        }
