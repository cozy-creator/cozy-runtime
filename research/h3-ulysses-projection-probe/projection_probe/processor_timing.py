"""Bounded, complete sparse-attention timing with actual H3 processors.

Global rotary tables are supplied explicitly to the candidate request state here.
The separate lifecycle fixture qualifies their production pre-split retention.
"""

from __future__ import annotations

from .metrics import compared
from .probe import DIM, HEADS, INPUT_WIDTH, TOKENS, exact, quantizer_route, token_bounds

CURRENT = "kitchen-sol-fp4-lowmem"
CANDIDATE = "kitchen-sol-producer-fp4-fixed-means-shared-qkv"
PREFIX = 257


def _module(torch, projections, device):
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

    # Only the supplied real FP8 leaves and two small RMS weights are materialized.
    with torch.device("meta"):
        module = MiniMaxH3Attention(INPUT_WIDTH, HEADS, DIM, qk_norm_eps=1e-6)
    for name in ("to_q", "to_k", "to_v"):
        setattr(module, name, getattr(projections, name))
    module.to_out = torch.nn.ModuleList([projections.to_out, torch.nn.Dropout(0.0)])
    for norm, low, high in ((module.norm_q, 0.875, 1.125), (module.norm_k, 1.125, 0.875)):
        norm.to_empty(device=device).to(dtype=torch.bfloat16)
        norm.weight.data.copy_(torch.linspace(low, high, DIM, device=device).bfloat16())
    return module.eval()


def run(torch, handle, parallel_config, hidden, attn_projections, seed):
    """Use the existing full hidden fixture and baked FP8/LoRA leaves on both ranks.

    Return all-rank metrics and three paired timings. No cross-recipe equality or
    performance pass threshold is imposed. Preparation and comparisons are untimed.
    """
    import hashlib
    import importlib.metadata
    import inspect
    import os
    import time
    from pathlib import Path

    import torch.distributed as dist
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

    from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope
    from cozy_runtime.internal import (
        accel,
        attention,
        attention_producer,
        attention_sol,
        attention_ulysses,
    )
    from cozy_runtime.internal.encoding import DeviceFacts, leaves
    from cozy_runtime.models.minimax_h3.turbo import _LoRAHook

    rank, degree = dist.get_rank(handle), dist.get_world_size(handle)
    if degree != 2 or attention_ulysses.group(parallel_config) is not handle:
        raise ValueError("Complete-processor comparison requires the actual two-rank group")
    identity = {
        "rank": rank,
        "world_size": degree,
        "pid": os.getpid(),
        "uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
        "sm": list(torch.cuda.get_device_capability()),
        "group_global_ranks": dist.get_process_group_ranks(handle),
        "runtime": importlib.metadata.version("cozy-runtime"),
        "tensorfs": importlib.metadata.version("tensorfs"),
    }
    identities = [None] * degree
    dist.all_gather_object(identities, identity, group=handle)
    if (
        sorted(item["rank"] for item in identities) != [0, 1]
        or len({item["uuid"] for item in identities}) != 2
        or len({item["pid"] for item in identities}) != 2
        or any(
            item["sm"] != [12, 0]
            or item["group_global_ranks"] != [0, 1]
            or item["runtime"] != identity["runtime"]
            or not item["runtime"].startswith("0.23.5+dev.")
            or item["tensorfs"] != identity["tensorfs"]
            for item in identities
        )
    ):
        raise ValueError(
            "Research fixture requires two distinct SM120 ranks on the same private SDK"
        )
    if tuple(hidden.shape) != (1, TOKENS, INPUT_WIDTH) or hidden.dtype != torch.bfloat16:
        raise ValueError("Complete-processor comparison requires the full H3 hidden geometry")
    if hidden.device.type != "cuda" or torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("This bounded fixture requires the assigned Blackwell CUDA device")
    free, total = torch.cuda.mem_get_info()
    reserved, allocated = torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
    admission = {
        "stage": "after caller-owned full hidden and projection weights",
        "free_bytes": free,
        "total_bytes": total,
        "reserved_bytes": reserved,
        "allocated_bytes": allocated,
        "available_additional_bytes": free + reserved - allocated,
        "required_additional_bytes": 22 << 30,
        "conservative_additional_tensor_estimate_bytes": 18_306_126_739,
    }
    # Source liveness sums ~17.05GiB, deliberately including nonoverlapping
    # stages;22GiB leaves ~4.95GiB for unmodeled workspaces/fragmentation. The
    # caller's already-live ~1.25GiB fixture does not need allocating again.
    if admission["available_additional_bytes"] < admission["required_additional_bytes"]:
        raise ValueError(f"Complete-processor comparison lacks additional headroom: {admission}")
    for name in ("to_q", "to_k", "to_v", "to_out"):
        projection = getattr(attn_projections, name)
        hooks = tuple(projection._forward_hooks.values())
        if len(hooks) != 1 or not isinstance(hooks[0], _LoRAHook):
            raise ValueError("Use the existing real Turbo hook on every supplied FP8 projection")
        if not hooks[0]._cozy_baked or hooks[0].owner._arming is None:
            raise ValueError("This single performance case requires the baked Turbo state")

    start, stop = token_bounds(rank)
    local = hidden[:, start:stop]
    device = DeviceFacts(
        "cuda",
        torch.cuda.get_device_name(),
        120,
        accel.driver_version(),
        "complete sparse H3 processor diagnostic",
        index=torch.cuda.current_device(),
    )
    ready = {name: attention.pinned(name, device)[0] for name in (CURRENT, CANDIDATE)}
    torch.cuda.reset_peak_memory_stats()
    states = {name: attention_producer.RequestState() for name in ready}
    modules = {}
    rotary_key = object()
    with torch.inference_mode():
        positions = torch.arange(TOKENS, device=hidden.device)
        positions = torch.stack(
            (positions // (84 * 48), (positions // 84) % 48, positions % 84),
            dim=-1,
        )
        rotary = MiniMaxH3RotaryPosEmbed().to(hidden.device)(positions)
        local_rotary = tuple(x[start:stop] for x in rotary)
        del positions
        for name, selection in ready.items():
            module = _module(torch, attn_projections, hidden.device)
            module.processor._attention_backend = selection.member
            if name == CANDIDATE:
                attention_producer.install(module)
                module._cozy_producer_rotary_key = rotary_key
                # Explicit research setup, not a substitute for the real retention test.
                states[name].parallel_rotary[rotary_key] = rotary
            module.processor._parallel_config = parallel_config
            attention_sol.install_site(module, "dit", "transformer_blocks.0.attn")
            modules[name] = module

        observations = []

        def call(name, step):
            with (
                attention_scope(AttentionLayout(TOKENS, PREFIX, step, 4)),
                attention_sol.observing(producer_state=states[name]) as counts,
                attention_ulysses.observing() as served,
            ):
                output = modules[name](local, local_rotary)
            observations.append(
                {
                    "recipe": name,
                    "step": step,
                    "counts": dict(counts),
                    "implementation": attention.implementations(served),
                }
            )
            return output

        references = {}
        timings = []
        metrics = {name: [] for name in ready}
        quantizer_before = quantizer_route(leaves)
        try:
            for name in ready:
                first = call(name, 4)
                metrics[name].append(
                    {"phase": "bootstrap finite", **compared(torch, first[0], first[0])}
                )
                del first
                references[name] = call(name, 5)
                metrics[name].append(
                    {
                        "phase": "previous-step finite",
                        **compared(torch, references[name][0], references[name][0]),
                    }
                )
            # The candidate's same-step replay retains the original step4 statistics;
            # repeating step5 does not silently turn this into step6/7/8.
            for repetition in range(3):
                order = (CURRENT, CANDIDATE) if repetition % 2 == 0 else (CANDIDATE, CURRENT)
                for name in order:
                    dist.barrier(group=handle)
                    torch.cuda.synchronize()
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    wall = time.perf_counter()
                    begin.record()
                    output = call(name, 5)
                    end.record()
                    end.synchronize()
                    timings.append(
                        {
                            "recipe": name,
                            "repetition": repetition,
                            "wall_ms": (time.perf_counter() - wall) * 1000,
                            "cuda_ms": float(begin.elapsed_time(end)),
                        }
                    )
                    metrics[name].append(
                        {
                            "phase": "same step5 repeat",
                            "repetition": repetition,
                            **compared(torch, output[0], references[name][0]),
                        }
                    )
                    del output
            regenerated = torch.randn(
                hidden.shape,
                dtype=hidden.dtype,
                device=hidden.device,
                generator=torch.Generator(device=hidden.device).manual_seed(seed),
            )
            input_unchanged = compared(torch, hidden[0], regenerated[0])
            del regenerated
            rows = [None] * degree
            row = {
                **identity,
                "token_range": [start, stop],
                "timings": timings,
                "metrics": metrics,
                "observations": observations,
                "lora_state": "baked synthetic weights",
                "input_unchanged": input_unchanged,
                "quantizer_before": quantizer_before,
                "quantizer_after": quantizer_route(leaves),
                "total_device_bytes": total,
                "memory_admission": admission,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "kernel_artifacts": {name: selection.artifact for name, selection in ready.items()},
                "source_sha256": {
                    module.__name__: hashlib.sha256(
                        Path(inspect.getfile(module)).read_bytes()
                    ).hexdigest()
                    for module in (attention_producer, attention_ulysses, leaves)
                },
            }
            dist.all_gather_object(rows, row, group=handle)
            return {
                "seed": seed,
                "ranks": rows,
                "repeat_finite_exact": all(
                    exact(item["input_unchanged"])
                    and all(
                        exact(metric) for values in item["metrics"].values() for metric in values
                    )
                    for item in rows
                ),
                "expected_sparse_calls_observed": all(
                    observation["counts"]
                    == {
                        "sparse": 1,
                        "dense_step": 0,
                        "dense_path": 0,
                        "dense_prefix": 1,
                    }
                    for item in rows
                    for observation in item["observations"]
                ),
                "scope": "Complete sparse H3 attention: FP8 QKV/output projections, "
                "real baked Turbo hooks, RMS/RoPE, attention and actual Ulysses exchange",
                "prefix_tokens": PREFIX,
                "candidate_dense_query_rows": 320,
                "limits": [
                    "Synthetic weights/hidden inputs; no model download, video or quality test.",
                    "Global rotary tables are supplied explicitly; a separate lifecycle fixture "
                    "tests the real retention hook.",
                    "Current materialized attention and the producer have different recipes. "
                    "No cross-recipe output equality asserted.",
                    "Three warm paired repetitions: use maximum-rank wall time. "
                    "No full-generation or universal no-slowdown claim.",
                    "Candidate repeats use replay-safe original step4 statistics; "
                    "actual implementation and call counters are retained.",
                ],
            }
        finally:
            for state in states.values():
                state.clear()
