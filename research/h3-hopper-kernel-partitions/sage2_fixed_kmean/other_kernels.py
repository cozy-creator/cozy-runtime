"""Actual public Hopper BF16 sparse and Kitchen dense calls, with local head groups."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any

from .metrics import accumulate, compared, finish, new_metric, operand_layout

DEGREES = (2, 4, 7, 8)
HEADS, DIM, PREFIX = 56, 128, 257


def fresh(torch: Any, values: Any, start: int = 0, stop: int = HEADS) -> list[Any]:
    return [x[:, :, start:stop].clone(memory_format=torch.contiguous_format) for x in values]


@contextmanager
def sparse_scope(sol: Any, tokens: int, prefix: int, step: int, dense: Any) -> Iterator[None]:
    from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope

    previous, identity = sol._DENSE, sol._DENSE_IDENTITY
    site = sol._SITE.set(sol.Site("diagnostic", "blocks.2.attn"))
    sol.bind_dense(dense, "sageattention")
    try:
        with attention_scope(AttentionLayout(tokens, prefix, step, dense_until_step=4)):
            yield
    finally:
        sol._DENSE, sol._DENSE_IDENTITY = previous, identity
        sol._SITE.reset(site)


def canonical(value: Any) -> Any:
    return value[0].transpose(0, 1).contiguous()


def _public_sparse(
    sol: Any,
    upstream: Any,
    values: Any,
    prefix: int,
    step: int,
    observations: list[dict[str, Any]] | None = None,
) -> Any:
    with (
        sparse_scope(sol, values[0].shape[1], prefix, step, upstream.SAGE.forward),
        sol.observing() as counts,
    ):
        output = sol.SOL.forward(*values, scale=DIM**-0.5)
    expected = {"sparse": 1, "dense_step": 0, "dense_path": 0, "dense_prefix": 1}
    if counts != expected or sol.backend(values[0].device) != "cute_sm90":
        raise RuntimeError(f"BF16 CuTe sparse did not serve the requested call: {counts}")
    if observations is not None:
        observations.append(
            {
                "step": step,
                "heads": values[0].shape[2],
                "counts": dict(counts),
                "implementation": sol.backend(values[0].device),
                "dense_prefix": upstream.SAGE.impl(*values, {}),
            }
        )
    return output


def capture(
    torch: Any,
    check: str,
    values: Any,
    prefix: int,
    step: int,
    loaded: Any,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    from cozy_runtime.internal import attention_sol as sol
    from cozy_runtime.internal import attention_upstream as upstream

    if check == "kitchen_dense":
        return {"output": canonical(upstream.KITCHEN.forward(*values, scale=DIM**-0.5))}
    # Raw sparse separates CuTe head geometry from the selected Sage2 protected prefix.
    raw = sol._native(*values, DIM**-0.5, prefix, loaded)
    combined = _public_sparse(sol, upstream, values, prefix, step, observations)
    return {"raw_sparse": canonical(raw), "output": canonical(combined)}


def timed(torch: Any, call: Callable[[], Any]) -> dict[str, float]:
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    result = call()
    end.record()
    end.synchronize()
    measured = {
        "cuda_ms": float(start.elapsed_time(end)),
        "wall_ms": (time.perf_counter() - wall) * 1000,
    }
    del result
    return measured


def controls(torch: Any, check: str, tokens: int, seed: int) -> dict[str, Any]:
    from cozy_runtime.internal import attention_sol as sol
    from cozy_runtime.internal import attention_upstream as upstream

    free, total = torch.cuda.mem_get_info()
    reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    required = 28 << 30 if tokens > 129 else 2 << 30
    if free + reusable < required:
        raise RuntimeError(f"Requires {required} free/reusable bytes; no shape reduction")
    prefix = min(PREFIX, tokens)
    loaded = None
    compile_rows: list[dict[str, Any]] = []
    if check == "bf16_sparse":
        with sol.observing(kernels=compile_rows):
            sol.specialize(tokens, torch.device("cuda"), 3)
            loaded = sol._ready(tokens, torch.device("cuda"), 4)
        if loaded is None or sol.backend(torch.device("cuda")) != "cute_sm90":
            raise RuntimeError(
                "The requested BF16 CuTe SM90 sparse kernel is not ready; no fallback"
            )
    torch.cuda.reset_peak_memory_stats()
    steps = []
    observations: list[dict[str, Any]] = []
    for step in (4, 5) if check == "bf16_sparse" else (4,):
        case_seed = seed + step - 4
        gen = torch.Generator(device="cuda").manual_seed(case_seed)
        values = [
            torch.randn(1, tokens, HEADS, DIM, dtype=torch.bfloat16, device="cuda", generator=gen)
            for _ in range(3)
        ]
        full = capture(torch, check, fresh(torch, values), prefix, step, loaded, observations)
        again = capture(torch, check, fresh(torch, values), prefix, step, loaded, observations)
        full_repeat = {name: compared(torch, again[name], value) for name, value in full.items()}
        del again
        parts = []
        for degree in DEGREES:
            width = HEADS // degree
            metrics = {name: new_metric() for name in full}
            repeated = {name: new_metric() for name in full}
            protected, suffix = new_metric(), new_metric()
            for start in range(0, HEADS, width):
                operands = fresh(torch, values, start, start + width)
                actual = capture(torch, check, operands, prefix, step, loaded, observations)
                again = capture(
                    torch,
                    check,
                    fresh(torch, values, start, start + width),
                    prefix,
                    step,
                    loaded,
                    observations,
                )
                for name, value in actual.items():
                    accumulate(torch, metrics[name], value, full[name][start : start + width])
                    accumulate(torch, repeated[name], again[name], value)
                accumulate(
                    torch,
                    protected,
                    actual["output"][:, :prefix],
                    full["output"][start : start + width, :prefix],
                )
                if prefix < tokens:
                    accumulate(
                        torch,
                        suffix,
                        actual["output"][:, prefix:],
                        full["output"][start : start + width, prefix:],
                    )
                del actual, again, operands
            parts.append(
                {
                    "local_head_groups": degree,
                    "heads_per_call": width,
                    "vs_full": {k: finish(v) for k, v in metrics.items()},
                    "repeat": {k: finish(v) for k, v in repeated.items()},
                    "protected_prefix": finish(protected),
                    "suffix": finish(suffix) if prefix < tokens else None,
                }
            )
        del full
        # Time actual public full/half-head callables, never instrumentation capture/copies.
        half = fresh(torch, values, 0, 28)
        if check == "bf16_sparse":
            calls = {
                "full56": partial(
                    _public_sparse, sol, upstream, values, prefix, step, observations
                ),
                "local28": partial(_public_sparse, sol, upstream, half, prefix, step, observations),
            }
        else:
            calls = {
                "full56": partial(upstream.KITCHEN.forward, *values, scale=DIM**-0.5),
                "local28": partial(upstream.KITCHEN.forward, *half, scale=DIM**-0.5),
            }
        for call in calls.values():
            result = call()
            del result
        timing = []
        for repetition in range(3):
            for name in list(calls) if repetition % 2 == 0 else list(reversed(calls)):
                timing.append(
                    {"geometry": name, "repetition": repetition, **timed(torch, calls[name])}
                )
        gen = torch.Generator(device="cuda").manual_seed(case_seed)
        custody = []
        for value in values:
            again = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=gen)
            custody.append(compared(torch, canonical(value), canonical(again)))
            del again
        if any(not row["bytes_equal"] for row in custody):
            raise RuntimeError("Kernel changed original input bytes")
        steps.append(
            {
                "step": step,
                "seed": case_seed,
                "full_repeat": full_repeat,
                "partitions": parts,
                "input_custody": custody,
                "input_layout": [operand_layout(v) for v in values],
                "timing": timing,
            }
        )
        del half, values, calls, call
    return {
        "check": check,
        "tokens": tokens,
        "heads": HEADS,
        "head_dim": DIM,
        "protected_prefix": prefix,
        "steps": steps,
        "compile_rows": compile_rows,
        "public_sparse_calls": observations,
        "implementation": sol.backend(torch.device("cuda"))
        if check == "bf16_sparse"
        else "comfy_kitchen.int8_attention",
        "dense_prefix_implementation": (
            "sageattn_qk_int8_pv_fp8_cuda_sm90(per_thread, fp32+fp32, smooth_k)"
        )
        if check == "bf16_sparse"
        else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "total_device_bytes": total,
        "timing_scope": (
            "Single-GPU warm public full/28-head call; no collective, no generation speed claim."
        ),
        "state_scope": (
            "BF16 CuTe computes current-call statistics independently at steps4/5; "
            "no previous-step producer state."
        ),
    }
