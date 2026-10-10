"""Actual public dense adapters; fixed inputs with long K/V and full/short Q."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import time
from pathlib import Path
from typing import Any

from .metrics import accumulate, compared, finish, new_metric, operand_layout

N, H, D = 109225, 56, 128
OPTIONS = {
    "attn_mask": None,
    "dropout_p": 0.0,
    "is_causal": False,
    "scale": D**-0.5,
    "enable_gqa": False,
}


def fresh(torch: Any, values: Any, start: int = 0, stop: int = H) -> list[Any]:
    return [x[:, :, start:stop].clone(memory_format=torch.contiguous_format) for x in values]


def canonical(value: Any) -> Any:
    return value[0].transpose(0, 1).contiguous()


def guarded(local: Any, values: Any, observations: list[dict[str, Any]]) -> Any:
    implementation = local.impl(*values, OPTIONS)
    if implementation == "math":
        raise RuntimeError(
            "SDPA selected math at109225 K/V rows; "
            "refusing large score materialization without switching backend"
        )
    result = local.forward(*values, **OPTIONS)
    if result.shape != values[0].shape or result.dtype != values[0].dtype:
        raise RuntimeError("Dense adapter changed output geometry or dtype")
    observations.append(
        {
            "heads": values[0].shape[2],
            "query_rows": values[0].shape[1],
            "kv_rows": values[1].shape[1],
            "implementation": implementation,
        }
    )
    return result


def run(kernel: str, query: str, seed: int) -> dict[str, Any]:
    import torch

    from cozy_runtime.internal import accel, attention
    from cozy_runtime.internal.encoding import DeviceFacts

    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This control is scoped to Hopper SM90; no fallback")
    qn = 257 if query == "prefix" else N
    device = DeviceFacts(
        "cuda",
        torch.cuda.get_device_name(),
        90,
        accel.driver_version(),
        "dense prefix diagnostic",
        index=torch.cuda.current_device(),
    )
    candidate = attention.BY_NAME[kernel]
    artifact = attention._compiled(candidate, device, wait=True)
    attention._import(candidate)
    local = candidate.local
    free, _ = torch.cuda.mem_get_info()
    if free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated() < 28 << 30:
        raise RuntimeError("Requires28GiB free/reusable; no shape reduction")
    torch.cuda.reset_peak_memory_stats()
    observations: list[dict[str, Any]] = []
    with torch.inference_mode():
        generator = torch.Generator(device="cuda").manual_seed(seed)
        full_inputs = [
            torch.randn(1, N, H, D, dtype=torch.bfloat16, device="cuda", generator=generator)
            for _ in range(3)
        ]
        values = [full_inputs[0][:, :qn].clone(), full_inputs[1], full_inputs[2]]
        del full_inputs
        if local.impl(*values, OPTIONS) == "math":
            return {
                "supported": False,
                "kernel": kernel,
                "query_rows": qn,
                "kv_rows": N,
                "reason": "SDPA selected math; no dense score allocation or backend substitution",
            }
        reference = canonical(guarded(local, fresh(torch, values), observations))
        again = canonical(guarded(local, fresh(torch, values), observations))
        full_repeat = compared(torch, again, reference)
        del again
        rows = []
        for degree in (2, 4, 7, 8):
            width = H // degree
            metric, repeat = new_metric(), new_metric()
            for start in range(0, H, width):
                actual = canonical(
                    guarded(local, fresh(torch, values, start, start + width), observations)
                )
                again = canonical(
                    guarded(local, fresh(torch, values, start, start + width), observations)
                )
                accumulate(torch, metric, actual, reference[start : start + width])
                accumulate(torch, repeat, again, actual)
                del actual, again
            rows.append(
                {
                    "local_head_groups": degree,
                    "heads_per_call": width,
                    "vs_full": finish(metric),
                    "repeat": finish(repeat),
                }
            )
        del reference
        # Warm wall/CUDA timing of actual public full-head and28-head calls. No metrics/copies.
        half = fresh(torch, values, 0, 28)
        for operands in (values, half):
            output = guarded(local, operands, observations)
            del output
        timings = []
        for rep in range(3):
            for label, operands in (
                [("full56", values), ("local28", half)]
                if rep % 2 == 0
                else [("local28", half), ("full56", values)]
            ):
                torch.cuda.synchronize()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                wall = time.perf_counter()
                start.record()
                output = guarded(local, operands, observations)
                end.record()
                end.synchronize()
                timings.append(
                    {
                        "geometry": label,
                        "repetition": rep,
                        "cuda_ms": float(start.elapsed_time(end)),
                        "wall_ms": (time.perf_counter() - wall) * 1000,
                    }
                )
                del output
        generator = torch.Generator(device="cuda").manual_seed(seed)
        custody = []
        for index, value in enumerate(values):
            original = torch.randn(
                1, N, H, D, dtype=torch.bfloat16, device="cuda", generator=generator
            )
            if index == 0:
                original = original[:, :qn]
            custody.append(compared(torch, canonical(value), canonical(original)))
            del original
        half_custody = [
            compared(torch, canonical(a), canonical(b[:, :, :28]))
            for a, b in zip(half, values, strict=True)
        ]
        if any(not row["bytes_equal"] for row in custody + half_custody):
            raise RuntimeError("Dense adapter changed timed input bytes")
    path = Path(inspect.getfile(attention)).with_name("attention_upstream.py")
    return {
        "supported": True,
        "kernel": kernel,
        "seed": seed,
        "shape_q": [1, qn, H, D],
        "shape_kv": [1, N, H, D],
        "software": {
            n: importlib.metadata.version(n) for n in ["cozy-runtime", "tensorfs", "torch"]
        },
        "artifact": artifact,
        "adapter_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "full_repeat": full_repeat,
        "partitions": rows,
        "input_custody": custody,
        "timed_half_custody": half_custody,
        "input_layouts": [operand_layout(v) for v in values],
        "calls": observations,
        "timings": timings,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "limits": [
            "One GPU local head groups, no actual collective or pipeline.",
            "No original/candidate change; this measures public adapter geometry sensitivity.",
            "FA3 FP8 remains an explicit unapproved-quality experiment.",
            "Warm three-sample kernel timings are not generation-speed benchmarks.",
        ],
    }
