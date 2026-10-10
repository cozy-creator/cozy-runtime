"""Use installed Runtime arithmetic on fixed operands; compare bytes in bounded slices."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SHAPES = {
    "qkv": (7168, 5376),
    "attention_out": (5376, 7168),
    "fc1": (28672, 5376),
    "fc2": (5376, 14336),
}
ROWS = {"small": 129, "full": 109225}
DEGREES = (2, 4, 7, 8)


def partitions(rows: int, degree: int) -> list[tuple[int, int]]:
    quotient, extra = divmod(rows, degree)
    result = []
    start = 0
    for rank in range(degree):
        stop = start + quotient + int(rank < extra)
        result.append((start, stop))
        start = stop
    return result


def new_metric() -> dict[str, Any]:
    return {
        "elements": 0,
        "different_elements": 0,
        "first_different_element": None,
        "nonfinite_elements": 0,
        "max_absolute_difference": 0.0,
        "absolute_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "reference_squared_sum": 0.0,
    }


def accumulate(torch: Any, result: dict[str, Any], actual: Any, reference: Any) -> None:
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError("Comparison requires identical geometry and dtype")
    columns = math.prod(actual.shape[1:])
    block_rows = max(1, min(512, (1 << 20) // columns))
    for start in range(0, actual.shape[0], block_rows):
        a = actual[start : start + block_rows].contiguous()
        b = reference[start : start + block_rows].contiguous()
        different = (
            (a.view(torch.uint8) != b.view(torch.uint8))
            .reshape(a.numel(), a.element_size())
            .any(dim=1)
        )
        count = int(different.sum().item())
        if count and result["first_different_element"] is None:
            first = int(different.to(torch.uint8).argmax().item())
            result["first_different_element"] = result["elements"] + first
        finite = torch.isfinite(a.float()) & torch.isfinite(b.float())
        nonfinite = int((~finite).sum().item())
        delta = torch.where(finite, a.float() - b.float(), 0.0)
        ref = torch.where(finite, b.float(), 0.0)
        result["elements"] += a.numel()
        result["different_elements"] += count
        result["nonfinite_elements"] += nonfinite
        result["max_absolute_difference"] = max(
            result["max_absolute_difference"], float(delta.abs().max().item())
        )
        result["absolute_error_sum"] += float(delta.abs().sum().item())
        result["squared_error_sum"] += float(delta.square().sum().item())
        result["reference_squared_sum"] += float(ref.square().sum().item())


def finish(result: dict[str, Any]) -> dict[str, Any]:
    return {
        **result,
        "bytes_equal": result["different_elements"] == 0,
        "mean_absolute_difference": result["absolute_error_sum"] / result["elements"],
        "relative_l2": (
            math.sqrt(result["squared_error_sum"] / result["reference_squared_sum"])
            if result["reference_squared_sum"]
            else None
        ),
    }


def compared(torch: Any, actual: Any, reference: Any) -> dict[str, Any]:
    result = new_metric()
    accumulate(torch, result, actual, reference)
    return finish(result)


def measured(torch: Any, function: Any, operand: Any) -> tuple[Any, float]:
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin.record()
    output = function(operand)
    end.record()
    end.synchronize()
    return output, float(begin.elapsed_time(end))


def case(torch: Any, leaves: Any, name: str, rows: int, seed: int) -> dict[str, Any]:
    n, k = SHAPES[name]
    # Two complete outputs coexist only for the full-shape repeat control. Comparison
    # temporaries hold at most 2^20 values each. Include operands, allocator/library slack.
    required = rows * n * 4 + rows * k * 3 + n * k * 4 + (3 << 30)
    free, total = torch.cuda.mem_get_info()
    reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    if required > free + reusable:
        raise RuntimeError(
            f"{name}/{rows} requires about {required} bytes of headroom; "
            f"device has {free + reusable}. No shape was reduced."
        )
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, k), generator=generator, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), generator=generator, dtype=torch.bfloat16, device="cuda")
    weight.mul_(0.02)
    wq, ws = leaves.quantize_activation_rowwise(torch, weight)
    del weight
    aq, scales = leaves.quantize_activation_rowwise(torch, x)
    leaf = SimpleNamespace(data=wq, scale=ws, out_dtype=torch.bfloat16)
    encoded = leaves.PreQuantized(aq, scales, tuple(x.shape))

    def forward(operand: Any) -> Any:
        return leaves.RowwiseNativeLeaf._forward(leaf, operand)

    reference, full_ms = measured(torch, forward, encoded)
    repeat, repeat_ms = measured(torch, forward, encoded)
    full_repeat = compared(torch, repeat, reference)
    del repeat
    plain, plain_ms = measured(torch, forward, x)
    plain_full = compared(torch, plain, reference)
    del plain
    results = []
    for degree in DEGREES:
        quant_codes, quant_scales, fixed_codes, complete, repeated = (
            new_metric() for _ in range(5)
        )
        durations = {"fixed_codes_gemm_ms": 0.0, "complete_leaf_ms": 0.0}
        for start, stop in partitions(rows, degree):
            chunk = x[start:stop]
            cq, cs = leaves.quantize_activation_rowwise(torch, chunk)
            accumulate(torch, quant_codes, cq, aq[start:stop])
            accumulate(torch, quant_scales, cs, scales[start:stop])
            # First keep the exact full-call quantized operands, so any difference is
            # after quantization; then run the ordinary leaf on the original BF16 rows.
            fixed = leaves.PreQuantized(aq[start:stop], scales[start:stop], tuple(chunk.shape))
            fixed_out, ms = measured(torch, forward, fixed)
            durations["fixed_codes_gemm_ms"] += ms
            accumulate(torch, fixed_codes, fixed_out, reference[start:stop])
            fixed_again, _ = measured(torch, forward, fixed)
            accumulate(torch, repeated, fixed_again, fixed_out)
            del fixed_again, fixed_out
            actual, ms = measured(torch, forward, chunk)
            durations["complete_leaf_ms"] += ms
            accumulate(torch, complete, actual, reference[start:stop])
            del actual, cq, cs
        results.append(
            {
                "degree_simulated_on_one_gpu": degree,
                "row_splits": [b - a for a, b in partitions(rows, degree)],
                "activation_codes": finish(quant_codes),
                "activation_scales": finish(quant_scales),
                "fixed_codes_gemm_output": finish(fixed_codes),
                "ordinary_chunk_leaf_output": finish(complete),
                "same_chunk_configuration_repeat": finish(repeated),
                "isolated_cuda_event_times": durations,
            }
        )
    return {
        "shape_mnk": [rows, n, k],
        "seed": seed,
        "estimated_required_headroom_bytes": required,
        "device_total_bytes": total,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "full_configuration_repeat": full_repeat,
        "ordinary_full_leaf_vs_prequantized": plain_full,
        "full_cuda_event_ms": full_ms,
        "repeat_full_cuda_event_ms": repeat_ms,
        "plain_full_cuda_event_ms": plain_ms,
        "partitions": results,
    }


def run(selected: str, size: str, seed: int) -> dict[str, Any]:
    import torch

    from cozy_runtime.internal.encoding import leaves

    versions = {
        name: importlib.metadata.version(name)
        for name in ("cozy-runtime", "tensorfs", "torch", "triton")
    }
    if versions["cozy-runtime"] != "0.23.2" or versions["tensorfs"] != "0.6.1":
        raise RuntimeError(f"The captured diagnostic requires the reviewed cohort: {versions}")
    if torch.cuda.get_device_capability() not in ((9, 0), (12, 0)):
        raise RuntimeError("This prepared probe is scoped to the owned H100 or RTX 5090")
    started = time.monotonic()
    names = list(SHAPES) if selected == "all" else [selected]
    sizes = list(ROWS) if size == "all" else [size]
    results = []
    with torch.inference_mode():
        for index, name in enumerate(names):
            for label in sizes:
                results.append(
                    {
                        "case": name,
                        "size": label,
                        **case(torch, leaves, name, ROWS[label], seed + index),
                    }
                )
    source = Path(inspect.getfile(leaves))
    return {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "software": versions,
        "cuda_build": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "runtime_leaf_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
        "cases": results,
        "limits": [
            "One-GPU synthetic fixed-operand probe; no collectives or full H3 generation.",
            "The encoded GEMM core is isolated; module bias, LoRA hooks and paging are excluded.",
            "Actual installed Runtime quantization and RowwiseNativeLeaf._forward are called.",
            "Fixed-code comparisons separate GEMM shape dependence from activation quantization.",
            "CPU-readable scalar metrics use bounded float32 reductions; byte equality is exact.",
            "CUDA event timings are diagnostic and exclude comparisons; "
            "not a performance benchmark.",
            "No TF32, precision, deterministic-algorithm, kernel "
            "or production-default setting changes.",
        ],
    }
