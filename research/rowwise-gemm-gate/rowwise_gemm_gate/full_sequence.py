"""Measure the complete GEMM call with bounded diagnostic memory and alternating order."""

import math
import statistics
import time

from .compare import SHAPES, _version


def _measure(torch, function):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    started = time.perf_counter()
    begin.record()
    result = function()
    end.record()
    end.synchronize()
    row = {
        "cuda_event_ms": begin.elapsed_time(end),
        "synchronized_host_ms": (time.perf_counter() - started) * 1000,
        "peak_allocated_above_operands_bytes": torch.cuda.max_memory_allocated() - baseline,
        "required_output_bytes": result.numel() * result.element_size(),
    }
    # Drop the old complete output before starting another timed call. Otherwise the
    # assignment to the next result temporarily retains both full-sequence outputs.
    del result
    return row


def _compare_outputs(torch, candidate, reference):
    finite = torch.ones((), device=candidate.device, dtype=torch.bool)
    maximum = torch.zeros((), device=candidate.device, dtype=torch.float32)
    different = torch.zeros((), device=candidate.device, dtype=torch.int64)
    for start in range(0, candidate.shape[0], 128):
        actual = candidate[start : start + 128]
        expected = reference[start : start + 128]
        finite.logical_and_(torch.isfinite(actual).all())
        different.add_((actual != expected).sum())
        maximum = torch.maximum(maximum, (actual.float() - expected.float()).abs().max())
    return {
        "bytes_equal": bool(torch.equal(candidate.view(torch.uint8), reference.view(torch.uint8))),
        "finite": bool(finite.item()),
        "different_elements": int(different.item()),
        "max_abs": float(maximum.item()),
    }


def _case(torch, leaves, gemm, generator, name, rows, n, k, repeats):
    x = torch.randn((rows, k), dtype=torch.bfloat16, device="cuda", generator=generator)
    w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda", generator=generator) * 0.02
    aq, a_scale = leaves.quantize_activation_rowwise(torch, x)
    bq, b_scale = leaves.quantize_activation_rowwise(torch, w)
    del x, w
    b, bs = bq.t(), b_scale.reshape(1, -1)

    def native():
        return gemm.legacy(torch, aq, b, a_scale, bs, torch.bfloat16)

    def candidate():
        return gemm.mm(torch, aq, b, a_scale, bs, torch.bfloat16)

    functions = {"native": native, "candidate": candidate}
    # Both routes compile and warm before any recorded trial.
    for function in functions.values():
        warmed = function()
        torch.cuda.synchronize()
        del warmed
    timings = {name: [] for name in functions}
    order = []
    for iteration in range(repeats):
        labels = ("native", "candidate") if iteration % 2 == 0 else ("candidate", "native")
        for label in labels:
            order.append(label)
            timings[label].append(_measure(torch, functions[label]))
    original, changed = native(), candidate()
    numerical = _compare_outputs(torch, changed, original)
    return {
        "name": name,
        "shape_mnk": [rows, n, k],
        "candidate_tiles": math.ceil(rows / gemm.tile_rows(n)),
        "candidate_tile_rows": gemm.tile_rows(n),
        "trial_order": order,
        "trials": timings,
        "median_cuda_event_ms": {
            label: statistics.median(row["cuda_event_ms"] for row in samples)
            for label, samples in timings.items()
        },
        "median_synchronized_host_ms": {
            label: statistics.median(row["synchronized_host_ms"] for row in samples)
            for label, samples in timings.items()
        },
        "numerical": numerical,
    }


def compare(selected, rows, seed, repeats):
    import torch

    from cozy_runtime.internal.encoding import leaves, rowwise_gemm

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This study requires the explicit SM120 candidate GPU")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    cases = SHAPES if selected == "all" else {selected: SHAPES[selected]}
    with torch.inference_mode():
        results = [
            _case(torch, leaves, rowwise_gemm, generator, name, rows, n, k, repeats)
            for name, (_small_m, n, k) in cases.items()
        ]
    return {
        "device": torch.cuda.get_device_name(),
        "software": {
            name: _version(name) for name in ("cozy-runtime", "torch", "tensorfs", "triton")
        },
        "rows": rows,
        "seed": seed,
        "repeats_per_path": repeats,
        "cases": results,
        "limits": [
            "Complete synthetic GEMMs on resident encoded operands, not full H3 rendering",
            "Candidate scalar, FP32 scratch allocation, and every chunk-loop dispatch included",
            "Native reference uses one complete full-sequence rowwise call",
            "Quantization, model paging and CPU transfer excluded identically",
            "Numerical diagnostics run after timed trials in bounded 128-row slices",
            "Peak allocated delta includes the required BF16 output and Torch-visible scratch",
        ],
    }
