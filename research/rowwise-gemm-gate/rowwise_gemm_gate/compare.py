"""Same quantized operands, strict byte equivalence and real kernel evidence."""

import hashlib
import importlib.metadata
import json
import statistics
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

# Deliberately cross the candidate's 4096-row or wide-output 1024-row tile.
SHAPES = {
    "qkv": (4097, 7168, 5376),
    "attention_out": (4103, 5376, 7168),
    "fc1": (1025, 28672, 5376),
    "fc2": (4097, 5376, 14336),
}


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _bytes_equal(torch, actual, expected):
    return bool(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))


def _difference(torch, actual, expected):
    delta = actual.float() - expected.float()
    return {
        "bytes_equal": _bytes_equal(torch, actual, expected),
        "finite": bool(torch.isfinite(actual).all().item()),
        "max_abs": float(delta.abs().max().item()),
        "different_elements": int((actual != expected).sum().item()),
    }


def _timed(torch, function, warmup, repeats):
    for _ in range(warmup):
        result = function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        result = function()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}, result


@contextmanager
def _observe_tiles(gemm):
    """Inspect the real scratch passed to the epilogue, without changing its math."""
    previous = gemm.rescale
    tiles = []

    def observe(accum, scale_a, scale_b, output):
        tiles.append({"shape": list(accum.shape), "bytes": accum.numel() * accum.element_size()})
        return previous(accum, scale_a, scale_b, output)

    gemm.rescale = observe
    try:
        yield tiles
    finally:
        gemm.rescale = previous


def _leaf_check(torch, leaves, gemm, x, aq, a_scale, bq, b_scale, native):
    before = []
    after = []
    quantized_inputs = []
    module = torch.nn.Linear(
        x.shape[1], bq.shape[0], bias=True, device=x.device, dtype=torch.bfloat16
    )
    module.bias.data.fill_(0.125)
    leaf = leaves.RowwiseNativeLeaf("fixture").leaf(
        torch, {"data": bq, "scale": b_scale}, module, torch.bfloat16
    )
    prequantized = leaves.PreQuantized(aq, a_scale, tuple(x.shape))

    def pre_hook(_module, args):
        before.append(args[0] is x)

    def post_hook(_module, args, output):
        after.append(args[0] is x)
        output.add_(0.25)

    original_quantizer = leaves.quantize_activation_rowwise

    def observe_quantizer(control, value):
        quantized_inputs.append(value is prequantized)
        return original_quantizer(control, value)

    handles = [leaf.register_forward_pre_hook(pre_hook), leaf.register_forward_hook(post_hook)]
    leaves.quantize_activation_rowwise = observe_quantizer
    try:
        with _observe_tiles(gemm) as tiles:
            reused = leaf(x, prequantized=prequantized)
        ordinary = leaf(x)
    finally:
        leaves.quantize_activation_rowwise = original_quantizer
        for handle in handles:
            handle.remove()
    expected = native + module.bias
    expected.add_(0.25)
    checks = {
        "prequantized": _difference(torch, reused, expected),
        "ordinary": _difference(torch, ordinary, expected),
        "original_positional_input_reaches_hooks": before == [True, True] and after == [True, True],
        "prequantized_reuse_observed": quantized_inputs == [True, False],
        "stored_operands_preserved_by_identity": leaf.data is bq and leaf.scale is b_scale,
        "actual_leaf_epilogue_tiles": tiles,
    }
    checks["passed"] = all(
        checks[name]["bytes_equal"] and checks[name]["finite"]
        for name in ("prequantized", "ordinary")
    ) and all(
        checks[name]
        for name in (
            "original_positional_input_reaches_hooks",
            "prequantized_reuse_observed",
            "stored_operands_preserved_by_identity",
        )
    )
    return checks


def _trace(torch, native_call, candidate_call, path):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    ranges = []
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        start = time.time_ns()
        native = native_call()
        torch.cuda.synchronize()
        ranges.append({"name": "native", "start_unix_ns": start, "end_unix_ns": time.time_ns()})
        start = time.time_ns()
        candidate = candidate_call()
        torch.cuda.synchronize()
        ranges.append(
            {"name": "scalar_bounded", "start_unix_ns": start, "end_unix_ns": time.time_ns()}
        )
    profiler.export_chrome_trace(str(path))
    document = json.loads(Path(path).read_text())
    kernels = [event for event in document.get("traceEvents", []) if event.get("cat") == "kernel"]
    return {
        "activities": ["CUDA"],
        "synchronized_host_ranges": ranges,
        "range_semantics": "Each named host interval starts and ends at a CUDA synchronize fence",
        "base_time_nanoseconds": document.get("baseTimeNanoseconds"),
        "path_bytes": Path(path).stat().st_size,
        "kernel_count": len(kernels),
        "kernel_names": dict(Counter(event["name"] for event in kernels)),
        "trace_outputs": _difference(torch, candidate, native),
        "scope": "Two warmed calls on identical resident operands; not an H3 trace",
    }


def _case(torch, leaves, gemm, generator, name, shape, warmup, repeats, trace_path):
    m, n, k = shape
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.02
    aq, a_scale = leaves.quantize_activation_rowwise(torch, x)
    bq, b_scale = leaves.quantize_activation_rowwise(torch, w)
    b, bs = bq.t(), b_scale.reshape(1, -1)
    before = tuple(tensor.clone() for tensor in (x, aq, a_scale, bq, b_scale))

    def native_call():
        return gemm.legacy(torch, aq, b, a_scale, bs, torch.bfloat16)

    def candidate_call():
        return gemm.mm(torch, aq, b, a_scale, bs, torch.bfloat16)

    native_time, native = _timed(torch, native_call, warmup, repeats)
    candidate_time, candidate = _timed(torch, candidate_call, warmup, repeats)
    with _observe_tiles(gemm) as tiles:
        repeated = candidate_call()
    exact = _difference(torch, candidate, native)
    repeat_exact = _bytes_equal(torch, repeated, candidate)
    leaf = _leaf_check(torch, leaves, gemm, x, aq, a_scale, bq, b_scale, native)
    trace = _trace(torch, native_call, candidate_call, trace_path) if name == "qkv" else None
    unchanged = all(
        _bytes_equal(torch, tensor, old)
        for tensor, old in zip((x, aq, a_scale, bq, b_scale), before, strict=True)
    )
    expected_rows = list(range(0, m, gemm.tile_rows(n)))
    bounded = (
        len(tiles) == len(expected_rows)
        and [tile["shape"][0] for tile in tiles]
        == [min(gemm.tile_rows(n), m - start) for start in expected_rows]
        and all(tile["bytes"] <= gemm.MAX_FP32_BYTES for tile in tiles)
    )
    return {
        "name": name,
        "shape_mnk": [m, n, k],
        "native": native_time,
        "candidate": candidate_time,
        "comparison": exact,
        "repeated_bytes_equal": repeat_exact,
        "inputs_unchanged": unchanged,
        "actual_epilogue_tiles": tiles,
        "scratch_bound_observed": bounded,
        "leaf": leaf,
        "trace": trace,
        "passed": exact["bytes_equal"]
        and exact["finite"]
        and repeat_exact
        and unchanged
        and bounded
        and leaf["passed"]
        and (
            trace is None or (trace["kernel_count"] > 0 and trace["trace_outputs"]["bytes_equal"])
        ),
    }


def compare(seed, warmup, repeats, trace_path):
    import torch

    from cozy_runtime import _build_provenance
    from cozy_runtime.internal import lora_bake
    from cozy_runtime.internal.encoding import leaves, rowwise_gemm

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This gate requires the explicit SM120 candidate GPU")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode():
        cases = [
            _case(torch, leaves, rowwise_gemm, generator, name, shape, warmup, repeats, trace_path)
            for name, shape in SHAPES.items()
        ]
    return {
        "passed": all(case["passed"] for case in cases),
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "software": {
            name: _version(name) for name in ("cozy-runtime", "torch", "tensorfs", "triton")
        },
        "implementation": {
            "modules": {
                module.__name__: {
                    "path": module.__file__,
                    "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                }
                for module in (leaves, rowwise_gemm, lora_bake)
            },
            "provenance": {
                name: getattr(_build_provenance, name, None)
                for name in (
                    "SAGE3_LIFETIMES_PATCH_COMMIT",
                    "SHARED_QKV_PATCH_COMMIT",
                    "BOUNDED_ROWWISE_GEMM_PATCH_COMMIT",
                    "LORA_BAKE_DEAD_WORK_PATCH_COMMIT",
                    "LORA_NATIVE_DECODE_PATCH_COMMIT",
                )
            },
        },
        "seed": seed,
        "warmup": warmup,
        "repeats": repeats,
        "cases": cases,
        "max_fp32_scratch_bytes": rowwise_gemm.MAX_FP32_BYTES,
        "limits": [
            "Synthetic matrices do not establish H3 video quality or full-request performance",
            "Both candidates consume identical rowwise-quantized activation and weight payloads",
            "CUDA event timings include allocation and epilogue but exclude quantization",
            "FP32 scratch cap excludes required BF16 output and opaque vendor workspace",
            "SM120 BF16 output only; SM89, SM90 and other outputs preserve the legacy route",
        ],
    }
