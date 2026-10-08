"""Tiny resident-tensor study; every result states its accumulation and timing boundary."""

from __future__ import annotations

import importlib.metadata
import math
import statistics
import time

# Checkpoint matrix geometry, not full request sequence lengths. Turbo output heads
# are FP32 and deliberately absent from this FP8 comparison.
SHAPES = {
    "qkv": (4096, 7168, 5376),
    "attention_out": (1024, 5376, 7168),
    "fc1": (1024, 28672, 5376),
    "fc2": (1024, 5376, 14336),
}


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _timed(torch, function, warmup, repeats):
    for _ in range(warmup):
        value = function()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeats)
    ]
    start = time.perf_counter()
    for begin, end in events:
        begin.record()
        value = function()
        end.record()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    samples = [begin.elapsed_time(end) for begin, end in events]
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "p90_ms": sorted(samples)[math.ceil(len(samples) * 0.9) - 1],
        "samples_ms": samples,
        "host_total_s": elapsed,
    }, value


def _error(torch, actual, reference):
    a, b = actual.float(), reference.float()
    finite = bool(torch.isfinite(a).all().item())
    if not finite:
        return {
            "finite": False,
            "max_abs": None,
            "rmse": None,
            "relative_l2": None,
            "reference_rms": float(b.square().mean().sqrt().item()),
        }
    delta = a - b
    return {
        "finite": finite,
        "max_abs": float(delta.abs().max().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "relative_l2": float(
            torch.linalg.vector_norm(delta).item() / max(torch.linalg.vector_norm(b).item(), 1e-30)
        ),
        "reference_rms": float(b.square().mean().sqrt().item()),
    }


def compare_paths(selected, seed, warmup, repeats):
    import torch

    from cozy_runtime.internal.encoding.leaves import quantize_activation_rowwise

    from .epilogue import rescale

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This explicit SM120 study requires compute capability12.0")
    cases = SHAPES if selected == "all" else {selected: SHAPES[selected]}
    generator = torch.Generator(device="cuda").manual_seed(seed)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    results = []
    try:
        with torch.inference_mode():
            for name, (m, n, k) in cases.items():
                results.append(
                    _case(
                        torch,
                        rescale,
                        quantize_activation_rowwise,
                        generator,
                        name,
                        m,
                        n,
                        k,
                        warmup,
                        repeats,
                    )
                )
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return {
        "study": "SM120 FP8 GEMM + separate scale epilogue",
        "seed": seed,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "software": {n: _version(n) for n in ("cozy-runtime", "tensorfs", "torch", "triton")},
        "warmup": warmup,
        "repeats": repeats,
        "cases": results,
        "native": "torch._scaled_mm rowwise As[M,1],Bs[1,N],BF16,use_fast_accum=True",
        "candidate": (
            "torch._scaled_mm scalar unity scales,FP32,use_fast_accum=False; "
            "then (acc*As)*Bs FP32 -> BF16"
        ),
        "timing": (
            "CUDA event elapsed time after warmup; random tensors already resident; "
            "quantization measured separately; combined path measured directly"
        ),
        "limits": [
            "Not checkpoint or video quality qualification",
            "Not paging or transfer throughput",
            "No bias, no static calibration",
            "Different fast-accum flags are explicit",
            "Dispatch inferred from scalar/rowwise API contract; kernel trace still required",
            "FP32 numerical reference uses highest precision, TF32 disabled",
            "Existing full-precision Turbo heads are not replaced",
        ],
    }


def _case(torch, rescale, quantize_activation_rowwise, generator, name, m, n, k, warmup, repeats):
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda", generator=generator)
    w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda", generator=generator) * 0.02
    aq, a_scale = quantize_activation_rowwise(torch, x)
    bq, b_scale = quantize_activation_rowwise(torch, w)
    b = bq.t()
    bs = b_scale.reshape(1, -1)
    one = torch.ones((), dtype=torch.float32, device="cuda")
    dest = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    assert aq.stride(1) == 1 and b.stride(0) == 1
    before = tuple(a.clone() for a in (aq, bq, a_scale, b_scale))

    def rowwise():
        return torch._scaled_mm(
            aq,
            b,
            scale_a=a_scale,
            scale_b=bs,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
            bias=None,
        )

    def scalar():
        return torch._scaled_mm(
            aq,
            b,
            scale_a=one,
            scale_b=one,
            out_dtype=torch.float32,
            use_fast_accum=False,
            bias=None,
        )

    native_time, native = _timed(torch, rowwise, warmup, repeats)
    scalar_time, accum = _timed(torch, scalar, warmup, repeats)
    if accum.dtype != torch.float32 or not accum.is_contiguous():
        raise RuntimeError("Scalar GEMM did not return the required contiguous FP32 result")
    epilogue_time, _scaled = _timed(
        torch, lambda: rescale(accum, a_scale, bs, dest), warmup, repeats
    )
    composed_time, combined = _timed(
        torch, lambda: rescale(scalar(), a_scale, bs, dest), warmup, repeats
    )
    quant_time, quantized = _timed(
        torch, lambda: quantize_activation_rowwise(torch, x), warmup, repeats
    )
    # Numerical-only FP32 reference, with TF32 disabled: same quantized bytes,
    # full FP32 dot, the same sequential scale order, then final BF16 rounding.
    reference = ((aq.float() @ bq.float().t()) * a_scale) * bs
    reference = reference.to(torch.bfloat16)
    original = x.float() @ w.float().t()
    checks = {
        "native_vs_quantized_reference": _error(torch, native, reference),
        "composed_vs_quantized_reference": _error(torch, combined, reference),
        "composed_vs_native": _error(torch, combined, native),
        "native_vs_unquantized_fp32": _error(torch, native, original),
        "composed_vs_unquantized_fp32": _error(torch, combined, original),
    }
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip((aq, bq, a_scale, b_scale), before, strict=True)
    )
    assert torch.equal(aq.view(torch.uint8), quantized[0].view(torch.uint8))
    assert torch.equal(a_scale, quantized[1])
    return {
        "name": name,
        "m": m,
        "n": n,
        "k": k,
        "quantized_operands_shared_by_identity": True,
        "input_bytes_unchanged": True,
        "a_shape_stride": [list(aq.shape), list(aq.stride())],
        "b_shape_stride": [list(b.shape), list(b.stride())],
        "activation_quantization": quant_time,
        "rowwise_native_gemm": native_time,
        "scalar_fp32_gemm": scalar_time,
        "fp32_rescale_bf16_epilogue": epilogue_time,
        "scalar_gemm_plus_epilogue": composed_time,
        "errors": checks,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "finite_pass": all(c["finite"] for c in checks.values()),
    }
