"""Exploratory resident GEMM timings; no checkpoint or H3 rendering is performed."""

from typing import Annotated, Literal

import msgspec

from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs

app = App()


class Request(msgspec.Struct):
    case: Literal["all", "qkv", "attention_out", "fc1", "fc2"] = "all"
    seed: int = 20261008
    warmup: Annotated[int, msgspec.Meta(ge=1, le=100)] = 10
    repeats: Annotated[int, msgspec.Meta(ge=3, le=200)] = 30


class Result(msgspec.Struct):
    report: Annotated[FileAsset, AssetBound(max_bytes=16 << 20, media_types=("application/json",))]


@app.job(accelerator=True)
def compare(payload: Request, ctx: Context, out: Outputs) -> Result:
    """Compare rowwise native GEMM with unit-scale FP32 GEMM and a fused epilogue."""
    if ctx.device.type != "cuda":
        raise RuntimeError("This accelerator job requires its normally admitted CUDA device")
    import json

    from .measure import compare_paths

    report = compare_paths(payload.case, payload.seed, payload.warmup, payload.repeats)
    return Result(
        out.save_bytes(json.dumps(report, allow_nan=False).encode(), media_type="application/json")
    )


class TraceRequest(msgspec.Struct):
    dimension: Annotated[int, msgspec.Meta(ge=64, le=2048)] = 512
    repeats: Annotated[int, msgspec.Meta(ge=1, le=10)] = 3


class TraceResult(msgspec.Struct):
    trace: Annotated[FileAsset, AssetBound(max_bytes=16 << 20, media_types=("application/json",))]
    report: Annotated[FileAsset, AssetBound(max_bytes=1 << 20, media_types=("application/json",))]


@app.job(accelerator=True)
def trace_cuda_only(payload: TraceRequest, ctx: Context, out: Outputs) -> TraceResult:
    """Probe which trace categories CUDA-only profiling actually retains on this worker."""
    if ctx.device.type != "cuda":
        raise RuntimeError("This accelerator job requires its normally admitted CUDA device")
    import json

    from .trace_probe import capture

    path = out.temporary_file(suffix=".json")
    report = capture(path, payload.dimension, payload.repeats)
    return TraceResult(
        out.save_file(path, media_type="application/json"),
        out.save_bytes(json.dumps(report, allow_nan=False).encode(), media_type="application/json"),
    )
