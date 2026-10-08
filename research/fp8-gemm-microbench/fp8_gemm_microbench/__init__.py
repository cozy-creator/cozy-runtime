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
    report: Annotated[FileAsset, AssetBound(max_bytes=16 << 20)]


@app.job(accelerator=True)
def compare(payload: Request, ctx: Context, out: Outputs) -> Result:
    """Compare rowwise native GEMM with unit-scale FP32 GEMM and a fused epilogue."""
    if ctx.device.type != "cuda":
        raise RuntimeError("This accelerator job requires its normally admitted CUDA device")
    from .measure import compare_paths

    report = compare_paths(payload.case, payload.seed, payload.warmup, payload.repeats)
    return Result(out.save_json(report))
