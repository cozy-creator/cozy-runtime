"""Ordinary accelerator gate for the installed bounded FP8 GEMM implementation."""

from typing import Annotated

import msgspec

from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs

app = App()


class Request(msgspec.Struct):
    seed: int = 20261008
    warmup: Annotated[int, msgspec.Meta(ge=1, le=20)] = 3
    repeats: Annotated[int, msgspec.Meta(ge=3, le=50)] = 10


class Result(msgspec.Struct):
    passed: bool
    report: Annotated[FileAsset, AssetBound(max_bytes=4 << 20, media_types=("application/json",))]
    trace: Annotated[FileAsset, AssetBound(max_bytes=32 << 20, media_types=("application/json",))]


@app.job(accelerator=True)
def verify(payload: Request, ctx: Context, out: Outputs) -> Result:
    if ctx.device.type != "cuda":
        raise RuntimeError("The GEMM gate requires its normally admitted CUDA device")
    import json

    from .compare import compare

    trace_path = out.temporary_file(suffix=".json")
    report = compare(payload.seed, payload.warmup, payload.repeats, trace_path)
    return Result(
        report["passed"],
        out.save_bytes(json.dumps(report, allow_nan=False).encode(), media_type="application/json"),
        out.save_file(trace_path, media_type="application/json"),
    )
