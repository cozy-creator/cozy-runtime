"""Ordinary-CLI, single-device diagnostic; never run beside timed quality requests."""

from typing import Annotated, Literal

import msgspec

from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs

app = App()


class Request(msgspec.Struct):
    seed: int = 581204
    case: Literal["all", "qkv", "attention_out", "fc1", "fc2"] = "all"
    size: Literal["all", "small", "full"] = "all"


class Result(msgspec.Struct):
    report: Annotated[FileAsset, AssetBound(max_bytes=4 << 20, media_types=("application/json",))]


@app.job(accelerator=True)
def verify(payload: Request, ctx: Context, out: Outputs) -> Result:
    """Compare the current FP8 leaf at full and degree-sized row partitions."""
    if ctx.device.type != "cuda":
        raise RuntimeError("This diagnostic requires its normally admitted CUDA device")
    import json

    from .probe import run

    report = run(payload.case, payload.size, payload.seed)
    return Result(
        out.save_bytes(json.dumps(report, allow_nan=False).encode(), media_type="application/json")
    )
