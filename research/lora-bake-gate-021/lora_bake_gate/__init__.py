"""Ordinary accelerator-job numerical gate; no performance or video inference claim."""

from typing import Annotated

import msgspec

from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs

app = App()


class Request(msgspec.Struct):
    seed: int = 20261008


class Result(msgspec.Struct):
    passed: bool
    report: Annotated[FileAsset, AssetBound(max_bytes=4 << 20, media_types=("application/json",))]


@app.job(accelerator=True)
def verify(payload: Request, ctx: Context, out: Outputs) -> Result:
    if ctx.device.type != "cuda":
        raise RuntimeError("The equivalence gate requires its normally admitted CUDA device")
    import json

    from .compare import compare

    report = compare(payload.seed)
    return Result(
        report["passed"],
        out.save_bytes(json.dumps(report, allow_nan=False).encode(), media_type="application/json"),
    )
