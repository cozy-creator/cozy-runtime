"""Small ordinary CLI GPU job: arithmetic and dispatch evidence, not video quality."""

from typing import Annotated

import msgspec

from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs

app = App()


class Request(msgspec.Struct):
    seed: int = 20261009


class Result(msgspec.Struct):
    passed: bool
    report: Annotated[FileAsset, AssetBound(max_bytes=4 << 20, media_types=("application/json",))]


@app.job(accelerator=True)
def verify(payload: Request, ctx: Context, out: Outputs) -> Result:
    if ctx.device.type != "cuda":
        raise ValueError("Attention matrix gate needs its normally allocated SM120 device")
    import json

    from .compare import compare

    report = compare(payload.seed, ctx.device.index or 0)
    return Result(
        report["passed"],
        out.save_bytes(json.dumps(report, allow_nan=False).encode(), media_type="application/json"),
    )
