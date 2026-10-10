"""Registered Hopper dense and sparse-prefix correctness in one bounded CLI job."""

from typing import Annotated

import msgspec

from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs

app = App()


class Request(msgspec.Struct):
    seed: int = 581204


class Result(msgspec.Struct):
    report: Annotated[FileAsset, AssetBound(max_bytes=4 << 20, media_types=("application/json",))]


@app.job(accelerator=True)
def verify(payload: Request, ctx: Context, out: Outputs) -> Result:
    if ctx.device.type != "cuda":
        raise RuntimeError("Registered native check needs its admitted CUDA device")
    import json

    from .probe import run

    result = run(payload.seed)
    return Result(
        out.save_bytes(json.dumps(result, allow_nan=False).encode(), media_type="application/json")
    )
