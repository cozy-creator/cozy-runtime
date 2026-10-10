"""One bounded dense attention diagnostic through ordinary Cozy CLI."""

from typing import Annotated, Literal

import msgspec

from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs

app = App()


class Request(msgspec.Struct):
    kernel: Literal["flash-attn3", "sdpa", "flash-attn3-fp8"] = "flash-attn3"
    query: Literal["prefix", "full"] = "prefix"
    seed: int = 581204


class Result(msgspec.Struct):
    report: Annotated[FileAsset, AssetBound(max_bytes=4 << 20, media_types=("application/json",))]


@app.job(accelerator=True)
def verify(payload: Request, ctx: Context, out: Outputs) -> Result:
    if ctx.device.type != "cuda":
        raise RuntimeError("This diagnostic requires its admitted CUDA device")
    import json

    from .probe import run

    result = run(payload.kernel, payload.query, payload.seed)
    return Result(
        out.save_bytes(json.dumps(result, allow_nan=False).encode(), media_type="application/json")
    )
