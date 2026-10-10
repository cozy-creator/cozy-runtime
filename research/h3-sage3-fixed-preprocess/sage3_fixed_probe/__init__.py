"""Ordinary-CLI research only; no model, serving override or global monkeypatch."""

from typing import Annotated, Literal

import msgspec

from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs

app = App()


class Request(msgspec.Struct):
    seed: int = 581204
    size: Literal["all", "small", "full"] = "all"


class Result(msgspec.Struct):
    report: Annotated[FileAsset, AssetBound(max_bytes=8 << 20, media_types=("application/json",))]


@app.job(accelerator=True)
def verify(payload: Request, ctx: Context, out: Outputs) -> Result:
    if ctx.device.type != "cuda":
        raise RuntimeError("Use the assigned CUDA rental through the ordinary CLI")
    import json

    from .probe import run

    result = run(payload.size, payload.seed)
    return Result(
        out.save_bytes(json.dumps(result, allow_nan=False).encode(), media_type="application/json")
    )
