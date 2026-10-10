"""Fixed-operand attention diagnostic, submitted through the ordinary Cozy CLI."""

from typing import Annotated, Literal

import msgspec

from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs

app = App()


class Request(msgspec.Struct):
    seed: int = 581204
    size: Literal["all", "small", "full"] = "full"
    check: Literal["sage2", "bf16_sparse", "kitchen_dense"] = "sage2"


class Result(msgspec.Struct):
    report: Annotated[FileAsset, AssetBound(max_bytes=4 << 20, media_types=("application/json",))]


@app.job(accelerator=True)
def verify(payload: Request, ctx: Context, out: Outputs) -> Result:
    """Compare identical full Q/K/V against fresh contiguous head-group operands."""
    if ctx.device.type != "cuda":
        raise RuntimeError("The diagnostic requires its normally admitted CUDA device")
    import json

    from .probe import run

    result = run(payload.size, payload.seed, payload.check)
    return Result(
        out.save_bytes(json.dumps(result, allow_nan=False).encode(), media_type="application/json")
    )
