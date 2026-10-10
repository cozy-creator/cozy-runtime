"""Actual Runtime four-rank Ulysses correctness check, without video inference."""

from typing import Annotated

import msgspec

from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs

from .model import ProbeModel

app = App()


class Request(msgspec.Struct):
    seed: int = 581204


class Result(msgspec.Struct):
    report: Annotated[FileAsset, AssetBound(max_bytes=8 << 20, media_types=("application/json",))]


@app.entrypoint(
    defaults={
        "model": [
            {
                "gpu": "RTX 5090",
                "gpus": 4,
                "lane": "paul/attention-kernel-benchmark-20260916@1/model",
            }
        ]
    },
    components={"model": ["dit"]},
)
def verify(payload: Request, model: ProbeModel, ctx: Context, out: Outputs) -> Result:
    import json

    report = model.verify(payload.seed)
    return Result(
        out.save_bytes(json.dumps(report, allow_nan=False).encode(), media_type="application/json")
    )
