"""Ordinary CPU checkpoint mint and real two-GPU model lifecycle check."""

import io
import json
from typing import Annotated

import msgspec
import torch
from tensorfs.derived import Config, Derivation, Part, Target, Tensor

from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    FileAsset,
    ModelArtifact,
    Outputs,
    WeightsOutput,
)
from cozy_runtime.internal.encoding import SPEC_PLAIN

from .model import Component, ProbeModel

app = App()


class MintRequest(msgspec.Struct):
    seed: int = 581204


@app.job(accelerator=False, weights=(WeightsOutput("model", max_new_bytes=1 << 20),))
def mint(payload: MintRequest, ctx: Context) -> ModelArtifact:
    # Deliberately CPU only. This normal derived-output job owns its native writer capability.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(payload.seed)
        state = Component().state_dict()
    tensors = {
        key: Tensor(
            "bf16", tuple(value.shape), SPEC_PLAIN, {"value": Part("bf16", tuple(value.shape))}
        )
        for key, value in state.items()
    }
    if len(tensors) != 6 or any(value.dtype != torch.bfloat16 for value in state.values()):
        raise RuntimeError("tiny checkpoint schema changed")
    with ctx.output("model").open(
        Derivation(
            sources={},
            targets={"dit": Target(add=tensors)},
            configs={"model": Config("add")},
            order=tuple(("dit", key) for key in tensors),
        )
    ) as writer:
        if writer.receipt is not None:
            return ctx.adopt_model(writer.receipt)
        done = frozenset(writer.completed_parts())
        for key, value in state.items():
            ctx.raise_if_cancelled()
            if ("dit", key, "value") not in done:
                writer.add_part(
                    "dit",
                    key,
                    "value",
                    io.BytesIO(value.contiguous().view(torch.uint8).numpy().tobytes()),
                )
        writer.add_config(
            "model",
            io.BytesIO(
                json.dumps(
                    {
                        "fixture": "h3-cp-producer-lifecycle/1",
                        "heads": 2,
                        "head_dim": 128,
                        "hidden_size": 128,
                    }
                ).encode()
            ),
        )
        return ctx.adopt_model(writer.commit())


class Request(msgspec.Struct):
    start_step: int = 4
    seed: int = 581204


class Result(msgspec.Struct):
    report: Annotated[FileAsset, AssetBound(max_bytes=1 << 20, media_types=("application/json",))]


@app.entrypoint(
    defaults={
        "model": [
            {"gpu": "RTX 5090", "gpus": 2, "lane": "paul/h3-cp-producer-lifecycle-20261010@1/model"}
        ]
    },
    components={"model": ["dit"]},
)
def verify(payload: Request, model: ProbeModel, out: Outputs) -> Result:
    report = model.verify(payload.start_step, payload.seed)
    return Result(
        out.save_bytes(json.dumps(report, allow_nan=False).encode(), media_type="application/json")
    )
