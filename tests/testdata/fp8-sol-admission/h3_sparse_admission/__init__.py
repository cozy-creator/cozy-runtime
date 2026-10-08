"""Author-facing GPU admission job; execute only through the normal Cozy CLI."""

from __future__ import annotations

import msgspec

from cozy_runtime.author import App, Context

app = App()


class Request(msgspec.Struct, forbid_unknown_fields=True):
    pass


class Case(msgspec.Struct):
    case: str
    tokens: int
    heads: int
    relative_l2_to_cpu_oracle: float
    maximum_absolute_error: float
    fp16_mma: bool
    finite: bool


class Result(msgspec.Struct):
    gpu: str
    kitchen_version: str
    cases: list[Case]


@app.job(accelerator=True)
def verify(payload: Request, ctx: Context) -> Result:
    if ctx.device.type != "cuda":
        raise ValueError("This admission job requires an allocated SM120 GPU")
    import torch

    from cozy_runtime.internal import attention
    from cozy_runtime.internal.encoding import DeviceFacts

    from .check import admit

    index = ctx.device.index or 0
    if torch.cuda.get_device_capability(index) != (12, 0):
        raise ValueError("This admission job is qualified for rented SM120 only")
    # Use Runtime's existing kernel acquisition/setup boundary, not a hand-built
    # import tree or an SSH-side Python environment.
    attention.pinned(
        "sol-fp8-fp16", DeviceFacts("cuda", torch.cuda.get_device_name(index), 120, "", "", index)
    )
    attention.pinned(
        "kitchen-sol-sage",
        DeviceFacts("cuda", torch.cuda.get_device_name(index), 120, "", "", index),
    )
    from importlib.metadata import version

    result = admit(index)
    result["kitchen_version"] = version("comfy-kitchen")
    return msgspec.convert(result, type=Result)
