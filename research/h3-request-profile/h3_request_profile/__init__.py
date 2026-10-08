"""Explicit diagnostic package; delegates full requests to the retained H3 package."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import msgspec

from cozy_runtime.author import (
    App,
    AssetBound,
    AssetLimits,
    Assets,
    Context,
    FileAsset,
    Image,
    Outputs,
    Telemetry,
    Tree,
    VideoAsset,
)
from cozy_runtime.models.minimax_h3.model import H3Model, H3TurboBase, H3TurboLoRA

from .instrument import profile_request, retained_handler, software_versions

app = App()


# Same one-5090 choices as retained minimax-h3@1.26.2, fully qualified because
# this explicit local diagnostic has its own package namespace.
_DEFAULT_MODEL_LADDER = [
    {"gpu": "5090", "gpus": 1, "lane": "paul/minimax-h3@1.0.0-rc.2/fp8-pruned"}
]
_DEFAULT_TURBO_LORA_LADDER = [
    {"gpu": "5090", "gpus": 1, "lane": "paul/minimax-h3-turbo-lora@1.0.0-audit.1/pdd8"}
]
KeyframeAssets = Annotated[Assets[Image], AssetLimits(images=2)]


class TurboInput(msgspec.Struct, kw_only=True):
    prompt: str
    duration_s: Literal[15]
    seed: int | None = None
    profile_detail: bool = False
    full_request_trace: bool = True
    trace_cpu_ops: bool = False


class RegularInput(TurboInput, kw_only=True):
    steps: Literal[30] = 30


class RenderResult(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(media_types=("video/mp4",))]
    warnings: list[str]


class ProfileResult(msgspec.Struct):
    render: RenderResult
    trace: Annotated[Tree, AssetBound(max_bytes=64 << 30)]
    summary: Annotated[FileAsset, AssetBound(media_types=("application/json",), max_bytes=64 << 20)]


class EnvironmentInput(msgspec.Struct, forbid_unknown_fields=True):
    pass


class EnvironmentResult(msgspec.Struct):
    device: str
    software: dict[str, str | None]
    delegated_handler_symbols: dict[str, list[str]]


@app.job
def environment(payload: EnvironmentInput, ctx: Context) -> EnvironmentResult:
    """CPU receipt proves direct implementation binding despite job-only public-export proxies."""
    if ctx.device.type != "cpu":
        raise ValueError("The environment receipt must run without a GPU allocation")
    return EnvironmentResult(
        str(ctx.device),
        software_versions(),
        {name: list(retained_handler(name).__code__.co_names) for name in ("fl2va", "fl2va_turbo")},
    )


def saved(result: tuple[Any, Tree, bytes], out: Outputs) -> ProfileResult:
    render, trace, summary = result
    return ProfileResult(
        RenderResult(render.video, render.warnings),
        trace,
        out.save_bytes(summary, media_type="application/json"),
    )


@app.entrypoint(defaults={"model": _DEFAULT_MODEL_LADDER})
def regular(
    ctx: Context,
    payload: RegularInput,
    assets: KeyframeAssets,
    model: H3Model,
    out: Outputs,
    tel: Telemetry,
) -> ProfileResult:
    if payload.duration_s != 15 or payload.steps != 30:
        raise ValueError("This diagnostic retains the full15-second regular30 request")
    handler = retained_handler("fl2va")
    return saved(
        profile_request(
            model,
            10,
            lambda observed_out, observed_tel: handler(
                ctx, payload, assets, model, observed_out, observed_tel
            ),
            outputs=out,
            telemetry=tel,
            detailed=payload.profile_detail,
            full_request_trace=payload.full_request_trace,
            trace_cpu_ops=payload.trace_cpu_ops,
        ),
        out,
    )


@app.entrypoint(
    defaults={"base_model": _DEFAULT_MODEL_LADDER, "turbo_lora": _DEFAULT_TURBO_LORA_LADDER}
)
def turbo(
    ctx: Context,
    payload: TurboInput,
    assets: KeyframeAssets,
    base_model: H3TurboBase,
    turbo_lora: H3TurboLoRA,
    out: Outputs,
    tel: Telemetry,
) -> ProfileResult:
    if payload.duration_s != 15:
        raise ValueError("This diagnostic retains the full15-second Turbo8 request")
    handler = retained_handler("fl2va_turbo")
    return saved(
        profile_request(
            base_model,
            4,
            lambda observed_out, observed_tel: handler(
                ctx, payload, assets, base_model, turbo_lora, observed_out, observed_tel
            ),
            outputs=out,
            telemetry=tel,
            detailed=payload.profile_detail,
            full_request_trace=payload.full_request_trace,
            trace_cpu_ops=payload.trace_cpu_ops,
        ),
        out,
    )
