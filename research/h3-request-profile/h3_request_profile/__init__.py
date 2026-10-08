"""Explicit diagnostic package; delegates full requests to the retained H3 package."""

from __future__ import annotations

from typing import Annotated, Any

import msgspec
from h3 import (
    _DEFAULT_MODEL_LADDER,
    _DEFAULT_TURBO_LORA_LADDER,
    ClipInput,
    H3VideoOutput,
    KeyframeAssets,
    StandardClipInput,
    fl2va,
    fl2va_turbo,
)

from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs, Telemetry
from cozy_runtime.models.minimax_h3.model import H3Model, H3TurboBase, H3TurboLoRA

from .instrument import profile_request

app = App()


class RegularInput(StandardClipInput, kw_only=True):
    profile_detail: bool = False


class TurboInput(ClipInput, kw_only=True):
    profile_detail: bool = False


class ProfileResult(msgspec.Struct):
    render: H3VideoOutput
    trace: Annotated[FileAsset, AssetBound(media_types=("application/gzip",), max_bytes=256 << 20)]
    summary: Annotated[FileAsset, AssetBound(media_types=("application/json",), max_bytes=64 << 20)]


def saved(result: tuple[Any, bytes, bytes], out: Outputs) -> ProfileResult:
    render, trace, summary = result
    return ProfileResult(
        render,
        out.save_bytes(trace, media_type="application/gzip"),
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
    return saved(
        profile_request(
            model,
            10,
            lambda: fl2va(ctx, payload, assets, model, out, tel),
            detailed=payload.profile_detail,
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
    return saved(
        profile_request(
            base_model,
            4,
            lambda: fl2va_turbo(ctx, payload, assets, base_model, turbo_lora, out, tel),
            detailed=payload.profile_detail,
        ),
        out,
    )
