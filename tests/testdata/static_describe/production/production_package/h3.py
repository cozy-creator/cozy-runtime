"""minimax-h3: seven CPU orchestrators that publish media as it lands."""

from __future__ import annotations

from typing import Annotated, Literal

import msgspec

from cozy_runtime.author import (
    AssetBound,
    AssetLimits,
    Assets,
    Context,
    Image,
    ImageAsset,
    MediaDecoder,
    Mixed,
    Outputs,
    Telemetry,
    VideoAsset,
)

KeyframeAssets = Annotated[Assets[Image], AssetLimits(images=2)]
DirectReferenceAssets = Annotated[Assets[Mixed], AssetLimits(images=4, videos=2, audio=2, total=6)]
Mp4 = Annotated[VideoAsset, AssetBound(media_types=("video/mp4",), max_bytes=1 << 30)]


class RenderInput(msgspec.Struct, kw_only=True):
    prompt: str
    duration_s: float
    resolution: Literal["native", "1080p"] = "native"


class StandardRenderInput(RenderInput, kw_only=True):
    steps: int = 30


class ReferenceRenderInput(RenderInput, kw_only=True):
    style: str = ""


class ReferenceStandardRenderInput(ReferenceRenderInput, kw_only=True):
    steps: int = 30


class RenderOutput(msgspec.Struct):
    video: Mp4
    originals: list[Mp4]
    warnings: list[str]


class ReferenceRenderOutput(RenderOutput):
    references: list[ImageAsset] = msgspec.field(default_factory=list)


class Reference(msgspec.Struct):
    name: str
    image: Annotated[ImageAsset, AssetBound(max_bytes=64 << 20, max_decoded_bytes=256 << 20)]


class LongFormInput(msgspec.Struct):
    segments: Annotated[list[str], msgspec.Meta(min_length=1)]
    references: list[Reference]
    style: str = ""


class LongFormCutsInput(LongFormInput):
    mode: Literal["turbo", "standard"] = "turbo"


class LongFormOutput(msgspec.Struct):
    video: Mp4
    references: list[ImageAsset]


class AssembleVideoRequest(msgspec.Struct):
    shots: Annotated[list[VideoAsset], AssetBound(max_bytes=1 << 30, max_decoded_bytes=1 << 30)]


class AssembleVideoResponse(msgspec.Struct):
    video: Mp4


async def fl2va(
    ctx: Context, payload: StandardRenderInput, assets: KeyframeAssets, out: Outputs, tel: Telemetry
) -> RenderOutput:
    raise NotImplementedError


async def fl2va_turbo(
    ctx: Context, payload: RenderInput, assets: KeyframeAssets, out: Outputs, tel: Telemetry
) -> RenderOutput:
    raise NotImplementedError


async def ref2va(
    ctx: Context,
    payload: ReferenceStandardRenderInput,
    out: Outputs,
    tel: Telemetry,
    assets: DirectReferenceAssets,
) -> ReferenceRenderOutput:
    raise NotImplementedError


async def ref2va_turbo(
    ctx: Context,
    payload: ReferenceRenderInput,
    out: Outputs,
    tel: Telemetry,
    assets: DirectReferenceAssets,
) -> ReferenceRenderOutput:
    raise NotImplementedError


async def long_form(
    ctx: Context, payload: LongFormInput, decoder: MediaDecoder, out: Outputs, tel: Telemetry
) -> LongFormOutput:
    raise NotImplementedError


async def long_form_cuts(
    ctx: Context, payload: LongFormCutsInput, decoder: MediaDecoder, out: Outputs, tel: Telemetry
) -> LongFormOutput:
    raise NotImplementedError


async def assemble_video(
    ctx: Context,
    *,
    payload: AssembleVideoRequest,
    decoder: MediaDecoder,
    out: Outputs,
    tel: Telemetry,
) -> AssembleVideoResponse:
    raise NotImplementedError
