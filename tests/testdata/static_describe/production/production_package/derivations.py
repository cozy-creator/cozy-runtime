"""seedvr2, birefnet, character-describer, sdxl and anima: raw weights in, a lane out."""

from __future__ import annotations

from typing import Annotated

from cozy_runtime.author import (
    AssetBound,
    Context,
    ModelArtifact,
    Telemetry,
    Tree,
    Weights,
    invocable,
)

from .models import AnimaModel, BackgroundRemovalModel, SdxlModel


@invocable
async def seedvr2_normalize(
    ctx: Context,
    *,
    dit: Weights,
    vae: Weights,
    conditioning: Annotated[Tree, AssetBound(max_bytes=2 << 20)],
) -> ModelArtifact:
    raise NotImplementedError


@invocable
async def birefnet_normalize(
    ctx: Context, *, source: Weights[BackgroundRemovalModel]
) -> ModelArtifact:
    raise NotImplementedError


@invocable
async def describer_normalize(ctx: Context, *, source: Weights, tel: Telemetry) -> ModelArtifact:
    raise NotImplementedError


@invocable
async def sdxl_fp8(
    ctx: Context,
    *,
    source: Weights[SdxlModel],
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    raise NotImplementedError


@invocable
async def sdxl_mxfp8(
    ctx: Context,
    *,
    source: Weights[SdxlModel],
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    raise NotImplementedError


@invocable
async def anima_fp8(
    ctx: Context,
    *,
    source: Weights[AnimaModel],
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    raise NotImplementedError


@invocable
async def anima_mxfp8(
    ctx: Context,
    *,
    source: Weights[AnimaModel],
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    raise NotImplementedError
