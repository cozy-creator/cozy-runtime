"""minimax-h3-tools: twelve lane derivations over raw H3 checkpoints, some on the GPU."""

from __future__ import annotations

from typing import Any, Literal

import msgspec

from cozy_runtime.author import Context, ModelArtifact, Telemetry, Weights, invocable

Task = Literal["fl2va", "ref2va"]


class Selection(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    projection: ModelArtifact | None
    ready: bool


class RetableResult(msgspec.Struct, frozen=True):
    pruned: ModelArtifact
    tables: ModelArtifact


@invocable
async def bf16_full(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    raise NotImplementedError


@invocable
async def bf16_pruned(ctx: Context, *, source: Weights[Any], tel: Telemetry) -> ModelArtifact:
    raise NotImplementedError


@invocable
async def fp8_pruned(
    ctx: Context,
    *,
    source: Weights[Any],
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    raise NotImplementedError


@invocable
async def mxfp8_pruned(
    ctx: Context,
    *,
    source: Weights[Any],
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    raise NotImplementedError


@invocable
async def fp8(
    ctx: Context,
    *,
    source: Weights[Any],
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    raise NotImplementedError


@invocable
async def mxfp8(
    ctx: Context,
    *,
    source: Weights[Any],
    max_relative_frobenius: float | None = None,
    tel: Telemetry,
) -> ModelArtifact:
    raise NotImplementedError


@invocable
async def retable(
    ctx: Context, *, pruned: Weights[Any], full: Weights[Any], tel: Telemetry
) -> RetableResult:
    raise NotImplementedError


@invocable
async def select_adaln_weights(ctx: Context, *, source: Weights[Any], task: Task) -> Selection:
    raise NotImplementedError


@invocable
async def compute_adaln_tables(
    ctx: Context, *, source: Weights[Any], task: Task, plan_digest: str, tel: Telemetry
) -> ModelArtifact:
    raise NotImplementedError


@invocable
async def apply_adaln(
    ctx: Context, *, source: Weights[Any], fl2va: Weights[Any], ref2va: Weights[Any]
) -> ModelArtifact:
    raise NotImplementedError


@invocable
async def retable_adaln(
    ctx: Context, *, source: Weights[Any], fl2va: Weights[Any], ref2va: Weights[Any]
) -> ModelArtifact:
    raise NotImplementedError


@invocable
async def turbo_lora(
    ctx: Context, *, adapters: Weights[Any], base: Weights[Any], tel: Telemetry
) -> ModelArtifact:
    raise NotImplementedError
