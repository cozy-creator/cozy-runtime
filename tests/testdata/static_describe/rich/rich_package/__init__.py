"""The app: two entrypoints (one hidden), two jobs (weights outputs, publication)."""

from __future__ import annotations

from typing import Annotated

from cozy_runtime.author import (
    App,
    AssetLimits,
    Assets,
    Context,
    ImageAsset,
    Outputs,
    Preflight,
    Telemetry,
    WeightsOutput,
    WeightsSink,
)

from .bounds import IMAGE_BOUND, LANE_BYTES
from .models import RenderModel, SourceModel
from .schemas import (
    Facts,
    GateRequest,
    GateResult,
    QuantizedLanes,
    QuantizeRequest,
    RenderRequest,
    RenderResult,
)

app = App()


def preflight_render(payload: RenderRequest) -> Facts:
    return Facts(reference_count=len(payload.references))


@app.entrypoint(preflight=preflight_render)
def render(
    ctx: Context,
    payload: RenderRequest,
    facts: Preflight[Facts],
    model: RenderModel,
    out: Outputs,
    tel: Telemetry,
    stills: Annotated[Assets[Annotated[ImageAsset, IMAGE_BOUND]], AssetLimits(images=2)],
) -> RenderResult:
    raise NotImplementedError


@app.entrypoint(name="render-draft", hidden=True)
def draft(payload: RenderRequest, model: RenderModel) -> RenderResult:
    raise NotImplementedError


@app.job(
    name="quantize",
    weights=(
        WeightsOutput("fp8", max_new_bytes=LANE_BYTES),
        WeightsOutput("mxfp8", max_new_bytes=LANE_BYTES),
    ),
)
def quantize(
    ctx: Context,
    payload: QuantizeRequest,
    source: SourceModel,
    weights: WeightsSink,
    tel: Telemetry,
) -> QuantizedLanes:
    raise NotImplementedError


@app.job(publishes=True)
def gate(payload: GateRequest, out: Outputs, tel: Telemetry) -> GateResult:
    raise NotImplementedError
