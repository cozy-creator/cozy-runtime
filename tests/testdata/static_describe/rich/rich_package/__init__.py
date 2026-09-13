"""The app: two entrypoints (one hidden), two jobs (weights outputs, publication)."""

from __future__ import annotations

from cozy_runtime.author import (
    App,
    Context,
    Outputs,
    Preflight,
    Telemetry,
    WeightsOutput,
)

from .bounds import LANE_BYTES, LANE_ENCODINGS, Stills
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
    stills: Stills,
) -> RenderResult:
    raise NotImplementedError


@app.entrypoint(name="render-draft", hidden=True)
def draft(payload: RenderRequest, model: RenderModel) -> RenderResult:
    raise NotImplementedError


@app.job(
    name="quantize",
    weights=tuple(WeightsOutput(lane, max_new_bytes=LANE_BYTES) for lane in LANE_ENCODINGS),
)
def quantize(
    ctx: Context,
    payload: QuantizeRequest,
    source: SourceModel,
    tel: Telemetry,
) -> QuantizedLanes:
    raise NotImplementedError


@app.job(publishes=True)
def gate(payload: GateRequest, out: Outputs, tel: Telemetry) -> GateResult:
    raise NotImplementedError
