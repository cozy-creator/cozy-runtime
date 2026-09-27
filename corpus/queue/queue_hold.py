"""Weightless entrypoints that hold the device for as long as they are asked to."""

import time

import msgspec

from cozy_runtime.author import App, Telemetry

app = App()


class HoldRequest(msgspec.Struct, forbid_unknown_fields=True):
    tag: str
    seconds: float = 0.0


class HoldResult(msgspec.Struct):
    tag: str


@app.entrypoint
def hold(payload: HoldRequest) -> HoldResult:
    time.sleep(payload.seconds)
    return HoldResult(tag=payload.tag)


class StepsRequest(msgspec.Struct, forbid_unknown_fields=True):
    tag: str
    steps: int = 1
    step_seconds: float = 0.0
    #: the step that never ends and never reports: a handler that ignores cancellation
    hang_at: int = -1


@app.entrypoint
def steps(payload: StepsRequest, tel: Telemetry) -> HoldResult:
    on_step = tel.step_callback(payload.steps, stage="denoise")
    for step in range(payload.steps):
        while step == payload.hang_at:
            time.sleep(3600)
        time.sleep(payload.step_seconds)
        on_step(step)
    return HoldResult(tag=payload.tag)
