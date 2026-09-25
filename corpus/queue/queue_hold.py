"""A weightless entrypoint that holds the device for as long as it is asked to."""

import time

import msgspec

from cozy_runtime.author import App

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
