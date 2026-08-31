"""A weightless Marco/Polo package for publication and placement examples."""

from __future__ import annotations

from typing import Literal

import msgspec

from cozy_runtime.author import App

app = App()


class MarcoRequest(msgspec.Struct, forbid_unknown_fields=True):
    message: Literal["marco"]


class PoloResponse(msgspec.Struct):
    message: Literal["polo"]


@app.entrypoint
def marco(payload: MarcoRequest) -> PoloResponse:
    return PoloResponse(message="polo")


@app.job
def marco_job(payload: MarcoRequest) -> PoloResponse:
    return PoloResponse(message="polo")
