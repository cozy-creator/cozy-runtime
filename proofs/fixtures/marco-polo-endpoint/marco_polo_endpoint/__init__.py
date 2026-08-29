"""A weightless endpoint used to prove dynamic endpoint publication and placement."""

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
