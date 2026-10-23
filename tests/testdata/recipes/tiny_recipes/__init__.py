"""Two tiny models, one function and the recipes of `recipes.py`."""

from __future__ import annotations

import msgspec

from cozy_runtime.author import App

from . import recipes as recipes

app = App()


class Echo(msgspec.Struct):
    text: str


@app.entrypoint(accelerator=False)
def echo(payload: Echo) -> Echo:
    return payload
