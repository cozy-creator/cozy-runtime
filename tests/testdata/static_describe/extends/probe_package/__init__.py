"""A package whose surface is built out of an INSTALLED cozy package's declarations.

`diagnostics/h3-vae-roundtrip` is the real one: it subclasses MiniMax-H3's `Model` and
reuses its request struct. The reader reads that dependency's own source out of the
environment, and reads nothing else there (cr-117).
"""

from __future__ import annotations

from typing import Any

from rich_package.models import RenderModel
from rich_package.schemas import RenderRequest, RenderResult

from cozy_runtime.author import App, Context, Outputs, Telemetry, uses_components

app = App()


class ProbeModel(RenderModel):
    @uses_components("vae")
    def probe(self, state: Any) -> Any:
        return None


@app.entrypoint
def probe(
    ctx: Context, payload: RenderRequest, model: ProbeModel, out: Outputs, tel: Telemetry
) -> RenderResult:
    raise NotImplementedError
