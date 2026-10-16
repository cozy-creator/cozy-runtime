"""A selection lane on a real native store: only the matched weights, the source's own objects,
with the source's configs; a pattern that matches nothing refuses."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import msgspec
import tensorfs

from cozy_runtime.author import (
    App,
    Context,
    Invocation,
    ModelArtifact,
    Telemetry,
    Weights,
    WeightsOutput,
    attempt,
    describe,
    invocable,
)
from cozy_runtime.derive.operations import SelectionPlan, select_into
from test_quantize_artifact_native import BOUND, _host, _source


@invocable
async def head(
    ctx: Context, *, source: Weights[Any], include: list[str], tel: Telemetry
) -> ModelArtifact:
    return select_into(ctx, tel, source=source, plan=SelectionPlan(tuple(include)), output="model")


def _select(root: Path, store: tensorfs.Store, source: ModelArtifact, include: list[str]) -> Any:
    app = App()
    app.entrypoint(head, weights=(WeightsOutput("model", BOUND),), accelerator=False)
    describe(app)
    with _host(store, root.name, source) as host:
        return attempt(
            app.get("head"),
            {"source": msgspec.to_builtins(source), "include": include},
            Invocation(
                host.attempt.request_id,
                root,
                math.inf,
                sources={"source": Weights(source.manifest.digest)},
                tensorfs_source=host.client.source,
                tensorfs_output=host.client.open_output,
                tensorfs_adopt=host.client.adopt_model,
            ),
        )


def test_a_selection_keeps_the_matched_weights_as_the_sources_objects(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    source = _source(store, "model", keys=("mtp.fc.weight", "layers.0.weight", "mtp.norm.weight"))
    result, outcome, _ = _select(tmp_path / "head", store, source, ["model.mtp.*"])
    assert outcome.terminal == "succeeded", outcome
    before = tensorfs.parse_header(store.manifest(source.manifest.digest)["header"])
    after = tensorfs.parse_header(store.manifest(result.result.manifest.digest)["header"])
    assert list(after["components"]) == ["model"]
    assert list(after["components"]["model"]) == ["mtp.fc.weight", "mtp.norm.weight"]
    for key in after["components"]["model"]:
        assert after["components"]["model"][key] == before["components"]["model"][key]
    assert after.get("configs") == before.get("configs")

    result, outcome, _ = _select(tmp_path / "none", store, source, ["model.absent.*"])
    assert result is None and outcome.terminal == "refused" and "match no weight" in outcome.message
