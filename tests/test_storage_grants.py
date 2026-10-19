"""A function's weights output is the attempt's grant, and the executor writes it itself: straight
into the store under its replay key, shown to the run as a checkpoint item, replayed by the
same work. Nothing about weights crosses the seam to TensorD but that item."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cozy_runtime import weights_store
from cozy_runtime.author._executor_requests import Answer, CheckpointItem, Request, refuse
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.seam import RESULT_DOCUMENT
from session import session
from store_fixtures import checkpoint, home

PACKAGE = """
from typing import Any

import msgspec

from cozy_runtime.author import App, CapabilityError, Context, Weights, WeightsOutput
from cozy_runtime.weights_store import Derivation

app = App()


class Request(msgspec.Struct):
    use_output: bool


class Result(msgspec.Struct):
    code: str
    checkpoint: str = ""
    transaction: str = ""


@app.entrypoint(accelerator=False, weights=(WeightsOutput("model", max_new_bytes=4096),))
def compute(payload: Request, ctx: Context, source: Weights[Any]) -> Result:
    if not payload.use_output:
        return Result("no-storage-needed")
    try:
        base = ctx.tensorfs_source(source)
        with ctx.output("model", Derivation(base)) as out:
            out.put("unet", "extra", "F32", [2], bytes(8))
            artifact = ctx.adopt_model(out)
    except CapabilityError as exc:
        return Result(exc.code)
    return Result("granted", artifact.checkpoint, artifact.transaction)
"""


@pytest.mark.parametrize("granted", [False, True])
def test_a_granted_output_is_written_into_the_store_shown_and_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, granted: bool
) -> None:
    store = home(tmp_path / "tensorfs", monkeypatch)
    source = checkpoint(store, {"unet": {"w": np.ones(2, np.float32)}}, configs={"unet": {"in": 2}})
    project = tmp_path / "project"
    project.mkdir()
    (project / "package.toml").write_text('[application]\nobject="storage_grant_fn:app"\n')
    (project / "storage_grant_fn.py").write_text(PACKAGE)
    shown: list[CheckpointItem] = []

    def machine(request: Request) -> Answer:
        if not isinstance(request, CheckpointItem):
            return refuse("no_durable_exchange", type(request).__name__)
        shown.append(request)
        return Answer(ok=True)

    results = []
    with session(tmp_path, discover(project)) as host:
        host.handler = machine
        # One run twice (an attempt again), then another run of the same operation.
        for spool, use_output, invoke in (
            ("first", False, {}),
            ("run", True, {}),
            ("run", True, {}),
            ("again", True, {"operation": "sha256:" + "a" * 64, "actor": "me"}),
            ("later", True, {"operation": "sha256:" + "a" * 64, "actor": "me"}),
        ):
            reply = host.run(
                "compute",
                {"use_output": use_output},
                tmp_path / spool,
                sources={"source": source},
                weights={"model": 4096} if granted else {},
                **invoke,
            )
            assert reply["outcome"]["terminal"] == "succeeded", reply
            results.append(json.loads((tmp_path / spool / RESULT_DOCUMENT).read_bytes()))
    assert results[0] == {"code": "no-storage-needed", "checkpoint": "", "transaction": ""}
    if not granted:
        assert {r["code"] for r in results[1:]} == {"weights_output_ungranted"} and not shown
        return
    assert {r["code"] for r in results[1:]} == {"granted"}
    assert results[1] == results[2], "the same run's attempt replays its committed output"
    assert results[3]["checkpoint"] == results[4]["checkpoint"] == results[1]["checkpoint"]
    assert results[3]["transaction"] == results[4]["transaction"] != results[1]["transaction"]
    assert [(item.output, item.repo_id) for item in shown] == [("model", weights_store.DERIVED)] * 4
    out = store.checkpoint(shown[0].revision)
    assert out.id == results[1]["checkpoint"]
    unet = out.components["unet"]
    assert set(unet.tensors) == {"w", "extra"} and unet.config == {"in": 2}
