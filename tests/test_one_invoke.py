"""One Invoke serves every function: `Weights[M]` sources and weights-output grants ride the
attempt, over a real store and a real seam; the executor reads and writes the store itself."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest
import tensorfs

from cozy_runtime import weights_store
from cozy_runtime.author import ModelArtifact, describe
from cozy_runtime.author._executor_requests import (
    Answer,
    CheckpointItem,
    ChildCall,
    Reply,
    Request,
    refuse,
)
from cozy_runtime.derive import safetensors_io as st
from cozy_runtime.derive.operations import MAX_QUANTIZED_BYTES, QuantizationPlan, app
from cozy_runtime.internal.discovery import Discovered, discover
from cozy_runtime.internal.seam import RESULT_DOCUMENT
from session import session
from store_fixtures import home

MODULE = "one_invoke_fixture"
QUANTIZE = "cozy_runtime.derive.operations"
SOURCE = """
from typing import Any
import array
import msgspec
from cozy_runtime.author import App, Context, ModelArtifact, Weights, invocable
from cozy_runtime.derive.operations import QuantizationPlan, quantize
app = App()
class Request(msgspec.Struct):
    component: str
class Read(msgspec.Struct):
    keys: list[str]
    values: list[float]
class Count(msgspec.Struct):
    n: int
@app.entrypoint(accelerator=False)
def read(payload: Request, source: Weights[Any]) -> Read:
    keys = sorted(source.keys(payload.component))
    values = array.array("f", bytes(16))
    source.read_values_into(payload.component, "a.weight", 0, values)
    return Read(keys, list(values))
@invocable
async def child(ctx: Context, *, source: Weights[Any], n: int) -> Count:
    return Count(n)
app.entrypoint(child)
@invocable
async def calls_child(ctx: Context, *, artifact: ModelArtifact) -> Count:
    return await child(source=artifact, n=2)
app.entrypoint(calls_child, accelerator=False)
@invocable
async def calls_quantize(ctx: Context, *, artifact: ModelArtifact) -> ModelArtifact:
    plan = QuantizationPlan(("unet",))
    return await quantize(source=artifact, plan=plan, encoding="fp8.rowwise")
app.entrypoint(calls_quantize, accelerator=False)
"""
VALUES = np.random.default_rng(22).normal(0, 0.05, (8, 64)).astype(np.float32)


@pytest.fixture
def package(tmp_path: Path) -> Iterator[Discovered]:
    project = tmp_path / "project"
    project.mkdir()
    (project / f"{MODULE}.py").write_text(SOURCE)
    (project / "package.toml").write_text(f'[application]\nobject="{MODULE}:app"\n')
    yield discover(project)
    sys.modules.pop(MODULE, None)


def _source(root: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A bf16 `unet` of three tensors, each the same values plus its index."""
    store = home(root, monkeypatch)
    bf16 = {
        key: np.frombuffer(st.from_f32(VALUES + index, "BF16"), np.uint16).reshape(8, 64)
        for index, key in enumerate(("z.weight", "a.weight", "bias"))
    }
    writer = tensorfs.Writer(cache_dir=store)
    unet = writer.component(
        "unet", cls=("diffusers", "", "UNet2DConditionModel"), config={"in": 64}
    )
    for key, value in bf16.items():
        unet.put(key, memoryview(value), dtype="BF16", shape=[8, 64])
    ref = writer.commit().id
    assert ref is not None
    return ref


def test_a_weights_argument_reads_the_checkpoint_its_attempt_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package: Discovered
) -> None:
    source = _source(tmp_path / "tensorfs", monkeypatch)
    with session(tmp_path, package) as host:
        spool = tmp_path / "read"
        reply = host.run("read", {"component": "unet"}, spool, sources={"source": source})
        assert reply["outcome"]["terminal"] == "succeeded", reply["outcome"]
        read = json.loads((spool / RESULT_DOCUMENT).read_bytes())
        assert read["keys"] == ["a.weight", "bias", "z.weight"]
        assert np.allclose(read["values"], VALUES[0, :4] + 1, atol=1e-2)
        # The next attempt names no checkpoint: the last one's never carries over.
        again = host.run("read", {"component": "unet"}, tmp_path / "again")
        assert again["outcome"]["code"] == "unbound_model", again["outcome"]


@pytest.mark.parametrize("granted", [True, False])
def test_a_writers_output_lands_only_through_its_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, granted: bool
) -> None:
    source = _source(tmp_path / "tensorfs", monkeypatch)
    module = sys.modules[app.get("quantize").fn.__module__]
    builtin = Discovered(app, f"{module.__name__}:app", tmp_path, module, describe(app))
    outputs = {"model": MAX_QUANTIZED_BYTES}
    shown: list[CheckpointItem] = []

    def machine(request: Request) -> Reply:
        if not isinstance(request, CheckpointItem):
            return refuse("no_durable_exchange", type(request).__name__)
        shown.append(request)
        return Answer(ok=True)

    spool = tmp_path / "quantize"
    with session(tmp_path, builtin) as host:
        host.handler = machine
        reply = host.run(
            "quantize",
            {"plan": msgspec.to_builtins(QuantizationPlan(("unet",))), "encoding": "fp8.rowwise"},
            spool,
            sources={"source": source},
            weights=outputs if granted else {},
        )
    if not granted:
        assert reply["outcome"]["code"] == "weights_output_ungranted", reply["outcome"]
        assert not shown
        return
    assert reply["outcome"]["terminal"] == "succeeded", reply["outcome"]
    result = msgspec.json.decode((spool / RESULT_DOCUMENT).read_bytes(), type=ModelArtifact)
    assert result.checkpoint != source
    (item,) = shown
    assert (item.output, item.repo_id) == ("model", weights_store.DERIVED)
    lane = tensorfs.open(item.revision, cache_dir=tmp_path / "tensorfs", local_files_only=True)
    assert lane.id == result.checkpoint
    encoded = lane.components["unet"].tensors["a.weight"].encoding
    assert encoded is not None and encoded.scheme == "fp8.rowwise"


@pytest.mark.parametrize(
    ("root", "row", "payload"),
    [
        ("calls_child", {"module": MODULE, "export": "child", "self": True}, {"n": 2}),
        (
            "calls_quantize",
            {"module": QUANTIZE, "export": "quantize", "builtin": "operations"},
            {
                "plan": json.loads(msgspec.json.encode(QuantizationPlan(("unet",)))),
                "encoding": "fp8.rowwise",
            },
        ),
    ],
)
def test_a_direct_call_carries_its_models_beside_its_request(
    tmp_path: Path, package: Discovered, root: str, row: dict[str, Any], payload: dict[str, Any]
) -> None:
    artifact = ModelArtifact("producer", "model", "sha256:" + "ab" * 32, "sha256:" + "cd" * 32)
    calls: list[ChildCall] = []

    def owner(request: Request) -> Reply:
        assert isinstance(request, ChildCall)
        calls.append(request)
        return refuse("child_call_refused", "recorded")

    with session(tmp_path, package, call_interfaces=[row]) as host:
        host.handler = owner
        reply = host.run(root, {"artifact": msgspec.to_builtins(artifact)}, tmp_path / root)
    assert reply["outcome"]["code"] == "child_call_refused", reply["outcome"]
    (call,) = calls
    sent = json.loads(call.payload)
    assert sent["models"] == {"source": msgspec.to_builtins(artifact)}
    assert {key: sent["payload"][key] for key in payload} == payload
    assert "source" not in sent["payload"]
