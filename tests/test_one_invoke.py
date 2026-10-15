"""One Invoke serves every function: `Weights[M]` sources and weights-output grants ride the
attempt, over a real Store, a real seam and the owner's writer channels."""

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

from cozy_runtime.author import ModelArtifact, describe
from cozy_runtime.author._artifacts import ObjectRef
from cozy_runtime.author._executor_requests import ChildCall, Reply, Request, refuse
from cozy_runtime.derive.operations import MAX_QUANTIZED_BYTES, QuantizationPlan, app
from cozy_runtime.internal import weights_sink
from cozy_runtime.internal.discovery import Discovered, discover
from cozy_runtime.internal.seam import RESULT_DOCUMENT
from native_weights import NativeExecution
from session import session
from test_quantize_artifact_native import _source
from weights_channel import broker_lane

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
    return await quantize(source=artifact, plan=QuantizationPlan(("unet",)), encoding="mxfp8/1")
app.entrypoint(calls_quantize, accelerator=False)
"""


@pytest.fixture
def package(tmp_path: Path) -> Iterator[Discovered]:
    project = tmp_path / "project"
    project.mkdir()
    (project / f"{MODULE}.py").write_text(SOURCE)
    (project / "package.toml").write_text(f'[application]\nobject="{MODULE}:app"\n')
    yield discover(project)
    sys.modules.pop(MODULE, None)


def test_a_weights_argument_reads_the_checkpoint_its_attempt_names(
    tmp_path: Path, package: Discovered
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    source = _source(store, "unet")
    root = tmp_path / "attempts"
    with (
        NativeExecution(store, root, "read", {"source": source}, {}) as owner,
        session(tmp_path, package) as host,
    ):
        host.handler = broker_lane(owner.broker, owner.attempt)
        reply = host.run(
            "read", {"component": "unet"}, owner.spool, sources={"source": source.manifest.digest}
        )
        assert reply["outcome"]["terminal"] == "succeeded", reply["outcome"]
        read = json.loads((owner.spool / RESULT_DOCUMENT).read_bytes())
        assert read["keys"] == ["a.weight", "bias", "z.weight"]
        written = np.random.default_rng(22).normal(0, 0.05, (8, 64)).astype(np.float32) + 1
        assert np.allclose(read["values"], written[0, :4], atol=1e-2)
        # The next attempt names no checkpoint: the last one's grant never carries over.
        again = host.run("read", {"component": "unet"}, root / "again")
        assert again["outcome"]["code"] == "unbound_model", again["outcome"]


@pytest.mark.parametrize("granted", [True, False])
def test_a_writers_output_lands_only_through_its_grant(tmp_path: Path, granted: bool) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    source = _source(store, "unet")
    module = sys.modules[app.get("quantize").fn.__module__]
    builtin = Discovered(app, f"{module.__name__}:app", tmp_path, module, describe(app), {})
    outputs = {"model": MAX_QUANTIZED_BYTES}
    with (
        NativeExecution(
            store, tmp_path / "attempts", "quantize", {"source": source}, outputs
        ) as owner,
        session(tmp_path, builtin) as host,
    ):
        host.handler = broker_lane(owner.broker, owner.attempt)
        reply = host.run(
            "quantize",
            {"plan": msgspec.to_builtins(QuantizationPlan(("unet",))), "encoding": "mxfp8/1"},
            owner.spool,
            sources={"source": source.manifest.digest},
            weights=outputs if granted else {},
        )
    if not granted:
        assert reply["outcome"]["code"] == "weights_output_ungranted", reply["outcome"]
        assert not owner.receipts
        return
    assert reply["outcome"]["terminal"] == "succeeded", reply["outcome"]
    result = msgspec.json.decode((owner.spool / RESULT_DOCUMENT).read_bytes(), type=ModelArtifact)
    (receipt,) = owner.receipts.values()
    assert result.manifest.digest == weights_sink.native_receipt(receipt).manifest.digest
    assert store.manifest(result.manifest.digest)["header"] is not None
    assert result.manifest.digest != source.manifest.digest


@pytest.mark.parametrize(
    ("root", "row", "payload"),
    [
        ("calls_child", {"module": MODULE, "export": "child", "self": True}, {"n": 2}),
        (
            "calls_quantize",
            {"module": QUANTIZE, "export": "quantize", "builtin": "operations"},
            {
                "plan": json.loads(msgspec.json.encode(QuantizationPlan(("unet",)))),
                "encoding": "mxfp8/1",
            },
        ),
    ],
)
def test_a_direct_call_carries_its_models_beside_its_request(
    tmp_path: Path, package: Discovered, root: str, row: dict[str, Any], payload: dict[str, Any]
) -> None:
    artifact = ModelArtifact(
        "producer", "model", ObjectRef("sha256:" + "ab" * 32, 9), "sha256:" + "cd" * 32
    )
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
