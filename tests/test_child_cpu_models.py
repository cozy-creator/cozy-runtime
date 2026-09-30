"""An explicitly CPU child derives owned weights without taking a GPU execution lane."""

from __future__ import annotations

import base64
import json

import pytest

from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import needs_executor
from test_machine_partial_work import machine

SOURCE = """import struct
import msgspec
import tensorfs
from tensorfs.derived import Derivation, Part, Source as NativeSource, Target, Tensor
from cozy_runtime.author import App, Context, Loader, Model, ModelArtifact, WeightsOutput, invocable
app = App()
class Request(msgspec.Struct): pass
class Result(msgspec.Struct):
    value: float
class Source(Model[object]):
    def load(self, loader: Loader) -> None:
        raise AssertionError("CPU model sources must not be constructed for serving")
@invocable
async def create(ctx: Context) -> ModelArtifact:
    plain = dict(tensorfs.seed_digests())["plain/1"]
    definition = Derivation({}, {"model": Target(add={"weight": Tensor("f32", (1,), plain, {"value": Part("f32", (1,))})})}, {}, (("model", "weight"),))
    with ctx.output("model").open(definition) as writer:
        writer.add_part("model", "weight", "value", struct.pack("<f", 7))
        return ctx.adopt_model(writer.commit())
app.job(create, accelerator=False, weights=(WeightsOutput("model", max_new_bytes=4096),))
@invocable
async def derive(ctx: Context, *, source: Source) -> ModelArtifact:
    with ctx.tensorfs_source(source) as held:
        facts = held.inspect()
        declaration = Derivation({"base": facts.source}, {"model": Target(source="base", source_component="model")}, {}, (("model", "weight"),))
        with ctx.output("model").open(declaration) as writer:
            return ctx.adopt_model(writer.commit())
app.job(derive, accelerator=False, weights=(WeightsOutput("model", max_new_bytes=0),))
@invocable
async def read(ctx: Context, *, source: Source) -> Result:
    raw = bytearray(4)
    with ctx.tensorfs_source(source) as held:
        held.read_part_into("model", "weight", "value", 0, raw)
    return Result(struct.unpack("<f", raw)[0])
app.job(read, accelerator=False)
@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    return await read(source=await derive(source=await create()))
"""


@needs_executor
def test_explicit_cpu_model_source_and_output_stay_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    with machine(monkeypatch, SOURCE) as pod:
        pod.start(("create", "derive", "read"))
        pod.submit("root")
        pod.wait(
            "the first child to be accepted",
            lambda: bool(pod.rows("SELECT preparation FROM executions WHERE request != 'root'")),
            120,
        )
        for row in pod.rows("SELECT preparation FROM executions WHERE request != 'root'"):
            raw = row["preparation"]
            assert isinstance(raw, bytes)
            preparation = json.loads(raw)
            state = pb.DesiredWorkerState.FromString(base64.b64decode(preparation["state"]))
            assert state.job.orchestration, (
                "explicit CPU metadata was overridden by a weights output"
            )
        outcome = pod.outcome("root")
        assert outcome["status"] == pb.OUTCOME_STATUS_SUCCEEDED, outcome
        result = outcome["result"]
        assert isinstance(result, dict) and isinstance(result["inline_result"], str)
        assert json.loads(base64.b64decode(result["inline_result"])) == {"value": 7.0}
        assert len(pod.rows("SELECT request FROM executions WHERE request != 'root'")) == 3
        for row in pod.rows("SELECT preparation FROM executions WHERE request != 'root'"):
            raw = row["preparation"]
            assert isinstance(raw, bytes)
            state = pb.DesiredWorkerState.FromString(base64.b64decode(json.loads(raw)["state"]))
            assert state.job.orchestration and not state.job.resource_caps.device_required
        assert pod.worker.gpu.view() == {"leases": {}, "grants": {}, "waiting": {}}
