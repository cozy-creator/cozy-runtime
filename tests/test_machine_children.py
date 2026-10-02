"""Captured Python composition executes on the worker without an observer."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import grpc
import msgspec
import pytest

import signed_claims
from cozy_runtime import canonical_json
from cozy_runtime.author._executor_requests import CallRequest, Reply
from cozy_runtime.internal import accel, child_env
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker import activity
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.control import GrpcControlHost
from cozy_runtime.internal.worker.machine_calls import CallRecord
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from local_owner import LocalRecordOwner, LocalRequest
from test_end_to_end import NO_EXECUTOR

SOURCE = """import msgspec
from cozy_runtime.author import App, Context, invocable
app = App()
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    value: int

@invocable(memoize=True)
async def leaf(ctx: Context, *, n: int) -> Result:
    return Result(n * 2)
app.job(leaf)

@invocable
async def compose(ctx: Context, *, n: int) -> Result:
    first = await leaf(n=n)
    second = await leaf(n=n)
    return Result(first.value + second.value)
app.job(compose)

@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    return await compose(n=3)
"""


MODEL_SOURCE = """import struct
import tensorfs
from tensorfs.derived import Config, Derivation, Part, Target, Tensor
from cozy_runtime.author import App, Context, Model, ModelArtifact, WeightsOutput, invocable

def assert_no_store():
    import os
    from pathlib import Path
    assert not any(
        "tensorfs.sqlite" in os.readlink(fd)
        for fd in Path("/proc/self/fd").iterdir() if fd.exists()
    )


app = App()
PLAIN = dict(tensorfs.seed_digests())["plain/1"]

class Source(Model[object], encoded_leaves="accept"):
    def load(self, loader):
        del loader

@invocable(memoize=True)
async def leaf(ctx: Context, *, n: int) -> ModelArtifact:
    assert_no_store()
    definition = Derivation(
        sources={},
        targets={"body": Target(add={
            "layer.weight": Tensor("f16", (16,32), PLAIN, {"value": Part("f16", (16,32))}),
            "layer.bias": Tensor("f16", (16,), PLAIN, {"value": Part("f16", (16,))}),
        })},
        configs={"pipeline": Config("add")},
        order=(("body", "layer.weight"), ("body", "layer.bias")),
    )
    with tensorfs.derive(ctx.output("model"), definition) as output:
        if output.receipt is None:
            output.add_part(
                "body", "layer.weight", "value",
                struct.pack("<512e", *(i/257-1 for i in range(512)))
            )
            output.add_part("body", "layer.bias", "value", struct.pack("<16e", *range(16)))
            output.add_config("pipeline", b"{}")
        receipt = output.receipt or output.commit()
    return ctx.adopt_model(receipt)

app.job(leaf, weights=(WeightsOutput("model", max_new_bytes=65536),))
import msgspec
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    value: int

@invocable
async def consume(ctx: Context, *, model: Source) -> Result:
    assert_no_store()
    source = ctx.tensorfs_source(model)
    view = source.inspect()
    assert_no_store()
    return Result(len(view.components["body"]))
app.job(consume)

@invocable
async def compose(ctx: Context, *, n: int) -> Result:
    first = await leaf(n=n)
    second = await leaf(n=n)
    assert first.manifest == second.manifest
    return await consume(model=first)
app.job(compose)

@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    return await compose(n=3)
"""


FILE_SOURCE = """import msgspec
from cozy_runtime.author import App, Context, FileAsset, Outputs, invocable
app = App()
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    value: int
class Report(msgspec.Struct):
    report: FileAsset
@invocable(memoize=True)
async def leaf(ctx: Context, out: Outputs, *, n: int) -> Report:
    path = out.temporary_file('.json')
    path.write_bytes(b'{"observed":7}')
    return Report(out.save_file(path, media_type='application/json'))
app.job(leaf)
@invocable
async def compose(ctx: Context, *, n: int) -> Result:
    first = await leaf(n=n)
    second = await leaf(n=n)
    assert first.report.read_bytes() == second.report.read_bytes() == b'{"observed":7}'
    return Result(12)
app.job(compose)
@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    return await compose(n=3)
"""


BYTE_INPUT_SOURCE = """import msgspec
from typing import Annotated
from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs, Tree, invocable
app = App()
ReportFile = Annotated[FileAsset, AssetBound(max_bytes=200000, media_types=('application/json',))]
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    value: int
class Report(msgspec.Struct):
    report: ReportFile
    bundle: Tree
@invocable(memoize=True)
async def leaf(ctx: Context, out: Outputs, *, n: int) -> Report:
    data = msgspec.json.encode({'observed': n, 'padding': 'verified' * 12000})
    root = out.temporary_file()
    root.mkdir()
    (root / 'report.json').write_bytes(data)
    (root / 'extra.txt').write_text('verified')
    return Report(out.save_bytes(data, media_type='application/json'), out.save_tree(root))
app.job(leaf)
@invocable
async def consume(ctx: Context, *, report: ReportFile, bundle: Tree) -> Result:
    raw = report.read_bytes()
    assert len(raw) > 48 * 1024
    assert raw == (bundle.path / 'report.json').read_bytes()
    assert (bundle.path / 'extra.txt').read_text() == 'verified'
    return Result(12)
app.job(consume)
@invocable
async def compose(ctx: Context, *, n: int) -> Result:
    first = await leaf(n=n)
    second = await leaf(n=n)
    assert first.report.read_bytes() == second.report.read_bytes()
    return await consume(report=first.report, bundle=first.bundle)
app.job(compose)
@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    return await compose(n=3)
"""


MODELED_SERVING_SOURCE = (
    "from cozy_runtime.author import ActivationCapture, Config, Telemetry, uses_components\n"
    + MODEL_SOURCE.replace(
        "        del loader",
        "        self.pipe = loader.construct(Pipeline, factory=build_pipeline)\n"
        "    @uses_components('body')\n"
        "    def infer(self, n, tel):\n"
        "        import torch\n"
        "        body = self.pipe.components['body']\n"
        "        on_step = tel.step_callback(1, stage='denoise')\n"
        "        with torch.inference_mode():\n"
        "            value = body(torch.ones((1,32), dtype=torch.float16,\n"
        "                                    device=body.layer.weight.device))\n"
        "            assert bool(torch.isfinite(value).all())\n"
        "        on_step(0)\n"
        "        return int(value.shape[1])",
    ).replace(
        "    return await consume(model=first)",
        "    call = generate(model=first, n=n, capture=ActivationCapture(('body',), (0,)))\n"
        "    result = await call\n"
        "    assert call.observation.environment.worker_boot_id\n"
        "    assert call.observation.capture.tree is not None\n"
        "    return result",
    )
    + """
class Pipeline:
    def __init__(self):
        import torch
        class Body(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(32, 16, dtype=torch.float16)
            def forward(self, value):
                return self.layer(value)
        self.components = {'body': Body()}
def build_pipeline(config: Config) -> Pipeline:
    return Pipeline()
@invocable
async def generate(ctx: Context, tel: Telemetry, *, model: Source, n: int) -> Result:
    return Result(model.infer(n, tel))
app.entrypoint(generate)
"""
)


MODELED_MEDIA_SOURCE = (
    "from cozy_runtime.author import ImageAsset, Outputs, Tree\n"
    + MODELED_SERVING_SOURCE.replace(
        "class Result(msgspec.Struct):\n    value: int",
        "class Result(msgspec.Struct):\n    value: int\n"
        "class Render(msgspec.Struct):\n    value: int\n    image: ImageAsset\n"
        "class Media(msgspec.Struct):\n    value: int\n    image: ImageAsset\n    capture: Tree",
    )
    .replace("        return int(value.shape[1])", "        return value")
    .replace(
        "async def compose(ctx: Context, *, n: int) -> Result:",
        "async def compose(ctx: Context, *, n: int) -> Media:",
    )
    .replace(
        "async def nested(ctx: Context, payload: Request) -> Result:",
        "async def nested(ctx: Context, payload: Request) -> Media:",
    )
    .replace(
        "async def generate(ctx: Context, tel: Telemetry, *, model: Source, n: int) -> Result:",
        "async def generate(ctx: Context, tel: Telemetry, out: Outputs,\n"
        "                   *, model: Source, n: int) -> Render:",
    )
    .replace(
        "    return Result(model.infer(n, tel))",
        "    from PIL import Image\n"
        "    value = model.infer(n, tel)\n"
        "    pixels = (value.reshape(4,4).float().sigmoid().cpu().numpy()*255).astype('uint8')\n"
        "    return Render(16, out.save_image(Image.fromarray(pixels).convert('RGB')))",
    )
    .replace(
        "    call = generate(model=first, n=n, capture=ActivationCapture(('body',), (0,)))\n"
        "    result = await call\n"
        "    assert call.observation.environment.worker_boot_id\n"
        "    assert call.observation.capture.tree is not None\n"
        "    return result",
        "    for render_n in (1,2,3,4,5,6,7,8) * 3:\n"
        "        call = generate(model=first, n=render_n,\n"
        "                        capture=ActivationCapture(('body',), (0,)))\n"
        "        result = await call\n"
        "        assert call.observation.environment.worker_boot_id\n"
        "        assert call.observation.capture.tree is not None\n"
        "    return Media(result.value, result.image, call.observation.capture.tree)",
    )
)


def native_origin(worker: Worker) -> tuple[ThreadingHTTPServer, threading.Thread]:
    fixture = Path(__file__).parent / "testdata/native_source"
    body = (fixture / "first.safetensors").read_bytes()
    digest = hashlib.sha256(body).hexdigest()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_GET(self) -> None:
            if self.path.startswith("/api/models/"):
                data = json.dumps(
                    [
                        {
                            "type": "file",
                            "path": "provider/first.safetensors",
                            "size": len(body),
                            "lfs": {"oid": digest, "size": len(body)},
                        }
                    ]
                ).encode()
                self.send_response(200)
            else:
                start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
                data = body[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    assert worker.source_calls is not None
    worker.source_calls.endpoints = {"huggingface": f"http://127.0.0.1:{server.server_port}"}
    worker.source_calls.native_registry = (fixture / "registry.json").read_bytes()
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    return server, thread


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
@pytest.mark.parametrize(
    "case",
    [
        "normal",
        "internal",
        "internal_root",
        "internal_serving",
        "pause",
        "cycle",
        "child_failure",
        "capacity",
        "model",
        "model_sequence",
        "model_result",
        "model_admission",
        "source",
        "source_conflict",
        "file",
        "byte_inputs",
        "byte_inputs_serving",
        "byte_input_admission",
        "serving",
        pytest.param("serving_model", marks=pytest.mark.real_gpu),
        pytest.param("serving_media", marks=pytest.mark.real_gpu),
    ],
)
def test_nested_capture_and_leaf_memoization_without_control_stream(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    proof_gate: Path | None = None,
    proof_info: Path | None = None,
) -> None:
    import test_job_preparation_isolation as fixture
    from conftest import image_python

    if case in ("serving_model", "serving_media"):
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            pytest.skip("modeled serving requires the Torch image qualification environment")
        import torch

        if torch.version.cuda is None or not accel.present(torch, "cuda"):
            pytest.skip("modeled serving requires a CUDA-capable qualification environment")

    with tempfile.TemporaryDirectory(prefix="cz-child.") as directory:
        root = Path(directory)
        source = (
            MODEL_SOURCE
            if case
            in (
                "model",
                "model_sequence",
                "model_result",
                "model_admission",
                "source",
                "source_conflict",
            )
            else SOURCE
        )
        if case == "model_sequence":
            source = source.replace(
                "    return await consume(model=first)",
                "    first_result = await consume(model=first)\n"
                "    from pathlib import Path\n    import asyncio\n"
                + f"    Path({str(root / 'between-children')!r}).touch()\n"
                + f"    while not Path({str(root / 'prepared-other')!r}).exists():\n"
                + "        ctx.raise_if_cancelled()\n        await asyncio.sleep(0.02)\n"
                + "    second_result = await consume(model=first)\n"
                + "    assert first_result == second_result\n"
                + "    return second_result",
            )
        if case == "model_admission":
            source = source.replace(
                "async def consume(ctx: Context, *, model: Source) -> Result:",
                "async def consume(ctx: Context, *, model: Source, z_other: Source) -> Result:",
            ).replace(
                "    return await consume(model=first)",
                "    bad = ModelArtifact(\n"
                "        'unowned', first.output_slot, first.manifest,\n"
                "        first.tensorfs_receipt_digest)\n"
                "    return await consume(model=first, z_other=bad)",
            )
        if case == "model_result":
            source = source.replace(
                "async def nested(ctx: Context, payload: Request) -> Result:",
                "async def nested(ctx: Context, payload: Request) -> ModelArtifact:",
            ).replace("    return await compose(n=3)", "    return await leaf(n=3)")
        if case == "file":
            source = FILE_SOURCE
        if case in ("byte_inputs", "byte_inputs_serving", "byte_input_admission"):
            source = BYTE_INPUT_SOURCE
        if case == "byte_inputs_serving":
            source = source.replace("app.job(consume)", "app.entrypoint(consume)")
        if case == "byte_input_admission":
            source = source.replace(
                "bundle: Tree) -> Result:",
                "bundle: Tree, z_rejected: Annotated[FileAsset, "
                "AssetBound(max_bytes=1)]) -> Result:",
            ).replace("bundle=first.bundle)", "bundle=first.bundle, z_rejected=first.report)")
        if case in ("serving", "internal_serving"):
            source = (
                SOURCE.replace("@invocable(memoize=True)", "@invocable")
                .replace("app.job(leaf)", "app.entrypoint(leaf)")
                .replace(
                    "    first = await leaf(n=n)",
                    "    call = leaf(n=n)\n"
                    "    first = await call\n"
                    "    assert call.observation.environment.worker_boot_id\n",
                )
            )
        if case == "serving":
            # Each leaf times a stage and two steps; compose calls them under author labels.
            source = (
                source.replace("App, Context, invocable", "App, Context, Telemetry, invocable")
                .replace(
                    "async def leaf(ctx: Context, *, n: int) -> Result:\n",
                    "async def leaf(ctx: Context, tel: Telemetry, *, n: int) -> Result:\n"
                    "    with tel.stage('condition'):\n        pass\n"
                    "    on_step = tel.step_callback(2, stage='denoise')\n"
                    "    on_step(0)\n    on_step(1)\n",
                )
                .replace(
                    "async def compose(ctx: Context, *, n: int) -> Result:\n    call = leaf(n=n)\n",
                    "async def compose(ctx: Context, tel: Telemetry, *, n: int) -> Result:\n"
                    "    with tel.scope('Leaf 1'):\n        call = leaf(n=n)\n",
                )
                .replace(
                    "    second = await leaf(n=n)",
                    "    with tel.scope('Leaf 2'):\n        second = await leaf(n=n)",
                )
            )
        if case == "serving_model":
            source = MODELED_SERVING_SOURCE
        if case == "serving_media":
            source = MODELED_MEDIA_SOURCE
            if proof_gate is not None:
                source = source.replace(
                    "    return Media(result.value, result.image, call.observation.capture.tree)",
                    "    from pathlib import Path\n    import asyncio\n"
                    + f"    ready = Path({str(proof_gate.with_suffix('.ready'))!r})\n"
                    + "    ready.write_text(ctx.request_id)\n"
                    + f"    while not Path({str(proof_gate)!r}).exists():\n"
                    + "        ctx.raise_if_cancelled()\n        await asyncio.sleep(0.05)\n"
                    + "    return Media(result.value, result.image, call.observation.capture.tree)",
                )
        if case in ("source", "source_conflict"):
            source = source.replace(
                'return Result(len(view.components["body"]))',
                "return Result(sum(len(v) for v in view.components.values()))",
            )
            source = source.replace(
                "    return await compose(n=3)",
                "    from cozy_runtime.author.sources import (\n"
                "        download_huggingface, source_files, convert_cozytensors)\n"
                "    source = await download_huggingface('example/model', revision='"
                + "a"
                * 40
                + "', carriers=('provider/first.safetensors',))\n"
                "    await source_files(source)\n"
                "    model = await convert_cozytensors(source, profiles=('fixture/first/1',))\n"
                "    return await consume(model=model)",
            )
        if case == "pause":
            source = source.replace(
                "    return Result(n * 2)",
                "    from pathlib import Path\n    import asyncio\n"
                + f"    Path({str(root / 'entered')!r}).touch()\n"
                + f"    while not Path({str(root / 'gate')!r}).exists():\n"
                + "        ctx.raise_if_cancelled()\n        await asyncio.sleep(0.02)\n"
                + "    return Result(n * 2)",
            )
        if case == "source_conflict":
            source = source.replace(
                "    from cozy_runtime.author.sources import (",
                "    from pathlib import Path\n    import asyncio\n"
                + f"    Path({str(root / 'source-ready')!r}).touch()\n"
                + f"    while not Path({str(root / 'source-release')!r}).exists():\n"
                + "        ctx.raise_if_cancelled()\n        await asyncio.sleep(0.01)\n"
                + "    from cozy_runtime.author.sources import (",
            )
        if case == "cycle":
            source = source.replace("first = await leaf(n=n)", "first = await compose(n=n)")
        if case == "child_failure":
            source = source.replace(
                "    return Result(n * 2)", "    raise RuntimeError('leaf broke')"
            )
        if case in ("internal", "internal_root", "internal_serving"):
            source = source.replace("app.job(leaf)", "app.job(leaf, internal=True)")
            source = source.replace("app.entrypoint(leaf)", "app.entrypoint(leaf, internal=True)")
            source = source.replace("app.job(compose)", "app.job(compose, internal=True)")
        if case == "internal_root":
            source = source.replace("@app.job\n", "@app.job(internal=True)\n")
        if case == "capacity":
            from cozy_runtime.internal.worker import machine_slots

            monkeypatch.setattr(machine_slots, "MAX_ACTIVE_CALLS", 2)
        monkeypatch.setattr(fixture, "SOURCE", source)
        environment, preparation = fixture.package(root, "nested")
        (root / "artifacts").mkdir()
        worker = Worker(
            RuntimeConfig(
                cozy_home=root / "home",
                credentials=Credentials(),
                record_owner_public_key=signed_claims.PUBLIC_KEY,
                child_base_env=tuple(
                    sorted(
                        (key, value)
                        for key, value in os.environ.items()
                        if not child_env.erased(key) and key != "PYTHONPATH"
                    )
                ),
            ),
            WorkerOptions(
                **signed_claims.IDENTITY,
                root=root / "worker",
                devices="",
                accelerator_backend="none",
                python=str(image_python()),
                install_root=environment,
                artifact_cache=root / "artifacts",
                tensorfs_root=root / "store",
                grant_roots=(str(root),),
            ),
            GrpcControlHost("127.0.0.1:0", root / "address"),
        )
        thread = threading.Thread(target=worker.run, daemon=True)
        if proof_info is not None:
            proof_info.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "store": str(root / "store"),
                        "artifacts": str(root / "artifacts"),
                        "owner": "owner",
                        "gate": str(proof_gate),
                        "ready": str(proof_gate.with_suffix(".ready")) if proof_gate else "",
                    }
                )
            )
        origin = native_origin(worker) if case in ("source", "source_conflict") else None
        try:
            prepared = worker.prepare_local_package(preparation)
            captured, capture_digest = documents.identity(
                pb.MachineExecutionCapture(
                    root_installation_id=prepared.installed_package.installation_id,
                    installed_packages=[prepared.installed_package],
                    bindings=[
                        pb.MachineCallableBinding(
                            caller_installation_id=prepared.installed_package.installation_id,
                            callee_installation_id=prepared.installed_package.installation_id,
                            module="prepare_nested",
                            export=export,
                            entrypoint=export,
                        )
                        for export in (
                            ("compose", "consume", "generate", "leaf")
                            if case in ("serving_model", "serving_media")
                            else ("compose", "consume", "leaf")
                            if case
                            in (
                                "model",
                                "model_sequence",
                                "model_result",
                                "model_admission",
                                "source",
                                "byte_inputs",
                                "byte_inputs_serving",
                                "byte_input_admission",
                            )
                            else ("compose", "leaf")
                        )
                    ],
                )
            )
            if worker.supervision.current is not None:
                worker.supervision.retire_current(worker.supervision.current, "begin lifecycle")
            plans = [
                json.loads(path.read_bytes()) for path in (root / "home/job-plans").glob("*/*.json")
            ]
            binding = JobBinding.read(next(row for row in plans if row.get("job") == "nested"))
            thread.start()
            # A hang is STILLNESS, not elapsed time: a loaded shared box may take any time
            # while the worker, its executors and its journal keep moving.
            still: list[Any] = ["", time.monotonic()]

            def advancing() -> bool:
                reading = str(activity.meter(worker))
                if reading != still[0]:
                    still[:] = [reading, time.monotonic()]
                assert time.monotonic() - still[1] < 180, "the worker stopped moving"
                return thread.is_alive()

            while not (root / "address").exists():
                assert advancing()
                time.sleep(0.02)
            claim = signed_claims.claim()
            worker.serve_stream(iter([pb.RecordOwnerFrame(claim=claim)]), lambda _: None)
            generator = LocalRecordOwner(
                LocalRequest(
                    entrypoint="nested",
                    payload={},
                    outputs=("image", "capture") if case == "serving_media" else (),
                    kind="job",
                    request_id="root",
                    job_descriptor_id=binding.job_descriptor_id,
                    timeout_ms=900_000 if case == "serving_media" or proof_gate else 240_000,
                ),
                {},
                root / "grants",
                package_installation_id=binding.installation_id,
            )
            offered = generator.offer()
            state = generator.desired_state()
            state.job.orchestration = True
            channel = grpc.insecure_channel((root / "address").read_text().strip())
            client = rpc.WorkerControlStub(channel)
            submitted = pb.MachineExecutionSubmit(
                claim=claim,
                submission_id="root",
                capture_digest=capture_digest,
                capture_canonical_bytes=captured,
                offer=offered,
                prepared_state=state,
                payload_canonical_bytes=canonical_json.encode({}),
                expected_execution_workspace_id=client.GetMachineExecutionWorkspace(
                    pb.MachineExecutionWorkspaceQuery(claim=claim)
                ).execution_workspace_id,
            )
            if case == "internal_root":
                with pytest.raises(grpc.RpcError, match="internal_callable") as refused:
                    client.SubmitMachineExecution(submitted)
                assert refused.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                assert worker.executions is not None
                assert not worker.executions.owns("owner", "root")
                return
            client.SubmitMachineExecution(submitted)
            if case == "source_conflict":
                from cozy_runtime.internal import source_interfaces
                from cozy_runtime.internal.call_intent import canonical_intent
                from cozy_runtime.internal.worker import workspace_sources

                while not (root / "source-ready").exists():
                    assert advancing()
                    time.sleep(0.01)
                # A native slot already accepted a different frozen source intent.
                # The real public facade must receive the durable refusal and end.
                call = pb.ChildCallRequest(
                    parent_request_id="root",
                    parent_attempt_ordinal=1,
                    parent_invocation_spec_digest=submitted.offer.invocation_spec_digest,
                    call_index=0,
                    module=source_interfaces.MODULE,
                    export="download_huggingface",
                    request_canonical_bytes=canonical_json.encode(
                        {"repository": "different/model", "revision": "b" * 40}
                    ),
                )
                call.intent_digest = hashlib.sha256(canonical_intent(call)).digest()
                assert worker.workspace is not None
                workspace_sources.accepted(
                    worker.workspace,
                    "owner",
                    pb.NativeSourceCommand(
                        service_id=workspace_sources.identity("owner", "root", 0),
                        operation=pb.NATIVE_SOURCE_OPERATION_HUGGINGFACE,
                        parent_call=call,
                    ),
                )
                (root / "source-release").touch()
            if case == "model_sequence":
                while not (root / "between-children").exists():
                    assert advancing()
                    time.sleep(0.02)
                _, other = fixture.package(root, "unrelated")
                worker.prepare_local_package(other)
                (root / "prepared-other").touch()
            if case == "pause":
                while not (root / "entered").exists():
                    assert advancing()
                    time.sleep(0.02)
                assert len(worker.lanes.composition) == 3
                # Two parents await a blocked leaf. They wait for a nudge, not a clock: a
                # 50 ms poll made ~40 child_poll exchanges in this second.
                exchanged: list[str] = []
                handle = worker.calls.handle

                def counted(attempt: AttemptRecord, request: CallRequest) -> Reply:
                    exchanged.append(type(request).__name__)
                    return handle(attempt, request)

                worker.calls.handle = counted  # type: ignore[method-assign]
                time.sleep(1)
                worker.calls.handle = handle  # type: ignore[method-assign]
                assert exchanged.count("ChildPoll") <= 2, exchanged
                worker.control_execution(claim, "root", "pause-root", 1, "pause")
                while True:
                    assert advancing()
                    assert worker.workspace is not None
                    with worker.workspace.locked() as db:
                        states = [row[0] for row in db.execute("SELECT state FROM executions")]
                    if states == ["paused"] * 3 and not worker.lanes.composition:
                        break
                    time.sleep(0.02)
                (root / "gate").touch()
                paused = worker.execution_status(claim, "root")
                worker.control_execution(claim, "root", "resume-root", paused.generation, "resume")
            while worker.execution_status(claim, "root").state not in ("succeeded", "failed"):
                assert advancing()
                time.sleep(0.02)
            outcome = worker.collect_execution(claim, "root")
            body = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
            if case == "child_failure":
                assert body["status"] == pb.OUTCOME_STATUS_FAILED, body
                assert worker.workspace is not None
                with worker.workspace.locked() as db:
                    rows = db.execute(
                        "SELECT e.request,e.body FROM execution_events e LEFT JOIN "
                        "execution_calls c ON c.child_request=e.request WHERE e.kind='log'"
                    ).fetchall()
                by_request = {
                    request: log["payload"]["fields"]
                    for request, raw in rows
                    if (log := canonical_json.decode(raw))["payload"].get("name")
                    == "executor release"
                }
                # The leaf's own failure poisons its process. Each parent only awaited it and
                # stopped at the call boundary, so its process stays reusable (run 1516).
                assert len(by_request) == 3, by_request
                reasons = sorted(row["reuse_miss_reason"] for row in by_request.values())
                assert reasons == ["", "", "failed/unhandled_exception"], by_request
                assert by_request["root"]["reusable"], by_request
                return
            if case == "source_conflict":
                assert body["status"] == pb.OUTCOME_STATUS_FAILED, body
                assert worker.workspace is not None
                with worker.workspace.locked() as db:
                    refused = db.execute(
                        "SELECT safe_code,safe_detail FROM execution_calls "
                        "WHERE parent_request='root' AND call_index=0"
                    ).fetchone()
                    assert refused["safe_code"] == "child_admission_refused"
                    assert "another native source intent" in refused["safe_detail"]
                return
            if case in ("cycle", "capacity", "model_admission", "byte_input_admission"):
                assert body["status"] == pb.OUTCOME_STATUS_FAILED, body
                assert worker.workspace is not None
                with worker.workspace.locked() as db:
                    count = db.execute("SELECT count(*) FROM executions").fetchone()[0]
                    assert count == (2 if case == "cycle" else 3)
                    if case == "cycle":
                        assert db.execute(
                            "SELECT 1 FROM execution_calls WHERE safe_detail "
                            "LIKE '%active ancestor%'"
                        ).fetchone()
                    if case == "byte_input_admission":
                        refused = db.execute(
                            "SELECT child_request FROM execution_calls WHERE safe_detail "
                            "LIKE '%declared byte bound%'"
                        ).fetchone()
                        assert refused is not None
                        records = db.execute(
                            "SELECT retention FROM execution_model_holds WHERE recipient=?",
                            (refused[0],),
                        ).fetchall()
                        assert records
                        for record in records:
                            hold = pb.DerivedRetentionRequest.FromString(record[0])
                            assert (
                                db.execute(
                                    "SELECT state FROM holds WHERE id=?", (hold.retention_id,)
                                ).fetchone()[0]
                                == "released"
                            )
                    if case == "model_admission":
                        refused = db.execute(
                            "SELECT child_request FROM execution_calls "
                            "WHERE safe_detail LIKE '%provenance%'"
                        ).fetchone()
                        assert refused is not None
                        (row,) = db.execute(
                            "SELECT retention FROM execution_model_holds WHERE recipient=?",
                            (refused[0],),
                        ).fetchall()
                        hold = pb.DerivedRetentionRequest.FromString(row[0])
                        assert (
                            db.execute(
                                "SELECT state FROM holds WHERE id=?", (hold.retention_id,)
                            ).fetchone()[0]
                            == "released"
                        )
                while worker.lanes.composition:
                    assert advancing()
                    time.sleep(0.02)
                assert not worker.stop.is_set()
                return
            assert body["status"] == pb.OUTCOME_STATUS_SUCCEEDED, body
            assert not worker.stop.is_set()
            import base64

            if case == "serving_media":
                value = canonical_json.decode(base64.b64decode(body["result"]["inline_result"]))
                assert (
                    value["value"] == 16
                    and value["image"]["kind"] == "image"
                    and value["capture"]["kind"] == "tree"
                )
                assert {output["output_id"] for output in body["output_manifest"]["outputs"]} == {
                    "image",
                    "capture",
                }
                assert worker.workspace is not None
                with worker.workspace.locked() as db:
                    assert db.execute("SELECT count(*) FROM executions").fetchone()[0] == 27
                    assert db.execute("SELECT count(*) FROM operation_cache").fetchone()[0] == 1
                return
            if case == "model_result":
                import tensorfs

                from test_machine_execution import ack

                value = canonical_json.decode(base64.b64decode(body["result"]["inline_result"]))
                assert not body.get("weights_receipts")
                (retained,) = body["result"]["retained_models"]
                assert retained.get("result_pointer", "") == ""
                assert (
                    canonical_json.decode(
                        base64.b64decode(retained["model_artifact_canonical_bytes"])
                    )
                    == value
                )
                assert worker.workspace is not None
                received = worker.workspace.retain(
                    "owner",
                    pb.DerivedRetentionRequest(
                        weights_transaction_id=retained["retention"]["weights_transaction_id"],
                        tensorfs_receipt_digest=documents.raw(
                            retained["retention"]["tensorfs_receipt_digest"]
                        ),
                        retention_id="sha256:" + "e7" * 32,
                    ),
                )
                worker.acknowledge_execution_collection(claim, ack(outcome))
                tensorfs.gc(str(worker.workspace.store_root))
                assert documents.body(received.manifest) == value["manifest"]
                assert tensorfs.Store.open(str(worker.workspace.store_root)).manifest(
                    value["manifest"]["digest"]
                )
                return
            assert canonical_json.decode(base64.b64decode(body["result"]["inline_result"])) == {
                "value": 16
                if case == "serving_model"
                else 1
                if case in ("source", "source_conflict")
                else 2
                if case in ("model", "model_sequence")
                else 12
            }
            assert worker.workspace is not None
            with worker.workspace.locked() as db:
                if case in ("serving", "internal_serving"):
                    logs = [
                        canonical_json.decode(row["body"])
                        for row in db.execute("SELECT body FROM execution_events WHERE kind='log'")
                    ]
                    invoked = [
                        log["payload"]["fields"]
                        for log in logs
                        if log.get("payload", {}).get("name") == "executor invoke"
                    ]
                    released = [
                        log["payload"]["fields"]
                        for log in logs
                        if log.get("payload", {}).get("name") == "executor release"
                    ]
                    assert invoked and released
                    assert all(row["pid"] > 0 and row["started_ticks"] > 0 for row in invoked)
                    assert all(row["ranks"] and row["reusable"] for row in released)
                    assert all(not row["reuse_miss_reason"] for row in released)
                    # Each attempt's execution record names its processes by GPU (`gpus`,
                    # what triage and the GPU release carry), beside the older `ranks`.
                    executed = [a.execution for a in worker.engine.history.values() if a.execution]
                    assert executed and all(
                        [g["pid"] for g in e["gpus"]] == [r["pid"] for r in e["ranks"]]
                        and all(g["gpu"] == -1 and "rank" not in g for g in e["gpus"])
                        for e in executed
                    ), executed
                if case == "serving":
                    # Every settled call is on the root's journal, grandchildren included:
                    # its function, the author's label, its status and where its time went.
                    records = [
                        msgspec.json.decode(row["body"], type=CallRecord)
                        for row in db.execute(
                            "SELECT body FROM execution_events WHERE request='root' "
                            "AND kind='call' ORDER BY sequence"
                        )
                    ]
                    compose, *leaves = sorted(records, key=lambda record: record.export)
                    assert (compose.export, compose.parent, compose.label) == (
                        "compose",
                        "root",
                        "",
                    )
                    assert [(leaf.label, leaf.index) for leaf in leaves] == [
                        ("Leaf 1", 0),
                        ("Leaf 2", 1),
                    ]
                    for leaf in leaves:
                        assert (leaf.export, leaf.parent, leaf.status, leaf.attempt) == (
                            "leaf",
                            compose.request,
                            "succeeded",
                            1,
                        )
                        assert leaf.stages["condition"].count == 1
                        assert leaf.steps["denoise"].count == len(leaf.steps["denoise"].series) == 2
                        assert 0 < leaf.called_unix_ms <= leaf.steps["denoise"].ended_unix_ms
                if case == "source":
                    # Native source work reports on the root's own journaled lane: byte
                    # samples per stage, then each stage's durable timing record.
                    events = [
                        canonical_json.decode(row["body"])
                        for row in db.execute(
                            "SELECT body FROM execution_events WHERE request='root' "
                            "AND kind IN ('progress','log') ORDER BY sequence"
                        )
                    ]
                    carrier = (Path(__file__).parent / "testdata/native_source").joinpath(
                        "first.safetensors"
                    )
                    for stage in ("Downloading source", "Converting to cozytensors"):
                        samples = [
                            event["payload"]
                            for event in events
                            if event["type"] == "progress"
                            and event["payload"].get("stage") == stage
                        ]
                        assert samples[-1]["unit"] == "bytes", samples
                        assert samples[-1]["position"] == samples[-1]["total"] > 0
                        (ended,) = [
                            event["payload"]["fields"]
                            for event in events
                            if event["type"] == "log" and event["payload"].get("name") == stage
                        ]
                        assert ended["completed"] and ended["elapsed_ms"] > 0
                        if stage == "Downloading source":
                            assert ended["bytes"] == carrier.stat().st_size
                assert db.execute("SELECT count(*) FROM executions").fetchone()[0] == (
                    2
                    if case in ("source", "source_conflict")
                    else 5
                    if case == "model_sequence"
                    else 4
                    if case
                    in (
                        "model",
                        "serving",
                        "internal_serving",
                        "serving_model",
                        "byte_inputs",
                        "byte_inputs_serving",
                    )
                    else 3
                )
                assert db.execute("SELECT count(*) FROM execution_calls").fetchone()[0] == (
                    5
                    if case == "model_sequence"
                    else 4
                    if case
                    in ("model", "source", "serving_model", "byte_inputs", "byte_inputs_serving")
                    else 3
                )
                assert db.execute("SELECT count(*) FROM operation_cache").fetchone()[0] == (
                    2
                    if case in ("source", "source_conflict")
                    else 0
                    if case in ("serving", "internal_serving")
                    else 1
                )
        finally:
            worker.request_stop()
            worker.host.stop()
            if thread.ident is not None:
                thread.join(30)
            assert not thread.is_alive()
            if origin is not None:
                origin[0].shutdown()
                origin[0].server_close()
                origin[1].join()


def test_managed_depth_counts_the_prospective_child(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from cozy_runtime.internal.worker.attempts import AttemptRecord
    from cozy_runtime.internal.worker.calls import Calls as Transport
    from cozy_runtime.internal.worker.machine_calls import Calls as Dispatcher
    from cozy_runtime.internal.worker.machine_child_target import Target
    from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
    from cozy_runtime.internal.worker.workspace_calls import Calls
    from cozy_runtime.internal.worker.workspace_executions import Executions
    from test_machine_calls import request, running
    from test_machine_execution import offer

    workspace = Workspace(tmp_path / "store")
    executions, calls = Executions(workspace), Calls(workspace)
    selected = "root"
    parents = []
    for _ in range(32):
        executions.submit(
            "owner",
            selected,
            b"c" * 32,
            offer(selected),
            expected_execution_workspace_id=executions.workspace_id,
        )
        parent = running(executions, selected)
        parents.append(selected)
        selected = calls.accept("owner", request(parent)).child_request
    dispatcher = Dispatcher(
        cast(
            Worker,
            SimpleNamespace(
                workspace=workspace,
                executions=executions,
                calls=Transport(lambda *_: pb.ChildCallResult(), workspace=workspace),
            ),
        )
    )
    target = cast(
        Target,
        SimpleNamespace(
            installation_id="different",
            binding=SimpleNamespace(job_descriptor_id="different"),
            declaration={},
        ),
    )
    dispatcher._check_ancestors(
        "owner", cast(AttemptRecord, SimpleNamespace(request_id=parents[-2])), target, b"{}"
    )
    with pytest.raises(WorkspaceRefusal, match="depth"):
        dispatcher._check_ancestors(
            "owner", cast(AttemptRecord, SimpleNamespace(request_id=parents[-1])), target, b"{}"
        )


def test_canceling_root_also_abandons_failed_descendants(tmp_path: Path) -> None:
    from dataclasses import replace

    from cozy_runtime.internal.worker.control import InMemoryControlHost
    from cozy_runtime.internal.worker.workspace_calls import Calls
    from test_device_lanes import _config
    from test_machine_calls import request, running
    from test_machine_execution import complete, offer

    worker = Worker(
        replace(_config(tmp_path / "home"), record_owner_public_key=signed_claims.PUBLIC_KEY),
        WorkerOptions(
            **signed_claims.IDENTITY, root=tmp_path / "worker", tensorfs_root=tmp_path / "store"
        ),
        InMemoryControlHost(),
    )
    claim = signed_claims.claim()
    try:
        worker.accept_claim(claim, lambda frame: None)
        executions = worker.executions
        assert executions is not None and worker.workspace is not None
        calls = Calls(worker.workspace)
        executions.submit(
            "owner",
            "root",
            b"c" * 32,
            offer(),
            expected_execution_workspace_id=executions.workspace_id,
        )
        call = calls.accept("owner", request(running(executions, "root")))
        executions.submit(
            "owner",
            "child",
            b"c" * 32,
            offer(call.child_request),
            expected_execution_workspace_id=executions.workspace_id,
        )
        complete(executions, call.child_request, status=pb.OUTCOME_STATUS_FAILED)
        complete(executions, "root", status=pb.OUTCOME_STATUS_FAILED)
        worker.control_execution(claim, "root", "cancel-root", 1, "cancel")
        deadline = time.monotonic() + 60
        while executions.retention_required("owner"):  # each settles on its own unit
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        worker.shutdown()
