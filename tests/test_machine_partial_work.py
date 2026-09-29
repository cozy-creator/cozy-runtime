"""A canceled machine execution stops its work and a re-run resumes its progress.

Real Worker, real delegated executor, real native source child and TensorFS store, driven
through the machine-execution RPCs a rented run uses. The download origin is the only
stand-in: it can hold a carrier's body open at the first 64 MiB checkpoint, so the test
sees exactly when the pod stops moving bytes and which bytes a re-run fetches again.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import socket
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import grpc
import pytest

import signed_claims
from cozy_runtime import canonical_json
from cozy_runtime.internal import child_env
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.control import GrpcControlHost
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.internal.worker.stage_progress import SAMPLE_SECONDS
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from local_owner import LocalRecordOwner, LocalRequest
from test_end_to_end import NO_EXECUTOR

SCRIPT = """import msgspec
from cozy_runtime.author import App, Context
from cozy_runtime.author.sources import download_huggingface
app = App()
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    value: int

@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    await download_huggingface(
        'example/model', revision='%s', carriers=('provider/big.safetensors',))
    return Result(1)
""" % ("a" * 40)

BLOCK = 64 << 20  # TensorFS makes a download durable every 64 MiB
CHUNK = bytes(range(256)) * 4096
LENGTH = BLOCK + 8 * len(CHUNK)


class Origin:
    """Serves provider/big.safetensors. While holding, every body stops at the first 64 MiB
    checkpoint and stays open until its client hangs up, so exactly that prefix is durable."""

    def __init__(self) -> None:
        self.holding = True
        self.held = threading.Event()
        self.left = threading.Event()
        self.closed = threading.Event()
        # (start, bytes sent) of each body served after holding ends.
        self.resumed: list[tuple[int, int]] = []
        digest = hashlib.sha256(CHUNK * (LENGTH // len(CHUNK))).hexdigest()
        origin = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                pass

            def do_GET(self) -> None:
                if self.path.startswith("/api/models/"):
                    data = json.dumps(
                        [
                            {
                                "type": "file",
                                "path": "provider/big.safetensors",
                                "size": LENGTH,
                                "lfs": {"oid": digest, "size": LENGTH},
                            }
                        ]
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
                hold = origin.holding
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{LENGTH}")
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                offset = start
                try:
                    while offset <= end:
                        # A split run past the checkpoint never lands while holding, so the
                        # durable prefix is exactly the first 64 MiB.
                        if hold and (offset == BLOCK or start > BLOCK):
                            self.wfile.flush()
                            origin.held.set()
                            self._await_hangup()
                            return
                        if hold and offset + len(CHUNK) == BLOCK:
                            # The durable report arrives after the progress lane's throttle.
                            time.sleep(1.5 * SAMPLE_SECONDS)
                        piece = CHUNK[offset % len(CHUNK) :][: end + 1 - offset]
                        self.wfile.write(piece)
                        offset += len(piece)
                except (BrokenPipeError, ConnectionResetError):
                    origin.left.set()
                finally:
                    if not hold:
                        origin.resumed.append((start, offset - start))

            def _await_hangup(self) -> None:
                connection: socket.socket = self.connection
                while not origin.closed.is_set():
                    readable, _, _ = select.select([connection], [], [], 0.05)
                    if readable and not connection.recv(1, socket.MSG_PEEK):
                        origin.left.set()
                        return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.closed.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def durable(pod: Machine) -> int:
    """The source bytes the first run's download progress reports durable."""
    rows = pod.rows("SELECT body FROM execution_events WHERE request='first' AND kind='progress'")
    positions = [
        canonical_json.decode(cast(bytes, row["body"])).get("payload", {}).get("position", 0)
        for row in rows
    ]
    return max(positions, default=0)


def registry() -> bytes:
    """The fixture registry plus one reviewed profile naming the big carrier."""
    document = json.loads(
        (Path(__file__).parent / "testdata/native_source/registry.json").read_text()
    )
    profile = json.loads(json.dumps(document["source_profiles"][0]))
    profile["name"] = "fixture/big/1"
    profile["components"][0]["source_member"] = "provider/big.safetensors"
    document["source_profiles"].append(profile)
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode()


def wait(what: str, done: Callable[[], bool], alive: Callable[[], bool], seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while not done():
        assert alive(), f"worker stopped while waiting for {what}"
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.02)


class Machine:
    """One real Worker holding one captured local package, driven as a rented pod is."""

    def __init__(self, root: Path, worker: Worker, thread: threading.Thread) -> None:
        self.root, self.worker, self.thread = root, worker, thread
        self.prepared = pb.PreparePackageSetResult()
        self.claim = signed_claims.claim()

    def start(self, exports: tuple[str, ...], *, entrypoint: str = "nested") -> None:
        self.entrypoint = entrypoint
        prepared = self.prepared
        installation = prepared.installed_package.installation_id
        self.captured, self.capture_digest = documents.identity(
            pb.MachineExecutionCapture(
                root_installation_id=installation,
                installed_packages=[prepared.installed_package],
                bindings=[
                    pb.MachineCallableBinding(
                        caller_installation_id=installation,
                        callee_installation_id=installation,
                        module="prepare_nested",
                        export=export,
                        entrypoint=export,
                    )
                    for export in exports
                ],
            )
        )
        worker = self.worker
        if worker.supervision.current is not None:
            worker.supervision.retire_current(worker.supervision.current, "begin lifecycle")
        plans = [
            json.loads(path.read_bytes())
            for path in (self.root / "home/job-plans").glob("*/*.json")
        ]
        self.binding = JobBinding.read(next(row for row in plans if row.get("job") == entrypoint))
        self.thread.start()
        self.wait("the control address", (self.root / "address").exists, 60)
        worker.serve_stream(iter([pb.RecordOwnerFrame(claim=self.claim)]), lambda _: None)
        self.client = rpc.WorkerControlStub(
            grpc.insecure_channel((self.root / "address").read_text().strip())
        )

    def wait(self, what: str, done: Callable[[], bool], seconds: float) -> None:
        wait(what, done, self.thread.is_alive, seconds)

    def submit(
        self,
        request: str,
        credentials: tuple[pb.SourceCredential, ...] = (),
        owner_memo: bool = False,
    ) -> pb.MachineExecutionSubmit:
        generator = LocalRecordOwner(
            LocalRequest(
                entrypoint=self.entrypoint,
                payload={},
                kind="job",
                request_id=request,
                outputs=(),
                job_descriptor_id=self.binding.job_descriptor_id,
                timeout_ms=120_000,
            ),
            {},
            self.root / ("grants-" + request),
            package_installation_id=self.binding.installation_id,
        )
        offered = generator.offer()
        state = generator.desired_state()
        state.job.orchestration = True
        submission = pb.MachineExecutionSubmit(
            claim=self.claim,
            submission_id=request,
            capture_digest=self.capture_digest,
            capture_canonical_bytes=self.captured,
            offer=offered,
            prepared_state=state,
            payload_canonical_bytes=canonical_json.encode({}),
            expected_execution_workspace_id=self.client.GetMachineExecutionWorkspace(
                pb.MachineExecutionWorkspaceQuery(claim=self.claim)
            ).execution_workspace_id,
            source_credentials=credentials,
            owner_memo=owner_memo,
        )
        self.client.SubmitMachineExecution(submission)
        return submission

    def state(self, request: str) -> str:
        return self.worker.execution_status(self.claim, request).state

    def cancel(self, request: str) -> None:
        """Creator's `cozy run cancel` of a machine execution: read, then control."""
        query = pb.MachineExecutionQuery(
            claim=self.claim,
            request_id=request,
            expected_execution_workspace_id=self.client.GetMachineExecutionWorkspace(
                pb.MachineExecutionWorkspaceQuery(claim=self.claim)
            ).execution_workspace_id,
        )
        state = self.client.GetMachineExecution(query)
        self.client.ControlMachineExecution(
            pb.MachineExecutionControl(
                execution=query,
                command_id="cancel-" + request,
                expected_generation=state.generation,
                action=pb.MACHINE_EXECUTION_ACTION_CANCEL,
            )
        )

    def outcome(self, request: str) -> dict[str, object]:
        self.wait(
            request + " to finish",
            lambda: self.state(request) in ("succeeded", "failed", "canceled"),
            120,
        )
        collected = self.worker.collect_execution(self.claim, request)
        return documents.read(collected.outcome_canonical_bytes, pb.AttemptOutcomeBody)

    def rows(self, sql: str) -> list[dict[str, object]]:
        assert self.worker.workspace is not None
        with self.worker.workspace.locked() as db:
            return [dict(row) for row in db.execute(sql)]


@contextmanager
def machine(monkeypatch: pytest.MonkeyPatch, script: str) -> Iterator[Machine]:
    import test_job_preparation_isolation as fixture
    from conftest import image_python

    monkeypatch.setattr(fixture, "SOURCE", script)
    with tempfile.TemporaryDirectory(prefix="cz-partial.") as directory:
        root = Path(directory)
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
        pod = Machine(root, worker, thread)
        try:
            pod.prepared = worker.prepare_local_package(preparation)
            yield pod
        finally:
            worker.request_stop()
            worker.host.stop()
            if thread.ident is not None:
                thread.join(30)


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_canceled_download_stops_and_a_rerun_resumes_its_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin = Origin()
    try:
        with machine(monkeypatch, SCRIPT) as pod:
            assert pod.worker.source_calls is not None
            pod.worker.source_calls.endpoints = {
                "huggingface": f"http://127.0.0.1:{origin.server.server_port}"
            }
            pod.worker.source_calls.native_registry = registry()
            pod.start(())
            pod.submit("first")
            pod.wait("a body to be held at the checkpoint", origin.held.is_set, 120)
            pod.wait("the first 64 MiB reported durable", lambda: durable(pod) >= BLOCK, 60)
            pod.cancel("first")

            # The pod stops moving bytes: the held body is abandoned, not drained, and the
            # native source child is gone.
            pod.wait("the held body to be abandoned", origin.left.is_set, 30)
            assert pod.outcome("first")["status"] == pb.OUTCOME_STATUS_CANCELED
            with pod.worker.source_calls.lock:
                live = [
                    p for p in pod.worker.source_calls.processes.values() if p and p.poll() is None
                ]
            assert not live, "a native source child outlived its canceled parent"

            # An identical re-run fetches exactly the bytes the canceled run left undurable.
            origin.holding = False
            pod.submit("second")
            body = pod.outcome("second")
            assert body["status"] == pb.OUTCOME_STATUS_SUCCEEDED, body
            assert origin.resumed and all(start >= BLOCK for start, _ in origin.resumed), (
                f"re-run fetched {origin.resumed} below the durable 64 MiB prefix"
            )
            assert sum(sent for _, sent in origin.resumed) == LENGTH - BLOCK
            (spent,) = pod.rows("SELECT state FROM native_calls WHERE parent_request='first'")
            assert spent["state"] == "released", "the adopted partial was not spent"
    finally:
        origin.close()


CONVERSION = """import asyncio, json, struct
from pathlib import Path
import msgspec
import tensorfs
from tensorfs.derived import Config, Derivation, Part, Target, Tensor
from cozy_runtime.author import App, Context, ModelArtifact, WeightsOutput, invocable
app = App()
PLAIN = dict(tensorfs.seed_digests())["plain/1"]
ENTERED, GATE, SEEN = %r, %r, %r

@invocable(memoize=True)
async def leaf(ctx: Context, *, n: int) -> ModelArtifact:
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
            done = sorted(list(part) for part in output.completed_parts())
            with open(SEEN, "a") as seen:
                seen.write(json.dumps(done) + "\\n")
            if ["body", "layer.weight", "value"] not in done:
                output.add_part("body", "layer.weight", "value",
                                struct.pack("<512e", *(i/257-1 for i in range(512))))
                output.checkpoint()
            Path(ENTERED).touch()
            while not Path(GATE).exists():
                ctx.raise_if_cancelled()
                await asyncio.sleep(0.02)
            output.add_part("body", "layer.bias", "value", struct.pack("<16e", *range(16)))
            output.add_config("pipeline", b"{}")
        receipt = output.receipt or output.commit()
    return ctx.adopt_model(receipt)
app.job(leaf, weights=(WeightsOutput("model", max_new_bytes=65536),))

class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    value: int

@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    await leaf(n=3)
    return Result(1)
"""


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
@pytest.mark.parametrize("stop", ["cancel", "failure"])
def test_stopped_memoized_conversion_is_adopted_by_an_identical_rerun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stop: str
) -> None:
    entered, gate, seen = tmp_path / "entered", tmp_path / "gate", tmp_path / "seen"
    script = CONVERSION % (str(entered), str(gate), str(seen))
    if stop == "failure":
        script = script.replace(
            "            while not Path(GATE).exists():",
            "            if not Path(GATE).exists():\n"
            "                raise RuntimeError('the conversion failed after committing a part')\n"
            "            while not Path(GATE).exists():",
        )
    with machine(monkeypatch, script) as pod:
        pod.start(("leaf",))
        pod.submit("first")
        pod.wait("the conversion to checkpoint its first part", entered.exists, 120)
        if stop == "cancel":
            pod.cancel("first")
        first = pod.outcome("first")
        assert first["status"] != pb.OUTCOME_STATUS_SUCCEEDED, first

        gate.touch()
        pod.submit("second")
        body = pod.outcome("second")
        assert body["status"] == pb.OUTCOME_STATUS_SUCCEEDED, body
        runs = [json.loads(line) for line in seen.read_text().splitlines()]
        # The identical re-run started from the stopped run's checkpoint: the written part
        # was reused, not recomputed.
        assert runs == [[], [["body", "layer.weight", "value"]]], runs
