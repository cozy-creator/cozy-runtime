"""Real Hub/AuthKit fixture: one Runtime, concurrent scoped uploads, and a denied destination.

The endpoint_live fixture supplies credentials on stdin. Run only against its disposable Hub.
"""

from __future__ import annotations

import base64
import os
import socket
import ssl
import subprocess
import sys
from pathlib import Path

import msgspec
import tensorfs

from cozy_runtime import weights_store
from cozy_runtime.author import ModelArtifact
from cozy_runtime.author._executor_requests import Answer, CheckpointItem, Request, respond
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor_commands import ResolvedModels
from cozy_runtime.internal.seam import RESULT_DOCUMENT, Channel, Frame


class Scope(msgspec.Struct):
    key: str
    capability: str
    upload_repo: str


class Denied(Scope):
    attempt_repo: str


class Fixture(msgspec.Struct):
    origin: str
    endpoint: str
    token_endpoint: str
    model: str
    tree: str
    ca: str
    cache_dir: str
    owner_token: str
    runs: list[Scope]
    denied: Denied
    component: str = "model"
    tensor: str = "w"
    variant: str = "int8"


APP = """
import threading
import msgspec
from typing import Any
from cozy_runtime.author import App, Context, ModelArtifact, Weights, WeightsOutput
from cozy_runtime.weights_store import Derivation

app = App()
barrier = threading.Barrier(2)
class Input(msgspec.Struct):
    value: int
    synchronize: bool = False
@app.entrypoint(accelerator=False, weights=(WeightsOutput("model", max_new_bytes=4096),))
def derive(payload: Input, ctx: Context, source: Weights[Any]) -> ModelArtifact:
    source.read_into("model", "w", "", 0, bytearray(16))
    if payload.synchronize:
        barrier.wait(timeout=30)
    with ctx.output("model", Derivation(ctx.tensorfs_source(source))) as output:
        output.put("model", "derived", "U8", [1], bytes([payload.value]))
        return ctx.adopt_model(output)
"""


class ExecutorProcess:
    def __init__(self, root: Path, store: Path, project: Path | None = None) -> None:
        root.mkdir(parents=True)
        path = root / "seam.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        listener.listen(1)
        listener.settimeout(30)
        env = {
            **os.environ,
            "TENSORFS_HOME": str(store),
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        bootstrap = "import runpy,sys; "
        if project is not None:
            bootstrap += f"sys.path.insert(0, {str(project)!r}); "
        bootstrap += "runpy.run_module('cozy_runtime.internal.executor', run_name='__main__')"
        self.log = (root / "stderr.log").open("wb")
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-c", bootstrap, "--socket", str(path), "--root", str(root)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=self.log,
        )
        try:
            connection, _ = listener.accept()
            self.channel = Channel(connection)
        except BaseException:
            self.process.kill()
            self.process.wait()
            self.log.close()
            raise
        finally:
            listener.close()
        self.items: dict[str, list[CheckpointItem]] = {}
        assert self.call({"cmd": "hello"})["torch_loaded"] is False

    def receive(self) -> Frame:
        while True:
            frame = self.channel.recv(timeout=120)
            assert frame is not None, "Runtime ended before its reply"
            if frame.get("event") == "request":
                request_id = str(frame.get("request_id", ""))

                def accept(request: Request, request_id: str = request_id) -> Answer:
                    assert isinstance(request, CheckpointItem), type(request).__name__
                    self.items.setdefault(request_id, []).append(request)
                    return Answer(ok=True)

                answer, handoff = respond(frame, accept)
                assert handoff is None
                self.channel.send(answer)
            elif "event" not in frame:
                return frame

    def call(self, frame: Frame) -> Frame:
        self.channel.send(frame)
        return self.receive()

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                assert self.call({"cmd": "shutdown"})["ok"]
                assert self.process.wait(timeout=30) == 0
            assert not Path(f"/proc/{self.process.pid}").exists()
        finally:
            self.channel.close()
            self.log.close()
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()


def hub(fixture: Fixture, scope: Scope, token: str | None = None) -> dict[str, object]:
    der = ssl.PEM_cert_to_DER_cert(Path(fixture.ca).read_text())
    value = msgspec.to_builtins(
        weights_store.Hub(
            endpoint=fixture.endpoint,
            capability=scope.capability,
            key=scope.key,
            token_endpoint=fixture.token_endpoint,
            ca_der=base64.b64encode(der).decode(),
            access_token=token,
        )
    )
    assert isinstance(value, dict)
    return value


def metadata(fixture: Fixture, scope: Scope, root: Path, store: Path) -> ResolvedModels:
    helper = ExecutorProcess(root, store)
    try:
        reply = helper.call(
            {
                "cmd": "resolve_models",
                "request_id": root.name,
                "store": str(store),
                "hub": hub(fixture, scope),
                "models": [
                    {
                        "slot": "source",
                        "repo_id": fixture.model,
                        "revision": fixture.tree,
                        "variant": fixture.variant,
                    }
                ],
            }
        )
        assert reply.get("ok"), {key: reply.get(key) for key in ("code", "detail")}
        assert helper.call({"cmd": "hello"})["torch_loaded"] is False
        return msgspec.convert(reply, ResolvedModels)
    finally:
        helper.close()


def run() -> None:
    fixture = msgspec.json.decode(sys.stdin.buffer.read(), type=Fixture)
    assert len(fixture.runs) == 2
    assert (fixture.component, fixture.tensor) == ("model", "w")
    root = Path(fixture.cache_dir) / "runtime-proof"
    store = root / "store"
    first = metadata(fixture, fixture.runs[0], root / "cold-helper", store)
    assert first.get("access_token") is not None, "cold metadata did not trade"
    second = metadata(fixture, fixture.runs[1], root / "cached-helper", store)
    assert second.get("access_token") is None, "cached metadata traded"
    header = "sha256:" + first["resolved"][0]["header"]["sha256"]
    assert tensorfs.Store(store).checkpoint(header).missing(), (
        "metadata helper fetched weight bytes"
    )
    project = root / "package"
    project.mkdir()
    (project / "package.toml").write_text('[application]\nobject="runtime_publication:app"\n')
    (project / "runtime_publication.py").write_text(APP)
    interface = root / "interface.json"
    interface.write_bytes(
        package_interface.canonical_bytes(package_interface.build(discover(project)))
    )
    executor = ExecutorProcess(root / "executor", store, project)

    def prepare(request_id: str, value: int, synchronize: bool) -> None:
        assert executor.call(
            {
                "cmd": "prepare_request",
                "request_id": request_id,
                "construction": "models",
                "entrypoint": "derive",
                "payload": {"value": value, "synchronize": synchronize},
            }
        ).get("ok")

    def invoke(request_id: str, scope: Scope, token: str | None, destination: str) -> Frame:
        spool = root / request_id
        spool.mkdir()
        return {
            "cmd": "invoke",
            "request_id": request_id,
            "construction": "models",
            "entrypoint": "derive",
            "spool": str(spool),
            "deadline_s": 120,
            "sources": {"source": header},
            "weights": {"model": 4096},
            "actor": request_id,
            "hub": hub(fixture, scope, token),
            "destination": destination,
        }

    try:
        binding = {"application": "runtime_publication:app", "package_interface": str(interface)}
        assert executor.call({"cmd": "start", **binding, "devices": ""}).get("ok")
        for name, scope, resolved in zip(("a", "b"), fixture.runs, (first, second), strict=True):
            assert executor.call(
                {
                    "cmd": "load",
                    "construction": "models",
                    "request_id": name,
                    "binding": binding,
                    "hub": hub(fixture, scope, resolved.get("access_token")),
                }
            ).get("ok")
            prepare(name, 1 if name == "a" else 2, True)
        assert executor.call({"cmd": "activate", "construction": "models"}).get("ok")
        for name, scope, resolved in zip(("a", "b"), fixture.runs, (first, second), strict=True):
            executor.channel.send(
                invoke(name, scope, resolved.get("access_token"), scope.upload_repo)
            )
        replies = [executor.receive(), executor.receive()]
        for reply in replies:
            outcome = reply.get("outcome")
            assert isinstance(outcome, dict) and outcome.get("terminal") == "succeeded", outcome
        for name, scope in zip(("a", "b"), fixture.runs, strict=True):
            local, remote = executor.items[name]
            assert local.repo_id == weights_store.DERIVED and remote.repo_id == scope.upload_repo
            artifact = msgspec.json.decode(
                (root / name / RESULT_DOCUMENT).read_bytes(), type=ModelArtifact
            )
            assert remote.revision != artifact.checkpoint
            ck = tensorfs.open(
                remote.repo_id,
                revision=remote.revision,
                token=fixture.owner_token,
                endpoint=fixture.endpoint,
                ca_der=ssl.PEM_cert_to_DER_cert(Path(fixture.ca).read_text()),
                cache_dir=root / f"readback-{name}",
            )
            assert ck.components["model"].tensors.read("derived").tobytes() == bytes(
                [1 if name == "a" else 2]
            )
        warm = metadata(fixture, fixture.runs[0], root / "warm-helper", store)
        assert warm.get("access_token") is None
        prepare("warm", 3, False)
        warm_reply = executor.call(invoke("warm", fixture.runs[0], None, ""))
        assert warm_reply["outcome"]["terminal"] == "succeeded", (
            "a warm run tried to redeem the used capability"
        )
        prepare("denied", 4, False)
        denied = executor.call(invoke("denied", fixture.denied, None, fixture.denied.attempt_repo))
        assert (
            denied["outcome"]["terminal"] == "failed"
            and denied["outcome"]["code"] == "upload_failed"
        )
        (kept,) = executor.items["denied"]
        assert kept.repo_id == weights_store.DERIVED
        local = tensorfs.Store(store).checkpoint(kept.revision)
        assert local.components["model"].tensors.read("derived").tobytes() == b"\x04"
        assert not local.missing()
        assert executor.call({"cmd": "hello"})["torch_loaded"] is False
    finally:
        executor.close()
    print("Runtime/AuthKit: helper token handoff and one executor concurrent scoped uploads passed")
    print("Warm request traded zero times; denied destination retained its local checkpoint")


if __name__ == "__main__":
    run()
