"""A rented ingest of a gated source presents the owner's credential, and never keeps it.

Real Worker, real delegated executor, real native source child and TensorFS store, driven
through the machine-execution RPCs a rented run uses. The only stand-in is a Civitai origin
that answers 401 unless the request carries the owner's bearer token.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest

from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR
from test_machine_partial_work import Machine, machine

TOKEN = "fake-civitai-token-5f0c9a"
BODY = bytes(range(256)) * 4096
SCRIPT = """import asyncio, json, os, sys
from pathlib import Path
import msgspec
from cozy_runtime.author import App, Context
from cozy_runtime.author.sources import download_civitai
app = App()
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    value: int

@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    while not Path(%r).exists():
        await asyncio.sleep(0.02)
    artifact = await download_civitai(777)
    # Everything the package process can see of the call: it never holds the bearer.
    seen = {"environ": dict(os.environ), "argv": sys.argv, "artifact": repr(artifact)}
    Path(%r).write_text(json.dumps(seen))
    return Result(1)
"""


class Civitai:
    """A gated model version: every route answers 401 without `Bearer TOKEN`."""

    def __init__(self) -> None:
        self.authorizations: list[str] = []
        listing = json.dumps(
            {
                "id": 777,
                "files": [
                    {
                        "id": 92696,
                        "name": "model.safetensors",
                        "primary": True,
                        "hashes": {"SHA256": hashlib.sha256(BODY).hexdigest().upper()},
                    }
                ],
            }
        ).encode()
        origin = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                pass

            def do_GET(self) -> None:
                authorization = self.headers.get("Authorization", "")
                origin.authorizations.append(authorization)
                if authorization != "Bearer " + TOKEN:
                    self.send_response(401)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if self.path == "/api/v1/model-versions/777":
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(listing)))
                    self.end_headers()
                    self.wfile.write(listing)
                    return
                assert self.path == "/api/download/models/777?fileId=92696", self.path
                start, end = 0, len(BODY) - 1
                if spec := self.headers.get("Range"):
                    first, _, last = spec.removeprefix("bytes=").partition("-")
                    start, end = int(first), min(int(last or end), end)
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(BODY)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                self.wfile.write(BODY[start : end + 1])

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@contextmanager
def gated_pod(monkeypatch: pytest.MonkeyPatch, gate: Path) -> Iterator[tuple[Machine, Civitai]]:
    origin = Civitai()
    try:
        with machine(monkeypatch, SCRIPT % (str(gate), str(gate.with_name("seen.json")))) as pod:
            assert pod.worker.source_calls is not None
            pod.worker.source_calls.endpoints = {
                "civitai": f"http://127.0.0.1:{origin.server.server_port}"
            }
            pod.start(())
            yield pod, origin
    finally:
        origin.close()


def credential(value: str = TOKEN) -> tuple[pb.SourceCredential, ...]:
    return (
        pb.SourceCredential(
            provider=pb.NATIVE_SOURCE_OPERATION_CIVITAI, credential="bearer " + value
        ),
    )


def awaiting(pod: Machine, request: str) -> list[int]:
    query = pb.MachineExecutionQuery(
        claim=pod.claim,
        request_id=request,
        expected_execution_workspace_id=pod.worker.executions.workspace_id,  # type: ignore[union-attr]
    )
    return list(pod.client.GetMachineExecution(query).awaiting_source_credentials)


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_gated_download_presents_the_submitted_credential_and_keeps_it_nowhere(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    caplog.set_level(logging.DEBUG)
    gate = tmp_path / "gate"
    gate.touch()
    with gated_pod(monkeypatch, gate) as (pod, origin):
        # Without the owner's credential the provider refuses, as it did before this fix.
        pod.submit("anonymous")
        assert pod.outcome("anonymous")["status"] == pb.OUTCOME_STATUS_FAILED
        assert origin.authorizations and not any(origin.authorizations)

        origin.authorizations.clear()
        pod.submit("gated", credential())
        body = pod.outcome("gated")
        assert body["status"] == pb.OUTCOME_STATUS_SUCCEEDED, body
        assert origin.authorizations and set(origin.authorizations) == {"Bearer " + TOKEN}

        events = pod.rows("SELECT body FROM execution_events")
        providers = pod.rows("SELECT request,source_providers FROM executions")
        assert {row["request"]: row["source_providers"] for row in providers} == {
            "anonymous": "",
            "gated": "civitai",
        }
        # Only the worker and its native source child hold the token: the package process
        # saw none in its environment, argv or call result, and it is in no journal row,
        # event, file under the machine's roots (the executor's scope included), log line
        # or process output.
        seen = (tmp_path / "seen.json").read_text()
        assert json.loads(seen)["artifact"] and TOKEN not in seen
        secret = TOKEN.encode()
        assert not any(secret in cast(bytes, row["body"]) for row in events)
        leaked = [p for p in pod.root.rglob("*") if p.is_file() and secret in p.read_bytes()]
        assert not leaked, f"the credential was written to {leaked}"
    out, err = capfd.readouterr()
    assert TOKEN not in caplog.text + out + err


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_a_download_is_a_call_of_its_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A settled model download is recorded on its root like any call: what it fetched, how it
    ended, and its stage's time and bytes."""
    gate = tmp_path / "gate"
    gate.touch()
    with gated_pod(monkeypatch, gate) as (pod, _origin):
        pod.submit("anonymous")
        pod.outcome("anonymous")
        pod.submit("gated", credential())
        pod.outcome("gated")
        rows = pod.rows("SELECT request,body,at_ms FROM execution_events WHERE kind='call'")
    calls = {row["request"]: json.loads(cast(bytes, row["body"])) for row in rows}
    assert set(calls) == {"anonymous", "gated"}, rows
    for root, call in calls.items():
        assert call["parent"] == root and call["index"] == 0
        assert call["module"] == "cozy_runtime.author.sources"
        assert call["export"] == "download_civitai"
        assert call["label"] == "Download Civitai model version 777"
        assert call["called_unix_ms"] > 0
    refused, fetched = calls["anonymous"], calls["gated"]
    assert refused["status"] == "failed" and refused["error"], refused
    assert fetched["status"] == "succeeded" and fetched["error"] == "", fetched
    (download,) = fetched["stages"].values()
    assert download["bytes"] == len(BODY) and download["count"] == 1
    assert fetched["called_unix_ms"] <= download["started_unix_ms"] <= download["ended_unix_ms"]
    assert download["total_ms"] > 0


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_a_credential_lost_at_restart_holds_the_call_until_the_owner_supplies_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = tmp_path / "gate"
    with gated_pod(monkeypatch, gate) as (pod, origin):
        submission = pod.submit("gated", credential())
        assert awaiting(pod, "gated") == []
        # Lose the volatile credential state while retaining the durable provider
        # declaration. A real restart rebuilds Sources and SourceCalls together;
        # replacing just Sources would strand the service's bound status callbacks.
        with pod.worker.machine_sources.lock:
            pod.worker.machine_sources.credentials.clear()
        assert awaiting(pod, "gated") == [pb.NATIVE_SOURCE_OPERATION_CIVITAI]

        gate.touch()
        calls = pod.worker.machine_calls
        assert calls is not None
        pod.wait("the source call to be pending", lambda: bool(calls.requests), 120)
        for _ in range(50):  # its unit waits: the call must not reach the provider anonymously
            assert pod.state("gated") == "running" and not origin.authorizations
            threading.Event().wait(0.02)

        # The owner's reattach resubmits the identical submission with the credential.
        pod.client.SubmitMachineExecution(submission)
        body = pod.outcome("gated")
        assert body["status"] == pb.OUTCOME_STATUS_SUCCEEDED, body
        assert set(origin.authorizations) == {"Bearer " + TOKEN}
        assert awaiting(pod, "gated") == []
