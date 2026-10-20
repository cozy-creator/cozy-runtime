"""A call outlives its bearer: TensorD forwards each signer's newer bearer for an endpoint as an
unanswered `bearer` frame, and every TensorFS request the call makes after it carries that one.

The endpoint is a bare HTTP gate in front of nothing: it records each request's bearer, refuses
a revoked one with 401 and answers any other with 404. Whether the upload then lands is the
Hub's to prove (Creator's TensorD end-to-end test); what is proved here is which bearer every
request of a real executor process and of TensorFS's own client presented.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import tensorfs

from cozy_runtime import weights_store
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover_installed
from cozy_runtime.internal.seam import Channel
from test_executor_fork import _accept, _environment, _listener

APP = """
import time
from pathlib import Path

import msgspec
from cozy_runtime.author import App, Context, Telemetry, WeightsOutput
from cozy_runtime.weights_store import Derivation

app = App()


class Request(msgspec.Struct):
    release: str


class Result(msgspec.Struct):
    checkpoint: str


@app.entrypoint(accelerator=False, weights=(WeightsOutput("model", max_new_bytes=4096),))
def derive(payload: Request, ctx: Context, tel: Telemetry) -> Result:
    while not Path(payload.release).exists():  # the long part, while TensorD refreshes
        tel.progress(0.5, stage="held")
        time.sleep(0.01)
    with ctx.output("model", Derivation(None, order=(("model", "w"),))) as out:
        out.put("model", "w", "U8", [4], b"data")
        artifact = ctx.adopt_model(out)
    return Result(artifact.checkpoint)
"""


class Gate:
    """An HTTP endpoint that keeps every request's bearer and answers through `answer`."""

    def __init__(self, answer: Callable[[str], int]) -> None:
        self.seen: list[str] = []
        gate = self

        class Handler(BaseHTTPRequestHandler):
            def _any(self) -> None:
                self.rfile.read(int(self.headers.get("content-length") or 0))
                bearer = self.headers.get("authorization", "")
                gate.seen.append(bearer)
                body = json.dumps({"error": "gate"}).encode()
                self.send_response(answer(bearer))
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST = do_PUT = do_HEAD = do_PATCH = _any

            def log_message(self, *_: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class Peer:
    """TensorD's side of the seam: every frame kept, every executor request answered ok."""

    def __init__(self, channel: Channel) -> None:
        self.channel = channel
        self.frames: list[dict[str, Any]] = []
        self.changed = threading.Condition()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        while (frame := self.channel.recv()) is not None:
            if frame.get("event") == "request":
                self.channel.send({"event": "answer", "seq": frame["seq"], "ok": True})
            with self.changed:
                self.frames.append(frame)
                self.changed.notify_all()

    def until(self, found: Callable[[], Any]) -> Any:
        with self.changed:
            assert self.changed.wait_for(found, 120), self.frames[-5:]
            return found()

    def reply(self, cmd: str, request_id: str = "") -> dict[str, Any]:
        def found() -> dict[str, Any] | None:
            return next(
                (
                    f
                    for f in self.frames
                    if f.get("reply") == cmd and f.get("request_id", "") == request_id
                ),
                None,
            )

        return dict(self.until(found))

    def call(self, frame: dict[str, Any]) -> dict[str, Any]:
        self.channel.send(frame)
        return self.reply(frame["cmd"], frame.get("request_id", ""))


@pytest.fixture
def executor(tmp_path: Path) -> Iterator[Peer]:
    (tmp_path / "bearerapp.py").write_text(APP)
    sys.path.insert(0, str(tmp_path))
    try:
        interface = package_interface.build(discover_installed("bearerapp:app"))
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("bearerapp", None)
    (tmp_path / "interface.json").write_text(json.dumps(interface))
    root = tmp_path / "root"
    root.mkdir()
    listener = _listener(tmp_path / "e.sock")
    process = subprocess.Popen(
        [
            sys.executable,
            *("-m", "cozy_runtime.internal.executor"),
            *("--socket", str(tmp_path / "e.sock"), "--root", str(root)),
        ],
        env={**_environment(tmp_path), "TENSORFS_HOME": str(tmp_path / "tensorfs")},
    )
    try:
        channel, _ = _accept(listener)
        peer = Peer(channel)
        binding = {
            "application": "bearerapp:app",
            "package_interface": str(tmp_path / "interface.json"),
        }
        assert peer.call({"cmd": "start", "devices": "", **binding})["ok"]
        load = {"cmd": "load", "construction": "c", "devices": "", "binding": binding}
        assert peer.call(load)["ok"]
        assert peer.call({"cmd": "activate", "construction": "c"})["ok"]
        yield peer
    finally:
        process.kill()
        process.wait()


def test_a_call_uploads_under_the_bearer_tensord_sent_while_it_ran(
    executor: Peer, tmp_path: Path
) -> None:
    gate = Gate(lambda bearer: 401 if bearer != "Bearer new" else 404)
    peer, release = executor, tmp_path / "release"
    try:
        frame = {"cmd": "prepare_request", "request_id": "r", "entrypoint": "derive"}
        assert peer.call({**frame, "payload": {"release": str(release)}})["ok"]
        (tmp_path / "r").mkdir()
        peer.channel.send(
            {
                **frame,
                "cmd": "invoke",
                "spool": str(tmp_path / "r"),
                "deadline_s": 120,
                "weights": {"model": 4096},
                "destination": "alice/model",
                "actor": "alice",
                "tensorfs": {"endpoint": gate.url, "token": "old"},
            }
        )
        peer.until(lambda: any(f.get("stage") == "held" for f in peer.frames))

        def bearer(actor: str, endpoint: str, token: str, expires: int) -> None:
            options = {"endpoint": endpoint, "token": token, "expires_unix_ms": expires}
            peer.channel.send({"cmd": "bearer", "actor": actor, "tensorfs": options})

        bearer("alice", gate.url, "new", 2_000)
        bearer("alice", gate.url, "stale", 1_000)  # expires sooner than the one held: kept out
        bearer("bob", gate.url, "bobs", 0)  # another signer's
        bearer("alice", "http://127.0.0.1:9", "elsewhere", 0)  # another endpoint's
        release.touch()
        reply = peer.reply("invoke", "r")
    finally:
        gate.close()
    assert reply["ok"], reply
    # The gate is no Hub, so the upload ends there, but only under the newest bearer.
    assert reply["outcome"]["code"] == "upload_failed", reply
    assert gate.seen and set(gate.seen) == {"Bearer new"}, gate.seen
    assert not any(f.get("reply") == "bearer" for f in peer.frames)


def test_a_401_for_a_bearer_replaced_in_flight_is_asked_once_more(tmp_path: Path) -> None:
    bearers = weights_store.Bearers()
    gate: Gate

    def answer(bearer: str) -> int:
        if bearer == "Bearer old":  # TensorD refreshed while this request was in flight
            bearers.note("alice", gate.url, "new", 0)
            return 401
        return 404

    gate = Gate(answer)
    provider = weights_store.Bearer(bearers, "alice", gate.url, "old")
    try:
        with pytest.raises(tensorfs.Error) as refused:
            tensorfs.open("alice/model", endpoint=gate.url, token=provider, cache_dir=tmp_path)
    finally:
        gate.close()
    assert gate.seen[:2] == ["Bearer old", "Bearer new"], gate.seen
    assert not isinstance(refused.value, tensorfs.DeniedError), refused.value
