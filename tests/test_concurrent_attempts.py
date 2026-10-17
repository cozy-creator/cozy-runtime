"""Calls side by side in one real executor process (cr-202 U3), CUDA hidden.

Three calls meet in one decode-phase scope at once and take their compute stages one at a
time; their requests are answered out of order and each reply, frame and result stays its
own call's. A cancel ends its own call only, whenever it arrives.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.author import Model, uses_components
from cozy_runtime.author._model import STAGES
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover_installed
from cozy_runtime.internal.seam import RESULT_DOCUMENT, Channel
from test_executor_fork import _accept, _environment, _listener

APP = """
import threading
import time

import msgspec
from cozy_runtime.author import App, Context, ImageAsset, Model, Outputs, Telemetry, uses_components
from cozy_runtime.author._services import ImageFrame

app = App()
TOGETHER = threading.Barrier(3, timeout=60)
LOCK = threading.Lock()
COMPUTING = [0, 0]  # now, most at once


class Decoder(Model[object], phase="decode"):
    @uses_components("lm")
    def decode(self, out: Outputs) -> ImageAsset:
        TOGETHER.wait()  # every call is inside this one scope at once
        image = out.save_image(ImageFrame(1, 1, bytes(3)))
        out.publish("image", image)
        return image


class Denoiser(Model[object]):
    @uses_components("dit")
    def denoise(self) -> None:
        with LOCK:
            COMPUTING[0] += 1
            COMPUTING[1] = max(COMPUTING)
        time.sleep(0.2)
        with LOCK:
            COMPUTING[0] -= 1


DECODER, DENOISER = Decoder.for_test(), Denoiser.for_test()


class Request(msgspec.Struct):
    n: int
    mode: str = "share"


class Result(msgspec.Struct):
    n: int
    computing: int
    image: ImageAsset


@app.entrypoint(accelerator=False)
def generate(ctx: Context, payload: Request, out: Outputs, tel: Telemetry) -> Result:
    tel.progress(0.5, stage=f"call-{payload.n}")
    if payload.mode == "hold":
        while True:
            time.sleep(0.01)
            tel.progress(0.5, stage=f"call-{payload.n}")
    if payload.mode == "plain":
        return Result(payload.n, 0, out.save_image(ImageFrame(1, 1, bytes(3))))
    image = DECODER.decode(out)
    DENOISER.denoise()
    return Result(payload.n, COMPUTING[1], image)
"""


class Peer:
    """TensorD's side of one executor's seam: every frame kept by kind, and the executor's
    Publish requests answered only once `hold` of them arrived, last first."""

    def __init__(self, channel: Channel, hold: int) -> None:
        self.channel = channel
        self.hold = hold
        self.frames: list[dict[str, Any]] = []
        self.asked: list[dict[str, Any]] = []
        self.answered: list[str] = []
        self.changed = threading.Condition()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        while (frame := self.channel.recv()) is not None:
            with self.changed:
                self.frames.append(frame)
                if frame.get("event") == "request":
                    self.asked.append(frame)
                    if len(self.asked) == self.hold:
                        for request in reversed(self.asked):
                            self.answered.append(request["request_id"])
                            answer = {"ok": True, "sequence": len(self.answered)}
                            self.channel.send({"event": "answer", "seq": request["seq"], **answer})
                self.changed.notify_all()

    def until(self, found: Callable[[], Any], seconds: float = 120) -> Any:
        with self.changed:
            assert self.changed.wait_for(found, seconds), self.frames[-5:]
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

    def progress(self, request_id: str, kind: str = "progress") -> list[dict[str, Any]]:
        with self.changed:
            return [
                f
                for f in self.frames
                if f.get("event") == "progress"
                and f.get("kind") == kind
                and f["request_id"] == request_id
            ]


@pytest.fixture
def executor(tmp_path: Path) -> Iterator[tuple[Peer, Path]]:
    (tmp_path / "callsapp.py").write_text(APP)
    sys.path.insert(0, str(tmp_path))
    try:
        interface = package_interface.build(discover_installed("callsapp:app"))
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("callsapp", None)
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
        env=_environment(tmp_path),
    )
    try:
        channel, _ = _accept(listener)
        peer = Peer(channel, hold=3)
        binding = {
            "application": "callsapp:app",
            "package_interface": str(tmp_path / "interface.json"),
        }
        assert peer.call({"cmd": "start", "devices": "", **binding})["ok"]
        load = {"cmd": "load", "construction": "c", "devices": "", "binding": binding}
        assert peer.call(load)["ok"]
        assert peer.call({"cmd": "activate", "construction": "c"})["ok"]
        yield peer, root
    finally:
        process.kill()
        process.wait()


def _prepare(peer: Peer, root: Path, request_id: str, n: int, mode: str) -> None:
    (root / request_id).mkdir()
    payload = {"n": n, "mode": mode}
    peer.channel.send(
        {"cmd": "prepare_request", "request_id": request_id, **_CALL, "payload": payload}
    )


def _invoke(peer: Peer, root: Path, request_id: str) -> None:
    spool = str(root / request_id)
    peer.channel.send({"cmd": "invoke", "request_id": request_id, **_CALL, "spool": spool})


_CALL = {"entrypoint": "generate", "deadline_s": 120}


def _outcome(peer: Peer, request_id: str) -> dict[str, Any]:
    reply = peer.reply("invoke", request_id)
    assert reply["ok"] and not reply["poisoned"], reply
    return dict(reply["outcome"])


def test_calls_share_a_decode_stage_and_take_compute_stages_one_at_a_time(
    executor: tuple[Peer, Path],
) -> None:
    peer, root = executor
    calls = {"a": 1, "b": 2, "c": 3}
    for request_id, n in calls.items():
        _prepare(peer, root, request_id, n, "share")
    for request_id in calls:
        assert peer.reply("prepare_request", request_id)["ok"]
    for request_id in calls:
        _invoke(peer, root, request_id)
    for request_id, n in calls.items():
        assert _outcome(peer, request_id)["terminal"] == "succeeded"
        result = json.loads((root / request_id / RESULT_DOCUMENT).read_bytes())
        # The three met in the decode stage (else the barrier breaks) and denoised alone.
        assert (result["n"], result["computing"]) == (n, 1), result
        assert [f["stage"] for f in peer.progress(request_id)] == [f"call-{n}"]
    # Each publish named its call, and its answer, sent last first, reached it.
    assert sorted(peer.answered) == sorted(calls) and len(peer.asked) == 3
    assert peer.answered == [request["request_id"] for request in reversed(peer.asked)]


def test_a_cancel_ends_its_own_call_whenever_it_arrives(executor: tuple[Peer, Path]) -> None:
    peer, root = executor
    peer.channel.send({"cmd": "cancel", "request_id": "early"})  # before the call exists
    _prepare(peer, root, "early", 1, "plain")
    assert peer.reply("prepare_request", "early")["ok"]
    _invoke(peer, root, "early")
    assert _outcome(peer, "early")["terminal"] == "canceled"

    _prepare(peer, root, "held", 2, "hold")
    _prepare(peer, root, "beside", 3, "plain")
    assert peer.reply("prepare_request", "held")["ok"]
    assert peer.reply("prepare_request", "beside")["ok"]
    _invoke(peer, root, "held")
    peer.until(lambda: peer.progress("held"))
    _invoke(peer, root, "beside")
    assert _outcome(peer, "beside")["terminal"] == "succeeded"  # while `held` runs
    # Measured whole: the worker `held` runs on is the executor's, not a thread of `beside`.
    activity = peer.progress("beside", "execution_activity")[-1]
    assert activity["finished"] and activity["known"], activity
    assert not any(f.get("reply") == "invoke" and f["request_id"] == "held" for f in peer.frames)
    peer.channel.send({"cmd": "cancel", "request_id": "held"})
    assert _outcome(peer, "held")["terminal"] == "canceled"
    assert not any(f.get("reply") == "cancel" for f in peer.frames)
    # A process-wide command is served once no call is in flight.
    assert peer.call({"cmd": "probe"})["ok"]


class Encoder(Model[object]):
    @uses_components("dit")
    def encode(self, entered: threading.Event, leave: threading.Event) -> None:
        entered.set()
        leave.wait(30)


class Decoder(Model[object], phase="decode"):
    @uses_components("lm")
    def decode(self, entered: threading.Event, leave: threading.Event) -> None:
        entered.set()
        leave.wait(30)


def _thread(call: Callable[[threading.Event, threading.Event], None]) -> tuple[Any, ...]:
    entered, leave = threading.Event(), threading.Event()
    thread = threading.Thread(target=call, args=(entered, leave), daemon=True)
    thread.start()
    return thread, entered, leave


def test_a_waiting_compute_stage_takes_its_turn_before_later_decode_joins() -> None:
    decoder, encoder = Decoder.for_test(), Encoder.for_test()
    first, first_in, first_out = _thread(decoder.decode)
    assert first_in.wait(10)
    joined, joined_in, joined_out = _thread(decoder.decode)
    assert joined_in.wait(10)  # shared: the second decode joins the open stage
    compute, compute_in, compute_out = _thread(encoder.encode)
    while not STAGES._queue:
        time.sleep(0.01)
    late, late_in, late_out = _thread(decoder.decode)
    time.sleep(0.2)
    assert not compute_in.is_set() and not late_in.is_set()  # it queues behind the compute
    first_out.set()
    joined_out.set()
    assert compute_in.wait(10)
    assert not late_in.wait(0.2)
    compute_out.set()
    assert late_in.wait(10)
    late_out.set()
    for thread in (first, joined, compute, late):
        thread.join(10)
    assert decoder._cozy_active is None and encoder._cozy_active is None
    assert STAGES._open is None and not STAGES._holders
