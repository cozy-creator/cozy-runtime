"""Calls side by side in one real executor process (cr-202 U3), CUDA hidden.

Three calls are in flight at once and take their component scopes one call at a time; their
requests, made from a helper thread, are answered out of order and each reply, frame and result
stays its own call's. A cancel ends its own call only, whenever it arrives, a process-wide
command waits for the calls before it without stopping the seam, and a handler's SystemExit
ends the process.
"""

from __future__ import annotations

import contextvars
import json
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.author import Model, uses_components
from cozy_runtime.author._context import CURRENT, Context
from cozy_runtime.author._model import WARNINGS
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover_installed
from cozy_runtime.internal.seam import RESULT_DOCUMENT, Channel
from test_executor_fork import _accept, _environment, _listener

APP = """
import sys
import threading
import time

import msgspec
from cozy_runtime.author import App, Context, ImageAsset, Model, Outputs, Telemetry, uses_components
from cozy_runtime.author._services import ImageFrame

app = App()
TOGETHER = threading.Barrier(3, timeout=60)
LOCK = threading.Lock()
SCOPED = [0, 0]  # calls in a component scope now, most at once


def inside() -> None:
    with LOCK:
        SCOPED[0] += 1
        SCOPED[1] = max(SCOPED)
    time.sleep(0.1)
    with LOCK:
        SCOPED[0] -= 1


class Decoder(Model[object], phase="decode"):
    @uses_components("lm")
    def decode(self) -> None:
        inside()


class Denoiser(Model[object]):
    @uses_components("dit")
    def denoise(self) -> None:
        inside()


DECODER, DENOISER = Decoder.for_test(), Denoiser.for_test()


class Request(msgspec.Struct):
    n: int
    mode: str = "share"


class Result(msgspec.Struct):
    n: int
    scoped: int
    image: ImageAsset


@app.entrypoint(accelerator=False)
def generate(ctx: Context, payload: Request, out: Outputs, tel: Telemetry) -> Result:
    tel.progress(0.5, stage=f"call-{payload.n}")
    while payload.mode == "hold":
        time.sleep(0.01)
        tel.progress(0.5, stage=f"call-{payload.n}")
    if payload.mode == "exit":
        sys.exit(3)
    image = out.save_image(ImageFrame(1, 1, bytes(3)))
    if payload.mode == "plain":
        return Result(payload.n, 0, image)
    TOGETHER.wait()  # every call is in flight at once
    helper = threading.Thread(target=out.publish, args=("image", image))
    helper.start()
    helper.join()
    DECODER.decode()
    DENOISER.denoise()
    return Result(payload.n, SCOPED[1], image)
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

    def replied(self, cmd: str, request_id: str = "") -> dict[str, Any] | None:
        with self.changed:
            return next(
                (
                    f
                    for f in self.frames
                    if f.get("reply") == cmd and f.get("request_id", "") == request_id
                ),
                None,
            )

    def reply(self, cmd: str, request_id: str = "") -> dict[str, Any]:
        return dict(self.until(lambda: self.replied(cmd, request_id)))

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


@dataclass
class Running:
    peer: Peer
    root: Path
    process: subprocess.Popen[bytes]

    def prepare(self, request_id: str, n: int, mode: str) -> None:
        (self.root / request_id).mkdir()
        payload = {"n": n, "mode": mode}
        frame = {"cmd": "prepare_request", "request_id": request_id, **_CALL, "payload": payload}
        self.peer.channel.send(frame)
        assert self.peer.reply("prepare_request", request_id)["ok"]

    def invoke(self, request_id: str) -> None:
        spool = str(self.root / request_id)
        self.peer.channel.send({"cmd": "invoke", "request_id": request_id, **_CALL, "spool": spool})

    def outcome(self, request_id: str) -> dict[str, Any]:
        reply = self.peer.reply("invoke", request_id)
        assert reply["ok"] and not reply["poisoned"], reply
        return dict(reply["outcome"])


_CALL = {"entrypoint": "generate", "deadline_s": 120}


@pytest.fixture
def executor(tmp_path: Path) -> Iterator[Running]:
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
        yield Running(peer, root, process)
    finally:
        process.kill()
        process.wait()


def test_calls_run_side_by_side_and_take_component_scopes_one_call_at_a_time(
    executor: Running,
) -> None:
    peer = executor.peer
    calls = {"a": 1, "b": 2, "c": 3}
    for request_id, n in calls.items():
        executor.prepare(request_id, n, "share")
    for request_id in calls:
        executor.invoke(request_id)
    for request_id, n in calls.items():
        assert executor.outcome(request_id)["terminal"] == "succeeded"
        result = json.loads((executor.root / request_id / RESULT_DOCUMENT).read_bytes())
        # The three were in flight at once (else the barrier breaks); one call was in a scope.
        assert (result["n"], result["scoped"]) == (n, 1), result
        assert [f["stage"] for f in peer.progress(request_id)] == [f"call-{n}"]
        # Its own activity, measured whole: no other call's worker reads as its thread.
        activity = peer.progress(request_id, "execution_activity")[-1]
        assert activity["finished"] and activity["known"], activity
    # Each publish, made on a helper thread, named its call; its answer, sent last first,
    # reached it.
    assert sorted(peer.answered) == sorted(calls) and len(peer.asked) == 3
    assert peer.answered == [request["request_id"] for request in reversed(peer.asked)]


def test_a_cancel_ends_its_own_call_whenever_it_arrives(executor: Running) -> None:
    peer, root = executor.peer, executor.root
    peer.channel.send({"cmd": "cancel", "request_id": "early"})  # before the call exists
    executor.prepare("early", 1, "plain")
    executor.invoke("early")
    assert executor.outcome("early")["terminal"] == "canceled"

    executor.prepare("held", 2, "hold")
    executor.prepare("beside", 3, "plain")
    executor.invoke("held")
    peer.until(lambda: peer.progress("held"))
    executor.invoke("beside")
    assert executor.outcome("beside")["terminal"] == "succeeded"  # while `held` runs
    # A process-wide command waits for the call before it, and the seam keeps being read.
    peer.channel.send({"cmd": "probe"})
    peer.channel.send({"cmd": "cancel", "request_id": "held"})
    assert executor.outcome("held")["terminal"] == "canceled"
    assert peer.reply("probe")["ok"]
    assert not any(f.get("reply") == "cancel" for f in peer.frames)
    # TensorD 0.10 marks a cancel by renaming `executor.cancel` into the root, not a command.
    executor.prepare("filed", 4, "hold")
    executor.invoke("filed")
    peer.until(lambda: peer.progress("filed"))
    (root / "executor.cancel.pending").write_text("filed")
    (root / "executor.cancel.pending").rename(root / "executor.cancel")
    assert executor.outcome("filed")["terminal"] == "canceled"


def test_a_shutdown_waits_for_the_calls_before_it_and_their_cancels_still_land(
    executor: Running,
) -> None:
    peer = executor.peer
    executor.prepare("held", 1, "hold")
    executor.invoke("held")
    peer.until(lambda: peer.progress("held"))
    peer.channel.send({"cmd": "shutdown"})
    peer.channel.send({"cmd": "cancel", "request_id": "held"})
    assert executor.outcome("held")["terminal"] == "canceled"
    assert peer.reply("shutdown")["ok"]
    assert executor.process.wait(60) == 0


def test_a_refused_invoke_keeps_its_prepared_record_to_come_again(executor: Running) -> None:
    """The record is taken only once nothing can refuse the attempt (admission, then its
    grant), so a call refused before it ran may come again."""
    executor.prepare("again", 5, "plain")
    spool = str(executor.root / "again")
    refused = executor.peer.call(
        {"cmd": "invoke", "request_id": "again", **_CALL, "entrypoint": "nope", "spool": spool}
    )
    assert (refused["ok"], refused["code"]) == (False, "unknown_entrypoint"), refused
    executor.peer.frames.remove(refused)
    executor.invoke("again")
    assert executor.outcome("again")["terminal"] == "succeeded"


def test_a_handlers_system_exit_ends_the_process(executor: Running) -> None:
    executor.prepare("gone", 1, "exit")
    executor.invoke("gone")
    assert executor.process.wait(60) == 3
    assert executor.peer.replied("invoke", "gone") is None


class Encoder(Model[object]):
    @uses_components("dit")
    def encode(self, entered: threading.Event, leave: threading.Event) -> None:
        entered.set()
        assert leave.wait(30)


class Decoder(Model[object], phase="decode"):
    @uses_components("lm")
    def decode(self, entered: threading.Event, leave: threading.Event) -> None:
        entered.set()
        assert leave.wait(30)


def _call(name: str) -> contextvars.Context:
    context = contextvars.Context()
    context.run(CURRENT.set, Context(name, float("inf")))
    return context


def _in(context: contextvars.Context, scope: Callable[..., None]) -> tuple[Any, ...]:
    entered, leave = threading.Event(), threading.Event()
    thread = threading.Thread(target=context.copy().run, args=(scope, entered, leave), daemon=True)
    thread.start()
    return thread, entered, leave


def test_a_call_waits_for_another_calls_scope_and_its_own_threads_never_do() -> None:
    encoder, decoder, other = Encoder.for_test(), Decoder.for_test(), Decoder.for_test()
    first, second = _call("first"), _call("second")
    WARNINGS.clear()
    holder, held, release = _in(first, encoder.encode)
    assert held.wait(10)
    waiter, waited, unblock = _in(second, decoder.decode)
    assert not waited.wait(0.3)  # another call's scope is open
    helper, helped, helper_out = _in(first, decoder.decode)
    assert helped.wait(10)  # the holding call's own thread: today's nesting
    # An author's plain thread carries no call (`_Scopes`' known limit): it never waits.
    out_in, out_out = threading.Event(), threading.Event()
    outside = threading.Thread(target=other.decode, args=(out_in, out_out), daemon=True)
    first.copy().run(outside.start)
    assert out_in.wait(10)
    assert [m for m, _ in WARNINGS] == [
        "a thread outside every call entered a component scope during another call's turn"
    ]
    for event in (helper_out, out_out, release):
        event.set()
    assert waited.wait(10)
    unblock.set()
    for thread in (holder, waiter, helper, outside):
        thread.join(10)
    assert encoder._cozy_active is None and decoder._cozy_active is None


def test_every_thread_of_a_call_waiting_on_another_call_passes_once_it_takes_the_turn() -> None:
    """Another call holds the turn and one call's two threads queue. When the first of them
    takes its call's turn, the second passes too, whatever order the two woke in."""
    for _ in range(20):
        holding, first, second = Encoder.for_test(), Decoder.for_test(), Decoder.for_test()
        call, other = _call("call"), _call("other")
        holder, held, release = _in(other, holding.encode)
        assert held.wait(10)
        one, one_in, one_out = _in(call, first.decode)
        two, two_in, two_out = _in(call, second.decode)
        assert not one_in.wait(0.1) and not two_in.wait(0.01)  # both queued
        release.set()
        assert one_in.wait(10) and two_in.wait(10)
        one_out.set()
        two_out.set()
        for thread in (holder, one, two):
            thread.join(10)
