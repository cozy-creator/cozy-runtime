"""A call's work is told to the machine (cl-648): its CPU no longer says that it moves, so the
executor tells each call in flight the work count once a second while it moves, over the real
seam, and nothing after the call's reply."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from test_concurrent_attempts import Peer
from test_executor_fork import _accept, _environment, _listener

from cozy_runtime.author import _work as work
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover_installed

APP = """
import time

import msgspec
from cozy_runtime.author import App, Context
from cozy_runtime.author import _work as work

app = App()


class Request(msgspec.Struct):
    seconds: float
    working: bool


class Done(msgspec.Struct):
    ok: bool


@app.entrypoint(accelerator=False)
def run(ctx: Context, payload: Request) -> Done:
    # A phase of the Runtime's own (an LM's decode steps), with nothing else to say; or a sleep.
    end = time.monotonic() + payload.seconds
    while time.monotonic() < end:
        if payload.working:
            work.done += 1
        time.sleep(0.05)
    return Done(True)
"""


def test_a_working_call_is_told_its_work_each_second_and_a_still_one_nothing(tmp_path: Path) -> None:
    (tmp_path / "workapp.py").write_text(APP)
    sys.path.insert(0, str(tmp_path))
    try:
        interface = package_interface.build(discover_installed("workapp:app"))
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("workapp", None)
    (tmp_path / "interface.json").write_text(json.dumps(interface))
    root = tmp_path / "root"
    root.mkdir()
    listener = _listener(tmp_path / "e.sock")
    command = [sys.executable, "-m", "cozy_runtime.internal.executor"]
    command += ["--socket", str(tmp_path / "e.sock"), "--root", str(root)]
    process = subprocess.Popen(command, env=_environment(tmp_path))
    try:
        channel, _ = _accept(listener)
        peer = Peer(channel, hold=1)
        binding = {"application": "workapp:app", "package_interface": str(tmp_path / "interface.json")}
        assert peer.call({"cmd": "start", "devices": "", **binding})["ok"]
        assert peer.call({"cmd": "load", "construction": "c", "devices": "", "binding": binding})["ok"]
        assert peer.call({"cmd": "activate", "construction": "c"})["ok"]

        def call(request_id: str, seconds: float, working: bool) -> int:
            (root / request_id).mkdir()
            frame = {"request_id": request_id, "entrypoint": "run", "deadline_s": 120}
            payload = {"seconds": seconds, "working": working}
            assert peer.call({"cmd": "prepare_request", **frame, "payload": payload})["ok"]
            reply = peer.call({"cmd": "invoke", **frame, "spool": str(root / request_id)})
            assert reply["ok"] and reply["outcome"]["terminal"] == "succeeded", reply
            return peer.frames.index(reply)

        replied = call("working", 3.5, True)
        told = peer.progress("working", "work")
        counts = [frame["work"] for frame in told]
        assert 2 <= len(counts) <= 5 and counts == sorted(set(counts)), counts
        assert all(peer.frames.index(frame) < replied for frame in told), "told after its reply"

        time.sleep(2 * work.SAMPLE_SECONDS)  # the working call's last units are told by now
        call("still", 2.5, False)
        assert peer.progress("still", "work") == []
    finally:
        process.kill()
        process.wait()
