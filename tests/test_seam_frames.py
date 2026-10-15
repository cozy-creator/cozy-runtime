"""A seam frame carries its structure whole: no size rule, only an allocation guard."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover_installed
from cozy_runtime.internal.seam import HEADER, MAX_FRAME, Channel, SeamError
from test_executor_fork import _accept, _call, _environment, _listener

#: Enough entrypoints that a load reply naming them passes the old 64 KiB frame.
WIDE = 1500

APP = f"""
import msgspec
from cozy_runtime.author import App, Context

app = App()


class Request(msgspec.Struct):
    n: int


class Result(msgspec.Struct):
    value: int


def generate(ctx: Context, payload: Request) -> Result:
    return Result(payload.n + 1)


for index in range({WIDE}):
    app.entrypoint(generate, name=f"generate_with_a_long_descriptive_name_{{index:04d}}")
"""


def test_a_load_reply_over_64_kib_crosses_a_real_executors_seam(tmp_path: Path) -> None:
    """FLUX.3 Klein's as-is load failed `reply_too_large`: its facts named every stored key the
    module skips, past the old 64 KiB frame. A real executor process here answers a load whose
    reply names 1,500 entrypoints, and the construction then serves a call."""
    (tmp_path / "wideapp.py").write_text(APP)
    sys.path.insert(0, str(tmp_path))
    try:
        interface = package_interface.build(discover_installed("wideapp:app"))
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("wideapp", None)
    path = tmp_path / "interface.json"
    path.write_text(json.dumps(interface))
    release = {"application": "wideapp:app", "package_interface": str(path)}
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
        assert _call(channel, {"cmd": "start", "devices": "", **release})["ok"]
        before = channel.recv_bytes
        load = {"cmd": "load", "construction": "wide", "devices": "", "binding": release}
        loaded = _call(channel, load)
        assert loaded["ok"], loaded
        assert channel.recv_bytes - before > 64 * 1024 and len(loaded["entrypoints"]) == WIDE
        assert _call(channel, {"cmd": "activate", "construction": "wide"})["ok"]
        name = "generate_with_a_long_descriptive_name_1499"
        attempt = {"request_id": "r", "entrypoint": name}
        assert _call(channel, {"cmd": "prepare_request", **attempt, "payload": {"n": 41}})["ok"]
        spool = root / "r"
        spool.mkdir()
        invoke = {"cmd": "invoke", **attempt, "spool": str(spool), "deadline_s": 60}
        reply = _call(channel, invoke)
        assert reply["outcome"]["terminal"] == "succeeded", reply
        assert json.loads((spool / "result.canonical").read_bytes()) == {"value": 42}
    finally:
        process.kill()
        process.wait()


def test_a_length_prefix_over_the_guard_is_refused_before_its_body() -> None:
    left, right = socket.socketpair()
    with left, right:
        left.sendall(HEADER.pack(MAX_FRAME + 1))
        with pytest.raises(SeamError) as refused:
            Channel(right).recv(timeout=5)
    assert refused.value.code == "frame_over_cap"
