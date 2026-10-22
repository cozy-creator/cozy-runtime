"""A cancel that lands inside a handler's asyncio.TaskGroup ends the call CANCELED, in a real
executor process: the group's ExceptionGroup is its tasks' cancel, never an unhandled exception
(cancelling an H3 long_form while it made its references ended FAILED)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover_installed
from cozy_runtime.internal.seam import Channel
from test_executor_fork import _accept, _environment, _listener

APP = """
import asyncio

import msgspec
from cozy_runtime.author import App, Context, Telemetry

app = App()


class Request(msgspec.Struct):
    tasks: int = 2


class Result(msgspec.Struct):
    done: int


async def work(ctx: Context, tel: Telemetry, index: int) -> None:
    while True:
        ctx.raise_if_cancelled()
        tel.progress(0.5, stage=f"task-{index}")
        await asyncio.sleep(0.02)


@app.entrypoint(accelerator=False)
async def grouped(ctx: Context, payload: Request, tel: Telemetry) -> Result:
    async with asyncio.TaskGroup() as group:
        for index in range(payload.tasks):
            group.create_task(work(ctx, tel, index))
    return Result(payload.tasks)
"""


def _reply(channel: Channel, cmd: str, request_id: str = "") -> dict[str, Any]:
    while (frame := channel.recv()) is not None:
        if frame.get("reply") == cmd and frame.get("request_id", "") == request_id:
            return dict(frame)
    raise AssertionError(f"the executor closed its seam before replying {cmd}")


def test_a_cancel_inside_a_task_group_ends_the_call_canceled(tmp_path: Path) -> None:
    (tmp_path / "groupapp.py").write_text(APP)
    sys.path.insert(0, str(tmp_path))
    try:
        interface = package_interface.build(discover_installed("groupapp:app"))
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("groupapp", None)
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
        binding = {
            "application": "groupapp:app",
            "package_interface": str(tmp_path / "interface.json"),
        }
        for frame in (
            {"cmd": "start", "devices": "", **binding},
            {"cmd": "load", "construction": "c", "devices": "", "binding": binding},
            {"cmd": "activate", "construction": "c"},
        ):
            channel.send(frame)
            assert _reply(channel, str(frame["cmd"]))["ok"]
        call = {"entrypoint": "grouped", "deadline_s": 120}
        (root / "grouped").mkdir()
        channel.send({"cmd": "prepare_request", "request_id": "grouped", **call, "payload": {}})
        assert _reply(channel, "prepare_request", "grouped")["ok"]
        channel.send(
            {"cmd": "invoke", "request_id": "grouped", **call, "spool": str(root / "grouped")}
        )
        # Both tasks are running before the cancel arrives.
        seen: set[str] = set()
        while len(seen) < 2:
            frame = channel.recv()
            assert frame is not None
            if frame.get("event") == "progress" and str(frame.get("stage", "")).startswith("task-"):
                seen.add(frame["stage"])
        time.sleep(0.1)
        channel.send({"cmd": "cancel", "request_id": "grouped"})
        reply = _reply(channel, "invoke", "grouped")
        assert reply["ok"] and not reply["poisoned"], reply
        assert reply["outcome"]["terminal"] == "canceled", reply["outcome"]
    finally:
        process.kill()
        process.wait()
