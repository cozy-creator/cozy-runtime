"""A device-room ask carries a positive byte count or is not sent: the machine reads it as
unsigned, and a negative one (-22 MiB, run 10078) ended the executor."""

from __future__ import annotations

import socket
from pathlib import Path

from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.seam import Channel


def test_only_a_positive_room_ask_reaches_the_worker(tmp_path: Path) -> None:
    left, right = socket.socketpair()
    with left, right:
        worker = Channel(right)
        executor = Executor(Channel(left), tmp_path)
        for nothing in (-23068672, 0):
            executor._device_room(nothing)
        right.settimeout(0.2)
        worker.send({"event": "answer", "seq": 1, "ok": True, "cap_bytes": -1})
        executor._device_room(5 << 20)
        asked = worker.recv()
        assert (asked["kind"], asked["seq"], asked["free_bytes"]) == ("device_room", 1, 5 << 20)
