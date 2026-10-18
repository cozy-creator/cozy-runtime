"""The owner's floor watchdog caps a running call through the executor's budget cell.

Real: the executor's own `BudgetCell` exchange over a seam socketpair, the Worker's reply path
(`respond` through `child.Executor`'s handler), and the memfd both map. The executor half
applies at its block boundaries (`WeightResidency._asked`; the card arm is `test_weight_plane`'s).
"""

from __future__ import annotations

import mmap
import os
import socket
import struct
import threading
from pathlib import Path
from typing import Any

from cozy_runtime.author._executor_requests import Handler, refuse, respond
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.memory import CAP, Cell
from cozy_runtime.internal.seam import Channel


def serve_once(channel: Channel, handler: Handler | None) -> None:
    """The Worker's half of `child.Executor.call`'s request branch, for one request."""
    frame = channel.recv()
    assert frame is not None
    received = channel.recv_memfd() if frame.get("descriptor") is True else None
    answer, _ = respond(frame, handler, received)
    channel.send(answer)


def test_an_older_worker_leaves_the_executor_without_a_cell(tmp_path: Path) -> None:
    ours, theirs = socket.socketpair()

    def older(request: object) -> Any:
        return refuse("unknown_durable_request", "an older worker")

    serving = threading.Thread(target=serve_once, args=(Channel(theirs), older))
    serving.start()
    try:
        assert Executor(Channel(ours), tmp_path)._budget_cell() is None
    finally:
        serving.join()
        ours.close()
        theirs.close()


def test_a_cell_reads_back_what_each_side_wrote() -> None:
    """The owner writes its cap, then the counter (TensorD's `ask_cap`); the executor reads it
    once and writes back what it applied, then the counter."""
    cell, fd = Cell.create()
    owner = mmap.mmap(fd, 32)
    os.close(fd)
    try:
        assert cell.poll() is None
        struct.pack_into("<q", owner, 8, CAP - (5 << 20))
        struct.pack_into("<q", owner, 0, 1)
        assert cell.poll() == CAP - (5 << 20) and cell.poll() is None
        cell.acknowledge(4 << 20)
        assert struct.unpack_from("<qqqq", owner) == (1, CAP - (5 << 20), 1, 4 << 20)
    finally:
        cell.close()
        owner.close()
