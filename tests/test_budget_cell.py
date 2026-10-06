"""The Worker lowers a running call's plane budget through the executor's budget cell.

Real: the executor's own `BudgetCell` exchange over a seam socketpair, the Worker's reply path
(`respond` through `child.Executor`'s handler), the memfd both map, and the Worker's `cut`
waiting on the call itself. The executor half applies at its block boundaries
(`WeightResidency._asked`; the card arm is `test_weight_plane`'s).
"""

from __future__ import annotations

import os
import socket
import threading
from pathlib import Path
from typing import Any

from cozy_runtime.author._executor_requests import Handler, refuse, respond
from cozy_runtime.internal import budget_cell
from cozy_runtime.internal.executor import Executor
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
    cell, fd = budget_cell.Cell.create()
    other = budget_cell.Cell(fd)
    os.close(fd)
    try:
        assert cell.poll() is None
        ticket = other.ask(5 << 20)
        assert other.answered(ticket) is None
        assert cell.poll() == 5 << 20 and cell.poll() is None
        cell.acknowledge(4 << 20)
        assert other.answered(ticket) == 4 << 20
        later = other.ask(-1)
        assert other.answered(later) is None and cell.poll() == -1
    finally:
        cell.close()
        other.close()
