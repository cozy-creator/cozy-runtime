"""Real descriptor transfer and Runtime failure handling; these are not GPU proofs."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

from cozy_runtime.author._errors import classify, is_varena_error
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.seam import Channel, SeamError, anonymous


@pytest.mark.skipif(sys.platform != "linux", reason="the GPU ledger uses Linux eventfds")
def test_rank_grant_transfers_exactly_three_real_capabilities() -> None:
    left, right = socket.socketpair()
    owner, custody = socket.socketpair()
    table, event = anonymous("slim-ledger"), os.eventfd(0, os.EFD_CLOEXEC)
    received: list[int] = []
    try:
        os.write(table, b"table")
        Channel(left).send_ledger(table, event, custody.fileno())
        received = Channel(right).recv_ledger()
        assert len(received) == 3
        assert all(not os.get_inheritable(fd) for fd in received)
        assert os.pread(received[0], 5, 0) == b"table"
        os.eventfd_write(event, 7)
        assert os.eventfd_read(received[1]) == 7
        owner.sendall(b"bind this rank")
        assert os.read(received[2], 14) == b"bind this rank"
    finally:
        for fd in [table, event, *received]:
            os.close(fd)
        left.close()
        right.close()
        owner.close()
        custody.close()


@pytest.mark.skipif(sys.platform != "linux", reason="the GPU ledger uses Linux eventfds")
def test_rank_grant_rejects_a_file_instead_of_a_custody_channel() -> None:
    left, right = socket.socketpair()
    table, event = anonymous("slim-ledger"), os.eventfd(0, os.EFD_CLOEXEC)
    try:
        Channel(left).send_fds([table, event, table])
        with pytest.raises(SeamError, match="custody channel must be a socket"):
            Channel(right).recv_ledger()
    finally:
        os.close(table)
        os.close(event)
        left.close()
        right.close()


class NativeFailure(RuntimeError):
    code: str
    sticky: bool

    def __init__(self, *, sticky: bool) -> None:
        super().__init__("native failure")
        self.code, self.sticky = "cuda_fault" if sticky else "fill_failed", sticky


# The classifier deliberately recognizes the native ABI without importing a GPU module.
NativeFailure.__module__, NativeFailure.__name__ = "tensorfs.varena", "VarenaError"


@pytest.mark.parametrize("sticky", [False, True])
def test_only_sticky_native_failures_poison_an_executor(tmp_path: Path, sticky: bool) -> None:
    left, right = socket.socketpair()
    try:
        executor = Executor(Channel(left), tmp_path)
        failure = NativeFailure(sticky=sticky)
        assert is_varena_error(failure)
        assert classify(failure).origin == "runtime"
        reply = executor.unsettled(executor, failure, "request")
        assert reply["outcome"]["code"] == failure.code
        assert bool(executor.poisoned) is sticky
        assert reply["quiescent"] is not sticky
        assert executor.hello()["ok"] is True
    finally:
        left.close()
        right.close()
