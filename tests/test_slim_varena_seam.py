"""Real descriptor transfer and Runtime failure handling; these are not GPU proofs."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.author._errors import classify, is_varena_error
from cozy_runtime.internal.executor import Executor, _reply
from cozy_runtime.internal.seam import Channel, SeamError, anonymous


@pytest.mark.skipif(sys.platform != "linux", reason="the GPU ledger uses Linux eventfds")
def test_named_ledger_grant_crosses_a_real_process(tmp_path: Path) -> None:
    left, right = socket.socketpair()
    owner, custody = socket.socketpair()
    table = os.open(tmp_path / "persistent-ledger", os.O_CREAT | os.O_RDWR, 0o600)
    event = os.eventfd(0, os.EFD_CLOEXEC)
    child = subprocess.Popen(
        [sys.executable, "-c", """
import os, socket, sys
from cozy_runtime.internal.seam import Channel
channel = Channel(socket.socket(fileno=int(sys.argv[1])))
fds = channel.recv_ledgers(1)
assert len(fds) == 3 and all(not os.get_inheritable(fd) for fd in fds)
assert os.pread(fds[0], 5, 0) == b'table'
assert os.eventfd_read(fds[1]) == 7
assert os.read(fds[2], 14) == b'bind this rank'
for fd in fds: os.close(fd)
channel.close()
print('grant received')
""", str(right.fileno())],
        pass_fds=(right.fileno(),), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        right.close()
        os.write(table, b"table")
        Channel(left).send_ledger(table, event, custody.fileno())
        os.eventfd_write(event, 7)
        owner.sendall(b"bind this rank")
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, stderr.decode()
        assert stdout.strip() == b"grant received"
        # Other seam capabilities still require anonymous memory, not this named table.
        with pytest.raises(SeamError, match="capability must be a memfd"):
            Channel(left).send_memfd(table)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        os.close(table)
        os.close(event)
        left.close()
        right.close()
        owner.close()
        custody.close()


@pytest.mark.skipif(sys.platform != "linux", reason="the GPU ledger uses Linux eventfds")
@pytest.mark.parametrize("invalid", ["table", "event", "custody", "readonly", "datagram"])
def test_each_ledger_descriptor_has_its_own_kind(tmp_path: Path, invalid: str) -> None:
    left, right = socket.socketpair()
    owner, custody = socket.socketpair()
    datagram, datagram_peer = socket.socketpair(type=socket.SOCK_DGRAM)
    path = tmp_path / "ledger"
    table = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    readonly = os.open(path, os.O_RDONLY)
    event = os.eventfd(0, os.EFD_CLOEXEC)
    try:
        fds = [table, event, custody.fileno()]
        if invalid == "table":
            fds[0] = event
        elif invalid == "event":
            fds[1] = table
        elif invalid == "custody":
            fds[2] = table
        elif invalid == "datagram":
            fds[2] = datagram.fileno()
        else:
            fds[0] = readonly
        Channel(left).send_ledger(table, event, custody.fileno())
        Channel(left).send_fds(fds)
        before = len(os.listdir("/proc/self/fd"))
        with pytest.raises(SeamError):
            Channel(right).recv_ledgers(2)
        assert len(os.listdir("/proc/self/fd")) == before
    finally:
        os.close(table)
        os.close(readonly)
        os.close(event)
        left.close()
        right.close()
        owner.close()
        custody.close()
        datagram.close()
        datagram_peer.close()


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_ledger_count_is_explicit_and_nonnegative(count: int) -> None:
    left, right = socket.socketpair()
    try:
        with pytest.raises(SeamError, match="ledger grant count"):
            Channel(right).recv_ledgers(count)
    finally:
        left.close()
        right.close()


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


def test_removed_keep_command_is_refused_without_stopping_the_executor(tmp_path: Path) -> None:
    left, right = socket.socketpair()
    try:
        executor = Executor(Channel(left), tmp_path)
        reply = _reply(executor, {"cmd": "keep_weights", "construction": "old"})
        assert reply is not None and reply["code"] == "unknown_command"
        assert executor.hello()["ok"] is True
    finally:
        left.close()
        right.close()
