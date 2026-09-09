from __future__ import annotations

import array
import json
import os
import socket
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import storage_admission as storage


@pytest.fixture
def channel(monkeypatch: pytest.MonkeyPatch) -> Iterator[socket.socket]:
    host, worker = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    monkeypatch.setattr(storage, "_channel", worker)
    with host, worker:
        yield host


def operation(channel: socket.socket) -> tuple[list[dict[str, Any]], socket.socket]:
    body, ancillary, flags, _ = channel.recvmsg(4096, socket.CMSG_SPACE(9 * 4))
    assert flags == 0
    descriptors = array.array("i")
    descriptors.frombytes(ancillary[0][2])
    for descriptor in descriptors[1:]:
        os.close(descriptor)
    return json.loads(body), socket.socket(fileno=descriptors[0])


def test_refusal_happens_before_any_file_write(channel: socket.socket, tmp_path: Path) -> None:
    target = tmp_path / "not-created"

    def host() -> None:
        rows, conn = operation(channel)
        assert rows == [{"path": str(tmp_path), "bytes": 100, "inodes": 1}]
        with conn:
            conn.sendall(b'{"ok":false,"facts":{"available":90,"required":100}}\n')

    thread = threading.Thread(target=host)
    thread.start()
    with (
        pytest.raises(storage.StorageRefusal) as refused,
        storage.admit(storage.Write(tmp_path, 100, 1)),
    ):
        target.write_bytes(b"unexpected")
    thread.join(2)
    assert not thread.is_alive()
    assert refused.value.facts["available"] == 90
    assert not target.exists()


def test_exception_releases_only_after_work_finishes(
    channel: socket.socket, tmp_path: Path
) -> None:
    acquired, released = threading.Event(), threading.Event()

    def host() -> None:
        _, conn = operation(channel)
        with conn:
            conn.sendall(b'{"ok":true}\n')
            acquired.set()
            assert conn.recv(1) == b""
            released.set()

    thread = threading.Thread(target=host)
    thread.start()
    with (
        pytest.raises(RuntimeError, match="writer failed"),
        storage.admit(storage.Write(tmp_path, 100, 1)),
    ):
        assert acquired.wait(2)
        assert not released.is_set()
        raise RuntimeError("writer failed")
    assert released.wait(2)
    thread.join(2)


def test_nested_admission_refuses_without_waiting_on_its_own_host_lock(
    channel: socket.socket, tmp_path: Path
) -> None:
    def host() -> None:
        _, conn = operation(channel)
        with conn:
            conn.sendall(b'{"ok":true}\n')
            assert conn.recv(1) == b""

    thread = threading.Thread(target=host)
    thread.start()
    with (
        storage.admit(storage.Write(tmp_path, 100, 1)),
        pytest.raises(storage.StorageRefusal, match="nested"),
        storage.admit(storage.Write(tmp_path, 100, 1)),
    ):
        pytest.fail("nested write ran")
    thread.join(2)
    assert not thread.is_alive()


@pytest.mark.parametrize(
    "write",
    [
        storage.Write(Path("relative"), 1, 1),
        storage.Write(Path("/tmp"), -1, 1),
        storage.Write(Path("/tmp"), 1, -1),
    ],
)
def test_invalid_bounds_do_not_send_a_request(channel: socket.socket, write: storage.Write) -> None:
    with pytest.raises(storage.StorageRefusal), storage.admit(write):
        pytest.fail("invalid write ran")
    channel.settimeout(0.01)
    with pytest.raises(TimeoutError):
        channel.recv(1)


def test_missing_controller_is_not_a_local_storage_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(storage, "_channel", None)
    with storage.admit(storage.Write(tmp_path, 1, 1)):
        (tmp_path / "local").write_bytes(b"x")
    assert (tmp_path / "local").read_bytes() == b"x"
