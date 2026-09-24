from __future__ import annotations

import array
import os
import socket
from pathlib import Path

import pytest

from cozy_runtime.internal.seam import Channel, SeamError


def _descriptor_count(fd: int) -> int:
    """Count this capability's aliases, independently of unrelated async cleanup."""
    identity = os.fstat(fd)
    count = 0
    for path in Path("/proc/self/fd").iterdir():
        try:
            observed = path.stat()
        except FileNotFoundError:
            continue  # Another thread closed an unrelated descriptor.
        count += (observed.st_dev, observed.st_ino) == (identity.st_dev, identity.st_ino)
    return count


def test_descriptor_handoff_preserves_control_frames_and_connected_custody() -> None:
    left, right = socket.socketpair()
    producer, owner = socket.socketpair()
    with left, right, producer, owner:
        sending, receiving = Channel(left), Channel(right)
        sending.send({"event": "answer", "ok": True, "descriptor": True})
        sending.send_descriptor(producer)
        sending.send({"event": "later", "value": 7})
        assert receiving.recv() == {"event": "answer", "ok": True, "descriptor": True}
        fd = receiving.recv_descriptor()
        assert not os.get_inheritable(fd)
        with socket.socket(fileno=fd) as granted:
            producer.close()
            granted.sendall(b"written through the granted channel")
            assert owner.recv(100) == b"written through the granted channel"
        assert owner.recv(1) == b""
        assert receiving.recv() == {"event": "later", "value": 7}
        assert sending.sent_bytes == receiving.recv_bytes


@pytest.mark.parametrize("count", [0, 2, 8])
def test_descriptor_count_refuses_without_leaking_received_fds(count: int) -> None:
    left, right = socket.socketpair()
    producer, owner = socket.socketpair()
    with left, right, producer, owner:
        before = _descriptor_count(producer.fileno())
        rights = (
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [producer.fileno()] * count))]
            if count
            else []
        )
        left.sendmsg([b"\0"], rights)
        with pytest.raises(SeamError):
            Channel(right).recv_descriptor()
        assert _descriptor_count(producer.fileno()) == before


def test_descriptor_rejects_regular_files_and_unconnected_sockets(tmp_path: Path) -> None:
    left, right = socket.socketpair()
    with left, right, (tmp_path / "private").open("wb") as source:
        before = _descriptor_count(source.fileno())
        left.sendmsg(
            [b"\0"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [source.fileno()]))]
        )
        with pytest.raises(SeamError):
            Channel(right).recv_descriptor()
        assert _descriptor_count(source.fileno()) == before
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as unconnected:
            before = _descriptor_count(unconnected.fileno())
            left.sendmsg(
                [b"\0"],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [unconnected.fileno()]))],
            )
            with pytest.raises(SeamError):
                Channel(right).recv_descriptor()
            assert _descriptor_count(unconnected.fileno()) == before


def test_descriptor_peer_loss_is_typed() -> None:
    left, right = socket.socketpair()
    with right:
        left.close()
        with pytest.raises(SeamError, match="exactly one scoped capability"):
            Channel(right).recv_descriptor()
