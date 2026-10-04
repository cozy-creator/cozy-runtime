"""A supervisor refusal reaches the host with its code, reason and measured budget."""

from __future__ import annotations

import array
import json
import os
import socket
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from cozy_runtime.internal import storage_admission as storage


@pytest.fixture
def supervisor(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, object]]]:
    """The supervisor end of the inherited pair, answering each request with the next refusal."""
    host, worker = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    monkeypatch.setattr(storage, "_channel", worker)
    answers: list[dict[str, object]] = []

    def serve() -> None:
        while True:
            try:
                _, ancillary, _, _ = host.recvmsg(4096, socket.CMSG_SPACE(9 * 4))
            except OSError:
                return
            if not ancillary:
                return
            descriptors = array.array("i")
            descriptors.frombytes(ancillary[0][2])
            for descriptor in descriptors[1:]:
                os.close(descriptor)
            with socket.socket(fileno=descriptors[0]) as conn:
                conn.sendall(json.dumps({"ok": False, **answers.pop(0)}).encode() + b"\n")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    with host, worker:
        yield answers
        host.shutdown(socket.SHUT_RDWR)


def test_a_reasonless_supervisor_answer_says_so(
    supervisor: list[dict[str, object]], tmp_path: Path
) -> None:
    supervisor.append({})
    with pytest.raises(storage.StorageRefusal) as refused:
        with storage.admit(storage.Write(tmp_path, 1, 1)):
            pytest.fail("refused write ran")
    assert refused.value.code == "storage_admission_refused"
    assert str(refused.value) == (
        "supervisor refused Runtime disk write: the supervisor gave no reason"
    )
