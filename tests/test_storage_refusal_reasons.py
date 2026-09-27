"""A supervisor refusal reaches the host with its code, reason and measured budget."""

from __future__ import annotations

import array
import json
import os
import socket
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import pytest

from cozy_runtime.internal import storage_admission as storage
from cozy_runtime.internal.worker.control import _PreparationServicer
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc

CAPACITY: dict[str, object] = {
    "code": "insufficient_storage",
    "reason": "needs 5.6 GiB; available 199.0 GiB - reserved 100.0 GiB"
    " - reserve 96.0 GiB = 3.0 GiB free",
    "facts": {"available": 213674622976, "required": 6012954214, "reserved": 107374182400},
}


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


Preparer = Callable[[pb.PreparePackageSetRequest], pb.PreparePackageSetResult]


def prepare(request: pb.PreparePackageSetRequest, handler: Preparer) -> grpc.RpcError:
    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    rpc.add_RuntimePreparationServicer_to_server(
        _PreparationServicer(handler, None, None, None),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            client = rpc.RuntimePreparationStub(channel)
            with pytest.raises(grpc.RpcError) as failure:
                client.PreparePackageSet(request, timeout=10)
            return failure.value
    finally:
        server.stop(None).wait()


def trailer(failure: grpc.RpcError, key: str) -> list[str]:
    return [value for name, value in failure.trailing_metadata() or () if name == key]


def test_capacity_refusal_reaches_the_host_with_its_budget(
    supervisor: list[dict[str, object]], tmp_path: Path
) -> None:
    supervisor.append(CAPACITY)

    def preparer(_: pb.PreparePackageSetRequest) -> pb.PreparePackageSetResult:
        with storage.admit(storage.Write(tmp_path, 6012954214, 20000)):
            pytest.fail("refused write ran")
        raise AssertionError("unreachable")

    failure = prepare(pb.PreparePackageSetRequest(), preparer)
    assert failure.code() == grpc.StatusCode.INTERNAL
    assert failure.details() == (
        "StorageRefusal: supervisor refused Runtime disk write: "
        + str(CAPACITY["reason"])
        + " [available=213674622976, required=6012954214, reserved=107374182400]"
    )
    assert trailer(failure, "cozy-error-code") == ["insufficient_storage"]


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
