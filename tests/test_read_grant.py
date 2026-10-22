"""Kept layouts and GPU descriptors cross the seam; cold bytes are varena's to read."""

from __future__ import annotations

import os
import socket
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

from cozy_runtime.author._executor_requests import HostGrant
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.seam import Channel, sealed
from tensord_peer import answer_read_grants

MANIFEST = "sha256:" + "a" * 64


@contextmanager
def grants(
    tmp_path: Path, deposited: dict[str, int], **peer: object
) -> Iterator[tuple[Executor, list[str]]]:
    left, right = socket.socketpair()
    asked: list[str] = []
    serving = threading.Thread(
        target=answer_read_grants, args=(Channel(right), deposited, asked), kwargs=peer
    )
    serving.start()
    try:
        yield Executor(Channel(left), tmp_path), asked
    finally:
        left.close()
        serving.join()
        right.close()


def test_a_held_layout_is_the_one_descriptor_mapped_as_it_is(tmp_path: Path) -> None:
    layout = sealed("layout", b"a host layout varena sealed in another executor")
    try:
        with grants(tmp_path, {"sha256:held": layout}) as (executor, asked):
            fd, pages = (held := executor._read_grant(MANIFEST, "sha256:held")).fd, held.pages
            assert pages == () and fd is not None and not held.seal and asked == ["sha256:held"]
            assert os.pread(fd, 64, 0) == b"a host layout varena sealed in another executor"
            os.close(fd)
    finally:
        os.close(layout)


def test_an_unheld_layout_grants_nothing_and_varena_reads_the_checkpoint(tmp_path: Path) -> None:
    with grants(tmp_path, {}) as (executor, asked):
        assert executor._read_grant(MANIFEST, "sha256:cold") == HostGrant()
        assert asked == ["sha256:cold"]


def test_an_unheld_layout_seals_only_where_the_machine_admits_it(tmp_path: Path) -> None:
    peer = {"admitted": frozenset({"sha256:fits"})}
    with grants(tmp_path, {}, **peer) as (executor, asked):
        assert executor._read_grant(MANIFEST, "sha256:fits") == HostGrant(seal=True)
        assert executor._read_grant(MANIFEST, "sha256:short") == HostGrant(seal=False)
        assert asked == ["sha256:fits", "sha256:short"]


def test_a_layout_another_process_seals_is_asked_again_until_its_one_copy_is_deposited(
    tmp_path: Path,
) -> None:
    layout = sealed("layout", b"sealed by GPU 0 while GPU 1 asked")
    try:
        peer = {"sealing": {"sha256:s": 2}}
        with grants(tmp_path, {"sha256:s": layout}, **peer) as (executor, asked):
            held = executor._read_grant(MANIFEST, "sha256:s")
            fd = held.fd
            assert fd is not None and held.pages == ()
            assert asked == ["sha256:s"] * 3, "two sealing answers, then the deposit"
            assert os.pread(fd, 64, 0) == b"sealed by GPU 0 while GPU 1 asked"
            os.close(fd)
    finally:
        os.close(layout)


def test_a_held_layout_carries_kept_regions_and_their_fingerprint(tmp_path: Path) -> None:
    layout = sealed("layout", b"held layout")
    page = sealed("kept-page", b"opaque page capability")
    left, right = socket.socketpair()
    executor, peer = Executor(Channel(left), tmp_path), Channel(right)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            received = pool.submit(executor._read_grant, MANIFEST, "held")
            frame = peer.recv()
            assert frame is not None and frame["kind"] == "read_grant"
            peer.send(
                {
                    "event": "answer",
                    "seq": frame["seq"],
                    "ok": True,
                    "held": True,
                    "descriptors": 2,
                    "fingerprint": "sha256:layout-parts",
                    "pages": [{"region": 3, "nbytes": 2 << 20}],
                }
            )
            peer.send_fds([layout, page])
            held = received.result()
            fd, pages, fingerprint = held.fd, held.pages, held.fingerprint
        assert fd is not None and fingerprint == "sha256:layout-parts"
        assert len(pages) == 1 and pages[0][0] == 3
        try:
            assert os.pread(fd, 64, 0) == b"held layout"
            assert os.pread(pages[0][1], 64, 0) == b"opaque page capability"
        finally:
            os.close(fd)
            os.close(pages[0][1])
    finally:
        left.close()
        right.close()
        os.close(layout)
        os.close(page)


def test_kept_pages_transfer_containers_and_close_the_callers_descriptors(tmp_path: Path) -> None:
    left, right = socket.socketpair()
    container, producer = socket.socketpair()
    descriptor = container.detach()
    executor, peer = Executor(Channel(left), tmp_path), Channel(right)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            deposited = pool.submit(
                executor._keep_pages, "layout", "parts", [(2, 4096, descriptor)]
            )
            frame = peer.recv()
            assert frame is not None and frame["kind"] == "keep_pages"
            assert frame["layout"] == "layout" and frame["fingerprint"] == "parts"
            assert frame["pages"] == [{"region": 2, "nbytes": 4096}]
            assert frame["descriptors"] == 1 and "fds" not in frame
            handed = peer.recv_fds(1)[0]
            peer.send({"event": "answer", "seq": frame["seq"], "ok": True})
            deposited.result()
        with pytest.raises(OSError):
            os.fstat(descriptor)
        with socket.socket(fileno=handed) as kept:
            producer.sendall(b"live")
            assert kept.recv(4) == b"live"
    finally:
        left.close()
        right.close()
        producer.close()


def test_a_followers_kept_containers_reach_its_gpu_rank(tmp_path: Path) -> None:
    leader_end, machine_end = socket.socketpair()
    follower_end, relay_end = socket.socketpair()
    container, producer = socket.socketpair()
    descriptor = container.detach()
    leader = Executor(Channel(leader_end), tmp_path)
    follower_host = Executor(Channel(follower_end), tmp_path, rank=1, world=2)
    follower = follower_host._construction_child("a-model-member")
    leader._attempt_request = ""
    machine, relay = Channel(machine_end), Channel(relay_end)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            depositing = pool.submit(
                follower._keep_pages, "layout", "parts", [(4, 8192, descriptor)]
            )
            request = relay.recv()
            assert request is not None
            forwarding = pool.submit(leader._relay, relay, request)
            frame = machine.recv()
            assert frame is not None and frame["kind"] == "keep_pages" and frame["rank"] == 1
            assert frame["request_id"] == "", "retention after the call is construction-scoped"
            assert frame["pages"] == [{"region": 4, "nbytes": 8192}]
            handed = machine.recv_fds(1)[0]
            machine.send({"event": "answer", "seq": frame["seq"], "ok": True})
            forwarding.result()
            depositing.result()
        with pytest.raises(OSError):
            os.fstat(descriptor)
        with socket.socket(fileno=handed) as kept:
            producer.sendall(b"rank1")
            assert kept.recv(5) == b"rank1"
    finally:
        for sock in [leader_end, machine_end, follower_end, relay_end, producer]:
            sock.close()


def test_a_refused_follower_request_drains_its_plural_descriptors(tmp_path: Path) -> None:
    leader_end, machine_end = socket.socketpair()
    follower_end, relay_end = socket.socketpair()
    container, producer = socket.socketpair()
    leader, follower, relay = (
        Executor(Channel(leader_end), tmp_path),
        Channel(follower_end),
        Channel(relay_end),
    )
    try:
        request = {"event": "request", "seq": 1, "kind": "unknown", "descriptors": 1}
        follower.send(request)
        follower.send_fds([container.fileno()])
        container.close()
        assert relay.recv() == request
        leader._relay(relay, request)
        answer = follower.recv()
        assert answer is not None and not answer["ok"] and answer["code"] == "request_unsupported"
        assert producer.recv(1) == b"", "the refused container was closed"
        follower.send({"reply": "next", "ok": True})
        assert relay.recv() == {"reply": "next", "ok": True}, "the channel remains framed"
    finally:
        for sock in [leader_end, machine_end, follower_end, relay_end, container, producer]:
            sock.close()
