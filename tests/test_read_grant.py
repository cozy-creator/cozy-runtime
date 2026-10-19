"""Kept layouts and GPU descriptors cross the seam; cold bytes open through TensorFS."""

from __future__ import annotations

import os
import socket
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import tensorfs

from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.fill import Checkpoint
from cozy_runtime.internal.seam import Channel, sealed
from tensord_peer import answer_read_grants


@pytest.fixture
def checkpoint(tmp_path: Path) -> tuple[Checkpoint, str, list[tuple[str, int]]]:
    writer = tensorfs.Writer(cache_dir=tmp_path / "store")
    writer.component("dit", cls="", config={}).put(
        "a.weight", np.arange(20 << 18, dtype=np.float32), dtype="F32", shape=[20 << 18]
    )
    writer.component("vae", cls="", config={}).put(
        "b.weight", np.ones((8, 4), np.float16), dtype="F16", shape=[8, 4]
    )
    ck = writer.commit()
    assert ck.id is not None
    return Checkpoint(ck), ck.id, sorted(ck.chunks)


@contextmanager
def grants(tmp_path: Path, deposited: dict[str, int]) -> Iterator[tuple[Executor, list[str]]]:
    left, right = socket.socketpair()
    asked: list[str] = []
    serving = threading.Thread(target=answer_read_grants, args=(Channel(right), deposited, asked))
    serving.start()
    try:
        yield Executor(Channel(left), tmp_path), asked
    finally:
        left.close()
        serving.join()
        right.close()


def test_a_held_layout_is_the_one_descriptor_mapped_as_it_is(tmp_path: Path, checkpoint) -> None:
    ck, manifest, objects = checkpoint
    layout = sealed("layout", b"a host layout varena sealed in another executor")
    try:
        with grants(tmp_path, {"sha256:held": layout}) as (executor, asked):
            fd, files, pages, _ = executor._read_grant(
                manifest, "sha256:held", objects, checkpoints={manifest: ck}
            )
            assert files == [] == pages and fd is not None and asked == ["sha256:held"]
            assert os.pread(fd, 64, 0) == b"a host layout varena sealed in another executor"
            os.close(fd)
    finally:
        os.close(layout)


def test_chunk_files_come_in_sha256_order_named_by_their_lists_digest(
    tmp_path: Path, checkpoint
) -> None:
    ck, manifest, objects = checkpoint
    with grants(tmp_path, {}) as (executor, _):
        fd, files, _, _ = executor._read_grant(
            manifest, "sha256:cold", objects, checkpoints={manifest: ck}
        )
        assert fd is None and [(sha, n) for sha, n, _ in files] == objects
        for _, n, file in files:
            assert os.fstat(file).st_size == n
            os.close(file)


def test_an_object_outside_the_checkpoint_cannot_be_opened(tmp_path: Path, checkpoint) -> None:
    ck, manifest, _ = checkpoint
    with grants(tmp_path, {}) as (executor, _), pytest.raises(KeyError):
        executor._read_grant(manifest, "sha256:cold", [("0" * 64, 1)], checkpoints={manifest: ck})


def test_a_held_layout_carries_kept_regions_and_their_fingerprint(
    tmp_path: Path, checkpoint
) -> None:
    ck, manifest, objects = checkpoint
    layout = sealed("layout", b"held layout")
    page = sealed("kept-page", b"opaque page capability")
    left, right = socket.socketpair()
    executor, peer = Executor(Channel(left), tmp_path), Channel(right)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            received = pool.submit(
                executor._read_grant, manifest, "held", objects, checkpoints={manifest: ck}
            )
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
            fd, files, pages, fingerprint = received.result()
        assert fd is not None and not files and fingerprint == "sha256:layout-parts"
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
    follower = Executor(Channel(follower_end), tmp_path, rank=1, world=2)
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
