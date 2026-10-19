"""An executor's `ReadGrant` against TensorD's own answers (`tensord_peer.answer_read_grants`,
field for field as `Callbacks::read_grant` sends them): a held layout is the one descriptor,
mapped as it is with no chunk list (D9's refill); chunk files come named by their list's digest;
any other list is refused."""

from __future__ import annotations

import os
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import tensorfs

from cozy_runtime.author._errors import RuntimeFailure
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.seam import Channel, sealed
from tensord_peer import answer_read_grants


@pytest.fixture
def checkpoint(tmp_path: Path) -> tuple[tensorfs.Store, str, list[tuple[str, int]]]:
    store = tensorfs.Store(tmp_path / "store")
    writer = store.writer()
    writer.put("dit", "a.weight", "F32", [20 << 18], np.arange(20 << 18, dtype=np.float32))
    writer.put("vae", "b.weight", "F16", [8, 4], np.ones((8, 4), np.float16))
    manifest = f"sha256:{writer.commit().sha256}"
    return store, manifest, sorted(store.checkpoint(manifest).chunks)


@contextmanager
def grants(
    tmp_path: Path, store: tensorfs.Store, deposited: dict[str, int]
) -> Iterator[tuple[Executor, list[str]]]:
    left, right = socket.socketpair()
    asked: list[str] = []
    serving = threading.Thread(
        target=answer_read_grants, args=(Channel(right), store, deposited, asked)
    )
    serving.start()
    try:
        yield Executor(Channel(left), tmp_path), asked
    finally:
        left.close()
        serving.join()
        right.close()


def test_a_held_layout_is_the_one_descriptor_mapped_as_it_is(tmp_path: Path, checkpoint) -> None:
    store, manifest, objects = checkpoint
    layout = sealed("layout", b"a host layout varena sealed in another executor")
    try:
        with grants(tmp_path, store, {"sha256:held": layout}) as (executor, asked):
            fd, files, pages, _ = executor._read_grant(manifest, "sha256:held", objects)
            assert files == [] == pages and fd is not None and asked == ["sha256:held"]
            assert os.pread(fd, 64, 0) == b"a host layout varena sealed in another executor"
            os.close(fd)
    finally:
        os.close(layout)


def test_chunk_files_come_in_sha256_order_named_by_their_lists_digest(
    tmp_path: Path, checkpoint
) -> None:
    store, manifest, objects = checkpoint
    with grants(tmp_path, store, {}) as (executor, _):
        fd, files, _, _ = executor._read_grant(manifest, "sha256:cold", objects)
        assert fd is None and [(sha, n) for sha, n, _ in files] == objects
        for _, n, file in files:
            assert os.fstat(file).st_size == n
            os.close(file)


def test_a_grant_of_another_chunk_list_is_refused(tmp_path: Path, checkpoint) -> None:
    store, manifest, objects = checkpoint
    with grants(tmp_path, store, {}) as (executor, _):
        with pytest.raises(RuntimeFailure, match="a different list") as refused:
            executor._read_grant(manifest, "sha256:cold", objects[:-1])
        assert refused.value.code == "weights_ungranted"
