"""The Worker keeps each weight set's pinned tier across executors (weight-plane.md §3).

Real: two host-only `tensorfs.plane` instances (an executor and its replacement), the
executor's own durable exchange over a seam socketpair, the Worker's reply path and its
`HostTiers`. The replacement adopts the filled tier: it reads no byte from the store.
"""

from __future__ import annotations

import ctypes
import os
import socket
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.author._executor_requests import respond  # noqa: E402
from cozy_runtime.internal import plane  # noqa: E402
from cozy_runtime.internal.executor import Executor  # noqa: E402
from cozy_runtime.internal.seam import Channel, SeamError  # noqa: E402
from cozy_runtime.internal.weights import HostTiers as Tiers  # noqa: E402
from cozy_runtime.internal.weights import Weights  # noqa: E402
from cozy_runtime.internal.worker.host_tier import HostTiers  # noqa: E402
from test_weight_plane import assert_pinned, load, write_checkpoint  # noqa: E402

pytestmark = pytest.mark.skipif(not plane.available(), reason="a TensorFS with the weight plane")


def serve(channel: Channel, tiers: HostTiers, stop: threading.Event) -> None:
    """The Worker's half of `child.Executor.call`'s request branch."""
    while not stop.is_set():
        frame = channel.recv()
        if frame is None:
            return
        received = channel.recv_memfd() if frame.get("descriptor") is True else None
        answer, handoff = respond(frame, tiers.answer, received)
        channel.send(answer)
        if isinstance(handoff, int):
            channel.send_memfd(handoff)
            os.close(handoff)


def test_a_replacement_executor_adopts_the_tier_the_worker_kept(tmp_path: Path) -> None:
    checkpoint, values = write_checkpoint(tmp_path)
    tiers = HostTiers(lambda: 1 << 40)
    ours, theirs = socket.socketpair()
    stop = threading.Event()
    worker = threading.Thread(target=serve, args=(Channel(theirs), tiers, stop), daemon=True)
    worker.start()
    executor = Executor(Channel(ours), tmp_path)
    seam = Tiers(ask=executor._ask_tier, offer=executor._offer_tier)
    try:
        first = Weights(torch, torch.device("cpu"), "cpu")
        first.set_budget(-1, pinned=1 << 30)
        filled = load(first, checkpoint, "tier", tiers=seam).backend.components["stack"]
        assert_pinned(first, filled, values)
        assert len(tiers.held) == 1 and tiers.nbytes() > 0
        filled_bytes = plane.stats(first.plane).host.fill_bytes
        assert filled_bytes > 0

        second = Weights(torch, torch.device("cpu"), "cpu")
        second.set_budget(-1, pinned=1 << 30)
        adopted = load(second, checkpoint, "tier", tiers=seam).backend.components["stack"]
        assert_pinned(second, adopted, values)
        assert plane.stats(second.plane).host.fill_bytes == 0  # nothing read from the store
    finally:
        stop.set()
        ours.close()
        worker.join()
        theirs.close()
        tiers.close()


def test_the_seam_carries_only_memfds_as_tiers(tmp_path: Path) -> None:
    left, right = socket.socketpair()
    with left, right, (tmp_path / "file").open("w") as regular:
        with pytest.raises(SeamError, match="memfd"):
            Channel(left).send_memfd(regular.fileno())
        memfd = ctypes.CDLL(None, use_errno=True).memfd_create(b"tier", 0)
        assert memfd >= 0
        try:
            Channel(left).send_memfd(memfd)
            received = Channel(right).recv_memfd()
            assert os.path.samestat(os.fstat(received), os.fstat(memfd))
            os.close(received)
        finally:
            os.close(memfd)
