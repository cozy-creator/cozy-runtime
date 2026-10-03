"""Executors adopt the machine's sealed host layouts (`host_tiers.sealed/1`).

Real: a TensorFS store, the machine's half as TensorFS does it (`tensorfs.plane.fill_sealed`:
the verified read path, then seals), the executor's own durable exchange over a seam
socketpair, and host-only planes. The executor reads no byte from the store, takes no lease,
and owns none of the layout's RAM; it computes exactly on it.
"""

from __future__ import annotations

import hashlib
import os
import socket
import threading
from pathlib import Path
from typing import Any

import msgspec
import pytest

torch = pytest.importorskip("torch")

import tensorfs  # noqa: E402
from tensorfs import Store  # noqa: E402
from tensorfs import plane as tfs_plane  # noqa: E402

from cozy_runtime.author._executor_requests import SealedPlan  # noqa: E402
from cozy_runtime.internal import canonical, plane  # noqa: E402
from cozy_runtime.internal.executor import Executor  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402
from cozy_runtime.internal.weights import HostTiers, Weights  # noqa: E402
from test_host_tier import GpuTier  # noqa: E402
from test_weight_plane import (  # noqa: E402
    HIDDEN,
    Stack,
    assert_pinned,
    load,
    run,
    same,
    write_checkpoint,
)

pytestmark = pytest.mark.skipif(
    not plane.sealed_available(), reason="a TensorFS that adopts sealed host layouts"
)


class Machine:
    """The machine's half: fill each asked layout once, keep it, hand out read-only views."""

    def __init__(self, root: str, room: bool = True) -> None:
        self.store = Store.open(root)
        self.room = room
        self.held: dict[tuple[str, str], int] = {}
        self.fills = 0
        self.prefetches = 0

    @staticmethod
    def plans(frame: dict[str, Any], memfd: int) -> Any:
        body = os.pread(memfd, frame["length"], 0)
        os.close(memfd)
        assert hashlib.sha256(body).hexdigest() == frame["sha256"]
        return canonical.parse(body)

    def answer(self, doc: SealedPlan) -> int | None:
        key = (doc.name, doc.layout)
        if key not in self.held:
            if not self.room:
                return None
            # the machine's own planner, over its own copy of the header
            header = self.store.manifest(doc.manifest)["header"]
            plan = tensorfs.plan(
                header, list(doc.traversal), list(doc.components), window=doc.window
            )
            if doc.parts:
                plan = plan.select(list(doc.parts))
            lease = self.store.acquire_cozytensors(doc.manifest)
            fd, _ = tfs_plane.fill_sealed(
                self.store, lease, plan, [list(r) for r in doc.regions], doc.name, 2
            )
            self.held[key] = fd
            self.fills += 1
        return os.open(f"/proc/self/fd/{self.held[key]}", os.O_RDONLY | os.O_CLOEXEC)

    def serve(self, channel: Channel, stop: threading.Event) -> None:
        while not stop.is_set():
            frame = channel.recv()
            if frame is None:
                return
            assert frame.get("descriptor") is True
            sent = self.plans(frame, channel.recv_memfd())
            if frame["kind"] == "sealed_prefetch":  # fill every plan now; asks then hit
                self.prefetches += 1
                for doc in msgspec.convert(sent, list[SealedPlan]):
                    fd = self.answer(doc)
                    if fd is not None:
                        os.close(fd)
                channel.send({"event": "answer", "seq": frame["seq"], "ok": True})
                continue
            assert frame["kind"] == "sealed_tier"
            granted = self.answer(msgspec.convert(sent, SealedPlan))
            channel.send(
                {"event": "answer", "seq": frame["seq"], "ok": True, "held": granted is not None}
                | ({"descriptor": True} if granted is not None else {})
            )
            if granted is not None:
                channel.send_memfd(granted)
                os.close(granted)

    def close(self) -> None:
        for fd in self.held.values():
            os.close(fd)


class Session:
    """One executor's seam to the machine."""

    def __init__(self, machine: Machine, tmp: Path) -> None:
        ours, theirs = socket.socketpair()
        self.stop = threading.Event()
        self.sockets = (ours, theirs)
        self.thread = threading.Thread(
            target=machine.serve, args=(Channel(theirs), self.stop), daemon=True
        )
        self.thread.start()
        executor = Executor(Channel(ours), tmp)
        self.tiers = HostTiers(
            ask=lambda *_: None,
            offer=lambda *_: None,
            seal=executor._seal_tier,
            prefetch=executor._prefetch_tiers,
        )

    def close(self) -> None:
        self.stop.set()
        self.sockets[0].close()
        self.thread.join()
        self.sockets[1].close()


def test_an_executor_computes_on_the_machines_sealed_layout_and_reads_no_store(
    tmp_path: Path,
) -> None:
    checkpoint, values = write_checkpoint(tmp_path)
    machine = Machine(checkpoint.root)
    first, second = Session(machine, tmp_path), Session(machine, tmp_path)
    try:
        # A CPU executor binds the layout in place and computes exactly on it.
        weights = Weights(torch, torch.device("cpu"), "cpu")
        model = load(weights, checkpoint, "sealed", tiers=first.tiers)
        component = model.backend.components["stack"]
        assert component.sealed and machine.fills == 1
        assert machine.prefetches == 1  # filled before the ask, which then found it held
        assert_pinned(weights, component, values)
        x = torch.rand(4, HIDDEN, generator=torch.Generator().manual_seed(3)).half()
        reference = Stack()
        reference.load_state_dict(values)
        with torch.inference_mode():
            expected = reference(x)
        assert same(run(model, x, passes=1), [expected])
        held = plane.stats(weights.plane).host
        assert held.fill_bytes == 0 and held.disk_read_bytes >= 0
        assert held.sealed_bytes >= component.ws.nbytes // 2, held
        assert model.backend.stats["leases"]["acquisitions"] == 0  # no store lease at all

        # A GPU executor's tier (no device here): pinned inside the machine's budget only.
        budget = 64 << 20
        gpu = GpuTier(torch, torch.device("cpu"), "cpu")
        gpu.set_budget(-1, budget)
        again = load(gpu, checkpoint, "sealed", tiers=second.tiers).backend.components["stack"]
        assert again.sealed and machine.fills == 1  # the machine's one fill serves both
        if again.filling is not None:
            again.filling.wait()
        tier = plane.stats(gpu.plane).host
        assert 0 < tier.used <= budget and tier.fill_bytes == 0, tier
        gpu.set_budget(-1, again.ws.nbytes)  # a raise pins the rest, still without a read
        assert_pinned(gpu, again, values)
        assert plane.stats(gpu.plane).host.fill_bytes == 0
    finally:
        first.close()
        second.close()
        machine.close()


def test_without_room_at_the_machine_a_weight_set_reads_the_store_itself(
    tmp_path: Path,
) -> None:
    checkpoint, values = write_checkpoint(tmp_path)
    machine = Machine(checkpoint.root, room=False)
    session = Session(machine, tmp_path)
    try:
        weights = Weights(torch, torch.device("cpu"), "cpu")
        model = load(weights, checkpoint, "private", tiers=session.tiers)
        component = model.backend.components["stack"]
        assert not component.sealed and machine.fills == 0
        assert_pinned(weights, component, values)
        assert plane.stats(weights.plane).host.fill_bytes > 0
    finally:
        session.close()
        machine.close()
