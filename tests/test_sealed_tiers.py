"""Executors adopt the machine's sealed host layouts (`host_tiers.sealed/1`), staged in part
when they read the holes from the object files the machine hands over (`host_tiers.staged/1`).

Real: a TensorFS store, the machine's half as TensorFS does it (`tensorfs.plane.fill_sealed`:
the verified read path, then seals), the executor's own durable exchange over a seam
socketpair, and host-only planes. The executor reads no byte from the store, takes no lease,
and owns none of the layout's RAM; it computes exactly on it.
"""

from __future__ import annotations

import hashlib
import os
import socket
import sys
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
from cozy_runtime.internal import canonical, plane, spawn  # noqa: E402
from cozy_runtime.internal.executor import Executor  # noqa: E402
from cozy_runtime.internal.fill import Checkpoint, FillRefusal  # noqa: E402
from cozy_runtime.internal.parallel.group import RankGroup  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402
from cozy_runtime.internal.weights import HostTiers, Weights  # noqa: E402
from test_host_tier import GpuTier  # noqa: E402
from test_weight_plane import (  # noqa: E402
    HIDDEN,
    Stack,
    assert_pinned,
    load,
    on_gpu,
    run,
    same,
    tight,
    write_checkpoint,
)

pytestmark = pytest.mark.skipif(
    not plane.sealed_available(), reason="a TensorFS that adopts sealed host layouts"
)


class Machine:
    """The machine's half: fill each asked layout once, keep it, hand out read-only views."""

    def __init__(self, root: str, room: bool = True, staged: bool = False) -> None:
        self.store = Store.open(root)
        self.room = room
        #: its executors read holes: a prefetch opens layouts with nothing staged (no fill)
        self.staged = staged
        self.held: dict[tuple[str, str], int] = {}
        self.fills = 0
        self.prefetches = 0
        self.sourced = 0
        self.stage_asks: list[tuple[int, ...]] = []
        self.handed = 0

    @staticmethod
    def plans(frame: dict[str, Any], memfd: int) -> Any:
        body = os.pread(memfd, frame["length"], 0)
        os.close(memfd)
        assert hashlib.sha256(body).hexdigest() == frame["sha256"]
        return canonical.parse(body)

    def answer(self, doc: SealedPlan, stage: list[int] | None = None) -> int | None:
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
            regions = [list(r) for r in doc.regions]
            staged: dict[str, Any] = {} if stage is None else {"staged": stage}
            fd, _ = tfs_plane.fill_sealed(self.store, lease, plan, regions, doc.name, 2, **staged)
            self.held[key] = fd
            self.fills += 1
        return os.open(f"/proc/self/fd/{self.held[key]}", os.O_RDONLY | os.O_CLOEXEC)

    def object_files(self, doc: SealedPlan) -> list[tuple[str, int, int]]:
        """The verified object files behind a layout's plan, read-only, in SHA-256 order: its
        holes read them."""
        header = self.store.manifest(doc.manifest)["header"]
        plan = tensorfs.plan(header, list(doc.traversal), list(doc.components), window=doc.window)
        if doc.parts:
            plan = plan.select(list(doc.parts))
        with self.store.acquire_cozytensors(doc.manifest):  # verifies the closure
            objects = sorted(
                {
                    i["object"].removeprefix("sha256:"): i["object_length"]
                    for i in plan.items()
                    if i.get("source") == "object"
                }.items()
            )
        files = []
        for digest, length in objects:
            path = Path(self.store.root, "blobs", digest[:2], digest[2:4], digest)
            files.append((digest, length, os.open(path, os.O_RDONLY | os.O_CLOEXEC)))
        return files

    def source(self, frame: dict[str, Any]) -> bytes:
        """A model file, as the machine serves it (`model_source`): the header or an asset."""
        manifest, name = frame["manifest"], frame.get("name", "")
        header = self.store.manifest(manifest)["header"]
        if not name:
            return bytes(header)
        with self.store.acquire_cozytensors(manifest) as lease:
            return bytes(lease.read_asset(header, name, max_bytes=64 << 20))

    def serve(self, channel: Channel, stop: threading.Event) -> None:
        while not stop.is_set():
            frame = channel.recv()
            if frame is None:
                return
            if frame["kind"] == "model_source":
                self.sourced += 1
                body = self.source(frame)
                memfd = plane.sealed_memfd("model-source", body)
                channel.send(
                    {"event": "answer", "seq": frame["seq"], "ok": True, "held": True}
                    | {"descriptor": True, "sha256": hashlib.sha256(body).hexdigest()}
                    | {"length": len(body)}
                )
                channel.send_memfd(memfd)
                os.close(memfd)
                continue
            assert frame.get("descriptor") is True
            sent = self.plans(frame, channel.recv_memfd())
            if frame["kind"] == "sealed_prefetch":  # fill every plan now; asks then hit
                self.prefetches += 1
                for doc in [] if self.staged else msgspec.convert(sent, list[SealedPlan]):
                    fd = self.answer(doc)
                    if fd is not None:
                        os.close(fd)
                channel.send({"event": "answer", "seq": frame["seq"], "ok": True})
                continue
            if frame["kind"] == "object_files":
                files = self.object_files(msgspec.convert(sent, SealedPlan))
                self.handed += len(files)
                listed = "".join(f"{sha}:{length}\n" for sha, length, _ in files).encode()
                channel.send(
                    {"event": "answer", "seq": frame["seq"], "ok": True}
                    | {"objects_sha256": hashlib.sha256(listed).hexdigest()}
                    | {"descriptors": len(files)}
                )
                for _, _, fd in files:
                    channel._send_fd(fd, "object file")
                    os.close(fd)
                continue
            if frame["kind"] == "sealed_stage":  # no room to stage more: they read the files
                self.stage_asks.append(tuple(frame["stage_regions"]))
                channel.send({"event": "answer", "seq": frame["seq"], "ok": True, "staged": []})
                continue
            assert frame["kind"] == "sealed_tier"
            stage = frame.get("stage_regions")
            if stage is not None:
                self.stage_asks.append(tuple(stage))
            granted = self.answer(msgspec.convert(sent, SealedPlan), stage)
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

    def __init__(self, machine: Machine, tmp: Path, stage: tuple[int, ...] | None = None) -> None:
        ours, theirs = socket.socketpair()
        self.stop = threading.Event()
        self.sockets = (ours, theirs)
        self.thread = threading.Thread(
            target=machine.serve, args=(Channel(theirs), self.stop), daemon=True
        )
        self.thread.start()
        executor = self.executor = Executor(Channel(ours), tmp)
        staged = stage is not None
        self.tiers = HostTiers(
            ask=lambda *_: None,
            offer=lambda *_: None,
            seal=(lambda p: executor._seal_tier(p, stage)) if staged else executor._seal_tier,
            prefetch=executor._prefetch_tiers,
            stage=executor._stage_tier if staged else None,
            files=executor._object_files if staged else None,
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
        for ticket in again.filling:
            ticket.wait()
        tier = plane.stats(gpu.plane).host
        assert 0 < tier.used <= budget and tier.fill_bytes == 0, tier
        gpu.set_budget(-1, again.ws.nbytes)  # a raise pins the rest, still without a read
        assert_pinned(gpu, again, values)
        assert plane.stats(gpu.plane).host.fill_bytes == 0
    finally:
        first.close()
        second.close()
        machine.close()


@on_gpu
def test_a_stage_streaming_a_sealed_layout_the_card_will_not_lock_dmas_from_a_copy(
    tmp_path: Path,
) -> None:
    """A consumer GPU locks no read-only memory, so the machine's sealed layout cannot be DMA'd
    as it is. What a stage streams is copied once into page-locked memory of the executor's own
    and DMA'd from there after its first pass, bit-identical. A resident fill copies nothing.
    The executor reports the copies (`copied_bytes`): its own RAM, outside the machine's tier,
    never more than it streams."""
    checkpoint, _ = write_checkpoint(tmp_path)
    machine = Machine(checkpoint.root)
    session = Session(machine, tmp_path)
    try:
        weights = Weights(torch, torch.device("cuda", 0), "cuda")
        weights.set_budget(1 << 30, pinned=1 << 30)
        model = load(weights, checkpoint, "sealed", tiers=session.tiers)
        component = model.backend.components["stack"]
        assert component.sealed
        x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
        resident = run(model, x, passes=1)
        assert weights.facts().copied_bytes == 0
        weights.set_budget(tight(model))
        assert same(resident, run(model, x, passes=1))  # observes the order the ring streams
        model.residency.unpark()  # the attempt's end: the next stage streams through the ring
        before = weights.facts()
        assert same(resident * 4, run(model, x, passes=4))
        facts = weights.facts()
        bounced = facts.bounce_copy_bytes - before.bounce_copy_bytes
        if facts.register_failed == 0:  # this card locks read-only memory: nothing to copy
            assert facts.copied_bytes == 0 and bounced == 0, facts
            return
        assert 0 < facts.copied_bytes < component.ws.nbytes, facts
        # the first of four passes bounced while its copies were made; the rest were DMA
        assert bounced * 2 < facts.h2d_bytes - before.h2d_bytes, facts
    finally:
        session.close()
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


def test_with_model_sources_an_executor_opens_no_store(tmp_path: Path) -> None:
    """`Load.model_sources`: the header comes from the machine and the weights from its sealed
    layout. The checkpoint's store path names nothing, so any store access would refuse."""
    stored, values = write_checkpoint(tmp_path)
    machine = Machine(stored.root)
    session = Session(machine, tmp_path)
    try:
        manifest = stored.manifest_id
        checkpoint = Checkpoint(
            tmp_path / "no-store-here",
            manifest,
            sourced=lambda name: session.executor._model_source(manifest, name),
        )
        assert checkpoint.store is None and checkpoint.header_bytes == stored.header_bytes
        assert checkpoint.model_assets() == stored.model_assets()
        with pytest.raises(FillRefusal, match="reads no store"):
            checkpoint.acquire()
        weights = Weights(torch, torch.device("cpu"), "cpu")
        model = load(weights, checkpoint, "sourced", tiers=session.tiers)
        assert model.backend.components["stack"].sealed and machine.sourced >= 1
        assert_pinned(weights, model.backend.components["stack"], values)
        x = torch.rand(4, HIDDEN, generator=torch.Generator().manual_seed(5)).half()
        reference = Stack()
        reference.load_state_dict(values)
        with torch.inference_mode():
            expected = reference(x)
        assert same(run(model, x, passes=1), [expected])
        assert model.backend.stats["leases"]["acquisitions"] == 0
    finally:
        session.close()
        machine.close()


def test_a_groups_follower_asks_the_machine_through_rank_0_and_opens_no_store(
    tmp_path: Path,
) -> None:
    """A real follower process loads as rank 0 does: its header, assets and sealed layouts are
    its own requests to the machine, relayed on rank 0's seam. Its store path names nothing.
    The machine fills each layout once for both GPUs, and a follower refining later, on its
    own, is served the same way."""
    stored, values = write_checkpoint(tmp_path)
    machine = Machine(stored.root)
    session = Session(machine, tmp_path)
    leader: Any = session.executor
    fixture = Path(__file__).parent / "testdata" / "sealed_follower.py"

    def launch(module_argv: list[str], fd: int) -> spawn.Child:
        return spawn.spawn_follower(
            python=sys.executable,
            module_argv=[str(fixture), *module_argv[2:]],
            inherit_fd=fd,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )

    group = RankGroup(
        degree=2, backend="cpu:gloo", root=str(tmp_path), launch=launch, relay=leader._relay
    )
    manifest = stored.manifest_id
    try:
        group.spawn()
        # Both GPUs load at once: rank 0's own requests and the relayed ones share its seam.
        group.send_one(1, {"cmd": "adopt", "manifest": manifest, "construction": "group"})
        checkpoint = Checkpoint(
            tmp_path / "no-store-here",
            manifest,
            sourced=lambda name: leader._model_source(manifest, name),
        )
        weights = Weights(torch, torch.device("cpu"), "cpu")
        model = load(weights, checkpoint, "group", tiers=leader._tiers(False, True))
        assert model.backend.components["stack"].sealed
        x = torch.rand(4, HIDDEN, generator=torch.Generator().manual_seed(3)).half()
        reference = Stack()
        reference.load_state_dict(values)
        with torch.inference_mode():
            expected = hashlib.sha256(reference(x).numpy().tobytes()).hexdigest()
        reply = group.reply_one(1, "adopt")
        assert reply["ok"] and not group.broken, (reply, group.broken)
        assert reply["store"] is False and reply["leases"] == 0
        assert reply["assets"] == len(stored.model_assets())
        assert reply["sealed"] == [True, True] and reply["refined"] is True
        assert reply["outputs"] == [expected, expected]
        assert reply["refused"] is True  # only the machine's layouts and files are relayed
        # one copy of each layout for both GPUs: the coarse one, and the follower's finer one
        assert machine.fills == 2 and len(machine.held) == 2
        assert_pinned(weights, model.backend.components["stack"], values)
    finally:
        group.close()
        session.close()
        machine.close()


@pytest.mark.skipif(not plane.sealed_holes(), reason="a TensorFS that reads holes from files")
def test_a_layout_staged_in_part_pins_only_what_was_staged_and_reads_no_store(
    tmp_path: Path,
) -> None:
    """`host_tiers.staged/1`: a GPU executor asks for its layout with nothing staged (the
    machine stages only what streams), or with some regions staged. The machine hands over the
    verified object files for the holes; the executor pins only staged regions, fills nothing
    itself and takes no lease."""
    checkpoint = write_checkpoint(tmp_path)[0]
    machine = Machine(checkpoint.root, staged=True)
    budget = 64 << 20
    nothing, some = Session(machine, tmp_path, stage=()), Session(machine, tmp_path, stage=(0,))
    try:
        gpu = GpuTier(torch, torch.device("cpu"), "cpu")
        gpu.set_budget(-1, budget)
        model = load(gpu, checkpoint, "holes", tiers=nothing.tiers)
        component = model.backend.components["stack"]
        assert component.sealed and component.staging is not None
        assert machine.stage_asks == [()] and machine.handed > 0
        for ticket in component.filling:
            ticket.wait()
        tier = plane.stats(gpu.plane).host
        assert tier.used == 0 and tier.fill_bytes == 0, tier  # holes: nothing here to pin
        assert model.backend.stats["leases"]["acquisitions"] == 0

        # Streaming blocks are asked for as stages plan them; this machine has no room left.
        plan, tiers = component.staging
        assert tiers.stage is not None and tiers.stage(plan, [1, 2]) == []
        assert machine.stage_asks == [(), (1, 2)]

        part = GpuTier(torch, torch.device("cpu"), "cpu")
        part.set_budget(-1, budget)
        again = load(part, checkpoint, "part", tiers=some.tiers).backend.components["stack"]
        assert again.sealed and machine.stage_asks[2:] == [(0,)]
        for ticket in again.filling:
            ticket.wait()
        tier = plane.stats(part.plane).host
        assert tier.used == again.regions[0].span and tier.fill_bytes == 0, tier
    finally:
        nothing.close()
        some.close()
        machine.close()
