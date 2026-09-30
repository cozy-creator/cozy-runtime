"""CPU tensor backings and native TensorFS reads; no accelerator or fake store."""

import os
import socket
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import msgspec
import pytest

from cozy_runtime.internal import fill, memory, proctree, tiers
from cozy_runtime.internal.executor_replies import AttemptReply, decode
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.worker.attempts import AttemptEngine
from cozy_runtime.internal.worker.child import Executor, ExecutorSupervision
from cozy_runtime.internal.worker.lanes import LaneSet
from cozy_runtime.internal.worker.plan import DeclaredBinding
from cozy_runtime.internal.worker.session import Placement, Tenant, Worker
from cozy_runtime.internal.worker.supervisor import Supervisor
from cozy_runtime.protocol import documents
from test_weights_resume_native import _committed

torch = pytest.importorskip("torch")


def test_host_bytes_count_unique_backings_in_slots_and_retained_copies() -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    backing = torch.arange(32, dtype=torch.float32)
    a, b = torch.nn.Module(), torch.nn.Module()
    a.weight = torch.nn.Parameter(backing[:8])
    b.weight = torch.nn.Parameter(backing[8:16])
    root = torch.nn.Module()
    root.a, root.b = a, b
    first = owner.adopt("a", "model", root, None, tiers.HOST)
    # Shared storage belongs to one managed component, including retained copies.
    first.part.common.host[(id(a), "weight")] = backing
    first.part.common.host[(id(b), "weight")] = backing
    assert owner.facts()["host_bytes"] == backing.untyped_storage().nbytes() == 128
    extra = torch.zeros(4)
    first.part.common.host[(id(a), "extra")] = extra
    assert owner.facts()["host_bytes"] == 144
    assert owner.document("a")["disk"] == {"a/model": 0}
    assert owner.facts()["disk_weight_bytes"] == 0


def test_cpu_drop_reports_real_store_transition_epochs() -> None:
    owner = memory.Planner(torch, torch.device("cpu"))
    root = torch.nn.Module()
    root.weight = torch.nn.Parameter(torch.ones(8))
    # Exercise the real fill-plane eviction with a completed CPU component. No
    # reader or GPU state is needed to drop its backing while retaining its schema.
    backend = fill.StreamingFillBackend.__new__(fill.StreamingFillBackend)
    backend.torch = torch
    backend.components = {"model": root}
    backend.parked = {}
    backend.replaced = {}
    backend.stage_log = []
    backend.component_memory = {"model": fill.ComponentMemory(destinations=32)}
    held = owner.adopt("generation", "model", root, None, tiers.HOST)
    held.part.store = backend
    assert owner.facts()["host_bytes"] == 32
    tiers.to_disk(torch, held.part, None)
    assert root.weight.is_meta
    assert owner.facts()["host_bytes"] == 0
    assert owner.document("generation")["disk"] == {"generation/model": 32}
    assert owner.document("generation")["disk_epoch"] == {"generation/model": 1}
    tiers.to_disk(torch, held.part, None)
    assert held.part.disk_epoch == 1  # Already on disk is no new transition.
    root.to_empty(device="cpu")
    backend.components["model"] = root
    tiers.to_host(torch, held.part.common)
    assert owner.facts()["host_bytes"] == 32
    tiers.to_disk(torch, held.part, None)
    assert owner.document("generation")["disk_epoch"] == {"generation/model": 2}


def test_prefetch_cannot_clear_a_replacement_executors_memory_facts() -> None:
    lanes = LaneSet.from_envelope("0", worker_pid=os.getpid())
    lane = lanes.lanes[0]
    lane.placements.add("tenant")
    row = lane.row("tenant")
    ours, theirs = socket.socketpair()
    channel = Channel(ours)
    executor = Executor(
        epoch=2,
        process=proctree.process_identity(os.getpid()),
        scope=proctree.ProcessTreeScope("prefetch-proof", os.getuid()),
        channel=channel,
        seal_digest="cpu-test",
    )
    row.supervision = cast(ExecutorSupervision, SimpleNamespace(current=executor))
    comparing = threading.Event()
    published = threading.Event()

    class PreviousGeneration(tuple[int, ...]):
        def __ne__(self, other: object) -> bool:
            different = tuple(self) != other
            if threading.current_thread().name == "prefetch-race":
                comparing.set()
                assert published.wait(5), "replacement facts were not published"
            return different

    row.memory_generation = cast(tuple[int, int], PreviousGeneration((executor.pid, 1)))
    item = ("store", "manifest", "model")
    worker = Worker.__new__(Worker)
    worker._first_use_waiting = ()
    worker._memory_templates = frozenset()
    worker._first_pages_warmed = {}
    worker._first_pages_attempted = {}
    worker._page_warm_lock = threading.Lock()
    worker.hosted = {}
    worker.placement = None
    worker.lanes = lanes
    worker._page_warm_targets = {"tenant": {item: ("model",)}}
    demand = memory.Demand("tenant", "representation", "artifact", (0,), 0, None)
    result = []
    errors = []

    def prefetch() -> None:
        try:
            result.append(worker._prefetch_tier(demand, item))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=prefetch, name="prefetch-race")
    try:
        thread.start()
        assert comparing.wait(5), "prefetch did not inspect the replacement generation"
        Worker._observe_memory(
            lane,
            row,
            {
                "rank_memory": {"0": {"reserved_bytes": 64, "overhead_bytes": 8}},
                "residency": {"disk": {"model": 32}, "disk_epoch": {"model": 1}},
            },
        )
        row.peak[("run", "shape")] = {0: 96}
        published.set()
        thread.join(5)
        assert not thread.is_alive()
        assert not errors
        assert result == [None]  # Old observations cannot authorize speculative work.
        assert row.held == {0: 72}
        assert row.overhead == {0: 8}
        assert row.peak == {("run", "shape"): {0: 96}}
        assert row.disk == {"model": 32}
        assert row.disk_epoch == {"model": 1}
        assert row.disk_revision == {"model": 1}
    finally:
        published.set()
        thread.join(5)
        channel.close()
        theirs.close()


def test_worker_prefetch_uses_live_disk_facts_and_keeps_warmth_on_reordering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _, _, receipt = _committed(tmp_path)
    item = (str(store.root), receipt.artifact.manifest.digest, "model")
    binding = DeclaredBinding(
        entrypoint_binding_digest="binding",
        entrypoint="run",
        model_class="CPUModel",
        model_binding_path="slot",
        model_parameter_name="model",
        release="cpu-proof",
        store=item[0],
        reference_snapshot=item[1],
        components=("model",),
    )
    construction = binding.construction_key()
    model_key = f"{construction}#slot"
    owner = memory.Planner(torch, torch.device("cpu"))
    root = torch.nn.Module()
    root.weight = torch.nn.Parameter(torch.empty(512, device="meta"))
    owner.adopt(model_key, "model", root, None, tiers.DISK)
    key = f"{model_key}/model"
    lanes = LaneSet.from_envelope("0", worker_pid=os.getpid())
    lane = lanes.lanes[0]
    lane.placements.add("tenant")
    row = lane.row("tenant")
    ours, theirs = socket.socketpair()
    channel = Channel(ours)
    executor = Executor(
        epoch=1,
        process=proctree.process_identity(os.getpid()),
        scope=proctree.ProcessTreeScope("prefetch-proof", os.getuid()),
        channel=channel,
        seal_digest="cpu-test",
    )
    row.supervision = cast(ExecutorSupervision, SimpleNamespace(current=executor))
    worker = Worker.__new__(Worker)
    worker._first_use_waiting = ()
    worker._memory_templates = frozenset()
    worker._first_pages_warmed = {}
    worker._first_pages_attempted = {}
    worker._page_warm_lock = threading.Lock()
    worker.hosted = {}
    worker.placement = None
    worker.lanes = lanes
    worker.stop = threading.Event()
    worker.engine = cast(AttemptEngine, SimpleNamespace(live={}))
    worker.memory = memory.Machine(
        {}, lambda _: None, lambda _: None, host_available=lambda: 1 << 30
    )
    worker.memory.optional_host_available = lambda: 1 << 30
    worker._page_warm_targets = {}
    worker._pages_warmed = {}
    worker._page_warm_frontier = ()
    worker.placement = cast(
        Placement,
        SimpleNamespace(
            placement_id="tenant", installation_id="installation", bindings_digest=b"a" * 32
        ),
    )
    worker.hosted = {}
    template = ("installation", documents.spell(b"a" * 32))
    worker._memory_waiting = (template,)
    # Run the real scheduled function synchronously for deterministic CPU evidence.
    worker.supervisor = cast(Supervisor, SimpleNamespace(spawn=lambda name, work: work(None)))
    notes = []
    monkeypatch.setattr(worker, "note", lambda topic, line: notes.append((topic, line)))

    def observe() -> None:
        reply = decode(
            msgspec.json.decode(
                msgspec.json.encode(
                    {"ok": True, "memory": owner.facts(), "residency": owner.document(construction)}
                )
            ),
            AttemptReply,
        )
        Worker._observe_memory(lane, row, {"memory": reply.memory, "residency": reply.residency})

    try:
        observe()
        worker._warm_pages(
            cast(Tenant, SimpleNamespace(placement=worker.placement, bindings={"binding": binding}))
        )
        assert worker._page_warm_targets == {"tenant": {item: (key,)}}
        assert len(notes) == 1 and "2048 B" in notes[-1][1]
        assert not worker.memory.host_reserved
        # Only queue position and measured preparation time change.
        worker._memory_waiting = (("unprepared", "other"), template)
        row.load_seconds = 99
        worker._refresh_memory_queue()
        assert len(notes) == 1
        # New HOST facts suppress TensorFS even after another queue change.
        hosted = torch.nn.Module()
        hosted.weight = torch.nn.Parameter(torch.zeros(512))
        owner.adopt(model_key, "model", hosted, None, tiers.HOST)
        observe()
        worker._refresh_memory_queue()
        assert len(notes) == 1 and owner.facts()["host_bytes"] == 2048
        # A genuine observed return to DISK invalidates the prior completed warm.
        owner.adopt(model_key, "model", root, None, tiers.DISK)
        observe()
        worker._refresh_memory_queue()
        assert len(notes) == 2
        # Paging between observations can end at the same DISK byte count.
        owner.held[key].part.disk_epoch += 1
        observe()
        worker._refresh_memory_queue()
        assert len(notes) == 3
        # Same byte count on a new process epoch is not old generation evidence.
        executor.epoch = 2
        worker._refresh_memory_queue()
        assert len(notes) == 3
        observe()
        worker._refresh_memory_queue()
        assert len(notes) == 4
        # Old peers' missing tier facts only disable speculative work.
        Worker._observe_memory(lane, row, {"memory": {"reserved_bytes": 0}})
        worker._refresh_memory_queue()
        assert len(notes) == 4 and not worker.memory.host_reserved
    finally:
        channel.close()
        theirs.close()


def test_activation_warms_native_store_once_before_any_load_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _, _, receipt = _committed(tmp_path)
    binding = DeclaredBinding(
        entrypoint_binding_digest="binding",
        entrypoint="run",
        model_class="CPUModel",
        model_binding_path="slot",
        model_parameter_name="model",
        release="cpu-proof",
        store=str(store.root),
        reference_snapshot=receipt.artifact.manifest.digest,
        components=("model",),
    )
    worker = Worker.__new__(Worker)
    worker._first_use_waiting = ()
    worker._memory_templates = frozenset()
    worker._first_pages_warmed = {}
    worker._first_pages_attempted = {}
    worker._page_warm_lock = threading.Lock()
    worker.hosted = {}
    worker.placement = None
    worker.stop = threading.Event()
    worker.memory = memory.Machine(
        {}, lambda _: None, lambda _: None, host_available=lambda: 1 << 30
    )
    worker.lanes = cast(LaneSet, SimpleNamespace(lane_of=lambda _: None))
    worker.memory.optional_host_available = lambda: 1 << 30
    worker._page_warm_targets = {}
    monkeypatch.setattr(worker, "_refresh_memory_queue", lambda: None)
    scheduled = []
    worker.supervisor = cast(
        Supervisor, SimpleNamespace(spawn=lambda name, work: scheduled.append((name, work)))
    )
    notes = []
    monkeypatch.setattr(worker, "note", lambda topic, line: notes.append((topic, line)))
    tenant = SimpleNamespace(
        placement=SimpleNamespace(placement_id="tenant"), bindings={"a": binding}
    )
    worker._warm_pages(cast(Tenant, tenant))
    worker._warm_pages(cast(Tenant, tenant))
    assert len(scheduled) == 1  # No queue, executor, tier report or CUDA context exists.
    scheduled[0][1](None)
    assert len(notes) == 1 and "2048 B" in notes[0][1]
    assert not worker.memory.host_reserved
    # Retiring the construction cancels a queued read before it opens the store.
    worker._page_warm_targets.clear()
    scheduled[0][1](None)
    assert len(notes) == 1
