"""CPU tensor backings and native TensorFS reads; no accelerator or fake store."""

import os
import socket
import threading
from types import SimpleNamespace

import msgspec
import pytest

from cozy_runtime.internal import fill, memory, proctree, tiers
from cozy_runtime.internal.executor_replies import AttemptReply, decode
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.worker.child import Executor
from cozy_runtime.internal.worker.lanes import LaneSet
from cozy_runtime.internal.worker.plan import DeclaredBinding
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.protocol import documents
from test_weights_resume_native import _committed

torch = pytest.importorskip("torch")


def test_host_bytes_count_unique_backings_in_slots_and_retained_copies():
    owner = memory.Planner(torch, torch.device("cpu"))
    backing = torch.arange(32, dtype=torch.float32)
    a, b = torch.nn.Module(), torch.nn.Module()
    a.weight = torch.nn.Parameter(backing[:8])
    b.weight = torch.nn.Parameter(backing[8:16])
    first = owner.adopt("a", "model", a, None, tiers.HOST)
    second = owner.adopt("b", "model", b, None, tiers.HOST)
    # A retained cache can alias a live HOST tensor and another component's cache.
    first.part.common.host[(id(a), "weight")] = backing
    second.part.common.host[(id(b), "weight")] = backing
    assert owner.facts()["host_bytes"] == backing.untyped_storage().nbytes() == 128
    extra = torch.zeros(4)
    first.part.common.host[(id(a), "extra")] = extra
    assert owner.facts()["host_bytes"] == 144
    assert owner.document("a")["disk"] == {"a/model": 0, "b/model": 0}
    assert owner.facts()["disk_weight_bytes"] == 0


def test_cpu_drop_reports_real_store_transition_epochs():
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


def test_worker_prefetch_uses_live_disk_facts_and_keeps_warmth_on_reordering(tmp_path):
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
        scope=None,
        channel=channel,
        seal_digest="cpu-test",
    )
    row.supervision = SimpleNamespace(current=executor)
    worker = Worker.__new__(Worker)
    worker.lanes = lanes
    worker.stop = threading.Event()
    worker.engine = SimpleNamespace(live={})
    worker.memory = memory.Machine(
        {}, lambda _: None, lambda _: None, host_available=lambda: 1 << 30
    )
    worker._page_warm_targets = {}
    worker._pages_warmed = {}
    worker._page_warm_frontier = ()
    worker.placement = SimpleNamespace(
        placement_id="tenant", installation_id="installation", bindings_digest=b"a" * 32
    )
    worker.hosted = {}
    template = ("installation", documents.spell(b"a" * 32))
    worker._memory_waiting = (template,)
    # Run the real scheduled function synchronously for deterministic CPU evidence.
    worker.supervisor = SimpleNamespace(spawn=lambda name, work: work(None))
    notes = []
    worker.note = lambda topic, line: notes.append((topic, line))

    def observe():
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
            SimpleNamespace(placement=worker.placement, bindings={"binding": binding})
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
