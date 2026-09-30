"""Accepted native component reads need neither a replica nor an imported model."""

from __future__ import annotations

import base64
import builtins
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import msgspec
import pytest

from cozy_runtime.internal import memory
from cozy_runtime.internal.worker import machine_lanes
from cozy_runtime.internal.worker.plan import DeclaredBinding
from cozy_runtime.internal.worker.session import Placement, Tenant, Worker
from cozy_runtime.internal.worker.workspace_executions import ExecutionRow, Preparation
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_weights_resume_native import _committed


@dataclass
class Accepted:
    row_value: ExecutionRow
    prepared_value: Preparation
    assigned: bool = True
    reads: int = 0
    cancel_at: int = 0

    def row(self, owner: str, request: str) -> ExecutionRow | None:
        assert owner == "owner" and request in {"next", "shared"}
        self.reads += 1
        if self.reads == self.cancel_at:
            self.row_value = msgspec.structs.replace(self.row_value, desired="cancel")
        return self.row_value

    def prepared(self, owner: str, request: str) -> Preparation:
        assert owner == "owner" and request in {"next", "shared"}
        return self.prepared_value

    def assigned_here(self, owner: str, request: str, ordinal: int, boot: str) -> bool:
        assert (owner, ordinal, boot) == ("owner", 1, "boot")
        return self.assigned


def queued(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Worker, Accepted, list[str]]:
    store, _, _, receipt = _committed(tmp_path)
    model = pb.Placement(
        placement_id="template", installation_id="installation", bindings_digest=b"r" * 32
    )
    model.models.add(
        id="weights",
        manifest=pb.Ref(digest=documents.raw(receipt.artifact.manifest.digest), length=1),
    )
    entry = model.entrypoints.add(name="run", entrypoint_binding_digest=b"b" * 32)
    entry.slots.add(slot="model", reference_model_id="weights").components.add(
        component="model", model_id="weights"
    )
    desired = pb.DesiredWorkerState()
    desired.placement_set.placement_set_canonical_bytes = documents.canonical_bytes(
        pb.PlacementSet(placements=[model])
    )
    prepared = Preparation(state=base64.b64encode(desired.SerializeToString()).decode())
    spec = pb.InvocationSpec()
    spec.serving.entrypoint_binding_digest = documents.spell(b"b" * 32)
    offer = pb.AttemptOffer(
        request_id="next",
        placement_id="template",
        invocation_spec_canonical_bytes=documents.canonical_bytes(spec),
    )
    accepted = Accepted(
        ExecutionRow(1, 1, "queued", "run", False, 2, offer.SerializeToString(), False), prepared
    )
    worker = Worker.__new__(Worker)
    worker.executions = cast(Any, accepted)
    worker.options = cast(Any, SimpleNamespace(tensorfs_root=str(store.root)))
    worker.fence = cast(Any, SimpleNamespace(record_owner_id="owner", worker_boot_id="boot"))
    worker.stop = threading.Event()
    # An older request is executing. Starting/preparing demand, in contrast, gets priority.
    worker.engine = cast(
        Any, SimpleNamespace(live={"older": SimpleNamespace(state="running")}, history={})
    )
    worker.memory = memory.Machine(
        {}, lambda _: None, lambda _: None, host_available=lambda: 1 << 30
    )
    worker.placement = None
    worker.hosted = {}
    worker.lanes = cast(Any, SimpleNamespace(lane_of=lambda _: None))
    worker._pages_warmed = {}
    worker.memory.optional_host_available = lambda: 1 << 30
    worker._page_warm_targets = {}
    worker._page_warm_frontier = ()
    worker._first_pages_warmed = {}
    worker._first_pages_attempted = {}
    worker._page_warm_lock = threading.Lock()
    worker._memory_prefetch_deferred = False
    worker._prefetch_reasons = {}
    template = ("installation", documents.spell(b"r" * 32))
    worker._memory_waiting = (template,)
    worker._first_use_waiting = (("next#1", template),)
    worker._memory_templates = frozenset((template,))
    worker.supervisor = cast(Any, SimpleNamespace(spawn=lambda _name, _work: None))
    notes: list[str] = []
    monkeypatch.setattr(worker, "note", lambda _topic, line: notes.append(line))
    worker._refresh_memory_queue()
    return worker, accepted, notes


def test_first_use_native_read_without_executor_then_activation_coalesces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, accepted, notes = queued(tmp_path, monkeypatch)
    original = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name.split(".")[0] != "torch", "pregrant warming imported torch"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    worker._prefetch_pages()
    assert len(notes) == 1 and "2048 B" in notes[0]
    assert not worker.memory.host_reserved and not worker.hosted
    assert accepted.row_value.desired == "run" and not accepted.row_value.dispatched
    # Scalar and group requests share immutable host bytes, not a GPU lane/grant.
    template = worker._memory_waiting[0]
    worker._first_use_waiting += (("shared#1", template),)
    worker._memory_waiting += (template,)
    worker._refresh_memory_queue()
    assert len(worker._page_warm_frontier) == 1
    worker._prefetch_pages()
    assert len(notes) == 1
    item = next(iter(worker._first_pages_warmed[template]))
    binding = DeclaredBinding(
        entrypoint_binding_digest="binding",
        entrypoint="run",
        model_class="CPUModel",
        model_binding_path="slot",
        model_parameter_name="model",
        release="proof",
        store=item[0],
        reference_snapshot=item[1],
        components=(item[2],),
    )
    worker.placement = cast(
        Placement,
        SimpleNamespace(
            placement_id="tenant",
            installation_id=template[0],
            bindings_digest=documents.raw(template[1]),
        ),
    )
    scheduled = []
    worker.supervisor = cast(
        Any, SimpleNamespace(spawn=lambda name, work: scheduled.append((name, work)))
    )
    worker._warm_pages(
        cast(Tenant, SimpleNamespace(placement=worker.placement, bindings={"binding": binding}))
    )
    for name, work in scheduled:
        if name.startswith("page-warm:"):
            work(None)
    assert len(notes) == 1  # No duplicate activation reader of the same completed first-use bytes.
    worker._memory_templates = frozenset()
    worker._prefetch_pages()
    assert not worker._first_pages_warmed


@pytest.mark.parametrize(
    "changed",
    [
        "canceled",
        "generation",
        "owner",
        "assignment",
        "queue",
        "frontier",
        "refs",
        "demand",
        "host",
        "missing",
    ],
)
def test_first_use_optional_failures_never_change_accepted_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    worker, accepted, notes = queued(tmp_path, monkeypatch)
    if changed == "canceled":
        accepted.row_value = msgspec.structs.replace(accepted.row_value, desired="cancel")
    elif changed == "generation":
        accepted.row_value = msgspec.structs.replace(accepted.row_value, ordinal=2)
    elif changed == "owner":
        worker.fence.record_owner_id = "another"
    elif changed == "assignment":
        accepted.assigned = False
    elif changed == "queue":
        worker._first_use_waiting = ()
    elif changed == "frontier":
        worker._page_warm_frontier = ()
    elif changed == "refs":
        desired = accepted.prepared_value.desired()
        placement = documents.parse(
            desired.placement_set.placement_set_canonical_bytes, pb.PlacementSet
        )
        placement.placements[0].bindings_digest = b"z" * 32
        desired.placement_set.placement_set_canonical_bytes = documents.canonical_bytes(placement)
        accepted.prepared_value = Preparation(
            state=base64.b64encode(desired.SerializeToString()).decode()
        )
    elif changed == "demand":
        worker.engine.live["older"].state = "preparing"
    elif changed == "host":
        worker.memory.host_available = lambda: 0
    elif changed == "missing":
        worker.options = cast(Any, SimpleNamespace(tensorfs_root=tmp_path / "missing-store"))
    before = accepted.row_value
    worker._prefetch_pages()
    assert not worker._first_pages_warmed
    assert not worker.memory.host_reserved
    assert accepted.row_value == before
    assert not any("page-warmed" in line for line in notes)


def test_cancel_during_native_read_preserves_bytes_and_stops_marking_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, accepted, notes = queued(tmp_path, monkeypatch)
    accepted.cancel_at = 5  # durable acceptance changes during the native stream's batch check
    worker._prefetch_pages()
    assert not worker._first_pages_warmed and not worker.memory.host_reserved
    assert accepted.row_value.desired == "cancel"
    assert any("queue changed" in line for line in notes)
    accepted.cancel_at = 0
    templates = worker._memory_templates
    worker._memory_templates = frozenset()
    worker._prefetch_pages()  # retire canceled cohort's partial-read evidence
    worker._memory_templates = templates
    accepted.row_value = msgspec.structs.replace(accepted.row_value, desired="run")
    worker._prefetch_pages()
    assert worker._first_pages_warmed and "2048 B" in notes[-1]


def test_selected_component_refs_exclude_other_entrypoints_and_keep_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, accepted, _ = queued(tmp_path, monkeypatch)
    desired = accepted.prepared_value.desired()
    rows = documents.parse(desired.placement_set.placement_set_canonical_bytes, pb.PlacementSet)
    placement = rows.placements[0]
    placement.models.add(id="adapter", manifest=pb.Ref(digest=b"a" * 32, length=1))
    placement.entrypoints[0].slots[0].adapters.add(
        component="model", model_id="adapter", source_component="delta", scale="1"
    )
    other = placement.entrypoints.add(name="other", entrypoint_binding_digest=b"x" * 32)
    other.slots.add(slot="model", reference_model_id="adapter").components.add(
        component="unselected", model_id="adapter"
    )
    desired.placement_set.placement_set_canonical_bytes = documents.canonical_bytes(rows)
    prepared = Preparation(state=base64.b64encode(desired.SerializeToString()).decode())
    targets = machine_lanes.page_targets(
        prepared,
        accepted.row_value.attempt_offer(),
        str(worker.options.tensorfs_root),
        worker._memory_waiting[0],
    )
    assert [item[2] for item in targets] == ["model", "delta"]
    assert targets[1][1] == documents.spell(b"a" * 32)


def test_accepted_broker_template_reaches_native_store_before_grant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real durable acceptance and scheduler ordering feed the reader without a replica."""
    from cozy_runtime import canonical_json
    from cozy_runtime.internal.worker.gpu_scheduler import Demand, GpuScheduler, Want
    from cozy_runtime.internal.worker.workspace import Workspace
    from cozy_runtime.internal.worker.workspace_executions import Executions

    worker, selected, notes = queued(tmp_path, monkeypatch)
    executions = Executions(Workspace(Path(str(worker.options.tensorfs_root))))
    offer = selected.row_value.attempt_offer()
    raw = offer.invocation_spec_canonical_bytes
    import hashlib

    offer.invocation_spec_digest = hashlib.sha256(raw).digest()
    offer.grant.invocation_spec_digest = offer.invocation_spec_digest
    executions.submit(
        "owner",
        "next-submission",
        b"c" * 32,
        offer,
        expected_execution_workspace_id=executions.workspace_id,
        worker_boot="boot",
        preparation=canonical_json.encode({"state": selected.prepared_value.state}),
        worker_id="worker",
    )
    worker.executions = executions
    scheduler = GpuScheduler(2)
    roots = {"older": 1, "next": 2}
    assert scheduler.sync(roots, [Demand("older#1", "older", 2), Demand("next#1", "next", 1)]) == {
        "older#1": (0, 1)
    }
    before = scheduler.view()
    template = worker._memory_waiting[0]
    worker.gpu = cast(
        Any,
        SimpleNamespace(wants={"older#1": Want("older", 2), "next#1": Want("next", 1, template)}),
    )
    worker.prespawns = cast(Any, SimpleNamespace(reconcile=lambda *_: None))
    worker._gpu_observed([], roots, before, {})
    assert worker._page_warm_frontier[0].artifact == template[1]
    worker._prefetch_pages()
    assert len(notes) == 1 and "2048 B" in notes[0]
    assert scheduler.view() == before
    row = executions.row("owner", "next")
    assert row is not None and not row.dispatched and row.desired == "run"
    assert not worker.hosted


def test_granted_first_use_survives_new_placement_until_durable_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cozy_runtime.internal.worker.gpu_scheduler import Demand
    from test_memory_prefetch_wakes import accepted_queue

    worker, notes, scheduler, _ = accepted_queue(tmp_path, monkeypatch)
    template = worker._memory_waiting[0]
    roots = {"next": 2}
    assert scheduler.sync(roots, [Demand("next#1", "next", 1)]) == {"next#1": (0,)}
    granted = scheduler.view()
    worker._gpu_observed([], roots, granted, {})
    assert worker._memory_waiting == ()
    assert list(worker._first_use_waiting) == [("next#1", template)]
    # Assignment happens before imports. No tier reports or page targets exist yet.
    worker.placement = cast(
        Placement,
        SimpleNamespace(
            placement_id="new",
            installation_id=template[0],
            bindings_digest=documents.raw(template[1]),
        ),
    )
    worker.lanes = cast(
        Any, SimpleNamespace(lane_of=lambda _: SimpleNamespace(ordinals=(0,), rows={}))
    )
    worker._refresh_memory_queue()
    frontier = worker._page_warm_frontier
    assert len(frontier) == 1 and frontier[0].tenant == "queued:next#1"
    worker._prefetch_pages()
    assert len(notes) == 1 and "2048 B" in notes[0]
    assert scheduler.view() == granted
    assert worker.executions is not None
    row = worker.executions.row("owner", "next")
    assert row is not None and not row.dispatched
    attempt = SimpleNamespace(request_id="next", attempt=1, executions=0)
    worker.engine.history[("next", 1)] = cast(Any, attempt)
    worker._dispatched(cast(Any, attempt))
    dispatched = worker.executions.row("owner", "next")
    assert dispatched is not None and dispatched.dispatched
    assert len(worker._first_use_waiting) == len(worker._page_warm_frontier) == 0
    # A scheduler observation racing the transition cannot keep running work at the
    # head of the first-use frontier. The ordinary running edge reconciles the hint.
    worker._gpu_observed([], roots, granted, {})
    attempt.executions = 1
    worker._wake_memory_prefetch()
    assert len(worker._first_use_waiting) == len(worker._page_warm_frontier) == 0
    worker._gpu_observed([], roots, granted, {})
    assert not worker._first_use_waiting
    assert scheduler.view() == granted


@pytest.mark.parametrize("phase", ["entering", "preparing", "accepted"])
@pytest.mark.parametrize("coverage", ["new", "old", "group", "job", "replaced"])
def test_only_current_covered_serving_entry_relaxes_preparation_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, coverage: str
) -> None:
    worker, _, notes = queued(tmp_path, monkeypatch)
    current = SimpleNamespace(
        alive=lambda: True,
        poisoned="",
        hello={
            "memory": {"store_read_gate", "store_read_prepare"}
            if coverage != "old"
            else {"store_read_gate"}
        },
    )
    supervision = SimpleNamespace(current=current)
    lane = SimpleNamespace(
        ordinals=(0, 1) if coverage == "group" else (0,),
        rows={
            "new": SimpleNamespace(executor=lambda: object() if coverage == "replaced" else current)
        },
    )
    worker.lanes = cast(Any, SimpleNamespace(by_id={"lane": lane}, lane_of=lambda _: None))
    attempt = SimpleNamespace(
        state=phase,
        spec={"job": {}} if coverage == "job" else {"serving": {}},
        slot=SimpleNamespace(supervision=supervision),
        lane_id="lane",
        placement_id="new",
        executor=None,
    )
    worker.engine.live = {"entering": cast(Any, attempt)}
    allowed = phase == "entering" and coverage == "new"
    assert (worker._prefetch_blocked() == "") == allowed
    worker._prefetch_pages()
    assert any("2048 B" in note for note in notes) == allowed
    if phase == "entering":
        worker.control_lock = threading.RLock()
        monkeypatch.setattr(worker.engine, "supervision_for", lambda _: supervision, raising=False)
        monkeypatch.setattr(worker, "_entry_refused", lambda _: False)
        monkeypatch.setattr(
            worker, "_enter_prepared", lambda _: worker.memory.optional_store_wanted()
        )
        attempt.request_id, attempt.attempt = "next", 1
        assert worker.enter(cast(Any, attempt)) == allowed
        assert worker.memory.optional_store_wanted()
