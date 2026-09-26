"""Real execution ancestry and retained pin history drive private GPU group policy."""

from __future__ import annotations

import base64
import hashlib
import itertools
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.internal.call_intent import canonical_intent
from cozy_runtime.internal.worker import machine_lanes
from cozy_runtime.internal.worker.machine_groups import Groups, layout
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_calls import Calls
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

OWNER = "owner"
DIGEST = "sha256:" + "1" * 64
PLACEMENT = {"installation_id": "installed", "bindings_digest": DIGEST}
MODELS = [{"sequence_parallel": {"degrees": [2, 4]}}]


def preparation(ordinals: tuple[int, ...]) -> dict[str, Any]:
    state = pb.DesiredWorkerState(
        placement_set=pb.DesiredPlacementSet(
            placement_set_canonical_bytes=canonical_json.encode(
                {
                    "format": "cozy.worker.v1.PlacementSet/1",
                    "placements": [],
                }
            ),
            device_pins=[pb.PlacementDevicePin(placement_id="placement", device_ordinals=ordinals)],
        )
    )
    return {"state": base64.b64encode(state.SerializeToString()).decode()}


def submit(executions: Executions, request: str, *, pins: tuple[int, ...] = ()) -> pb.AttemptOffer:
    spec = pb.InvocationSpec(
        installation_id="installed",
        job=pb.JobInvocationSpec(
            installation_id="installed",
            job_descriptor_id=DIGEST,
        ),
    )
    raw, identity = documents.identity(spec)
    offer = pb.AttemptOffer(
        request_id=request,
        attempt_ordinal=1,
        invocation_spec_canonical_bytes=raw,
        invocation_spec_digest=identity,
    )
    executions.submit(
        OWNER,
        request,
        b"c" * 32,
        offer,
        expected_execution_workspace_id=executions.workspace_id,
        preparation=canonical_json.encode(preparation(pins)) if pins else b"",
    )
    return offer


def running(executions: Executions, request: str) -> None:
    offer = executions.offer(OWNER, request)
    accepted = pb.AttemptAccepted(
        request_id=request,
        attempt_ordinal=offer.attempt_ordinal,
        invocation_spec_digest=offer.invocation_spec_digest,
    )
    executions.workspace.mark_running(OWNER, accepted)
    executions.observe(OWNER, pb.WorkerFrame(attempt_accepted=accepted))


def child(executions: Executions, parent: str, index: int, pins: tuple[int, ...]) -> str:
    offer = executions.offer(OWNER, parent)
    request = pb.ChildCallRequest(
        parent_request_id=parent,
        parent_attempt_ordinal=offer.attempt_ordinal,
        parent_invocation_spec_digest=offer.invocation_spec_digest,
        call_index=index,
        module="models",
        export="generate",
        request_canonical_bytes=b"{}",
    )
    request.intent_digest = hashlib.sha256(canonical_intent(request)).digest()
    call = Calls(executions.workspace).accept(OWNER, request)
    submit(executions, call.child_request, pins=pins)
    return call.child_request


def finish(executions: Executions, request: str) -> None:
    offer = executions.offer(OWNER, request)
    raw, identity = documents.identity(
        pb.AttemptOutcomeBody(
            request_id=request,
            attempt_ordinal=offer.attempt_ordinal,
            invocation_spec_digest=documents.spell(offer.invocation_spec_digest),
            status=pb.OUTCOME_STATUS_SUCCEEDED,
            result=pb.ResultEnvelope(inline_result=b"null"),
        )
    )
    outcome = pb.AttemptOutcome(
        request_id=request,
        attempt_ordinal=offer.attempt_ordinal,
        invocation_spec_digest=offer.invocation_spec_digest,
        outcome_id="result",
        outcome_canonical_bytes=raw,
        outcome_digest=identity,
    )
    executions.workspace.outcome(OWNER, outcome)
    executions.observe(OWNER, pb.WorkerFrame(attempt_outcome=outcome))


def worker(tmp_path: Path, width: int = 2) -> tuple[Worker, Groups, Executions]:
    clock = itertools.count(100)
    executions = Executions(Workspace(tmp_path / "store"), clock_ms=lambda: next(clock))
    observed: list[Any] = []
    host = cast(
        Worker,
        SimpleNamespace(
            executions=executions,
            lanes=SimpleNamespace(entries=list(range(width)), wide=width > 1, by_id={}),
            engine=SimpleNamespace(
                history={}, PRE_RELEASE=("queued", "running"), HELD=("released",)
            ),
            machine_calls=None,
            emit_progress=lambda *args: observed.append(args),
            observed=observed,
        ),
    )
    groups = Groups(host)
    host.machine_groups = groups
    for name in ("older", "newer"):
        submit(executions, name)
        running(executions, name)
    groups.refresh(OWNER)
    return host, groups, executions


def test_children_inherit_root_priority_and_brief_handoff_keeps_group(tmp_path: Path) -> None:
    host, groups, executions = worker(tmp_path)
    groups.announce(OWNER, "older", 2)
    assert groups.available(OWNER, "older", 1) == (0, 1)
    first = child(executions, "older", 0, (0, 1))
    newer = child(executions, "newer", 0, (0, 1))
    finish(executions, first)
    groups.refresh(OWNER)
    # No older GPU child exists during the Python handoff; its CPU root still runs.
    assert groups.available(OWNER, "newer", 2) == ()
    later = child(executions, "older", 1, (0, 1))
    assert executions.pending(OWNER).index(newer) < executions.pending(OWNER).index(later)
    assert groups.priority(OWNER, later) < groups.priority(OWNER, newer)
    assert not groups.admits(OWNER, newer, preparation((0, 1)))
    assert groups.admits(OWNER, later, preparation((0, 1)))
    groups.wait(OWNER, newer, "newer", 1)
    groups.wait(OWNER, newer, "newer", 1)
    assert len(cast(Any, host).observed) == 1


def test_hint_does_not_claim_gpu_but_announced_width_precedes_fanout(tmp_path: Path) -> None:
    host, groups, _ = worker(tmp_path, 4)
    groups.announce(OWNER, "older", 4)
    assert layout(groups.roots, 4) == {}
    assert groups.available(OWNER, "newer", 2) == (0, 1)
    assert groups.available(OWNER, "older", 1) == (0, 1, 2, 3)
    assert groups.available(OWNER, "newer", 2) == ()
    assert not cast(Any, host).engine.history  # Policy changes no running attempt.


def test_two_gpu_owner_leaves_third_gpu_for_singleton_work(tmp_path: Path) -> None:
    host, groups, _ = worker(tmp_path, 3)
    groups.announce(OWNER, "older", 2)
    assert groups.available(OWNER, "older", 1) == (0, 1)
    selected = machine_lanes.replica(host, OWNER, PLACEMENT, [{}], parent_request="newer")
    assert selected is not None and list(selected[1][0].device_ordinals) == [2]
    assert groups.available(OWNER, "older", 2) == (0, 1)


@pytest.mark.parametrize("hint", [False, True])
def test_singleton_children_fan_out_with_and_without_a_width_hint(
    tmp_path: Path, hint: bool
) -> None:
    host, groups, executions = worker(tmp_path, 4)
    if hint:
        groups.announce(OWNER, "older", 4)
    seen = []
    for index in range(4):
        selected = machine_lanes.replica(host, OWNER, PLACEMENT, [{}], parent_request="older")
        assert selected is not None
        pins = tuple(selected[1][0].device_ordinals)
        seen.append(pins)
        child(executions, "older", index, pins)
    assert seen == [(0,), (1,), (2,), (3,)]


def test_free_non_aligned_group_is_usable(tmp_path: Path) -> None:
    host, _, executions = worker(tmp_path, 3)
    child(executions, "older", 0, (0,))
    selected = machine_lanes.replica(
        host, OWNER, PLACEMENT, MODELS, [{"gpus": 2}], parent_request="newer"
    )
    assert selected is not None and list(selected[1][0].device_ordinals) == [1, 2]


def test_older_group_upgrade_does_not_deadlock_behind_younger_reservation(tmp_path: Path) -> None:
    _, groups, _ = worker(tmp_path, 4)
    assert groups.available(OWNER, "older", 2) == (0, 1)
    assert groups.available(OWNER, "newer", 2) == (2, 3)
    groups.announce(OWNER, "older", 4)
    assert groups.available(OWNER, "older", 4) == (0, 1, 2, 3)
    assert not groups.admits(OWNER, "newer", preparation((2, 3)))


def test_observed_transfer_can_lend_but_owner_ready_recalls_at_child_boundary(
    tmp_path: Path,
) -> None:
    host, groups, executions = worker(tmp_path)
    assert groups.available(OWNER, "older", 2) == (0, 1)
    first = child(executions, "older", 0, (0, 1))
    finish(executions, first)
    groups.refresh(OWNER)
    assert groups.available(OWNER, "newer", 2) == ()
    groups.transfer(OWNER, "older", 1)
    assert groups.available(OWNER, "newer", 2) == (0, 1)
    borrowed = child(executions, "newer", 0, (0, 1))
    running(executions, borrowed)
    groups.transfer(OWNER, "older", -1)
    assert groups.available(OWNER, "older", 2) == (0, 1)
    assert not groups.admits(OWNER, borrowed, preparation((0, 1)))
    # Recall fences future admission only. The already running child's state is unchanged.
    assert executions.status(OWNER, borrowed).state == "running"
    assert not cast(Any, host).engine.history


@pytest.mark.parametrize("action", ["pause", "cancel"])
def test_restart_recovers_affinity_and_stopped_root_releases_claim(
    tmp_path: Path, action: str
) -> None:
    host, groups, executions = worker(tmp_path)
    groups.announce(OWNER, "older", 2)
    first = child(executions, "older", 0, (0, 1))
    finish(executions, first)
    restored = Groups(host)
    restored.refresh(OWNER)
    assert restored.available(OWNER, "newer", 2) == ()
    executions.control(OWNER, "older", action, 1, cast(Any, action))
    restored.refresh(OWNER)
    assert restored.available(OWNER, "newer", 2) == (0, 1)


def test_pending_child_reservation_already_has_root_priority(tmp_path: Path) -> None:
    _, groups, executions = worker(tmp_path)
    first = child(executions, "older", 0, (0,))
    running(executions, first)
    nested = child(executions, first, 0, (0,))
    assert executions.scheduling_root(OWNER, nested) == executions.scheduling_root(OWNER, "older")
    assert groups.priority(OWNER, nested) < groups.priority(OWNER, "newer")


def test_actual_dispatch_defers_younger_topology_before_model_activation(tmp_path: Path) -> None:
    host, groups, executions = worker(tmp_path)
    groups.announce(OWNER, "older", 2)
    assert groups.available(OWNER, "older", 2) == (0, 1)
    first = child(executions, "older", 0, (0, 1))
    newer = child(executions, "newer", 0, (0, 1))
    finish(executions, first)
    wanted: list[str] = []

    def prepare(_preparation: dict[str, Any], request: str) -> bool:
        wanted.append(request)
        return False  # The CPU harness observes entry, without starting an executor.

    current = cast(Any, host)
    current.fence = SimpleNamespace(
        record_owner_recorded=True, record_owner_id=OWNER, worker_boot_id=""
    )
    current.control_lock = threading.RLock()
    current.stop = threading.Event()
    current.machine_effects = None
    current._prepare_execution = prepare
    Worker._drive_executions(host)
    assert wanted == [] and executions.status(OWNER, newer).state == "queued"
    later = child(executions, "older", 1, (0, 1))
    Worker._drive_executions(host)
    assert wanted == [later]
    assert executions.status(OWNER, newer).state == "queued"
