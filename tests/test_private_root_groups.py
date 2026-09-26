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
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
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


def finish(executions: Executions, request: str, status: int = pb.OUTCOME_STATUS_SUCCEEDED) -> None:
    offer = executions.offer(OWNER, request)
    raw, identity = documents.identity(
        pb.AttemptOutcomeBody(
            request_id=request,
            attempt_ordinal=offer.attempt_ordinal,
            invocation_spec_digest=documents.spell(offer.invocation_spec_digest),
            status=status,
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


def test_singleton_parent_keeps_its_gpu_across_children_and_restart(tmp_path: Path) -> None:
    host, groups, executions = worker(tmp_path)
    selected = machine_lanes.replica(host, OWNER, PLACEMENT, [{}], parent_request="older")
    assert selected is not None and list(selected[1][0].device_ordinals) == [0]
    first = child(executions, "older", 0, (0,))
    finish(executions, first)
    groups.refresh(OWNER)
    assert groups.available(OWNER, "newer", 1) == (1,)
    assert not groups.admits(OWNER, "newer", preparation((0,)))
    restored = Groups(host)
    restored.refresh(OWNER)
    assert restored.available(OWNER, "newer", 1) == (1,)
    later = child(executions, "older", 1, (0,))
    assert restored.admits(OWNER, later, preparation((0,)))


def test_one_device_worker_keeps_parent_priority_through_child_handoff(tmp_path: Path) -> None:
    host, groups, executions = worker(tmp_path, 1)
    selected = machine_lanes.replica(host, OWNER, PLACEMENT, [{}], parent_request="older")
    assert selected is not None and list(selected[1][0].device_ordinals) == [0]
    first = child(executions, "older", 0, (0,))
    finish(executions, first)
    groups.refresh(OWNER)
    assert machine_lanes.replica(host, OWNER, PLACEMENT, [{}], parent_request="newer") is None
    later = child(executions, "older", 1, (0,))
    assert groups.admits(OWNER, later, preparation((0,)))
    assert not groups.admits(OWNER, "newer", preparation((0,)))
    assert groups.admits(OWNER, "newer", {})  # CPU orchestration remains independent.
    finish(executions, later)
    finish(executions, "older")
    groups.refresh(OWNER)
    assert machine_lanes.replica(host, OWNER, PLACEMENT, [{}], parent_request="newer") is not None


def test_one_device_restart_recovers_previously_unpinned_model_children(tmp_path: Path) -> None:
    host, _, executions = worker(tmp_path, 1)
    state = machine_lanes.state(preparation(()))
    state.placement_set.placement_set_canonical_bytes = canonical_json.encode(
        {"placements": [{"models": [{"id": "model"}]}]}
    )
    retained = {"state": base64.b64encode(state.SerializeToString()).decode()}
    first = child(executions, "older", 0, ())
    with executions.workspace.locked() as db:
        db.execute(
            "UPDATE executions SET preparation=? WHERE request=?",
            (canonical_json.encode(retained), first),
        )
    finish(executions, first)
    restored = Groups(host)
    restored.refresh(OWNER)
    assert not restored.admits(OWNER, "newer", retained)
    assert restored.admits(OWNER, "older", retained)


def test_no_hint_fanout_recovers_all_used_devices(tmp_path: Path) -> None:
    host, _, executions = worker(tmp_path, 3)
    children = []
    for index in range(3):
        selected = machine_lanes.replica(host, OWNER, PLACEMENT, [{}], parent_request="older")
        assert selected is not None
        children.append(child(executions, "older", index, tuple(selected[1][0].device_ordinals)))
    for request in children:
        finish(executions, request)
    restored = Groups(host)
    restored.refresh(OWNER)
    assert restored.available(OWNER, "newer", 1) == ()
    assert restored.available(OWNER, "older", 1) == (0, 1, 2)


def test_singleton_fanout_does_not_outgrow_capacity_reserved_by_older_root(tmp_path: Path) -> None:
    host, groups, executions = worker(tmp_path, 3)
    groups.announce(OWNER, "older", 2)
    assert groups.available(OWNER, "older", 2) == (0, 1)
    for index in range(2):
        selected = machine_lanes.replica(host, OWNER, PLACEMENT, [{}], parent_request="newer")
        assert selected is not None and list(selected[1][0].device_ordinals) == [2]
        child(executions, "newer", index, (2,))
    submit(executions, "youngest")
    assert groups.available(OWNER, "youngest", 1) == ()


def test_group_growth_preserves_selected_device_when_another_gpu_is_busy(tmp_path: Path) -> None:
    host, groups, executions = worker(tmp_path, 3)
    # Existing work makes the older root initially select the middle GPU.
    child(executions, "newer", 0, (0,))
    selected = machine_lanes.replica(host, OWNER, PLACEMENT, [{}], parent_request="older")
    assert selected is not None and list(selected[1][0].device_ordinals) == [1]
    first = child(executions, "older", 0, (1,))
    selected = machine_lanes.replica(host, OWNER, PLACEMENT, [{}], parent_request="older")
    assert selected is not None and list(selected[1][0].device_ordinals) == [2]
    assert groups.admits(OWNER, first, preparation((1,)))


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


def test_transient_child_reservations_fan_out_before_durable_submit(tmp_path: Path) -> None:
    """Concurrent serving calls reserve lanes before their execution rows exist.

    ``machine_serving`` selects a placement and only then journals ``Executions.submit``.
    A drive tick can therefore admit several siblings while the durable queue still looks
    empty.  The affinity owner must account for those short-lived selections exactly as it
    accounts for retained pins, or every sibling lands on ordinal zero.
    """
    host, groups, _ = worker(tmp_path, 4)
    selected: list[tuple[int, ...]] = []
    for index in range(4):
        replica = machine_lanes.replica(host, OWNER, PLACEMENT, [{}], parent_request="older")
        assert replica is not None
        pins = tuple(replica[1][0].device_ordinals)
        selected.append(pins)
        groups.reserve(OWNER, "older", f"transient-{index}", pins)
    assert selected == [(0,), (1,), (2,), (3,)]
    for index in range(4):
        groups.release_reservation(OWNER, f"transient-{index}")


def test_transient_singletons_do_not_change_later_group_degree(tmp_path: Path) -> None:
    """Reference replicas fan out, then a sequence-parallel child still gets its group."""
    host, groups, _ = worker(tmp_path, 4)
    for index in range(3):
        selected = machine_lanes.replica(host, OWNER, PLACEMENT, [{}], parent_request="older")
        assert selected is not None
        pins = tuple(selected[1][0].device_ordinals)
        assert pins == ((0,), (1,), (2,))[index]
        groups.reserve(OWNER, "older", f"reference-{index}", pins)

    h3 = machine_lanes.replica(
        host,
        OWNER,
        PLACEMENT,
        MODELS,
        [{"gpus": 4}],
        parent_request="older",
    )
    assert h3 is not None
    assert tuple(h3[1][0].device_ordinals) == (0, 1, 2, 3)


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


def test_impossible_retained_pin_refuses_instead_of_waiting_forever(tmp_path: Path) -> None:
    _, groups, _ = worker(tmp_path, 2)
    with pytest.raises(WorkspaceRefusal, match="granted device pool"):
        groups.admits(OWNER, "older", preparation((0, 2)))


@pytest.mark.parametrize("status", [pb.OUTCOME_STATUS_SUCCEEDED, pb.OUTCOME_STATUS_FAILED])
def test_finished_root_does_not_keep_a_group_for_its_cached_models(
    tmp_path: Path, status: int
) -> None:
    _, groups, executions = worker(tmp_path)
    assert groups.available(OWNER, "older", 2) == (0, 1)
    first = child(executions, "older", 0, (0, 1))
    finish(executions, first)
    finish(executions, "older", status)
    groups.refresh(OWNER)
    assert groups.available(OWNER, "newer", 2) == (0, 1)


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
