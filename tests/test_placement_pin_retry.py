"""One real worker retries corrected placement pins without rebuilding unchanged failures."""

from __future__ import annotations

import pytest

from cozy_runtime.internal.local import LocalRecordOwner
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import _slice, _workspace, needs_executor


@needs_executor
@pytest.mark.parametrize("corrected", [None, (1,)])
def test_failed_group_pin_then_same_document_corrected_pin_serves(
    corrected: tuple[int, ...] | None,
) -> None:
    workers: list[Worker] = []
    sent: list[int] = []
    bad_attempts: list[int] = []

    def owner_hook(owner: LocalRecordOwner) -> None:
        desired = owner._desired_state
        observe = owner._on_observed_state
        initial: pb.DesiredWorkerState | None = None
        stage = 0

        def first() -> pb.DesiredWorkerState:
            nonlocal initial
            initial = desired()
            assert owner.placement is not None
            initial.placement_set.device_pins.append(
                pb.PlacementDevicePin(
                    placement_id=owner.placement.placement_id, device_ordinals=[0, 1]
                )
            )
            return initial

        def report(report: pb.ObservedWorkerState) -> None:
            nonlocal stage
            worker = workers[0]
            if (
                worker.placement is not None
                and worker.placement.materialization == pb.MATERIALIZATION_STATE_FAILED
            ):
                if stage < 2 and report.accepted_desired_state_revision == stage + 1:
                    assert initial is not None
                    bad_attempts.append(
                        sum("is materializing" in event.step for event in worker.activity)
                    )
                    revision = stage + 2
                    next_state = pb.DesiredWorkerState()
                    next_state.CopyFrom(initial)
                    next_state.revision = revision
                    if stage == 1:
                        next_state.placement_set.ClearField("device_pins")
                        if corrected is not None:
                            next_state.placement_set.device_pins.append(
                                pb.PlacementDevicePin(
                                    placement_id=worker.placement.placement_id,
                                    device_ordinals=corrected,
                                )
                            )
                    sent.append(revision)
                    stage += 1
                    owner._send(pb.RecordOwnerFrame(desired_state=next_state))
                return
            if (
                stage == 2
                and worker.placement is not None
                and worker.placement.serving == pb.SERVING_STATE_DISPATCHABLE
            ):
                # The owner has explicitly superseded the failed revision; historical
                # fault events do not prevent a new dispatchable placement from serving.
                clean = pb.ObservedWorkerState()
                clean.CopyFrom(report)
                clean.ClearField("faults")
                for placement in clean.placements:
                    placement.ClearField("faults")
                observe(clean)

        owner._desired_state = first  # type: ignore[method-assign]
        owner._on_observed_state = report  # type: ignore[method-assign]

    with _workspace() as root:
        outcome, seals = _slice(
            root, devices="0,1", kind="entrypoint", hook=workers.append, owner_hook=owner_hook
        )
    assert sent == [2, 3]
    assert outcome.status == "OUTCOME_STATUS_SUCCEEDED", outcome.error
    assert outcome.result is not None
    assert workers[0].accepted.accepted_desired_state_revision == 3
    assert workers[0].placement is not None
    assert workers[0].placement.lane_id == ("lane-0" if corrected is None else "lane-1")
    assert seals[-1] == ("0" if corrected is None else "1")
    assert bad_attempts[0] > 0
    assert bad_attempts[0] == bad_attempts[1], (
        "unchanged invalid pins triggered another materialization"
    )


@needs_executor
@pytest.mark.parametrize("hosted", [False, True])
def test_changed_pin_reassigns_live_or_failed_hosted_placement(hosted: bool) -> None:
    workers: list[Worker] = []
    sent: list[int] = []
    before: list[int] = []

    def owner_hook(owner: LocalRecordOwner) -> None:
        desired = owner._desired_state
        observe = owner._on_observed_state
        initial: pb.DesiredWorkerState | None = None
        stage = 0

        def first() -> pb.DesiredWorkerState:
            nonlocal initial
            from cozy_runtime.protocol import documents

            initial = desired()
            assert owner.placement is not None
            placements = [owner.placement]
            if hosted:
                sibling = pb.Placement()
                sibling.CopyFrom(owner.placement)
                sibling.placement_id = "hosted"
                placements.append(sibling)
                raw, digest = documents.identity(pb.PlacementSet(placements=placements))
                initial.placement_set.placement_set_canonical_bytes = raw
                initial.placement_set.placement_set_digest = digest
            initial.placement_set.device_pins.append(
                pb.PlacementDevicePin(
                    placement_id="hosted" if hosted else owner.placement.placement_id,
                    device_ordinals=[0, 1] if hosted else [0],
                )
            )
            return initial

        def report(report: pb.ObservedWorkerState) -> None:
            nonlocal stage
            worker = workers[0]
            target = worker.hosted.get("hosted") if hosted else None
            ready_for_change = (
                target is not None
                and target.placement.materialization == pb.MATERIALIZATION_STATE_FAILED
                if hosted
                else worker.placement is not None
                and worker.placement.serving == pb.SERVING_STATE_DISPATCHABLE
            )
            if stage == 0 and ready_for_change and report.accepted_desired_state_revision == 1:
                assert initial is not None
                if not hosted:
                    assert worker.supervision.current is not None
                    before.append(worker.supervision.current.pid)
                fixed = pb.DesiredWorkerState()
                fixed.CopyFrom(initial)
                fixed.revision = 2
                fixed.placement_set.device_pins[0].device_ordinals[:] = [1]
                stage = 1
                sent.append(2)
                owner._send(pb.RecordOwnerFrame(desired_state=fixed))
                return
            target_ready = (
                worker.hosted.get("hosted") is not None
                and worker.hosted["hosted"].placement.serving == pb.SERVING_STATE_DISPATCHABLE
                if hosted
                else worker.placement is not None
                and worker.placement.lane_id == "lane-1"
                and worker.placement.serving == pb.SERVING_STATE_DISPATCHABLE
            )
            if stage == 1 and target_ready:
                clean = pb.ObservedWorkerState()
                clean.CopyFrom(report)
                clean.ClearField("faults")
                for placement in clean.placements:
                    placement.ClearField("faults")
                observe(clean)

        owner._desired_state = first  # type: ignore[method-assign]
        owner._on_observed_state = report  # type: ignore[method-assign]

    with _workspace() as root:
        outcome, seals = _slice(
            root, devices="0,1", kind="entrypoint", hook=workers.append, owner_hook=owner_hook
        )
    assert sent == [2]
    assert outcome.status == "OUTCOME_STATUS_SUCCEEDED", outcome.error
    worker = workers[0]
    if hosted:
        assert worker.hosted["hosted"].placement.lane_id == "lane-1"
    else:
        assert seals == ["0", "1"]
        assert worker.supervision.spawns == 2
