"""Managed placement uses local authority and reports terminal preparation failure."""

from __future__ import annotations

import base64
import os

import pytest

from cozy_runtime.internal import accel
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.lanes import LaneRefusal, LaneSet
from cozy_runtime.internal.worker.machine_serving import device_pins
from cozy_runtime.internal.worker.session import Worker, WorkerOptions, desired_state_body
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import _config, _workspace
from test_group_lanes import _placement


def test_managed_h3_degree_joins_existing_lane_admission() -> None:
    lanes = LaneSet.from_envelope("0,1", worker_pid=os.getpid())
    models = [{"sequence_parallel": {"degrees": [2, 4]}}]
    measured = {index: accel.DeviceMemory("measured", 6 << 30, 8 << 30) for index in range(2)}
    # The former managed path omitted this pin and failed at the real lane boundary.
    with pytest.raises(LaneRefusal, match="carries no device pin"):
        lanes.assign("h3", model_bearing=True, declared_bytes=1, measured=measured, degrees=(2, 4))
    pins = device_pins(lanes, "h3", models)
    lane = lanes.assign(
        "h3",
        model_bearing=True,
        declared_bytes=1,
        measured=measured,
        degrees=(2, 4),
        pin=tuple(pins[0].device_ordinals),
    )
    assert lane.ordinals == (0, 1) and lane.group
    assert tuple(device_pins(lanes, "h3", models)[0].device_ordinals) == lane.ordinals
    # Local ownership does not authorize stealing those devices from a live placement.
    with pytest.raises(LaneRefusal, match="only empty devices"):
        lanes.assign(
            "another",
            model_bearing=True,
            declared_bytes=1,
            measured=measured,
            degrees=(2,),
            pin=(0, 1),
        )


def test_managed_degrees_intersect_and_preserve_single_device_models() -> None:
    lanes = LaneSet.from_envelope("0,1,2,3", worker_pid=os.getpid())
    models = [{"sequence_parallel": {"degrees": [2, 4]}}, {"sequence_parallel": {"degrees": [2]}}]
    assert list(device_pins(lanes, "pair", models)[0].device_ordinals) == [0, 1]
    assert list(device_pins(lanes, "single", [{}])[0].device_ordinals) == [0]
    assert device_pins(lanes, "weightless", []) == []
    with pytest.raises(WorkspaceRefusal, match="no declared"):
        device_pins(lanes, "unsupported", [{"sequence_parallel": {"degrees": [8]}}])


def test_failed_machine_preparation_is_not_an_endless_capacity_wait() -> None:
    with _workspace() as root:
        worker = Worker(
            _config(root / "home"), WorkerOptions(root=root / "worker"), InMemoryControlHost()
        )
        try:
            placement = _placement("failed")
            worker.placement = placement
            desired = pb.DesiredWorkerState(
                placement_set=pb.DesiredPlacementSet(
                    placement_set_digest=placement.placement_set_digest
                )
            )
            worker.accepted.desired_state_body = desired_state_body(desired)
            preparation = {"state": base64.b64encode(desired.SerializeToString()).decode()}
            placement.materialization = pb.MATERIALIZATION_STATE_FAILED
            worker.engine.fault(
                pb.Fault(
                    kind=pb.FAULT_KIND_CONFIG_REFUSED,
                    subject="failed",
                    reason="device_group_unsupported",
                    detail="the selected group is unsupported",
                )
            )
            with pytest.raises(WorkspaceRefusal, match="device_group_unsupported"):
                worker._prepare_execution(preparation, "managed-child")
            # Ongoing or recoverable acquisition is not converted into a terminal failure.
            for state in (pb.MATERIALIZATION_STATE_MATERIALIZING, pb.MATERIALIZATION_STATE_ABSENT):
                placement.materialization = state
                assert worker._prepare_execution(preparation, "managed-child")
        finally:
            worker.supervision.close()
            worker.weights.close()


@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize(
    "fault_kind",
    ["selected", "sibling", "old-environment", "old-bindings", "absent-identity", "preparing"],
)
def test_queued_child_receives_only_its_exact_preparation_failure(
    hosted: bool, fault_kind: str
) -> None:
    from dataclasses import replace

    from cozy_runtime.internal.worker.session import HostedPlacement
    from cozy_runtime.protocol import WIRE_MINOR, documents
    from test_machine_calls import request, running
    from test_machine_execution import offer

    with _workspace() as root:
        worker = Worker(
            replace(_config(root / "home"), bootstrap_credential="test-bootstrap"),
            WorkerOptions(root=root / "worker", tensorfs_root=root / "store"),
            InMemoryControlHost(),
        )
        claim = pb.Claim(
            record_owner_id="owner",
            record_owner_epoch=1,
            wire_minor=WIRE_MINOR,
            proof=b"test-bootstrap",
        )
        try:
            assert worker.accept_claim(claim, lambda frame: None)[0] == 1
            executions, calls = worker.executions, worker.machine_calls
            assert executions is not None and calls is not None
            executions.submit(
                "owner",
                "parent",
                b"p" * 32,
                offer("parent"),
                expected_execution_workspace_id=executions.workspace_id,
                worker_boot=worker.fence.worker_boot_id,
            )
            incoming = request(running(executions))
            child = calls.journal.accept("owner", incoming)
            placement = _placement("gpu")
            placement.environment_digest = b"e" * 32
            placement.bindings_digest = b"b" * 32
            placement.materialization = pb.MATERIALIZATION_STATE_STAGED
            placement.serving = pb.SERVING_STATE_OFFLINE
            if hosted:
                worker.hosted["gpu"] = HostedPlacement(placement, worker.supervision)
                failed = worker.hosted["gpu"].failed_bindings
            else:
                worker.placement = placement
                failed = worker.failed_bindings
            selected = documents.spell(b"s" * 32)
            if fault_kind != "preparing":
                failed[selected if fault_kind != "sibling" else documents.spell(b"x" * 32)] = (
                    "fusion_kernels_corrupt: the artifact launcher module is absent"
                )
            raw, digest = documents.identity(
                pb.InvocationSpec(
                    environment_digest=documents.spell(
                        b"o" * 32
                        if fault_kind == "old-environment"
                        else placement.environment_digest
                    ),
                    serving=pb.ServingInvocationSpec(
                        entrypoint_binding_digest=selected,
                        attempt_binding_id=selected,
                        bindings_digest=documents.spell(
                            b"o" * 32 if fault_kind == "old-bindings" else placement.bindings_digest
                        ),
                    ),
                )
            )
            offered = pb.AttemptOffer(
                request_id=child.child_request,
                attempt_ordinal=1,
                placement_id="gpu",
                invocation_spec_canonical_bytes=raw,
                invocation_spec_digest=digest,
                grant=pb.DeliveryGrant(invocation_spec_digest=digest),
            )
            if fault_kind == "absent-identity":
                placement.environment_digest = b""
                placement.bindings_digest = b""
            executions.submit(
                "owner",
                "child",
                b"c" * 32,
                offered,
                expected_execution_workspace_id=executions.workspace_id,
                worker_boot=worker.fence.worker_boot_id,
            )
            # There is no executor or available GPU seat. A known refusal must still
            # become a durable outcome; ordinary waiting must not become failure.
            worker.drive_executions()
            if fault_kind != "selected":
                assert executions.status("owner", child.child_request).state == "queued"
                return
            assert executions.status("owner", child.child_request).state == "failed"
            terminal = worker.collect_execution(claim, child.child_request)
            body = documents.read(terminal.outcome_canonical_bytes, pb.AttemptOutcomeBody)
            assert body["status"] == pb.OUTCOME_STATUS_REFUSED
            assert "binding_faulted" in body["safe_message"]
            assert "fusion_kernels_corrupt" in body["safe_message"]
            assert not body.get("execution_started", False)
            reply = calls.result("owner", child, incoming)
            assert reply.state == pb.CHILD_CALL_STATE_FAILED
            assert "fusion_kernels_corrupt" in reply.safe_detail
            # Repeated watchdog passes/reconnects return the same recorded outcome.
            worker.drive_executions()
            assert worker.collect_execution(claim, child.child_request) == terminal
        finally:
            worker.shutdown()
