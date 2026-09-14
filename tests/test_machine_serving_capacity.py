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
