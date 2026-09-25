"""Actual durable reservations and worker ownership; GPU inference is a separate live gate."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker import machine_lanes
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import HostedPlacement, Worker, WorkerOptions
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import _config
from test_machine_execution import complete, offer


def desired(placement: dict[str, Any], ordinals: tuple[int, ...]) -> pb.DesiredWorkerState:
    raw = canonical_json.encode(
        {"format": "cozy.worker.v1.PlacementSet/1", "placements": [placement]}
    )
    return pb.DesiredWorkerState(
        placement_set=pb.DesiredPlacementSet(
            placement_set_canonical_bytes=raw,
            device_pins=[
                pb.PlacementDevicePin(
                    placement_id=placement["placement_id"], device_ordinals=ordinals
                )
            ],
        )
    )


@pytest.mark.parametrize("width", [2, 4])
def test_eight_durable_children_spread_then_reuse_exact_replicas(
    tmp_path: Path, width: int
) -> None:
    worker = Worker(
        _config(tmp_path / "home"),
        WorkerOptions(root=tmp_path / "worker", devices=",".join(str(i) for i in range(width))),
        InMemoryControlHost(),
    )
    try:
        workspace = Workspace(tmp_path / "store")
        worker.executions = Executions(workspace)
        placement = {
            "placement_id": "prepared",
            "environment_digest": "sha256:" + "a" * 64,
            "bindings_digest": "sha256:" + "b" * 64,
        }
        identities = []
        for index in range(8):
            selected, pins = machine_lanes.replica(worker, "owner", placement, [{}])
            ordinals = tuple(pins[0].device_ordinals)
            assert ordinals == (index % width,)
            identities.append(selected["placement_id"])
            prepared = {
                "state": base64.b64encode(desired(selected, ordinals).SerializeToString()).decode()
            }
            child = offer(f"child-{index}")
            worker.executions.submit(
                "owner",
                child.request_id,
                b"c" * 32,
                child,
                expected_execution_workspace_id=worker.executions.workspace_id,
                preparation=canonical_json.encode(prepared),
            )
            # A fresh journal reader must make the same capacity decision after restart.
            worker.executions = Executions(Workspace(tmp_path / "store"))
        assert len(set(identities)) == width
        assert identities[:width] == identities[width : 2 * width]
        for index in range(8):
            complete(worker.executions, f"child-{index}", status=pb.OUTCOME_STATUS_CANCELED)
        selected, pins = machine_lanes.replica(worker, "owner", placement, [{}])
        assert selected["placement_id"] == identities[0]
        assert list(pins[0].device_ordinals) == [0]
        group, pins = machine_lanes.replica(
            worker, "owner", placement, [{"sequence_parallel": {"degrees": [2, 4]}}]
        )
        assert tuple(pins[0].device_ordinals) == tuple(range(width))
        assert group["placement_id"] not in identities
    finally:
        worker.shutdown()


def test_union_preserves_busy_disjoint_owners_and_group_waits_for_release(tmp_path: Path) -> None:
    worker = Worker(
        _config(tmp_path / "home"),
        WorkerOptions(root=tmp_path / "worker", devices="0,1,2,3"),
        InMemoryControlHost(),
    )
    try:
        documents_by_id = {
            f"replica-{index}": {"placement_id": f"replica-{index}"} for index in range(4)
        }
        for index, (identity, document) in enumerate(documents_by_id.items()):
            placement = worker._placement_from_entry(document, b"")
            placement.device_pin = (index,)
            if index == 0:
                worker.placement = placement
            else:
                worker.hosted[identity] = HostedPlacement(
                    placement, worker._new_placement_supervision(identity)
                )
            attempt = AttemptRecord(
                identity,
                1,
                b"",
                {},
                placement_id=identity,
                lane_id=f"lane-{index}",
                state="running",
            )
            worker.engine.history[(identity, 1)] = attempt
        parent = AttemptRecord(
            "parent", 1, b"", {}, lane_id="cpu-parent", state="running", kind="job"
        )
        worker.engine.history[("parent", 1)] = parent
        combined = machine_lanes.compose(worker, desired(documents_by_id["replica-0"], (0,)))
        assert combined is not None
        body = canonical_json.decode(combined.placement_set.placement_set_canonical_bytes)
        assert {row["placement_id"] for row in body["placements"]} == documents_by_id.keys()
        assert (
            documents.digest_of(combined.placement_set.placement_set_canonical_bytes)
            == combined.placement_set.placement_set_digest
        )
        group = desired({"placement_id": "fused"}, (0, 1, 2, 3))
        assert machine_lanes.compose(worker, group) is None
        # Neither cancellation requested nor released-but-unacknowledged work permits stealing.
        for attempt in worker.engine.history.values():
            if attempt is parent:
                continue
            attempt.canceling = "client"
            attempt.state = "outcome"
        assert machine_lanes.compose(worker, group) is None
        for attempt in worker.engine.history.values():
            if attempt is not parent:
                attempt.state = "closed"
        combined = machine_lanes.compose(worker, group)
        assert combined is not None
        assert canonical_json.decode(combined.placement_set.placement_set_canonical_bytes)[
            "placements"
        ] == [{"placement_id": "fused"}]
        # A CPU parent can await the group; changing it to a device owner is still blocked.
        parent.lane_id = "envelope"
        assert machine_lanes.compose(worker, group) is None
    finally:
        worker.shutdown()
