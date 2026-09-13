"""Actual published H3 binding sets at the native header and worker admission seam."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal import accel, base_observation, canonical
from cozy_runtime.internal.worker.acquire import Acquirer
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.plan import DeclaredBinding
from cozy_runtime.internal.worker.session import (
    Placement,
    Worker,
    WorkerOptions,
    executor_prepare_command,
    read_placement_set,
)
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import _config
from test_group_stage_residency import binding as independent_slots

FIXTURE = Path(__file__).parent / "testdata/h3-public-binding-sets"
BASE = FIXTURE.parent / "h3-group-residency"
ROOM = 85_017_493_504
JOINT_WEIGHTS = 52_304_245_216


def public_bindings(root: Path) -> tuple[Placement, dict[str, DeclaredBinding]]:
    store_root = root / "store"
    store = tensorfs.Store.ensure(str(store_root))
    for manifest, header in (
        (BASE / "manifest.json", BASE / "header.cbor"),
        (FIXTURE / "turbo-manifest.json", FIXTURE / "turbo-header.cbor"),
    ):
        raw = manifest.read_bytes()
        store.put_manifest(raw, documents.spell(documents.digest_of(raw)), len(raw))
        store.put_file(
            str(header),
            documents.spell(documents.digest_of(header.read_bytes())),
            header.stat().st_size,
        )
    raw = (FIXTURE / "placement.json").read_bytes()
    (entry,) = read_placement_set(
        pb.DesiredPlacementSet(
            placement_set_digest=documents.digest_of(raw),
            placement_set_canonical_bytes=raw,
        )
    )
    interface = (FIXTURE / "package-interface.json").read_bytes()
    assert canonical.digest(json.loads(interface)) == entry["package_interface"]["digest"]
    cache = root / "artifacts"
    cache.mkdir()
    (cache / entry["package_interface"]["digest"].removeprefix("sha256:")).write_bytes(interface)
    placement = Placement(
        placement_id=entry["placement_id"],
        placement_set_digest=documents.digest_of(raw),
        package=entry["package"],
        development={},
        environment_digest=documents.raw(entry["environment_digest"]),
        package_interface=(entry["package_interface"]["digest"], len(interface)),
        bindings_digest=documents.raw(entry["bindings_digest"]),
        models=tuple(entry["models"]),
        entrypoints=tuple(entry["entrypoints"]),
        environment=entry["environment"],
        document=entry,
    )
    result, _ = Acquirer(
        cache_root=cache,
        install_root=root / "environment",
        selection_root=root / "selection",
        tensorfs_root=store_root,
        base=base_observation.current(),
        python=Path("/usr/bin/python3"),
    )._bindings(placement)
    return placement, dict(result)


@pytest.mark.parametrize("degree", [2, 4])
def test_public_alternative_entrypoints_preserve_the_joint_call_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, degree: int
) -> None:
    placement, bindings = public_bindings(tmp_path)
    assert {b.entrypoint for b in bindings.values()} == {"fl2va_turbo", "ref2va_turbo"}
    assert len(placement.models) == 4  # Equal checkpoints were not deduplicated.
    assert all(len(b.model_bindings()) == 2 for b in bindings.values())
    assert {b.group_weight_bytes() for b in bindings.values()} == {JOINT_WEIGHTS}
    assert sum(b.group_weight_bytes() for b in bindings.values()) == 2 * JOINT_WEIGHTS
    assert len({b.construction_key() for b in bindings.values()}) == 1
    placement.device_pin = tuple(range(degree))
    monkeypatch.setattr(accel, "host_backend_family", lambda: "cuda")
    monkeypatch.setattr(
        accel, "device_memory", lambda *_: accel.DeviceMemory("measured", ROOM, ROOM)
    )
    worker = Worker(
        _config(tmp_path / "home"),
        WorkerOptions(root=tmp_path / "worker", devices=",".join(map(str, range(degree)))),
        InMemoryControlHost(),
    )
    try:
        assert worker._assign_lane(placement, bindings, worker.supervision)
        lane = worker.lanes.lane_of(placement.placement_id)
        assert lane is not None and lane.degree == degree
    finally:
        worker.shutdown()


def test_two_simultaneous_slots_still_add_even_for_the_same_checkpoint(tmp_path: Path) -> None:
    selected = independent_slots(tmp_path, slots=2)
    assert selected.group_weight_bytes() == 2 * 51_506_191_840 > ROOM


def test_public_h3_single_gpu_admits_staged_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    placement, bindings = public_bindings(tmp_path)
    room = 80_000_000_000
    assert all(b.model_weight_bytes > room for b in bindings.values())
    assert all(b.group_weight_bytes() < room for b in bindings.values())
    assert all(b.sequence_parallel_degrees() == (2, 4) for b in bindings.values())
    placement.device_pin = (0,)
    monkeypatch.setattr(accel, "host_backend_family", lambda: "cuda")
    monkeypatch.setattr(
        accel, "device_memory", lambda *_: accel.DeviceMemory("measured", room, room)
    )
    worker = Worker(
        _config(tmp_path / "home"),
        WorkerOptions(root=tmp_path / "worker", devices="0"),
        InMemoryControlHost(),
    )
    try:
        assert worker._assign_lane(placement, bindings, worker.supervision)
        lane = worker.lanes.lane_of(placement.placement_id)
        assert lane is not None and lane.degree == 1 and not lane.group
        for binding in bindings.values():
            command = executor_prepare_command(
                binding,
                devices=lane.devices,
                authorized_device_limit_bytes=room,
                sequence_parallel_degree=lane.degree,
            )
            assert "sequence_parallel_degree" not in command
            assert command["devices"] == "0"
            assert len(command["models"]) == 2
            assert command["authorized_device_limit_bytes"] == room
    finally:
        worker.shutdown()
