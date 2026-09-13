"""Real H3 metadata and measured H100 facts at the native binding/group boundary."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal import accel, base_observation, canonical
from cozy_runtime.internal.worker.acquire import Acquirer
from cozy_runtime.internal.worker.lanes import DeviceLane, LaneRefusal, LaneSet
from cozy_runtime.internal.worker.plan import DeclaredBinding
from cozy_runtime.internal.worker.session import Placement, read_placement_set
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

FIXTURE = Path(__file__).parent / "testdata/h3-group-residency"
HEADROOM = 85_017_493_504
TEXT_ENCODER = 51_506_191_840


def binding(tmp_path: Path, *, scope: str = "authored", slots: int = 1) -> DeclaredBinding:
    """Resolve the actual native header without fetching its 104 GB tensor payloads."""
    store_root = tmp_path / "store"
    store = tensorfs.Store.ensure(str(store_root))
    raw = (FIXTURE / "manifest.json").read_bytes()
    store.put_manifest(raw, documents.spell(documents.digest_of(raw)), len(raw))
    header = FIXTURE / "header.cbor"
    store.put_file(
        str(header),
        documents.spell(documents.digest_of(header.read_bytes())),
        header.stat().st_size,
    )
    interface = json.loads((FIXTURE.parent / "h3-code-only/package-interface.json").read_bytes())
    raw = (FIXTURE / "placement.json").read_bytes()
    (entry,) = read_placement_set(
        pb.DesiredPlacementSet(
            placement_set_digest=documents.digest_of(raw),
            placement_set_canonical_bytes=raw,
        )
    )
    described = next(row for row in interface["entrypoints"] if row["name"] == "fl2va")
    model = described["models"][0]
    if scope == "simultaneous":
        model["component_use"] = {
            "all": [row["component"] for row in entry["entrypoints"][0]["slots"][0]["components"]]
        }
    elif scope == "unknown":
        model["component_use"] = {"unknown": ["not-a-selected-component"]}
    elif scope == "uncovered":
        model["component_use"] = {"text": ["text_encoder"]}
    elif scope == "absent":
        model["component_use"] = {}
    if slots == 2:
        second = copy.deepcopy(model)
        second["path"] = "fl2va.models.other"
        described["models"].append(second)
        selected = copy.deepcopy(entry["entrypoints"][0]["slots"][0])
        selected["slot"] = "other"
        entry["entrypoints"][0]["slots"].append(selected)
    body = canonical.write(interface)
    digest = canonical.digest(interface)
    cache = tmp_path / "artifacts"
    cache.mkdir()
    (cache / digest.removeprefix("sha256:")).write_bytes(body)
    placement = Placement(
        placement_id=entry["placement_id"],
        placement_set_digest=documents.digest_of(raw),
        package={},
        development=entry["development"],
        environment_digest=documents.raw(entry["environment_digest"]),
        package_interface=(digest, len(body)),
        bindings_digest=documents.raw(entry["bindings_digest"]),
        models=tuple(entry["models"]),
        entrypoints=tuple(entry["entrypoints"]),
        environment=entry["environment"],
        document=entry,
        device_pin=(0, 1),
    )
    resolved, _ = Acquirer(
        cache_root=cache,
        install_root=tmp_path / "environment",
        selection_root=tmp_path / "selection",
        tensorfs_root=store_root,
        base=base_observation.current(),
        python=Path("/usr/bin/python3"),
    )._bindings(placement)
    return next(iter(resolved.values()))


def assign(resolved: DeclaredBinding, degree: int = 2, room: int = HEADROOM) -> DeviceLane:
    lanes = LaneSet.from_envelope(",".join(str(i) for i in range(degree)), worker_pid=os.getpid())
    return lanes.assign(
        "h3",
        model_bearing=True,
        declared_bytes=resolved.group_weight_bytes(),
        measured={i: accel.DeviceMemory("measured", room, room) for i in range(degree)},
        pin=tuple(range(degree)),
        degrees=resolved.sequence_parallel_degrees(),
    )


@pytest.mark.parametrize("degree", [2, 4])
def test_actual_h3_scopes_fit_each_rank_without_dividing_weights(
    tmp_path: Path, degree: int
) -> None:
    selected = binding(tmp_path)
    assert selected.model_weight_bytes == 104_738_993_849
    lane = assign(selected, degree)
    assert lane.degree == degree
    assert selected.group_weight_bytes() == TEXT_ENCODER
    with pytest.raises(LaneRefusal, match="device_group_infeasible"):
        assign(selected, degree, TEXT_ENCODER - 1)


@pytest.mark.parametrize("scope", ["simultaneous", "unknown", "uncovered", "absent"])
def test_full_or_unknown_scopes_remain_too_large(tmp_path: Path, scope: str) -> None:
    selected = binding(tmp_path, scope=scope)
    with pytest.raises(LaneRefusal, match="device_group_infeasible"):
        assign(selected)


def test_independent_slots_of_the_same_checkpoint_still_sum(tmp_path: Path) -> None:
    selected = binding(tmp_path, slots=2)
    assert len(selected.model_bindings()) == 2
    assert selected.group_weight_bytes() == 2 * TEXT_ENCODER
    with pytest.raises(LaneRefusal, match="device_group_infeasible"):
        assign(selected)
