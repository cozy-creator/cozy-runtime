"""Real H3 metadata and measured H100 facts at the native binding/group boundary."""

from __future__ import annotations

import base64
import copy
import json
import math
import os
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal import accel, base_observation, canonical
from cozy_runtime.internal.worker.acquire import Acquirer
from cozy_runtime.internal.worker.lanes import DeviceLane, LaneSet
from cozy_runtime.internal.worker.plan import DeclaredBinding
from cozy_runtime.internal.worker.session import read_placement_set
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

FIXTURE = Path(__file__).parent / "testdata/h3-group-residency"
HEADROOM = 85_017_493_504
TEXT_ENCODER = 51_506_191_840
INSTALLATION = "local-h3-fixture"


def current_placement(raw: bytes, interface: bytes) -> tuple[bytes, pb.Placement]:
    """The captured H3 placement on the current wire: an installation, and interface bytes."""
    document = json.loads(raw)
    (row,) = document["placements"]
    del row["environment"], row["environment_digest"]
    row["installation_id"] = INSTALLATION
    row["package_interface"] = base64.b64encode(interface).decode()
    if "development" in row:
        development = row["development"]
        row["development"] = {
            "package": development["package"],
            "release": development["release"],
            "installation_id": INSTALLATION,
        }
    raw = canonical.write(document)
    (entry,) = read_placement_set(
        pb.DesiredPlacementSet(
            placement_set_digest=documents.digest_of(raw), placement_set_canonical_bytes=raw
        )
    )
    return raw, entry


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
    captured = (FIXTURE.parent / "h3-code-only/package-interface.json").read_bytes()
    interface = json.loads(captured)
    raw, entry = current_placement((FIXTURE / "placement.json").read_bytes(), captured)
    described = next(row for row in interface["entrypoints"] if row["name"] == "fl2va")
    model = described["models"][0]
    if scope == "simultaneous":
        model["component_use"] = {
            "all": [row.component for row in entry.entrypoints[0].slots[0].components]
        }
    elif scope == "unknown":
        model["component_use"] = {"unknown": ["not-a-selected-component"]}
    elif scope == "uncovered":
        model["component_use"] = {"text": ["text_encoder"]}
    elif scope == "absent":
        model["component_use"] = {}
    elif scope == "missing_component":
        missing = entry.entrypoints[0].slots[0].components[0].component
        entry.entrypoints[0].slots[0].components[0].component = "absent"
        model["component_use"] = {
            method: ["absent" if name == missing else name for name in names]
            for method, names in model["component_use"].items()
        }
    if slots == 2:
        second = copy.deepcopy(model)
        second["path"] = "fl2va.models.other"
        described["models"].append(second)
        entry.entrypoints[0].slots.add().CopyFrom(entry.entrypoints[0].slots[0])
        entry.entrypoints[0].slots[-1].slot = "other"
    cache = tmp_path / "artifacts"
    cache.mkdir()
    entry.package_interface = canonical.write(interface)
    resolved, _ = Acquirer(
        cache_root=cache,
        install_root=tmp_path / "environment",
        selection_root=tmp_path / "selection",
        tensorfs_root=store_root,
        base=base_observation.current(),
        python=Path("/usr/bin/python3"),
    )._bindings(entry)
    return next(iter(resolved.values()))


def assign(resolved: DeclaredBinding, degree: int = 2, room: int = HEADROOM) -> DeviceLane:
    lanes = LaneSet.from_envelope(",".join(str(i) for i in range(degree)), worker_pid=os.getpid())
    assert degree in resolved.sequence_parallel_degrees()
    return lanes.bind(
        "h3",
        tuple(range(degree)),
        model_bearing=True,
        measured={i: accel.DeviceMemory("measured", room, room) for i in range(degree)},
    )


@pytest.mark.parametrize("degree", [2, 4])
def test_actual_h3_scopes_fit_each_rank_without_dividing_weights(
    tmp_path: Path, degree: int
) -> None:
    selected = binding(tmp_path)
    lane = assign(selected, degree)
    assert lane.degree == degree
    # Header estimates must not reject the text encoder before construction can
    # discover its smaller no-split block working set.
    assert assign(selected, degree, TEXT_ENCODER - 1).degree == degree


def test_the_binding_declares_logical_bytes_not_the_stored_fp8_closure(tmp_path: Path) -> None:
    """The H3 lane is fp8-rowwise: its closure stores 104.7 GB and its constructed
    components' destinations hold 144.8 GB. The binding declares what the destinations hold."""
    selected = binding(tmp_path)
    store = tensorfs.Store.ensure(str(tmp_path / "store"))
    stored = sum(int(row["length"]) for row in store.walk_cozytensors(selected.reference_snapshot))
    header = tensorfs.parse_header((FIXTURE / "header.cbor").read_bytes())
    logical = sum(
        math.prod(entry["logical"]["shape"]) * tensorfs.DTYPES[entry["logical"]["logical_dtype"]]
        for component in selected.components
        for entry in header["components"][component].values()
    )
    assert stored == 104_738_993_849
    assert selected.logical_weight_bytes == logical == 144_844_705_956


@pytest.mark.parametrize(
    "scope", ["simultaneous", "unknown", "uncovered", "absent", "missing_component"]
)
def test_header_scopes_defer_fit_to_bounded_construction(tmp_path: Path, scope: str) -> None:
    selected = binding(tmp_path, scope=scope)
    assert assign(selected).degree == 2


def test_independent_slots_of_the_same_checkpoint_still_sum(tmp_path: Path) -> None:
    selected = binding(tmp_path, slots=2)
    assert len(selected.model_bindings()) == 2
    assert assign(selected).degree == 2


def test_missing_component_still_refuses_native_model_fit(tmp_path: Path) -> None:
    selected = binding(tmp_path, scope="missing_component")
    assert selected.logical_weight_bytes > 0
    # The same native fit `PlaneBackend.fit` asks before construction registers anything
    # still judges the deliberately incomplete construction.
    verdict = tensorfs.fit(
        [tensorfs.TensorRequirement("absent", "weight", [1], "f32")],
        (FIXTURE / "header.cbor").read_bytes(),
        custody="canonical",
        encoded_leaves=True,
    )
    assert not verdict["ok"] and verdict["code"] == "component_missing"
