"""Actual published H3 binding sets at the native header and worker admission seam."""

from __future__ import annotations

from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal import accel, base_observation
from cozy_runtime.internal.worker.acquire import Acquirer
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.plan import DeclaredBinding
from cozy_runtime.internal.worker.session import Placement, Worker, WorkerOptions
from cozy_runtime.protocol import documents
from test_device_lanes import _config
from test_group_stage_residency import current_placement

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
    interface = (FIXTURE / "package-interface.json").read_bytes()
    raw, entry = current_placement((FIXTURE / "placement.json").read_bytes(), interface)
    cache = root / "artifacts"
    cache.mkdir()
    placement = Placement(entry, documents.digest_of(raw))
    result, _ = Acquirer(
        cache_root=cache,
        install_root=root / "environment",
        selection_root=root / "selection",
        tensorfs_root=store_root,
        base=base_observation.current(),
        python=Path("/usr/bin/python3"),
    )._bindings(entry)
    return placement, dict(result)


@pytest.mark.parametrize("degree", [2, 4])
def test_public_alternative_entrypoints_preserve_the_joint_call_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, degree: int
) -> None:
    placement, bindings = public_bindings(tmp_path)
    assert {b.entrypoint for b in bindings.values()} == {"fl2va_turbo", "ref2va_turbo"}
    assert len(placement.document.models) == 4  # Equal checkpoints were not deduplicated.
    assert all(len(b.model_bindings()) == 2 for b in bindings.values())
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
