"""The unified-memory plane (Apple GPUs) over a real TensorFS store: every region's bytes are
the stored bytes, inline and object parts alike, read from a store lease and from the
machine's object files; budgets evict in the plane's order and refuse what cannot fit; a
cursor lets its oldest streamed region go. On the CPU everywhere, and on MPS where there is one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tensorfs  # noqa: E402

from cozy_runtime.internal import plane  # noqa: E402
from cozy_runtime.internal.unified_plane import UnifiedPlane  # noqa: E402
from fetch_hub import write  # noqa: E402

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])

#: `tiny` is stored inline in the header; every other tensor is an object
VALUES = {
    "tiny": np.arange(4, dtype=np.float16),
    **{
        f"blocks.{i}.weight": np.random.default_rng(i).normal(size=(256, 512)).astype(np.float16)
        for i in range(3)
    },
}
REGIONS = [["c/tiny"], *[[f"c/blocks.{i}.weight"] for i in range(3)]]


def registered(tmp_path: Path, device: str, files: bool = False) -> tuple[UnifiedPlane, Any]:
    store, manifest = write(tmp_path / "store", {"c": VALUES})
    header = store.manifest(manifest)["header"]
    assert header is not None
    read = tensorfs.plan(header, [("c", key) for key in VALUES], ["c"], window=1 << 16)
    unified = UnifiedPlane(torch, torch.device(device))
    if files:
        handed = [(sha, n, os.open(_blob(store, sha), os.O_RDONLY)) for sha, n in _objects(read)]
        source = unified.files_source(handed, tensorfs.parse_header(header))
        for _, _, fd in handed:
            os.close(fd)
    else:
        lease = store.acquire(manifest, [(sha, n) for sha, n in _objects(read)])
        source = unified.source(store, lease)
    return unified, unified.register("c", source, read, REGIONS)


def _objects(read: Any) -> list[tuple[str, int]]:
    return sorted(
        {(i["object"], i["object_length"]) for i in read.items() if i["source"] == "object"}
    )


def _blob(store: Any, sha: str) -> Path:
    digest = sha.removeprefix("sha256:")
    return Path(store.root, "blobs", digest[:2], digest[2:4], digest)


@pytest.mark.parametrize("files", [False, True], ids=["lease", "object-files"])
@pytest.mark.parametrize("device", DEVICES)
def test_every_region_holds_the_stored_bytes(tmp_path: Path, device: str, files: bool) -> None:
    unified, ws = registered(tmp_path, device, files)
    unified.set_vram_budget(0, ws.nbytes)
    unified.want(ws, 0, None, priority=1)
    for part in ws.parts:
        region = ws.regions[part["region"]]
        with unified.acquire(ws, 0, part["region"], 0) as lease:
            assert lease.view().device.type == device
            at = part["offset"] - region["offset"]
            got = lease.view()[at : at + part["nbytes"]].cpu().numpy().tobytes()
        key = part["what"].removeprefix("c/").removesuffix("#value")
        assert got == VALUES[key].tobytes(), key
        assert part["offset"] % 4096 == 0
    assert unified.committed == ws.nbytes
    ws.close()


@pytest.mark.parametrize("device", DEVICES)
def test_budgets_evict_in_order_and_refuse_what_cannot_fit(tmp_path: Path, device: str) -> None:
    unified, ws = registered(tmp_path, device)
    span = ws.regions[1]["span"]
    unified.set_vram_budget(0, 2 * span)
    unified.want(ws, 0, [1], priority=1)
    unified.want(ws, 0, [2], priority=2)
    unified.events()
    # A region filled at a higher priority evicts the lowest first.
    with unified.acquire(ws, 0, 3, 0, priority=3):
        assert [(e["region"], e["kind"]) for e in unified.events()] == [(1, "evicted")]
        # Nothing strictly below it can go: refused, nothing moved.
        with pytest.raises(plane.error("Shortfall")):
            unified.acquire(ws, 0, 1, 0, priority=0)
        with pytest.raises(plane.error("LeaseViolation")):
            unified.drop(ws, 0, [3])
    with pytest.raises(plane.error("BelowFloor")):
        unified.want(ws, 0, [1, 2, 3], priority=9)
    trim = unified.set_vram_budget(0, span)
    assert trim["evicted"] == [("c", 2)] and unified.committed == span
    assert unified.drop(ws, 0, None) == span and unified.committed == 0
    ws.close()


@pytest.mark.parametrize("device", DEVICES)
def test_a_cursor_lets_its_oldest_streamed_region_go(tmp_path: Path, device: str) -> None:
    unified, ws = registered(tmp_path, device)
    unified.set_vram_budget(0, ws.nbytes)
    unified.want(ws, 0, [0], priority=5)
    cursor = unified.stream(ws, 0, [0, 1, 2, 3], window=1, priority=2)
    assert cursor.home == [True, False, False, False]
    for index in (0, 1, 2, 3):
        cursor.acquire(index).release()
    resident = [r["index"] for r in ws.regions if ws._regions[r["index"]].tensor is not None]
    assert resident == [0, 3], "home stays; one streamed region at a time"
    cursor.close()
    ws.close()
