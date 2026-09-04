"""A read lease is a hold on a SNAPSHOT, and a model takes one per snapshot it reads.

cr-102 measured an sdxl prepare acquiring the identical whole-artifact lease four times —
1091/1104/1160/1257 ms of the `lease_acquire` leg — because the fill plane's lease map was
keyed by COMPONENT NAME while a lease covers a MANIFEST. The four components of a
single-snapshot model all read the same 2,602 objects, so the same verification ran four
times and 10,408 file descriptors were pinned for 2,602 distinct objects, against the
FD_HEADROOM admission `acquire_cozytensors` performs on its way in.

Everything below runs against a REAL TensorFS store on disk holding a REAL CozyTensors
snapshot, through the real `Checkpoint.acquire`. Nothing is doubled, and no assertion here
names a number this file also computes: the descriptor claim is one arm measured against
another arm, and the lease counts come from the store.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal import fill

#: TensorFS's own canonical CozyTensors header, byte for byte (the same bytes
#: `test_model_runtime_closure` pins): one inline config, one inline f32 tensor under
#: `unet`, one six-byte tokenizer asset. Runtime does not author this format.
_HEADER = base64.b64decode(
    "hW1jb3p5dGVuc29ycy8xgYJkdW5ldFgceyJoaWRkZW5fc2l6ZSI6NCwibGF5ZXJzIjoxfYGFc3Rv"
    "a2VuaXplci92b2NhYi50eHRYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63Bmp0ZXh0"
    "L3BsYWlugYJYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63BoGEjAMLAgEABAUIBwYJC"
    "oCBg2V2YWx1ZYEAgQCBglggR2b5Mbu3DtQx40s05Pn8XdqBdOlvQ6is152Hme/NQ9IZAgSBgmR1bm"
    "V0gYVmd2VpZ2h0AYEBAIGEZXZhbHVlAYEBRAAAAAA="
)

#: The four component names of the model that measured the defect.
SDXL = ("text_encoder", "text_encoder_2", "unet", "vae")


def _open_descriptors() -> int:
    """This process's open file descriptors, read from the kernel."""
    return len(os.listdir("/proc/self/fd"))


def _snapshot(store: tensorfs.Store, tmp_path: Path, asset: bytes) -> str:
    """Put one real CozyTensors snapshot in the store and return its manifest id."""
    header = tmp_path / f"header-{hashlib.sha256(asset).hexdigest()[:8]}.cbor"
    header.write_bytes(_HEADER)
    header_digest = hashlib.sha256(_HEADER).hexdigest()
    store.put_file(str(header), "sha256:" + header_digest, len(_HEADER))

    vocab = tmp_path / f"vocab-{hashlib.sha256(asset).hexdigest()[:8]}.txt"
    vocab.write_bytes(asset)
    asset_digest = hashlib.sha256(asset).hexdigest()
    store.put_file(str(vocab), "sha256:" + asset_digest, len(asset))

    manifest = {
        "entries": [
            {
                "blob": {"length": len(_HEADER), "sha256": header_digest},
                "kind": "cozytensors",
                "path": "model.cozytensors",
            },
            {
                "blob": {"length": len(asset), "sha256": asset_digest},
                "kind": "file",
                "path": "tokenizer/vocab.txt",
            },
        ]
    }
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_id = "sha256:" + hashlib.sha256(raw).hexdigest()
    store.put_manifest(raw, manifest_id, len(raw))
    return manifest_id


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    tensorfs.Store.init(str(root))
    return root


def test_four_components_of_one_snapshot_take_one_lease(store_root: Path, tmp_path: Path) -> None:
    """The count that used to be four. It comes from the store, not from this file."""
    store = tensorfs.Store.open(str(store_root))
    manifest = _snapshot(store, tmp_path, b"vocab\n")
    leases = fill.ReadLeases({name: fill.Checkpoint(store_root, manifest) for name in SDXL})

    for name in SDXL:
        leases.acquire(name)

    held = leases.document()
    assert held["acquisitions"] == 1, held
    assert held["leases"] == 1, held

    # The objects and bytes held are the SNAPSHOT'S, counted once — measured against the
    # same snapshot leased for a single component rather than against a number written
    # here. Under the component key these read four times the truth, which is how an
    # artifact of 2,602 objects reported 10,408 leased.
    alone = fill.ReadLeases({"unet": fill.Checkpoint(store_root, manifest)})
    alone.acquire("unet")
    assert held["objects"] == alone.document()["objects"], (held, alone.document())
    assert held["bytes"] == alone.document()["bytes"], (held, alone.document())
    alone.release()
    leases.release()


def test_four_components_pin_no_more_descriptors_than_one(store_root: Path, tmp_path: Path) -> None:
    """THE FD CLAIM, measured against itself.

    Two arms over the same real snapshot: one component, then four. The expected value is
    the first arm's own measurement — nothing here knows or asserts how many descriptors a
    TensorFS lease pins per object, only that leasing a snapshot four times over costs no
    more of them than leasing it once, which is exactly what the old key did not hold.
    """
    store = tensorfs.Store.open(str(store_root))
    manifest = _snapshot(store, tmp_path, b"vocab\n")

    alone = fill.ReadLeases({"unet": fill.Checkpoint(store_root, manifest)})
    before = _open_descriptors()
    alone.acquire("unet")
    one_component = _open_descriptors() - before
    alone.release()
    assert one_component > 0, "a real lease pins descriptors; this store handed back none"

    every = fill.ReadLeases({name: fill.Checkpoint(store_root, manifest) for name in SDXL})
    before = _open_descriptors()
    for name in SDXL:
        every.acquire(name)
    four_components = _open_descriptors() - before

    assert four_components == one_component
    every.release()
    assert _open_descriptors() == before
    # The peak SURVIVES the release, which is the whole reason it is recorded: a
    # full-resident generation gives its leases back at its first commit, so the prepare
    # facts would otherwise report that a 5 GiB fill leased nothing at all.
    held = every.document()
    assert held["leases"] == 0
    assert held["objects"] == alone.document()["objects"] > 0
    assert held["bytes"] > 0


def test_a_component_per_snapshot_binding_still_takes_one_lease_each(
    store_root: Path, tmp_path: Path
) -> None:
    """The key is the SNAPSHOT, not the model: cr-008b's binding is not collapsed.

    Without this arm, "one lease" and "one lease per manifest" are indistinguishable, and
    the cheapest way to pass the first test would break every component-per-snapshot bind.
    """
    store = tensorfs.Store.open(str(store_root))
    first = _snapshot(store, tmp_path, b"vocab\n")
    second = _snapshot(store, tmp_path, b"a different tokenizer\n")
    assert first != second

    leases = fill.ReadLeases(
        {
            "text_encoder": fill.Checkpoint(store_root, first),
            "text_encoder_2": fill.Checkpoint(store_root, first),
            "unet": fill.Checkpoint(store_root, second),
            "vae": fill.Checkpoint(store_root, second),
        }
    )
    for name in SDXL:
        leases.acquire(name)

    held = leases.document()
    assert held["acquisitions"] == 2, held
    assert held["leases"] == 2, held
    leases.release()


def test_an_unbound_component_is_a_typed_destination_refusal(
    store_root: Path, tmp_path: Path
) -> None:
    """The refusal the map owed its caller survives the move off the backend."""
    store = tensorfs.Store.open(str(store_root))
    manifest = _snapshot(store, tmp_path, b"vocab\n")
    leases = fill.ReadLeases({"unet": fill.Checkpoint(store_root, manifest)})

    with pytest.raises(fill.FillRefusal) as raised:
        leases.acquire("controlnet")
    assert raised.value.code == "destination_absent"
    assert "controlnet" in str(raised.value)
