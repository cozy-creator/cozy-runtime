"""Model preparation consumes one CozyTensors file, not every snapshot sibling."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict

import pytest
import tensorfs

from cozy_runtime.internal import base_observation
from cozy_runtime.internal.worker.acquire import Acquirer, AcquisitionRefusal
from cozy_runtime.internal.worker.downloads import DownloadPlan

# Canonical CozyTensors header produced by TensorFS itself. It contains one inline f32 tensor and
# one six-byte tokenizer asset. Keeping the bytes fixed makes this a border test: Runtime does not
# author or reinterpret the header format.
_HEADER = base64.b64decode(
    "hW1jb3p5dGVuc29ycy8xgIGFc3Rva2VuaXplci92b2NhYi50eHRYIJ5ekBAsaZRV6QOf+QMoTgaJ"
    "OU3TRbsRRWcG8IeYTS63Bmp0ZXh0L3BsYWlugYJYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG"
    "8IeYTS63BoGFjAMLAgEABAUIBwYJCoCBg2V2YWx1ZYEAgQCBeJNQbGFpbiBsaXR0bGUtZW5kaWFu"
    "IHJvdy1tYWpvciBzdG9yYWdlOiBvbmUgYHZhbHVlYCByb2xlIGNhcnJ5aW5nIHRoZSBsb2dpY2Fs"
    "IGJ5dGVzIHZlcmJhdGltLiBFbnRyeSB6ZXJvIG9mIHRoZSByZWdpc3RyeTsgbm90IGEgc3BlY2lh"
    "bCBhYnNlbnQgZm9ybS6BglggR2b5Mbu3DtQx40s05Pn8XdqBdOlvQ6is152Hme/NQ9IZAgSBgmR1"
    "bmV0gYVmd2VpZ2h0AYEBAIGEZXZhbHVlAYEBRAAAAAA="
)
_ASSET = b"vocab\n"


class _ObjectRef(TypedDict):
    length: int
    sha256: str


def _ref(raw: bytes) -> _ObjectRef:
    return {"length": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _snapshot(tmp_path: Path, *, include_asset: bool) -> tuple[Path, str, int, Any]:
    store_root = tmp_path / "store"
    store = tensorfs.Store.init(str(store_root))

    header_path = tmp_path / "header.cbor"
    header_path.write_bytes(_HEADER)
    header = _ref(_HEADER)
    store.put_file(str(header_path), "sha256:" + header["sha256"], header["length"])

    asset = _ref(_ASSET)
    if include_asset:
        asset_path = tmp_path / "vocab.txt"
        asset_path.write_bytes(_ASSET)
        store.put_file(str(asset_path), "sha256:" + asset["sha256"], asset["length"])

    # These files are retained by the repository snapshot but intentionally absent locally. A
    # worker preparing the model has no reason to fetch either one.
    readme = _ref(b"example model\n")
    sample = _ref(b"not really a png")
    manifest = {
        "entries": [
            {"blob": readme, "kind": "file", "path": "README.txt"},
            {"blob": header, "kind": "cozytensors", "path": "model.cozytensors"},
            {"blob": sample, "kind": "file", "path": "sample.png"},
        ]
    }
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_id = "sha256:" + hashlib.sha256(raw).hexdigest()
    store.put_manifest(raw, manifest_id, len(raw))
    return store_root, manifest_id, len(raw), store


def _acquirer(tmp_path: Path, store_root: Path) -> Acquirer:
    roots = {name: tmp_path / name for name in ("cache", "environment", "selection")}
    for path in roots.values():
        path.mkdir()
    return Acquirer(
        DownloadPlan(),
        cache_root=roots["cache"],
        environment_root=roots["environment"],
        selection_root=roots["selection"],
        artifact_store=store_root,
        python=Path("/usr/bin/python3"),
        base=base_observation.current(),
    )


def test_model_prep_ignores_absent_snapshot_siblings(tmp_path: Path) -> None:
    store_root, manifest_id, manifest_length, store = _snapshot(tmp_path, include_asset=True)
    placement = SimpleNamespace(
        models=({"manifest": {"digest": manifest_id, "length": manifest_length}},)
    )

    count = _acquirer(tmp_path, store_root)._models(placement)

    assert count == 3  # Manifest, CozyTensors header, tokenizer asset.
    assert {row["kind"] for row in store.walk_cozytensors(manifest_id)} == {
        "header",
        "model_asset",
    }
    assert not store.contains(hashlib.sha256(b"example model\n").hexdigest())
    assert not store.contains(hashlib.sha256(b"not really a png").hexdigest())


def test_model_prep_still_requires_header_assets(tmp_path: Path) -> None:
    store_root, manifest_id, manifest_length, _store = _snapshot(tmp_path, include_asset=False)
    placement = SimpleNamespace(
        models=({"manifest": {"digest": manifest_id, "length": manifest_length}},)
    )

    with pytest.raises(AcquisitionRefusal) as refused:
        _acquirer(tmp_path, store_root)._models(placement)

    assert refused.value.code == "tensorfs_refused"
    assert hashlib.sha256(_ASSET).hexdigest() in refused.value.detail
