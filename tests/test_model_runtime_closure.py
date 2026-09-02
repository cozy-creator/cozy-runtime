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

from cozy_runtime.internal import base_observation, canonical, package_interface
from cozy_runtime.internal.worker.acquire import Acquirer, AcquisitionRefusal
from cozy_runtime.internal.worker.downloads import DownloadPlan

# Canonical CozyTensors header produced by TensorFS itself. It contains one inline JCS config, one
# inline f32 tensor, and one six-byte tokenizer asset. Keeping the bytes fixed makes this a border
# test: Runtime does not author or reinterpret the header format.
_HEADER = base64.b64decode(
    "hW1jb3p5dGVuc29ycy8xgYJkdW5ldFgceyJoaWRkZW5fc2l6ZSI6NCwibGF5ZXJzIjoxfYGFc3Rv"
    "a2VuaXplci92b2NhYi50eHRYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63Bmp0ZXh0"
    "L3BsYWlugYJYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63BoGFjAMLAgEABAUIBwYJC"
    "oCBg2V2YWx1ZYEAgQCBeJNQbGFpbiBsaXR0bGUtZW5kaWFuIHJvdy1tYWpvciBzdG9yYWdlOiBv"
    "bmUgYHZhbHVlYCByb2xlIGNhcnJ5aW5nIHRoZSBsb2dpY2FsIGJ5dGVzIHZlcmJhdGltLiBFbnRy"
    "eSB6ZXJvIG9mIHRoZSByZWdpc3RyeTsgbm90IGEgc3BlY2lhbCBhYnNlbnQgZm9ybS6BglggR2b5"
    "Mbu3DtQx40s05Pn8XdqBdOlvQ6is152Hme/NQ9IZAgSBgmR1bmV0gYVmd2VpZ2h0AYEBAIGEZXZh"
    "bHVlAYEBRAAAAAA="
)
_CONFIG = b'{"hidden_size":4,"layers":1}'
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


def test_binding_reads_config_only_from_the_cozytensors_header(tmp_path: Path) -> None:
    store_root, manifest_id, manifest_length, store = _snapshot(tmp_path, include_asset=True)
    acquirer = _acquirer(tmp_path, store_root)
    body = {
        "application": "test:app",
        "entrypoints": [
            {
                "name": "generate",
                "request": {"fields": []},
                "result": {"fields": []},
                "models": [
                    {
                        "path": "generate.models.model",
                        "class": "test.Model",
                        "component_use": {"generate": ["unet"]},
                    }
                ],
            }
        ],
        "format": package_interface.SCHEMA,
        "jobs": [],
    }
    interface_bytes = package_interface.canonical_bytes(body)
    interface_digest = "sha256:" + hashlib.sha256(interface_bytes).hexdigest()
    (acquirer.cache_root / interface_digest.removeprefix("sha256:")).write_bytes(interface_bytes)
    slot = {
        "slot": "model",
        "reference_model_id": "model-0000",
        "components": [{"component": "unet", "model_id": "model-0000"}],
    }
    binding_digest = (
        "sha256:"
        + hashlib.sha256(canonical.write({"name": "generate", "slots": [slot]})).hexdigest()
    )
    placement = SimpleNamespace(
        package_interface=(interface_digest, len(interface_bytes)),
        models=(
            {
                "id": "model-0000",
                "repo": "example/model",
                "version": "1",
                "lane": "main",
                "manifest": {"digest": manifest_id, "length": manifest_length},
            },
        ),
        entrypoints=(
            {"name": "generate", "entrypoint_binding_digest": binding_digest, "slots": [slot]},
        ),
        package={
            "package": "example/package",
            "release": "1",
            "project_wheel": {"ref": {"digest": "sha256:" + "2" * 64}},
        },
        development={},
        environment_digest=b"\x33" * 32,
    )

    bindings, documents_count = acquirer._bindings(placement)

    (model,) = bindings[binding_digest].model_bindings()
    assert tensorfs.parse_header(bytes(store.manifest(manifest_id)["header"]))["configs"] == {
        "unet": _CONFIG
    }
    assert Path(model.config).read_bytes() == _CONFIG
    assert model.config_digest == "sha256:" + hashlib.sha256(_CONFIG).hexdigest()
    assert documents_count == 1  # PackageInterface only; config is not a downloaded document.
