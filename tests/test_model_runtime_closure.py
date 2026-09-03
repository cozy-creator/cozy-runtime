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

from cozy_runtime.author import Artifact, CapabilityError, Config, Loader
from cozy_runtime.author._loader import loading
from cozy_runtime.internal import base_observation, canonical, fill, package_interface
from cozy_runtime.internal.encoding import SPEC_PLAIN, launch_providers
from cozy_runtime.internal.worker.acquire import Acquirer, AcquisitionRefusal, HeldManifests
from cozy_runtime.internal.worker.package_prepare import (
    PreparationRefusal,
    _compatibility_models,
)

# Canonical CozyTensors header produced by TensorFS itself. It contains one inline JCS config, one
# inline f32 tensor, and one six-byte tokenizer asset. Keeping the bytes fixed makes this a border
# test: Runtime does not author or reinterpret the header format.
_HEADER = base64.b64decode(
    "hW1jb3p5dGVuc29ycy8xgYJkdW5ldFgceyJoaWRkZW5fc2l6ZSI6NCwibGF5ZXJzIjoxfYGFc3Rv"
    "a2VuaXplci92b2NhYi50eHRYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63Bmp0ZXh0"
    "L3BsYWlugYJYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63BoGEjAMLAgEABAUIBwYJC"
    "oCBg2V2YWx1ZYEAgQCBglggR2b5Mbu3DtQx40s05Pn8XdqBdOlvQ6is152Hme/NQ9IZAgSBgmR1bm"
    "V0gYVmd2VpZ2h0AYEBAIGEZXZhbHVlAYEBRAAAAAA="
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
        cache_root=roots["cache"],
        install_root=roots["environment"],
        selection_root=roots["selection"],
        tensorfs_root=store_root,
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


def test_held_manifest_census_comes_from_tensorfs(tmp_path: Path) -> None:
    _store_root, manifest_id, _manifest_length, store = _snapshot(tmp_path, include_asset=True)
    held = HeldManifests()

    assert held.rebuild(store) == 1
    assert held.digests() == [manifest_id]


def test_model_prep_still_requires_header_assets(tmp_path: Path) -> None:
    store_root, manifest_id, manifest_length, _store = _snapshot(tmp_path, include_asset=False)
    placement = SimpleNamespace(
        models=({"manifest": {"digest": manifest_id, "length": manifest_length}},)
    )

    with pytest.raises(AcquisitionRefusal) as refused:
        _acquirer(tmp_path, store_root)._models(placement)

    assert refused.value.code == "tensorfs_refused"
    assert hashlib.sha256(_ASSET).hexdigest() in refused.value.detail


def test_loader_consumes_verified_header_assets(tmp_path: Path) -> None:
    store_root, manifest_id, _manifest_length, _store = _snapshot(tmp_path, include_asset=True)
    assets = fill.Checkpoint(store_root, manifest_id).model_assets()
    assert assets == {"tokenizer/vocab.txt": _ASSET}

    class Owner:
        def _cozy_require_scope(self, what: str) -> None:
            raise AssertionError(what)

    loader = Loader(
        Artifact("checkpoint", {}, Config({}), assets=assets),
        owner=Owner(),
    )
    capability = loader.assets
    with loading(loader):
        assert capability.names() == ("tokenizer/vocab.txt",)
        assert capability.read("tokenizer/vocab.txt") == _ASSET
        assert capability.open("tokenizer/vocab.txt").read() == _ASSET
        with capability.materialized("tokenizer") as tokenizer:
            vocab = tokenizer / "vocab.txt"
            assert vocab.read_bytes() == _ASSET
            assert vocab.stat().st_mode & 0o222 == 0
        assert not tokenizer.exists()
    with pytest.raises(CapabilityError, match="outside load/unload"):
        capability.read("tokenizer/vocab.txt")


def test_worker_prepare_reads_tensorfs_03_logical_dtype(tmp_path: Path) -> None:
    """Exercise the exact Checkpoint.rows path the executor's worker prepare calls."""

    store_root, manifest_id, _manifest_length, _store = _snapshot(tmp_path, include_asset=True)
    rows = fill.Checkpoint(store_root, manifest_id).rows("unet")
    assert [(row.key, row.dtype, row.shape) for row in rows] == [("unet.weight", "f32", (1,))]
    assert rows[0].encoded.encoding == SPEC_PLAIN
    assert SPEC_PLAIN in launch_providers()


def test_cold_fit_reads_manifest_header_and_model_assets_from_the_store(tmp_path: Path) -> None:
    """Wire 30: the cold check receives no files. The host admits the manifest, the
    CozyTensors header and the header-declared assets to the local Store; the check reads
    them through an explicit partial object-set lease while tensor bodies are still in
    flight, and an absent small object is a resumable refusal, not a verdict."""

    store_root, manifest_id, _manifest_length, _store = _snapshot(tmp_path, include_asset=True)
    selected = [
        {
            "slot": "generate.models.model",
            "manifest": manifest_id,
            "model": "example/model",
            "release": "1",
            "lane": "main",
        }
    ]
    candidate = _compatibility_models(selected, tensorfs_root=store_root)["generate.models.model"]
    assert candidate["config"] == _CONFIG
    assert candidate["assets"] == {"tokenizer/vocab.txt": _ASSET}

    mismatch = tensorfs.fit(
        [tensorfs.TensorRequirement("unet", "weight", [2], "f32")],
        candidate["header"],
        custody="canonical",
        encoded_leaves=False,
    )
    assert not mismatch["ok"] and mismatch["code"] == "shape_mismatch"

    # The asset object was never admitted: the partial lease refuses resumably.
    cold_root, cold_manifest, _cold_length, _cold_store = _snapshot(
        tmp_path / "cold", include_asset=False
    )
    cold = [dict(selected[0], manifest=cold_manifest)]
    with pytest.raises(PreparationRefusal, match="model_asset_unavailable"):
        _compatibility_models(cold, tensorfs_root=cold_root)


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
        package={"package": "example/package", "release": "1"},
        development={},
        environment_digest=b"\x33" * 32,
    )

    bindings, documents_count = acquirer._bindings(placement)

    (model,) = bindings[binding_digest].model_bindings()
    assert tensorfs.parse_header(bytes(store.manifest(manifest_id)["header"]))["configs"] == {
        "unet": _CONFIG
    }
    assert model.reference_snapshot == manifest_id
    assert documents_count == 1  # PackageInterface only; config is not a downloaded document.
