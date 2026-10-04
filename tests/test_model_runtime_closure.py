"""Model preparation consumes one CozyTensors file, not every snapshot sibling."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict

import pytest
import tensorfs

from cozy_runtime.author import Artifact, CapabilityError, Config, Loader
from cozy_runtime.author._loader import loading
from cozy_runtime.internal import fill
from cozy_runtime.internal.encoding import SPEC_PLAIN, launch_providers
from cozy_runtime.internal.worker import grants
from cozy_runtime.internal.worker.attempts import AttemptEngine, AttemptRefusal
from cozy_runtime.protocol import worker_pb2 as pb

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


def _snapshot(
    tmp_path: Path, *, include_asset: bool, checkpoint_only: bool = False
) -> tuple[Path, str, int, Any]:
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
    if checkpoint_only:
        manifest["entries"] = [row for row in manifest["entries"] if row["kind"] == "cozytensors"]
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_id = "sha256:" + hashlib.sha256(raw).hexdigest()
    store.put_manifest(raw, manifest_id, len(raw))
    return store_root, manifest_id, len(raw), store


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
    writes: list[str] = []

    def record(event: str, args: tuple[Any, ...]) -> None:
        if watching and event == "open" and any(c in str(args[1] or "r") for c in "wxa+"):
            writes.append(str(args[0]))
        elif watching and event in {"os.mkdir", "os.rename", "os.remove", "shutil.rmtree"}:
            writes.append(f"{event}{args[:1]}")

    watching = False
    sys.addaudithook(record)
    watching = True
    try:
        with loading(loader):
            assert capability.names() == ("tokenizer/vocab.txt",)
            assert capability.read("tokenizer/vocab.txt") == _ASSET
            assert capability.open("tokenizer/vocab.txt").read() == _ASSET
    finally:
        watching = False
    assert writes == []  # Serving load reads asset bytes; it writes no file.
    assert not hasattr(capability, "materialized")
    with pytest.raises(CapabilityError, match="outside load/unload"):
        capability.read("tokenizer/vocab.txt")


def test_worker_prepare_reads_tensorfs_03_logical_dtype(tmp_path: Path) -> None:
    """Exercise the exact Checkpoint.rows path the executor's worker prepare calls."""

    store_root, manifest_id, _manifest_length, _store = _snapshot(tmp_path, include_asset=True)
    rows = fill.Checkpoint(store_root, manifest_id).rows("unet")
    assert [(row.key, row.dtype, row.shape) for row in rows] == [("unet.weight", "f32", (1,))]
    assert rows[0].encoded.encoding == SPEC_PLAIN
    assert SPEC_PLAIN in launch_providers()


def test_job_input_lease_owns_bytes_before_acceptance(tmp_path: Path) -> None:
    root, manifest, length, native = _snapshot(tmp_path, include_asset=False)
    entry = grants.BoundInput(
        "model.source", "", bytes.fromhex(manifest[7:]), length, grants.MODEL_MIME, 0
    )
    grant = grants.BoundGrant(inputs={"model:source": entry})
    attempt: Any = SimpleNamespace(grant=grant, model_leases=[], job_models={})
    engine: Any = SimpleNamespace(
        tensorfs_root=root, _job_model_declarations=lambda *args: {"source": "Model"}
    )
    binding: Any = None
    with pytest.raises(AttemptRefusal) as missing:
        AttemptEngine._hold_job_models(engine, attempt, binding, "descriptor")
    assert missing.value.code == "model_materialization_required"
    assert missing.value.cause == pb.CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE
    assert attempt.model_leases == []
    # The same already-bound input succeeds after only the missing asset arrives.
    asset = tmp_path / "late-asset"
    asset.write_bytes(_ASSET)
    native.put_file(str(asset), "sha256:" + hashlib.sha256(_ASSET).hexdigest(), len(_ASSET))
    AttemptEngine._hold_job_models(engine, attempt, binding, "descriptor")
    assert len(attempt.model_leases) == 1
    try:
        with pytest.raises(Exception) as busy:
            tensorfs.gc(str(root))
        assert getattr(busy.value, "code", "") == "STORE_BUSY"
    finally:
        AttemptEngine._release_job_models(engine, attempt)
    assert tensorfs.gc(str(root))["reclaimed_bytes"] > 0


def test_job_model_identity_mismatch_is_not_a_cache_miss(tmp_path: Path) -> None:
    root, manifest, length, _native = _snapshot(tmp_path, include_asset=True)
    entry = grants.BoundInput(
        "model:source", "", bytes.fromhex(manifest[7:]), length + 1, grants.MODEL_MIME, 0
    )
    attempt: Any = SimpleNamespace(
        grant=grants.BoundGrant(inputs={"model:source": entry}), model_leases=[], job_models={}
    )
    engine: Any = SimpleNamespace(
        tensorfs_root=root, _job_model_declarations=lambda *args: {"source": "Model"}
    )
    binding: Any = None
    with pytest.raises(AttemptRefusal) as invalid:
        AttemptEngine._hold_job_models(engine, attempt, binding, "descriptor")
    assert invalid.value.code == "job_model_manifest_unavailable"
    assert invalid.value.cause == pb.CAUSE_CODE_LOCAL_SAFETY
    assert attempt.model_leases == []
