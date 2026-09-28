"""Model preparation consumes one CozyTensors file, not every snapshot sibling."""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict

import pytest
import tensorfs
from tensorfs.errors import RepositoryAbsent

from cozy_runtime.author import Artifact, CapabilityError, Config, Loader
from cozy_runtime.author._loader import loading
from cozy_runtime.internal import base_observation, canonical, fill, package_interface
from cozy_runtime.internal.census_cache import CensusRefusal
from cozy_runtime.internal.encoding import SPEC_PLAIN, launch_providers
from cozy_runtime.internal.worker import grants
from cozy_runtime.internal.worker.acquire import Acquirer, AcquisitionRefusal, HeldManifests
from cozy_runtime.internal.worker.attempts import AttemptEngine, AttemptRefusal
from cozy_runtime.internal.worker.package_prepare import (
    Construction,
    PreparationRefusal,
    _selected_models,
    selections,
)
from cozy_runtime.protocol import documents
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
    placement = pb.Placement(
        models=[
            pb.Model(manifest=pb.Ref(digest=documents.raw(manifest_id), length=manifest_length))
        ]
    )

    count = _acquirer(tmp_path, store_root)._models(placement)

    assert count == 1  # One manifest held; its closure is the header and tokenizer asset.
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
    placement = pb.Placement(
        models=[
            pb.Model(manifest=pb.Ref(digest=documents.raw(manifest_id), length=manifest_length))
        ]
    )

    with pytest.raises(AcquisitionRefusal) as refused:
        _acquirer(tmp_path, store_root)._models(placement)

    assert refused.value.code == "model_materialization_required"
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


def _prepared(root: Path, selection: dict[str, Any]) -> Construction:
    built = _selected_models(
        "example/package",
        selections([dict(selection, package="example/package")]),
        root,
        lambda *_args: None,
        construction_slots={selection["slot"]},
    )[selection["slot"]].construction
    assert built is not None
    return built


def test_prepare_reads_header_and_assets_without_a_closure_lease(tmp_path: Path) -> None:
    """Preparation reads the manifest, the CozyTensors header and -- only when the census
    misses -- the header-declared assets through a lease on those segments alone, never the
    tensor closure; an absent asset is a resumable refusal."""

    store_root, manifest_id, _manifest_length, _store = _snapshot(tmp_path, include_asset=True)
    selected = {
        "slot": "generate.models.model",
        "manifest": manifest_id,
        "model": "example/model",
        "release": "",
        "lane": "",
    }
    candidate = _prepared(store_root, selected)
    assert candidate.config_bytes == _CONFIG
    assert candidate.assets() == {"tokenizer/vocab.txt": _ASSET}
    assert candidate.tensor_dtypes == {"unet.weight": "f32"}

    mismatch = tensorfs.fit(
        [tensorfs.TensorRequirement("unet", "weight", [2], "f32")],
        candidate.header,
        custody="canonical",
        encoded_leaves=False,
    )
    assert not mismatch["ok"] and mismatch["code"] == "shape_mismatch"

    # The asset object was never admitted: the header still reads, the asset lease refuses.
    cold_root, cold_manifest, _cold_length, _cold_store = _snapshot(
        tmp_path / "cold", include_asset=False
    )
    cold = _prepared(cold_root, dict(selected, manifest=cold_manifest))
    with pytest.raises(CensusRefusal, match="model_asset_unavailable"):
        cold.assets()


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
    slot = pb.Slot(
        slot="model",
        reference_model_id="model-0000",
        components=[pb.Component(component="unet", model_id="model-0000")],
    )
    binding_digest = canonical.digest({"name": "generate", "slots": [documents.body(slot)]})
    placement = pb.Placement(
        package_interface=interface_bytes,
        models=[
            pb.Model(
                id="model-0000",
                repo="example/model",
                version="1",
                lane="main",
                manifest=pb.Ref(digest=documents.raw(manifest_id), length=manifest_length),
            )
        ],
        entrypoints=[
            pb.Entrypoint(
                name="generate",
                entrypoint_binding_digest=documents.raw(binding_digest),
                slots=[slot],
            )
        ],
        package=pb.PackageSelection(package="example/package", release="1"),
        installation_id="local-" + "33" * 16,
    )

    assert acquirer._models(placement) == 1
    bindings, documents_count = acquirer._bindings(placement)

    (model,) = bindings[binding_digest].model_bindings()
    assert tensorfs.parse_header(bytes(store.manifest(manifest_id)["header"]))["configs"] == {
        "unet": _CONFIG
    }
    assert model.reference_snapshot == manifest_id
    assert documents_count == 1  # PackageInterface only; config is not a downloaded document.


def _checkpoint_selection(manifest: str, **fields: Any) -> dict[str, Any]:
    return {
        "package": "example/package",
        "slot": "generate.models.model",
        "model": "example/model",
        "manifest": manifest,
        **fields,
    }


@pytest.mark.parametrize("named_fields", [{}, {"release": "", "lane": ""}])
def test_exact_checkpoint_prepares_and_streams_without_any_release(
    tmp_path: Path, named_fields: dict[str, str]
) -> None:
    root, manifest, length, store = _snapshot(tmp_path, include_asset=True, checkpoint_only=True)
    selection = _checkpoint_selection(manifest, **named_fields)
    original_objects = store.objects()
    for _ in range(2):
        # Uploaded checkpoints have no release aliases; preparing one must not
        # invent such an alias, even when all bytes are already warm locally.
        with pytest.raises(RepositoryAbsent):
            store.resolve_release("example", "model", "1", "main")
        prepared = _selected_models(
            "example/package",
            selections([selection]),
            root,
            lambda *_args: None,
            construction_slots={selection["slot"]},
        )
        chosen = prepared[selection["slot"]]
        assert chosen.manifest_length == length
        assert chosen.construction is not None
        assert chosen.construction.config_bytes == _CONFIG
        assert chosen.construction.assets() == {"tokenizer/vocab.txt": _ASSET}
        assert chosen.construction.tensor_dtypes == {"unet.weight": "f32"}
        checkpoint = fill.Checkpoint(root, manifest)
        plan = checkpoint.read_plan([("unet", "weight")], 64, ["unet"])
        slot = bytearray(b"\xa5" * 64)
        with checkpoint.acquire() as lease:

            def consume(batch: Any) -> None:
                assert batch.nbytes == 4
                assert batch.release()

            stats = lease.stream(plan, [slot], consume, readers=1)
        assert stats["items"] == 1 and stats["bytes"] == 4
        assert struct.unpack("<f", slot[:4]) == (0.0,)
        assert slot[4:] == b"\xa5" * 60
        assert store.objects() == original_objects


@pytest.mark.parametrize(
    "fields",
    [
        {"release": "1"},
        {"lane": "bf16"},
        {"release": "1", "lane": ""},
        {"release": "", "lane": "bf16"},
        {"model": "/model"},
        {"model": "example/"},
    ],
)
def test_checkpoint_selection_refuses_partial_release_or_incomplete_model(
    tmp_path: Path, fields: dict[str, str]
) -> None:
    root, manifest, _length, _store = _snapshot(tmp_path, include_asset=True, checkpoint_only=True)
    selection = _checkpoint_selection(manifest, **fields)
    with pytest.raises(PreparationRefusal, match="model_selection_mismatch"):
        _selected_models(
            "example/package",
            selections([selection]),
            root,
            lambda *_args: None,
            construction_slots={selection["slot"]},
        )


def test_checkpoint_selection_keeps_identity_and_geometry_checks(tmp_path: Path) -> None:
    root, manifest, length, _store = _snapshot(tmp_path, include_asset=True, checkpoint_only=True)
    selection = _checkpoint_selection(manifest, release="", lane="")
    # The digest is the identity; the length comes from the held bytes, not the selection.
    for declared_length in (-1, length + 1):
        prepared = _selected_models(
            "example/package",
            selections([dict(selection, manifest_length=declared_length)]),
            root,
            lambda *_args: None,
            construction_slots={selection["slot"]},
        )
        assert prepared[selection["slot"]].manifest_length == length
    with pytest.raises(PreparationRefusal, match="model_selection_mismatch"):
        _selected_models(
            "other/package",
            selections([selection]),
            root,
            lambda *_args: None,
            construction_slots={selection["slot"]},
        )
    missing = dict(selection, manifest="sha256:" + "f" * 64)
    with pytest.raises(PreparationRefusal):
        _selected_models(
            "example/package",
            selections([missing]),
            root,
            lambda *_args: None,
            construction_slots={selection["slot"]},
        )
    candidate = _prepared(root, selection)
    mismatch = tensorfs.fit(
        [tensorfs.TensorRequirement("unet", "weight", [2], "f32")],
        candidate.header,
        custody="canonical",
        encoded_leaves=False,
    )
    assert not mismatch["ok"] and mismatch["code"] == "shape_mismatch"


def test_checkpoint_selection_refuses_corrupt_header(tmp_path: Path) -> None:
    root, manifest, _length, _store = _snapshot(tmp_path, include_asset=True, checkpoint_only=True)
    # Locate the fixture object without encoding TensorFS's storage fanout rules.
    stored = next(root.rglob(hashlib.sha256(_HEADER).hexdigest()))
    stored.chmod(stored.stat().st_mode | 0o200)
    stored.write_bytes(b"x" * len(_HEADER))
    selection = _checkpoint_selection(manifest, release="", lane="")
    with pytest.raises(PreparationRefusal):
        _selected_models(
            "example/package",
            selections([selection]),
            root,
            lambda *_args: None,
            construction_slots={selection["slot"]},
        )


def test_evicted_preparation_rechecks_native_bytes(tmp_path: Path) -> None:
    root, manifest, length, _native = _snapshot(tmp_path, include_asset=True)
    placement = pb.Placement(
        models=[pb.Model(manifest=pb.Ref(digest=documents.raw(manifest), length=length))]
    )
    acquirer = _acquirer(tmp_path, root)
    assert acquirer._models(placement) == 1
    # These are unowned cached bytes. An earlier preparation is not a root.
    assert tensorfs.gc(str(root))["reclaimed_bytes"] > 0
    with pytest.raises(AcquisitionRefusal) as missing:
        acquirer._models(placement)
    assert missing.value.code == "model_materialization_required"
    assert missing.value.kind == pb.FAULT_KIND_ARTIFACT_FETCH_FAILED


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
