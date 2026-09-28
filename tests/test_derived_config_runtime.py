"""Derived configs cross the real Runtime -> published TensorFS 0.3.0 boundary."""

from __future__ import annotations

import hashlib
import json
import struct
from io import BytesIO
from pathlib import Path
from typing import Any

import tensorfs
from tensorfs.derived import Config, Derivation, Part, Source, Target, Tensor

from cozy_runtime.author import ModelArtifact, ObjectRef
from cozy_runtime.internal.encoding import SPEC_PLAIN
from native_weights import NativeExecution
from test_model_runtime_closure import _CONFIG, _snapshot

_ADDED_INPUT = b'{ "value": 1, "kind": "added" }'
_ADDED_JCS = b'{"kind":"added","value":1}'
_DERIVED_INPUT = b'{ "layers": 2, "hidden_size": 8 }'
_DERIVED_JCS = b'{"hidden_size":8,"layers":2}'


def _released_source(tmp_path: Path) -> tuple[Any, str, int]:
    _root, snapshot, _snapshot_length, store = _snapshot(tmp_path, include_asset=True)
    header = bytes(store.manifest(snapshot)["header"])
    manifest_bytes = json.dumps(
        {
            "entries": [
                {
                    "blob": {"length": len(header), "sha256": hashlib.sha256(header).hexdigest()},
                    "kind": "cozytensors",
                    "path": "model.cozytensors",
                }
            ]
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    store.put_manifest(manifest_bytes, manifest, len(manifest_bytes))
    operation = store.begin_operation("derived-config-source", "test", "model")
    operation.hold_manifest(manifest, len(manifest_bytes))
    operation.commit_release(None, "1", "main", manifest, len(manifest_bytes))
    return store, manifest, len(manifest_bytes)


def test_add_and_derive_configs_land_as_inline_jcs(tmp_path: Path) -> None:
    store, manifest, manifest_length = _released_source(tmp_path)
    derived_stream = BytesIO(_DERIVED_INPUT)
    configs = {
        "added": Config("add"),
        "copied": Config("copy", "source", "unet"),
        "derived": Config("derive", "source", "unet"),
    }
    definition = Derivation(
        {"source": Source(manifest, manifest_length)},
        {
            "unet": Target(
                source="source",
                source_component="unet",
                add={"bias": Tensor("f32", (2,), SPEC_PLAIN, {"value": Part("f32", (2,))})},
            )
        },
        configs,
        (("unet", "weight"), ("unet", "bias")),
    )
    source = ModelArtifact(
        "source", "model", ObjectRef(manifest, manifest_length), "sha256:" + "11" * 32
    )
    with NativeExecution(
        store, tmp_path, "request-1", {"source": source}, {"result": 1 << 20}
    ) as owner:
        with owner.client.source(manifest) as capability:
            structure = capability.inspect()
        assert [
            (key, tensor.logical_dtype) for key, tensor in structure.components["unet"].items()
        ] == [("weight", "f32")]
        with owner.client.open_output("result", definition) as transaction:
            transaction.add_config("added", _ADDED_INPUT)
            transaction.add_config("derived", derived_stream)
            transaction.add_part("unet", "bias", "value", struct.pack("<2f", 1.0, 2.0))
            facts = transaction.commit()
        declared = owner.declaration(facts)
    assert declared["configs"] == {
        "added": {"kind": "add"},
        "copied": {"kind": "copy", "source": "source", "source_config": "unet"},
        "derived": {"kind": "derive", "source": "source", "source_config": "unet"},
    }
    produced = facts["manifest"]
    held = store.manifest("sha256:" + produced["sha256"])
    assert produced["length"] == len(held["manifest"])
    assert tensorfs.parse_header(bytes(held["header"]))["configs"] == {
        "added": _ADDED_JCS,
        "copied": _CONFIG,
        "derived": _DERIVED_JCS,
    }


def test_companion_files_cross_scoped_writer_and_survive_derivation(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "companions")
    files = {"LICENSE": b"Research licence\n", "Notice": b"Exact required attribution\n"}
    definition = Derivation(
        {},
        {"model": Target(add={"w": Tensor("f32", (1,), SPEC_PLAIN, {"value": Part("f32", (1,))})})},
        {},
        (("model", "w"),),
        files=files,
    )
    with NativeExecution(store, tmp_path, "documents", {}, {"model": 1024}) as owner:
        with owner.client.open_output("model", definition) as writer:
            writer.add_part("model", "w", "value", struct.pack("<f", 1.0))
            receipt = writer.commit()
        artifact = owner.client.adopt_model(receipt)
    output = tmp_path / "first-checkout"
    store.checkout(artifact.manifest.digest, output, symlink=False)
    for name, expected in files.items():
        assert (output / name).read_bytes() == expected
    inherited = Derivation(
        {"source": Source(artifact.manifest.digest, artifact.manifest.length)},
        {"model": Target(source="source", source_component="model")},
        {},
        (("model", "w"),),
    )
    with NativeExecution(
        store, tmp_path, "next-operation", {"source": artifact}, {"model": 0}
    ) as owner:
        with owner.client.open_output("model", inherited) as writer:
            receipt = writer.commit()
        derived = owner.client.adopt_model(receipt)
    assert receipt["added_objects"] == []
    output = tmp_path / "derived-checkout"
    store.checkout(derived.manifest.digest, output, symlink=False)
    for name, expected in files.items():
        assert (output / name).read_bytes() == expected
