"""Derived configs cross the real Runtime -> published TensorFS 0.3.0 boundary."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import tensorfs

from cozy_runtime.author._services import Attempt
from cozy_runtime.author._weights import (
    WeightsCommit,
    WeightsConfig,
    WeightsTarget,
    WeightsTransaction,
)
from cozy_runtime.internal.weights_sink import WeightsTransactionHost
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
        "added": WeightsConfig(data=_ADDED_INPUT),
        "copied": WeightsConfig(source="source", source_config="unet"),
        "derived": WeightsConfig(
            source="source",
            source_config="unet",
            data=derived_stream,
            length=len(_DERIVED_INPUT),
        ),
    }
    targets = {"unet": WeightsTarget(source="source", source_component="unet")}
    request = WeightsCommit(
        output_slot="result",
        sources={"source": manifest},
        targets=targets,
        configs=configs,
        order=(("unet", "weight"),),
        max_new_bytes=1 << 20,
    )
    host = WeightsTransactionHost(
        store=store,
        owner_scope="derived-config-test",
        request_id="request-1",
        invocation_spec_digest="sha256:" + "11" * 32,
        writer_session_id=1,
        allowed_sources={manifest: manifest_length},
        output_bounds={"result": 1 << 20},
    )
    structure = host.structure(manifest)
    assert [(tensor.key, tensor.logical_dtype) for tensor in structure.tensors] == [
        ("weight", "f32")
    ]
    transaction = WeightsTransaction(
        attempt=Attempt("request-1", tmp_path / "spool"),
        cancel=lambda: False,
        output="result",
        sources=request.sources,
        targets=targets,
        configs=configs,
        host=host.open(request),
    )

    transaction.add_config("added", _ADDED_INPUT)
    transaction.add_config("derived", derived_stream)
    receipt = transaction.commit()

    facts = json.loads(receipt.tensorfs_receipt)
    assert facts["declaration"]["configs"] == {
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
