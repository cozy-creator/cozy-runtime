"""Declared part grafts cross the real Runtime/native grant and receipt boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.author import (
    CapabilityError,
    WeightsPart,
    WeightsPartSource,
    WeightsTarget,
    WeightsTensor,
)
from cozy_runtime.author._weights import WeightsCommit
from cozy_runtime.internal.weights_sink import WeightsTransactionHost


def _host(
    store: tensorfs.Store, request: str, grants: dict[str, int], limit: int
) -> WeightsTransactionHost:
    return WeightsTransactionHost(
        store=store,
        owner_scope="graft-proof",
        request_id=request,
        invocation_spec_digest="sha256:" + "91" * 32,
        work_fingerprint="sha256:" + "92" * 32,
        writer_session_id=1,
        allowed_sources=grants,
        output_bounds={"result": limit},
    )


def _tensor(source: WeightsPartSource | None = None) -> WeightsTensor:
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    return WeightsTensor("f32", (512,), plain, {"value": WeightsPart("f32", (512,), source=source)})


def _source(store: tensorfs.Store, name: str, byte: int) -> tuple[str, int, str]:
    request = WeightsCommit(
        "result",
        {},
        {"model": WeightsTarget(add={"weight": _tensor()})},
        {},
        (("model", "weight"),),
        2048,
    )
    writer = _host(store, name, {}, 2048).open(request)
    writer.add_part("model", "weight", "value", bytes([byte]) * 2048)
    receipt = writer.commit()
    manifest = json.loads(receipt.tensorfs_receipt)["manifest"]
    return "sha256:" + manifest["sha256"], manifest["length"], receipt.weights_transaction_id


def test_graft_is_native_declared_granted_and_retained(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    body, body_length, body_id = _source(store, "body", 0x11)
    bank, bank_length, bank_id = _source(store, "bank", 0x22)
    selected = WeightsPartSource("bank", "model", "weight", "value")
    request = WeightsCommit(
        "result",
        {"body": body, "bank": bank},
        {"model": WeightsTarget("body", "model", add={"table": _tensor(selected)})},
        {},
        (("model", "weight"), ("model", "table")),
        0,
    )
    grants = {body: body_length, bank: bank_length}
    host = _host(store, "graft", grants, 0)
    writer = host.open(request)
    # Both source producers can release their result while this writer owns the sources.
    store.derived_dispose(body_id)
    store.derived_dispose(bank_id)
    receipt = writer.commit()
    facts = json.loads(receipt.tensorfs_receipt)
    assert facts["declaration"]["components"]["model"]["add"]["table"]["parts"]["value"][
        "source"
    ] == {"source": "bank", "component": "model", "tensor": "weight", "role": "value"}
    assert facts["added_objects"] == []
    assert facts["inherit_observation"] == {"bytes": 4096, "hashes": 0, "objects": 2, "reads": 0}
    assert host.open(request).commit().tensorfs_receipt == receipt.tensorfs_receipt
    bank_header = tensorfs.parse_header(bytes(store.manifest(bank)["header"]))
    header = tensorfs.parse_header(
        bytes(store.manifest("sha256:" + facts["manifest"]["sha256"])["header"])
    )
    assert header["components"]["model"]["table"] == bank_header["components"]["model"]["weight"]

    with pytest.raises(CapabilityError, match="immutable grant"):
        _host(store, "ungranted", {body: body_length}, 0).open(request)
    with pytest.raises(CapabilityError):
        host.open(
            replace(
                request,
                targets={
                    "model": WeightsTarget(
                        "body", "model", add={"table": _tensor(replace(selected, tensor="absent"))}
                    )
                },
            )
        )


def test_graft_source_is_closed_and_cannot_supply_bytes() -> None:
    source = WeightsPartSource("bank", "model", "table", "value")
    with pytest.raises(CapabilityError, match="never both"):
        WeightsPart("f32", (512,), data=bytes(2048), source=source)
    with pytest.raises(CapabilityError):
        WeightsPartSource("", "model", "table", "value")
