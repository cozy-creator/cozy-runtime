"""Declared part grafts cross the real Runtime/native grant and receipt boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import tensorfs
from tensorfs.derived import Derivation, Part, PartSource, Source, Target, Tensor

from cozy_runtime.author import CapabilityError, ModelArtifact, ObjectRef
from native_weights import NativeExecution


def _host(
    store: tensorfs.Store, request: str, grants: dict[str, int], limit: int
) -> NativeExecution:
    return NativeExecution(
        store,
        Path(store.root).parent,
        request,
        {
            str(i): ModelArtifact(
                "source", "model", ObjectRef(digest, length), "sha256:" + "11" * 32
            )
            for i, (digest, length) in enumerate(grants.items())
        },
        {"result": limit},
    )


def _tensor(source: PartSource | None = None) -> Tensor:
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    return Tensor("f32", (512,), plain, {"value": Part("f32", (512,), source=source)})


def _source(store: tensorfs.Store, name: str, byte: int) -> tuple[str, int, str]:
    definition = Derivation(
        {}, {"model": Target(add={"weight": _tensor()})}, {}, (("model", "weight"),)
    )
    with (
        _host(store, name, {}, 2048) as owner,
        owner.client.open_output("result", definition) as writer,
    ):
        writer.add_part("model", "weight", "value", bytes([byte]) * 2048)
        receipt = writer.commit()
    return (
        "sha256:" + receipt["manifest"]["sha256"],
        receipt["manifest"]["length"],
        receipt["transaction_id"],
    )


def test_graft_is_native_declared_granted_and_retained(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    body, body_length, body_id = _source(store, "body", 0x11)
    bank, bank_length, bank_id = _source(store, "bank", 0x22)
    selected = PartSource("bank", "model", "weight", "value")
    request = Derivation(
        {"body": Source(body, body_length), "bank": Source(bank, bank_length)},
        {"model": Target("body", "model", add={"table": _tensor(selected)})},
        {},
        (("model", "weight"), ("model", "table")),
    )
    grants = {body: body_length, bank: bank_length}
    host = _host(store, "graft", grants, 0)
    writer = host.client.open_output("result", request)
    # Both source producers can release their result while this writer owns the sources.
    store.derived_dispose(body_id)
    store.derived_dispose(bank_id)
    receipt = writer.commit()
    facts = receipt
    assert host.declaration(facts)["components"]["model"]["add"]["table"]["parts"]["value"][
        "source"
    ] == {"source": "bank", "component": "model", "tensor": "weight", "role": "value"}
    assert facts["added_objects"] == []
    assert facts["inherit_observation"] == {"bytes": 4096, "hashes": 0, "objects": 2, "reads": 0}
    with host.client.open_output("result", request) as replay:
        assert replay.receipt == receipt
    bank_bytes = store.manifest(bank)["header"]
    result_bytes = store.manifest("sha256:" + facts["manifest"]["sha256"])["header"]
    assert bank_bytes is not None and result_bytes is not None
    bank_header = tensorfs.parse_header(bank_bytes)
    header = tensorfs.parse_header(result_bytes)
    assert header["components"]["model"]["table"] == bank_header["components"]["model"]["weight"]

    with pytest.raises(CapabilityError) as ungranted:
        _host(store, "ungranted", {body: body_length}, 0).client.open_output("result", request)
    assert ungranted.value.code == "weights_source_ungranted"
    with pytest.raises(CapabilityError):
        host.client.open_output(
            "result",
            replace(
                request,
                targets={
                    "model": Target(
                        "body", "model", add={"table": _tensor(replace(selected, tensor="absent"))}
                    )
                },
            ),
        )


def test_graft_source_is_validated_natively_and_has_no_payload_field(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    source = PartSource("bank", "model", "table", "value")
    with pytest.raises(TypeError):
        Part("f32", (512,), data=bytes(2048), source=source)  # type: ignore[call-arg]
    definition = Derivation(
        {"bank": Source("sha256:" + "11" * 32, 1)},
        {"model": Target(add={"table": _tensor(replace(source, source=""))})},
        {},
        (("model", "table"),),
    )
    with pytest.raises(tensorfs.errors.Refusal):
        store.derived_declaration(
            *definition.native_arguments(0), work_fingerprint="sha256:" + "11" * 32
        )
