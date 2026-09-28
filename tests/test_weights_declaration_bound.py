"""A weights declaration is bounded by what it declares, not by a small fixed constant.

SDXL's whole-model assembly declares ~2,500 tensors in 1.09 MB. Each declaration crosses the
real executor spool file, the worker's writer broker and TensorFS, the workspace journal and
the native writer open.
"""

from __future__ import annotations

import hashlib
import io
import struct
from pathlib import Path

import pytest
import tensorfs
from tensorfs.derived import Derivation, Part, PartSource, Source, Target, Tensor

from cozy_runtime.author import ModelArtifact, ObjectRef
from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.protocol import documents
from native_weights import NativeExecution


def _keys(count: int, width: int) -> list[str]:
    return [f"down_blocks.{index:05d}.".ljust(width, "w") + ".weight" for index in range(count)]


def _source(store: tensorfs.Store) -> ModelArtifact:
    tensor = {
        "logical_dtype": "f16",
        "shape": [4],
        "encoding": SPEC_PLAIN,
        "parts": {"value": {"dtype": "f16", "shape": [4]}},
    }
    writer = store.begin_derived(
        "sha256:" + "71" * 32, 1, {}, {"unet": {"drop": [], "add": {"base": tensor}}}, {},
        [("unet", "base")], 8, work_fingerprint="sha256:" + "72" * 32,
    )  # fmt: skip
    writer.add_part("unet", "base", "value", io.BytesIO(struct.pack("<4e", 0, 1, 2, 3)))
    manifest = writer.commit()["manifest"]
    reference = ObjectRef("sha256:" + manifest["sha256"], manifest["length"])
    return ModelArtifact("source", "model", reference, "sha256:" + "73" * 32)


@pytest.mark.parametrize(
    ("shape", "above"),
    [("sdxl-grafts", 1 << 20), ("additions", 16 << 20)],
)
def test_a_large_declaration_is_journaled_and_opens_its_writer(
    tmp_path: Path, shape: str, above: int
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    if shape == "sdxl-grafts":
        # SDXL `_targets`: every tensor grafted from its source, none written anew.
        source = _source(store)
        keys = _keys(2800, 60)
        graft = Part("f16", (4,), source=PartSource("source", "unet", "base", "value"))
        sources = {"source": Source(source.manifest.digest, source.manifest.length)}
        models, bound = {"source": source}, 0
    else:
        keys = _keys(9000, 1100)
        graft = Part("f16", (4,))
        sources, models, bound = {}, {}, 8 * len(keys)
    definition = Derivation(
        sources,
        {
            "unet": Target(
                add={key: Tensor("f16", (4,), SPEC_PLAIN, {"value": graft}) for key in keys}
            )
        },
        {},
        tuple(("unet", key) for key in keys),
    )
    with (
        NativeExecution(store, tmp_path, shape, models, {"model": bound}) as owner,
        owner.client.open_output("model", definition) as writer,
    ):
        if shape == "additions":
            writer.add_part("unet", keys[0], "value", struct.pack("<4e", 0, 1, 2, 3))
        (transaction,) = {key[2] for key in owner.owner.writer_broker.bindings}
        assert owner.owner.workspace is not None
        row = owner.owner.workspace.weights_row("package-proof", transaction)
        declaration = row["declaration"]
        assert len(declaration) > above
        assert row["declaration_digest"] == hashlib.sha256(declaration).digest()


def test_a_whole_model_declaration_commits_with_a_constant_size_receipt(tmp_path: Path) -> None:
    # SDXL `assemble_normalized`: every tensor inherited from its source, all in order.
    store = tensorfs.Store.init(tmp_path / "store")
    source = _source(store)
    keys = _keys(2800, 60)
    graft = Part("f16", (4,), source=PartSource("source", "unet", "base", "value"))
    definition = Derivation(
        {"source": Source(source.manifest.digest, source.manifest.length)},
        {
            "unet": Target(
                add={key: Tensor("f16", (4,), SPEC_PLAIN, {"value": graft}) for key in keys}
            )
        },
        {},
        tuple(("unet", key) for key in keys),
    )
    with NativeExecution(store, tmp_path, "assemble", {"source": source}, {"model": 0}) as owner:
        with owner.client.open_output("model", definition) as writer:
            receipt = writer.commit()
        artifact = owner.client.adopt_model(receipt)
    assert owner.owner.workspace is not None
    row = owner.owner.workspace.weights_row("package-proof", str(receipt["transaction_id"]))
    assert len(row["declaration"]) > 1 << 20
    # The receipt names the declaration by digest, so it stays small however large the model.
    assert row["receipt"] and len(row["receipt"]) < 16 << 10
    assert receipt["declaration_digest"] == documents.spell(row["declaration_digest"])
    header_bytes = store.manifest(artifact.manifest.digest)["header"]
    assert header_bytes is not None
    header = tensorfs.parse_header(header_bytes)
    assert list(header["components"]["unet"]) == keys
