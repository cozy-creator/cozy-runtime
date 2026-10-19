"""A cozytensors/1 checkpoint read through the wheel: rows in construction order with their
scheme and parts, the census fit, configs from files, and the chunks a fetch plan names."""

from __future__ import annotations

import socket
import threading
from pathlib import Path

import numpy as np
import pytest
import tensorfs

from cozy_runtime.internal import fetch_plan, model_config
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.fill import (
    HEADER,
    Checkpoint,
    FillRefusal,
    TensorRequirement,
    fit,
    normalized,
    tensorfs_requirement_dtype,
)
from cozy_runtime.internal.seam import Channel
from tensord_peer import answer_model_sources


def _store(root: Path) -> tuple[tensorfs.Store, Checkpoint]:
    store = tensorfs.Store(root)
    writer = store.writer()
    writer.put("unet", "conv_in.weight", "F16", [8, 4], np.ones((8, 4), np.float16))
    big = np.arange(20 << 18, dtype=np.float32)  # 20 MiB: chunks of its own
    writer.put("unet", "mid.weight", "F32", [20 << 18], big)
    writer.put_encoded(
        "unet",
        "proj.weight",
        "BF16",
        [4, 32],
        "fp8.rowwise",
        {
            "weight": ("F8_E4M3", [4, 32], np.full((4, 32), 0x38, np.uint8)),
            "scale": ("F32", [4], np.ones(4, np.float32)),
        },  # fmt: skip
        {"axis": 0, "scale_dtype": "F32"},
    )
    writer.put("vae", "decoder.weight", "F32", [3], np.ones(3, np.float32))
    writer.file("unet/config.json", b'{"in": 4}')
    writer.file("scheduler/scheduler_config.json", b'{"steps": 2}')
    writer.file("tokenizer/vocab.json", b'{"a": 0}')
    writer.file("tokenizer/merges.txt", b"#version: 0.2\na b\n")
    writer.file("assets/palette.bin", bytes(range(256)))
    return store, Checkpoint.from_store(store, f"sha256:{writer.commit().sha256}")


def test_rows_carry_order_scheme_params_and_parts(tmp_path: Path) -> None:
    _, ck = _store(tmp_path)
    rows = ck.rows("unet")
    assert [r.name for r in rows] == list(ck.ck.tensors["unet"])
    encoded = next(r for r in rows if r.name == "proj.weight")
    assert (encoded.encoded.scheme, dict(encoded.encoded.params)) == (
        "fp8.rowwise",
        {"axis": 0, "scale_dtype": "F32"},
    )
    assert {p.role: (p.dtype, p.shape) for p in encoded.encoded.parts} == {
        "weight": ("f8_e4m3fn", (4, 32)),
        "scale": ("f32", (4,)),
    }
    assert (encoded.dtype, encoded.nbytes) == ("bf16", 4 * 32 * 2)
    plain = next(r for r in rows if r.name == "conv_in.weight")
    assert plain.encoded.scheme == "plain" and plain.encoded.parts[0].role == "value"
    with pytest.raises(FillRefusal) as refused:
        ck.rows("text_encoder")
    assert refused.value.code == "destination_absent"


def test_an_executor_gets_every_manifest_file_through_model_source(tmp_path: Path) -> None:
    """Through TensorD (`ModelSource`) an executor reads the header and every file the manifest
    lists (configs, tokenizer, assets), byte for byte, each asked once."""
    store, ck = _store(tmp_path / "store")
    left, right = socket.socketpair()
    asked: list[str] = []
    machine = Channel(right)
    serving = threading.Thread(target=answer_model_sources, args=(machine, store, asked))
    serving.start()
    try:
        granted = Executor(Channel(left), tmp_path)._checkpoint(ck.manifest_id)
        stored = {path: ck.file(path) for path in ck.files if path != HEADER}
        assert granted.files == ck.files and len(stored) == 5
        assert granted.model_assets() == stored
        assert (
            granted.configs()
            == ck.configs()
            == {
                "unet": b'{"in": 4}',
                "scheduler": b'{"steps": 2}',
            }
        )
        assert [r.key for r in granted.rows("unet")] == [r.key for r in ck.rows("unet")]
        assert granted.ck.chunks == ck.ck.chunks
        granted.model_assets()
        assert sorted(asked) == sorted(["", ".", *stored]), "each name asked once, never by HEADER"
    finally:
        left.close()
        serving.join()
        right.close()


def test_configs_come_from_files_and_the_fit_judges_the_census(tmp_path: Path) -> None:
    _, ck = _store(tmp_path)
    assert ck.configs() == {"unet": b'{"in": 4}', "scheduler": b'{"steps": 2}'}
    assert (
        model_config.construction_config(ck.configs())
        == b'{"scheduler":{"steps":2},"unet":{"in":4}}'
    )
    assert normalized(ck) == ""
    built = [
        TensorRequirement("unet", "conv_in.weight", (8, 4)),
        TensorRequirement("unet", "proj.weight", (4, 32)),
    ]
    verdict = fit(built, ck, encoded_leaves=True, custody="canonical", device="cpu")
    assert verdict["ok"] and verdict["ignored"] == ["unet/mid.weight"]
    assert verdict["routes"] == {"verbatim": 1, "decoded_float": 0, "encoded_gemm": 1}
    assert (verdict["custody"], verdict["device"]) == ("canonical", "cpu")
    assert verdict["order"] == "", "the census builds in the header's construction order"
    swapped = fit(built[::-1], ck, encoded_leaves=True)
    assert swapped["ok"] and "builds proj.weight where" in str(swapped["order"])
    refused = fit(built, ck, encoded_leaves=False)
    assert (refused["code"], refused["key"], refused["encoding"]) == (
        "encoding_unsupported",
        "proj.weight",
        "fp8.rowwise",
    )
    assert (
        fit([TensorRequirement("unet", "conv_in.weight", (4, 8))], ck, encoded_leaves=True)["code"]
        == "shape_mismatch"
    )
    assert fit([TensorRequirement("text", "x", (1,))], ck, encoded_leaves=True)["code"] == (
        "component_missing"
    )


def test_fetch_plans_name_each_tensors_chunks(tmp_path: Path) -> None:
    store, ck = _store(tmp_path)
    keys = list(ck.ck.tensors["unet"])
    every = fetch_plan.component_objects(ck.ck, "unet", keys)
    big = fetch_plan.tensor_chunks(ck.ck, "unet", "mid.weight")
    assert len(big) == 1 and big[0] in every
    assert set(every) <= set(ck.ck.chunks)
    regions = fetch_plan.region_objects(ck.ck, [["unet/conv_in.weight"], ["unet/mid.weight"]])
    assert regions[1] == big
    tail = fetch_plan.component_objects(ck.ck, "unet", keys, keys.index("proj.weight"))
    assert set(tail) <= set(every)
    for sha, length in every:
        assert Path(store.root, "blobs", sha[:2], sha[2:4], sha).stat().st_size == length


def test_a_transformers_model_binds_its_stored_names_through_its_conversion_mapping(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.CLIPTextConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, max_position_embeddings=8, projection_dim=8,
    )  # fmt: skip
    built = {
        "text_encoder": transformers.CLIPTextModel(config),
        "text_encoder_2": transformers.CLIPTextModelWithProjection(config),
    }
    store = tensorfs.Store(tmp_path)
    writer = store.writer()
    for component, model in built.items():
        model.config.architectures = [type(model).__name__]  # as `save_pretrained` writes it
        for key, value in model.state_dict().items():  # upstream on-disk names
            stored = (
                key if key.startswith(("text_model.", "text_projection.")) else f"text_model.{key}"
            )
            writer.put(component, stored, "F32", list(value.shape), value.numpy())
        writer.file(f"{component}/config.json", model.config.to_json_string().encode())
    ck = Checkpoint.from_store(store, f"sha256:{writer.commit().sha256}")
    assert next(iter(ck.ck.tensors["text_encoder"])).startswith("text_model.")
    for component, model in built.items():
        keys = list(model.state_dict())
        rows = ck.rows(component)
        assert sorted(r.key for r in rows) == sorted(f"{component}.{k}" for k in keys), component
        assert all(r.name.startswith(("text_model.", "text_projection.")) for r in rows)
    requirements = [
        TensorRequirement(component, key, tuple(value.shape))
        for component, model in built.items()
        for key, value in model.state_dict().items()
    ]
    verdict = fit(requirements, ck, encoded_leaves=False)
    assert verdict["ok"] and verdict["ignored"] == [], verdict
    assert torch.equal(
        torch.from_dlpack(ck.ck.read("text_encoder", "text_model.final_layer_norm.weight")),
        built["text_encoder"].final_layer_norm.weight.detach(),
    )


@pytest.mark.parametrize(
    ("torch_spelling", "stored"),
    [("torch.bfloat16", "BF16"), ("torch.float16", "F16"), ("torch.float32", "F32")],
)
def test_the_census_dtype_a_fit_pins_is_the_runtimes_name(
    tmp_path: Path, torch_spelling: str, stored: str
) -> None:
    store = tensorfs.Store(tmp_path)
    writer = store.writer()
    writer.put(
        "unet",
        "weight",
        stored,
        [1],
        np.zeros(1, {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[stored]),
    )
    ck = Checkpoint.from_store(store, f"sha256:{writer.commit().sha256}")
    pinned = tensorfs_requirement_dtype(torch_spelling)
    assert pinned == (None if stored != "F32" else "f32"), "bf16 and f16 stay convertible"
    assert fit([TensorRequirement("unet", "weight", (1,), pinned)], ck, encoded_leaves=False)["ok"]
    refused = fit([TensorRequirement("unet", "weight", (1,), "bfloat16")], ck, encoded_leaves=False)
    assert refused["code"] == "dtype_mismatch", "a torch name is not a Runtime dtype"
