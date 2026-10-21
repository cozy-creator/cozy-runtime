"""A D10 checkpoint opened from the store through the wheel: rows in construction order with their
scheme and parts, configs from its components, assets at their Diffusers/Transformers paths, and
the census fit."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import numpy as np
import pytest
import tensorfs

from cozy_runtime.author._loader import ModelAssets
from cozy_runtime.internal import model_config
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.fill import (
    Checkpoint,
    FillRefusal,
    TensorRequirement,
    fit,
    normalized,
    tensorfs_requirement_dtype,
)
from cozy_runtime.internal.seam import Channel
from store_fixtures import checkpoint, home

UNET = {"in": 4, "eps": 1e-05, "act": None}
SCHEDULER = {"steps": 2, "beta": 0.00085}
TOKENIZER = {"model_max_length": 77}
VOCAB = b'{"a": 0}'
MERGES = b"#version: 0.2\na b\n"
INDEX = {"_class_name": "StableDiffusionPipeline", "unet": ["diffusers", "UNet2DConditionModel"]}


def _pipeline(root: Path, monkeypatch: pytest.MonkeyPatch) -> tensorfs.Checkpoint:
    store = home(root, monkeypatch)
    writer = tensorfs.Writer(cache_dir=store)
    writer.pipeline(("diffusers", "", "StableDiffusionPipeline"), INDEX)
    unet = writer.component("unet", cls=("diffusers", "", "UNet2DConditionModel"), config=UNET)
    unet.put("conv_in.weight", np.ones((8, 4), np.float16), dtype="F16", shape=[8, 4])
    big = np.arange(20 << 18, dtype=np.float32)  # 20 MiB: chunks of its own
    unet.put("mid.weight", big, dtype="F32", shape=[20 << 18])
    unet.put_encoded(
        "proj.weight", dtype="BF16", shape=[4, 32], scheme="fp8.rowwise",
        parts={"weight": ("F8_E4M3", [4, 32], np.full((4, 32), 0x38, np.uint8)),
         "scale": ("F32", [4], np.ones(4, np.float32))},
        params={},
    )  # fmt: skip
    writer.component(
        "scheduler", cls=("diffusers", "", "EulerDiscreteScheduler"), config=SCHEDULER,
    )  # fmt: skip
    tokenizer = writer.component(
        "tokenizer", cls=("transformers", "", "CLIPTokenizer"), config=TOKENIZER,
    )  # fmt: skip
    tokenizer.asset("vocab.json", VOCAB)
    tokenizer.asset("merges.txt", MERGES)
    vae = writer.component("vae", cls=("diffusers", "", "AutoencoderKL"), config={"z": 4})
    vae.put("decoder.weight", np.ones(3, np.float32), dtype="F32", shape=[3])
    return writer.commit()


def test_rows_carry_order_scheme_params_and_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ck = Checkpoint(_pipeline(tmp_path, monkeypatch))
    rows = ck.rows("unet")
    assert [r.name for r in rows] == list(ck.tensors("unet"))
    encoded = next(r for r in rows if r.name == "proj.weight")
    assert (encoded.encoded.scheme, dict(encoded.encoded.params)) == (
        "fp8.rowwise",
        {},
    )
    assert {p.role: (p.dtype, p.shape) for p in encoded.encoded.parts} == {
        "weight": ("f8_e4m3fn", (4, 32)),
        "scale": ("f32", (4,)),
    }
    assert (encoded.dtype, encoded.nbytes) == ("bf16", 4 * 32 * 2)
    plain = next(r for r in rows if r.name == "conv_in.weight")
    assert plain.encoded.scheme == "plain" and plain.encoded.parts[0].role == "value"
    assert ck.rows("tokenizer") == [], "a component without tensors has no rows"
    with pytest.raises(FillRefusal) as refused:
        ck.rows("text_encoder")
    assert refused.value.code == "destination_absent"
    held = set(ck.ck.chunks)
    for table in ck.table("unet").values():
        parts = table[4]
        assert isinstance(parts, list)
        for part in parts:
            assert isinstance(part, list)
            assert part[3] in {sha for sha, _ in held}


def test_an_executor_opens_a_checkpoint_from_the_store_with_its_configs_and_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An executor opens a request's checkpoint by its header ref from the shared store: configs
    directly from its components, exact supporting bytes without filename export, and
    the same immutable identity for every later load."""
    ref = _pipeline(tmp_path / "tensorfs", monkeypatch).id
    assert ref is not None
    redirected = tmp_path / "package-changed-store"
    monkeypatch.setenv("TENSORFS_HOME", str(redirected))
    left, right = socket.socketpair()
    try:
        executor = Executor(Channel(left), tmp_path)
        ck = executor._checkpoint(ref)
        assert executor._checkpoint(ref).id == ck.id == ref
        assert not redirected.exists(), "the executor uses its sealed store root"
        assert ck.configs() == {
            "unet": UNET,
            "scheduler": SCHEDULER,
            "tokenizer": TOKENIZER,
            "vae": {"z": 4},
        }
        assets = ck.assets()
        assert sorted(assets) == [
            "scheduler/config.json",
            "tokenizer/config.json",
            "tokenizer/merges.txt",
            "tokenizer/vocab.json",
            "unet/config.json",
            "vae/config.json",
        ]
        assert (assets["tokenizer/vocab.json"], assets["tokenizer/merges.txt"]) == (VOCAB, MERGES)
        assert json.loads(assets["unet/config.json"]) == UNET
        assert ck.ck.pipeline is not None and ck.ck.pipeline.config == INDEX
        assert not list((tmp_path / "tensorfs").rglob("config.json")), "loading exported files"
        assert set(ck.asset_digests()) == {"tokenizer/vocab.json", "tokenizer/merges.txt"}
    finally:
        left.close()
        right.close()


def test_a_model_without_a_pipeline_is_its_root_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = home(tmp_path, monkeypatch)
    ref = checkpoint(
        store,
        {"model": {"w": np.ones(2, np.float32)}},
        classes={"model": ("transformers", "", "Qwen3ForCausalLM")},
        configs={"model": {"hidden_size": 2}},
        assets={"model": {"tokenizer.json": b"{}", "tokenizer_config.json": b'{"x": 1}'}},
    )
    assets = Checkpoint(tensorfs.open(ref, cache_dir=store, local_files_only=True)).assets()
    assert sorted(assets) == ["config.json", "tokenizer.json", "tokenizer_config.json"]
    assert assets["tokenizer_config.json"] == b'{"x": 1}'
    assert json.loads(assets["config.json"]) == {"hidden_size": 2}


def test_a_processor_reads_its_config_as_the_file_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Transformers processor folder has no config.json: its preprocessor_config.json is the
    component's config, and a package still reads it by that path (qwen-image-2 on D10)."""
    image = {"image_processor_type": "Qwen2VLImageProcessor", "patch_size": 16}
    video = b'{"video_processor_type": "Qwen2VLVideoProcessor"}'
    store = home(tmp_path, monkeypatch)
    writer = tensorfs.Writer(cache_dir=store)
    writer.pipeline(("diffusers", "", "QwenImagePipeline"), {"_class_name": "QwenImagePipeline"})
    processor = writer.component(
        "processor", cls=("transformers", "", "Qwen2VLProcessor"), config=image,
        config_file="preprocessor_config.json",
    )  # fmt: skip
    processor.asset("video_preprocessor_config.json", video)
    vae = writer.component("vae", cls=("diffusers", "", "AutoencoderKL"), config={"z": 4})
    vae.put("decoder.weight", np.ones(3, np.float32), dtype="F32", shape=[3])
    ref = writer.commit().id
    ck = Checkpoint(tensorfs.open(ref, cache_dir=store, local_files_only=True))
    model = ModelAssets(ck.assets(), lambda _: None)
    assert json.loads(model.read("processor/preprocessor_config.json")) == image
    assert model.read("processor/video_preprocessor_config.json") == video
    assert "processor/config.json" not in model.names()
    assert set(ck.asset_digests()) == {"processor/video_preprocessor_config.json"}


def test_configs_construct_and_the_fit_judges_the_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ck = Checkpoint(_pipeline(tmp_path, monkeypatch))
    document = json.loads(model_config.construction_config(ck.configs()))
    assert document == {
        "scheduler": SCHEDULER,
        "tokenizer": TOKENIZER,
        "unet": UNET,
        "vae": {"z": 4},
    }
    assert model_config.construction_config({"unet": UNET}) == (
        b'{"act":null,"eps":0.00001,"in":4}'
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


def test_a_transformers_model_binds_its_stored_names_through_its_conversion_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    store = home(tmp_path, monkeypatch)
    config = transformers.CLIPTextConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, max_position_embeddings=8, projection_dim=8,
    )  # fmt: skip
    built = {
        "text_encoder": transformers.CLIPTextModel(config),
        "text_encoder_2": transformers.CLIPTextModelWithProjection(config),
    }
    writer = tensorfs.Writer(cache_dir=store)
    for component, model in built.items():
        model.config.architectures = [type(model).__name__]  # as `save_pretrained` writes it
        held = writer.component(
            component,
            cls=("transformers", "", type(model).__name__),
            config=json.loads(model.config.to_json_string()),
        )
        for key, value in model.state_dict().items():  # upstream on-disk names
            stored = (
                key if key.startswith(("text_model.", "text_projection.")) else f"text_model.{key}"
            )
            held.put(stored, value.numpy(), dtype="F32", shape=list(value.shape))
    ck = Checkpoint(writer.commit())
    assert next(iter(ck.tensors("text_encoder"))).startswith("text_model.")
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
    stored = ck.ck.components["text_encoder"].tensors.read("text_model.final_layer_norm.weight")
    assert torch.equal(
        torch.from_dlpack(stored), built["text_encoder"].final_layer_norm.weight.detach()
    )


@pytest.mark.parametrize(
    ("torch_spelling", "stored"),
    [("torch.bfloat16", "BF16"), ("torch.float16", "F16"), ("torch.float32", "F32")],
)
def test_the_census_dtype_a_fit_pins_is_the_runtimes_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torch_spelling: str, stored: str
) -> None:
    store = home(tmp_path, monkeypatch)
    writer = tensorfs.Writer(cache_dir=store)
    value = np.zeros(1, {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[stored])
    writer.component("unet", cls=("", "", ""), config={}).put(
        "weight", memoryview(value), dtype=stored, shape=[1]
    )
    ck = Checkpoint(writer.commit())
    pinned = tensorfs_requirement_dtype(torch_spelling)
    assert pinned == (None if stored != "F32" else "f32"), "bf16 and f16 stay convertible"
    assert fit([TensorRequirement("unet", "weight", (1,), pinned)], ck, encoded_leaves=False)["ok"]
    refused = fit([TensorRequirement("unet", "weight", (1,), "bfloat16")], ck, encoded_leaves=False)
    assert refused["code"] == "dtype_mismatch", "a torch name is not a Runtime dtype"
