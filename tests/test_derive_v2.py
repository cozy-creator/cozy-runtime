"""Runtime derivations straight into a real store: lanes encode by scheme name and decode back
through the wheel's reference decoder, a resumed output skips what an interrupted attempt
wrote, a committed output replays as `local/derived`, a selection keeps only its tensors, and
`Weights[M]` reads values."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import tensorfs

from cozy_runtime import weights_store
from cozy_runtime.author import CapabilityError, UnsupportedInput, Weights
from cozy_runtime.author.fakes import fake_telemetry
from cozy_runtime.derive import plan, quantize
from cozy_runtime.derive.operations import SelectionPlan, select_into
from cozy_runtime.internal.model_values import read_values_into
from store_fixtures import Shown, adopted, checkpoint, context, home

torch = pytest.importorskip("torch")

CONFIG = {"hidden": 256, "eps": 1e-05, "act": None}
VOCAB = b'{"a": 0}'
RNG = np.random.default_rng(7)
TENSORS: dict[str, np.ndarray] = {
    "blocks.0.proj.weight": (RNG.standard_normal((64, 256)) * 0.05).astype(np.float32),
    "blocks.0.proj.bias": RNG.standard_normal(64).astype(np.float32),
    "blocks.1.proj.weight": (RNG.standard_normal((32, 256)) * 0.05).astype(np.float32),
    "norm.weight": RNG.standard_normal(256).astype(np.float16),
}


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return home(tmp_path / "tensorfs", monkeypatch)


@pytest.fixture
def source(store: Path) -> str:
    """A pipeline whose `unet` holds `TENSORS` and whose tokenizer holds only an asset."""
    return checkpoint(
        store,
        {"unet": TENSORS},
        classes={
            "unet": "diffusers.UNet2DConditionModel", "tokenizer": "transformers.CLIPTokenizer"
        },
        configs={"unet": CONFIG},
        assets={"tokenizer": {"vocab.json": VOCAB}},
        pipeline=(
            "diffusers.StableDiffusionPipeline", {"unet": ["diffusers", "UNet2DConditionModel"]}
        ),
    )  # fmt: skip


def _read(ck: tensorfs.Checkpoint, key: str) -> np.ndarray:
    return torch.from_dlpack(ck.components["unet"].tensors.read(key)).float().numpy()


@pytest.mark.parametrize(
    ("scheme", "params", "bound"),
    [
        ("fp8.rowwise", {"axis": 0, "scale_dtype": "F32"}, 0.07),
        (
            "int8.convrot",
            {"rotation": "hadamard", "rotation_size": 256, "scale_dtype": "F32"},
            0.02,
        ),
        (
            "nvfp4",
            {
                "block_size": 16,
                "global_scale": "multiply",
                "nibble_order": "low_first",
                "scale_layout": "row_major",
            },
            0.2,
        ),
    ],
)
def test_a_lane_encodes_by_scheme_and_decodes_through_the_reference(
    store: Path, source: str, scheme: str, params: dict[str, object], bound: float
) -> None:
    shown: Shown = []
    ctx = context(store, shown=shown)
    result = quantize(Weights(source), plan(["unet"], scheme), ctx=ctx, tel=fake_telemetry())
    assert (result.encoded_keys, result.converted_keys, result.replayed) == (2, 1, False)
    out = adopted(store, shown)
    assert out.id == result.checkpoint
    unet = out.components["unet"]
    info = unet.tensors["blocks.0.proj.weight"]
    assert info.encoding is not None and info.encoding.scheme == scheme
    assert info.encoding.params == params and info.dtype == "BF16"
    for key in ("blocks.0.proj.weight", "blocks.1.proj.weight"):
        want = TENSORS[key]
        assert float(np.linalg.norm(_read(out, key) - want) / np.linalg.norm(want)) < bound, key
    assert unet.tensors["blocks.0.proj.bias"].dtype == "BF16"
    assert unet.tensors["norm.weight"].dtype == "F16", "f16 is not a bf16 source here"
    assert (unet.cls, unet.config) == ("diffusers.UNet2DConditionModel", CONFIG)
    assert out.components["tokenizer"].asset("vocab.json") == VOCAB, "the base's assets carry over"
    assert out.pipeline is not None and out.pipeline.cls == "diffusers.StableDiffusionPipeline"


def test_a_committed_output_replays_and_an_interrupted_one_resumes(
    store: Path, source: str
) -> None:
    lane = plan(["unet"], "fp8.rowwise")
    first = quantize(Weights(source), lane, ctx=context(store), tel=fake_telemetry())
    again = quantize(Weights(source), lane, ctx=context(store), tel=fake_telemetry())
    assert again.replayed and again.new_bytes_written_this_run == 0
    assert (again.weights_transaction_id, again.checkpoint) == (
        first.weights_transaction_id,
        first.checkpoint,
    )
    other = quantize(Weights(source), lane, ctx=context(store, "another"), tel=fake_telemetry())
    assert other.weights_transaction_id != first.weights_transaction_id, "another run's work"

    ctx = context(store, "interrupted")
    base = ctx.tensorfs_source(Weights(source))
    order = [(c, k) for c, held in base.components.items() for k in held.tensors]
    with ctx.output("model", weights_store.Derivation(base=base, order=order)) as out:
        out.put("unet", "blocks.0.proj.bias", "BF16", [64], np.zeros(64, np.uint16))
    resumed = quantize(
        Weights(source), lane, ctx=context(store, "interrupted"), tel=fake_telemetry()
    )
    assert resumed.reused_keys == 1 and not resumed.replayed


def test_a_selection_keeps_only_its_weights_and_every_component(store: Path, source: str) -> None:
    shown: Shown = []
    select_into(
        context(store, shown=shown), fake_telemetry(), source=Weights(source),
        plan=SelectionPlan(include=("unet.blocks.1.*",)), output="model",
    )  # fmt: skip
    out = adopted(store, shown)
    unet = out.components["unet"]
    assert list(unet.tensors) == ["blocks.1.proj.weight"]
    assert np.array_equal(_read(out, "blocks.1.proj.weight"), TENSORS["blocks.1.proj.weight"])
    assert unet.config == CONFIG and out.components["tokenizer"].asset("vocab.json") == VOCAB
    with pytest.raises(UnsupportedInput):
        select_into(
            context(store, "again"), fake_telemetry(), source=Weights(source),
            plan=SelectionPlan(include=("nothing.*",)), output="model",
        )  # fmt: skip


def test_values_read_from_plain_and_rowwise_sources(store: Path, source: str) -> None:
    ctx = context(store)
    lane = quantize(Weights(source), plan(["unet"], "fp8.rowwise"), ctx=ctx, tel=fake_telemetry())
    for ref, key, tolerance in (
        (source, "blocks.0.proj.bias", 0.0),
        (lane.checkpoint, "blocks.0.proj.weight", 0.07),
    ):
        out = np.zeros(64 * 256 if "weight" in key else 64, np.float32)
        ck = tensorfs.open(ref, cache_dir=store, local_files_only=True)
        read_values_into(torch, ck, "unet", key, 0, memoryview(out).cast("B"))
        want = TENSORS[key].reshape(-1)
        assert float(np.linalg.norm(out - want) / np.linalg.norm(want)) <= tolerance + 1e-3, key


def test_a_part_reads_across_its_chunks_from_any_offset(store: Path) -> None:
    big = np.arange(68 << 20, dtype=np.uint8)  # crosses the 64 MiB chunk grid
    ck = tensorfs.open(
        checkpoint(store, {"unet": {"big": big}}), cache_dir=store, local_files_only=True
    )
    assert len(ck.chunks) > 1
    into = bytearray(4 << 20)
    part = ck.components["unet"].tensors["big"].parts[""]
    offset = ck.chunks[part.chunk][1] - part.offset - 1001
    weights_store.read_into(ck, "unet", "big", "", offset, into)
    assert bytes(into) == memoryview(big)[offset : offset + len(into)]
    with pytest.raises(CapabilityError) as past:
        weights_store.read_into(ck, "unet", "big", "", big.nbytes - 3, bytearray(4))
    assert past.value.code == "weights_reader_bounds"


def test_a_put_takes_a_buffer_a_file_or_an_in_memory_stream(
    store: Path, source: str, tmp_path: Path
) -> None:
    ctx = context(store)
    base = ctx.tensorfs_source(Weights(source))
    bias = np.arange(64, dtype=np.float32)
    norm = np.arange(256, dtype=np.float16)
    weight = TENSORS["blocks.1.proj.weight"]
    on_disk = tmp_path / "bias.bin"
    on_disk.write_bytes(b"skip" + bias.tobytes())
    with ctx.output("model", weights_store.Derivation(base=base)) as out:
        with on_disk.open("rb") as file:
            file.seek(4)  # from its position to its end
            out.put("unet", "blocks.0.proj.bias", "F32", [64], file)
        out.put("unet", "norm.weight", "F16", [256], io.BytesIO(norm.tobytes()))
        out.put("unet", "blocks.1.proj.weight", "F32", list(weight.shape), weight)
        written = out.commit()
    for key, want in (
        ("blocks.0.proj.bias", bias),
        ("norm.weight", norm),
        ("blocks.1.proj.weight", weight),
    ):
        value = written.components["unet"].tensors.read(key)
        assert np.array_equal(torch.from_dlpack(value).numpy(), want), key


def test_an_output_refuses_past_its_granted_bytes(store: Path, source: str) -> None:
    ctx = context(store, outputs={"model": 100})
    base = ctx.tensorfs_source(Weights(source))
    with (
        ctx.output("model", weights_store.Derivation(base=base)) as out,
        pytest.raises(CapabilityError, match="granted new bytes") as refused,
    ):
        out.put("unet", "blocks.0.proj.bias", "F32", [64], np.zeros(64, np.float32))
    assert refused.value.code == "weights_output_bound"


def test_values_read_a_rowwise_scale_in_its_stored_dtype(store: Path) -> None:
    payload = torch.linspace(-2, 2, 4 * 32).reshape(4, 32).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.5, 0.25, 3.0, 0.0078125], dtype=torch.bfloat16)
    writer = tensorfs.Writer(cache_dir=store)
    writer.component("unet", cls="", config={}).put_encoded(
        "w", dtype="BF16", shape=[4, 32], scheme="fp8.rowwise",
        parts={"weight": ("F8_E4M3", [4, 32], payload.view(torch.uint8).numpy()),
         "scale": ("BF16", [4], scale.view(torch.uint16).numpy())},
        params={"axis": 0, "scale_dtype": "BF16"},
    )  # fmt: skip
    out = np.zeros(4 * 32, np.float32)
    read_values_into(torch, writer.commit(), "unet", "w", 0, memoryview(out).cast("B"))
    want = (payload.float() * scale.float().reshape(4, 1)).reshape(-1).numpy()
    assert np.array_equal(out, want)
