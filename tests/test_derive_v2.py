"""Runtime derivations on TensorD's v2 channels, over a real store: lanes encode by scheme name
and decode back through the wheel's reference decoder, resume skips what TensorD journaled, a
committed output replays, a selection keeps only its tensors, and `Weights[M]` reads values."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorfs

from cozy_runtime.author import UnsupportedInput, Weights
from cozy_runtime.author.fakes import fake_telemetry
from cozy_runtime.derive import plan, quantize
from cozy_runtime.derive.operations import SelectionPlan, select_into
from cozy_runtime.internal.model_values import read_values_into
from cozy_runtime.tensord_channels import Derivation
from tensord_peer import Peer, checkpoint, context

torch = pytest.importorskip("torch")

BOUND = 64 << 20
CONFIG = b'{"hidden": 256}'


def _source(root: Path) -> tuple[tensorfs.Store, tensorfs.Manifest, dict[str, np.ndarray]]:
    rng = np.random.default_rng(7)
    tensors: dict[str, np.ndarray] = {
        "blocks.0.proj.weight": (rng.standard_normal((64, 256)) * 0.05).astype(np.float32),
        "blocks.0.proj.bias": rng.standard_normal(64).astype(np.float32),
        "blocks.1.proj.weight": (rng.standard_normal((32, 256)) * 0.05).astype(np.float32),
        "norm.weight": rng.standard_normal(256).astype(np.float16),
    }
    store = tensorfs.Store(root / "store")
    return store, checkpoint(store, {"unet": tensors}, {"unet/config.json": CONFIG}), tensors


def _peer(root: Path, store: tensorfs.Store, source: tensorfs.Manifest) -> Peer:
    return Peer(store, root / "spool", {f"sha256:{source.sha256}": source.length}, {"model": BOUND})


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
    tmp_path: Path, scheme: str, params: dict[str, object], bound: float
) -> None:
    store, source, tensors = _source(tmp_path)
    peer = _peer(tmp_path, store, source)
    weights: Weights[Any] = Weights(f"sha256:{source.sha256}")
    result = quantize(
        weights, plan(["unet"], scheme), ctx=context(peer), tel=fake_telemetry(), output="model"
    )
    assert (result.encoded_keys, result.converted_keys, result.replayed) == (2, 1, False)
    out = store.checkpoint(f"sha256:{peer.adopted['model'][0]}")
    info = out.tensors["unet"]["blocks.0.proj.weight"]
    assert info.encoding is not None and info.encoding.scheme == scheme
    assert info.encoding.params == params and info.dtype == "BF16"
    for key in ("blocks.0.proj.weight", "blocks.1.proj.weight"):
        want = torch.from_numpy(tensors[key])
        got = torch.from_dlpack(out.read("unet", key)).float()
        assert float((got - want).norm() / want.norm()) < bound, key
    assert out.tensors["unet"]["blocks.0.proj.bias"].dtype == "BF16"
    assert out.tensors["unet"]["norm.weight"].dtype == "F16", "f16 is not a bf16 source here"
    assert out.file("unet/config.json") == CONFIG, "the base's files carry over"


def test_a_committed_output_replays_and_an_interrupted_one_resumes(tmp_path: Path) -> None:
    store, source, _ = _source(tmp_path)
    peer = _peer(tmp_path, store, source)
    weights: Weights[Any] = Weights(f"sha256:{source.sha256}")
    lane = plan(["unet"], "fp8.rowwise")
    first = quantize(weights, lane, ctx=context(peer), tel=fake_telemetry())
    again = quantize(weights, lane, ctx=context(peer), tel=fake_telemetry())
    assert again.replayed and again.new_bytes_written_this_run == 0
    assert again.weights_transaction_id == first.weights_transaction_id

    fresh = Peer(store, tmp_path / "spool2", peer.sources, peer.outputs)
    out = fresh.client.open_output("model", _derivation(fresh, source))
    out.put("unet", "blocks.0.proj.bias", "BF16", [64], np.zeros(64, np.uint16))
    out.close()  # the attempt dies: TensorD's writer keeps its journal
    resumed = quantize(weights, lane, ctx=context(fresh), tel=fake_telemetry())
    assert resumed.reused_keys == 1 and not resumed.replayed


def _derivation(peer: Peer, source: tensorfs.Manifest) -> Derivation:
    opened = peer.client.source(f"sha256:{source.sha256}")
    order = [(c, k) for c, tensors in opened.checkpoint.tensors.items() for k in tensors]
    return Derivation(base=opened, order=order)


def test_a_selection_keeps_only_its_weights_and_every_file(tmp_path: Path) -> None:
    store, source, tensors = _source(tmp_path)
    peer = _peer(tmp_path, store, source)
    selection = SelectionPlan(include=("unet.blocks.1.*",))
    select_into(
        context(peer), fake_telemetry(), source=Weights(f"sha256:{source.sha256}"),
        plan=selection, output="model",
    )  # fmt: skip
    out = store.checkpoint(f"sha256:{peer.adopted['model'][0]}")
    assert list(out.tensors["unet"]) == ["blocks.1.proj.weight"]
    got = torch.from_dlpack(out.read("unet", "blocks.1.proj.weight")).numpy()
    assert np.array_equal(got, tensors["blocks.1.proj.weight"])
    assert out.file("unet/config.json") == CONFIG
    with pytest.raises(UnsupportedInput):
        select_into(
            context(peer), fake_telemetry(), source=Weights(f"sha256:{source.sha256}"),
            plan=SelectionPlan(include=("nothing.*",)), output="model",
        )  # fmt: skip


def test_values_read_from_plain_and_rowwise_sources(tmp_path: Path) -> None:
    store, source, tensors = _source(tmp_path)
    peer = _peer(tmp_path, store, source)
    quantize(
        Weights(f"sha256:{source.sha256}"),
        plan(["unet"], "fp8.rowwise"),
        ctx=context(peer),
        tel=fake_telemetry(),
    )
    lane = store.checkpoint(f"sha256:{peer.adopted['model'][0]}").manifest
    assert lane is not None
    reader = Peer(
        store, tmp_path / "spool3", {f"sha256:{lane.sha256}": lane.length, **peer.sources}, {}
    )
    for manifest, key, tolerance in (
        (source, "blocks.0.proj.bias", 0.0),
        (lane, "blocks.0.proj.weight", 0.07),
    ):
        opened = reader.client.source(f"sha256:{manifest.sha256}")
        out = np.zeros(64 * 256 if "weight" in key else 64, np.float32)
        read_values_into(torch, opened, "unet", key, 0, memoryview(out).cast("B"))
        want = tensors[key].reshape(-1)
        assert float(np.linalg.norm(out - want) / np.linalg.norm(want)) <= tolerance + 1e-3, key


def test_a_put_takes_a_buffer_a_file_or_an_in_memory_stream(tmp_path: Path) -> None:
    store, source, tensors = _source(tmp_path)
    peer = _peer(tmp_path, store, source)
    out = peer.client.open_output("model", _derivation(peer, source))
    bias = np.arange(64, dtype=np.float32)
    norm = np.arange(256, dtype=np.float16)
    on_disk = tmp_path / "bias.bin"
    on_disk.write_bytes(b"skip" + bias.tobytes())
    with on_disk.open("rb") as file:
        file.seek(4)  # from its position to its end
        out.put("unet", "blocks.0.proj.bias", "F32", [64], file)
    out.put("unet", "norm.weight", "F16", [256], io.BytesIO(norm.tobytes()))
    weight = tensors["blocks.1.proj.weight"]
    out.put("unet", "blocks.1.proj.weight", "F32", list(weight.shape), weight)
    sha, _ = out.commit()
    written = store.checkpoint(f"sha256:{sha}")
    for key, want in (
        ("blocks.0.proj.bias", bias),
        ("norm.weight", norm),
        ("blocks.1.proj.weight", weight),
    ):
        assert np.array_equal(torch.from_dlpack(written.read("unet", key)).numpy(), want), key


def test_values_read_a_rowwise_scale_in_its_stored_dtype(tmp_path: Path) -> None:
    store = tensorfs.Store(tmp_path / "store")
    payload = torch.linspace(-2, 2, 4 * 32).reshape(4, 32).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.5, 0.25, 3.0, 0.0078125], dtype=torch.bfloat16)
    writer = store.writer()
    writer.put_encoded(
        "unet", "w", "BF16", [4, 32], "fp8.rowwise",
        {"weight": ("F8_E4M3", [4, 32], payload.view(torch.uint8).numpy()),
         "scale": ("BF16", [4], scale.view(torch.uint16).numpy())},
        {"axis": 0, "scale_dtype": "BF16"},
    )  # fmt: skip
    manifest = writer.commit()
    ref = f"sha256:{manifest.sha256}"
    reader = Peer(store, tmp_path / "spool", {ref: manifest.length}, {})
    out = np.zeros(4 * 32, np.float32)
    read_values_into(torch, reader.client.source(ref), "unet", "w", 0, memoryview(out).cast("B"))
    want = (payload.float() * scale.float().reshape(4, 1)).reshape(-1).numpy()
    assert np.array_equal(out, want)
