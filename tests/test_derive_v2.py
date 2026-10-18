"""Runtime derivations on TensorD's v2 channels, over a real store: lanes encode by scheme name
and decode back through the wheel's reference decoder, resume skips what TensorD journaled, a
committed output replays, a selection keeps only its tensors, and `Weights[M]` reads values."""

from __future__ import annotations

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
        ("fp8-rowwise", {"axis": 0}, 0.07),
        ("int8-convrot", {"block": 256, "rotation": "hadamard"}, 0.02),
        ("nvfp4", {"block": 16}, 0.2),
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
    lane = plan(["unet"], "fp8-rowwise")
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
        plan(["unet"], "fp8-rowwise"),
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
