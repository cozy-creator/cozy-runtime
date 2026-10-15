"""`nvfp4-w4a4/1`: the tensor-scaled producer over real TensorFS channels, decode and leaf."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorfs

from cozy_runtime.author._model import _derive_model
from cozy_runtime.author.fakes import fake_attempt, fake_telemetry
from cozy_runtime.derive import safetensors_io as st
from cozy_runtime.derive.microscale import encode_nvfp4
from cozy_runtime.derive.operations import (
    EncodingRule,
    QuantizationPlan,
    QuantizationSource,
    quantize_into,
)
from cozy_runtime.internal.encoding import SPEC_NVFP4
from native_weights import NativeExecution
from test_derive_quantize_facade import BOUND, header, mint

ENCODING = "nvfp4-w4a4/1"


def weight(rows: int, cols: int, seed: int) -> np.ndarray:
    values = np.random.default_rng(seed).standard_normal((rows, cols)).astype(np.float32) * 0.02
    values[1, :] = 0.0
    values[rows - 1, 5] = 3.0  # the tensor amax sits in the LAST row chunk
    return values


def test_the_leaf_is_a_leaf_provider() -> None:
    from cozy_runtime.internal.encoding import LeafProvider, Nvfp4NativeLeaf

    # Qualification runs a native oracle only for a LeafProvider; a missing member skips it.
    assert isinstance(Nvfp4NativeLeaf(SPEC_NVFP4), LeafProvider)


def test_chunked_encode_uses_the_tensor_amax() -> None:
    values = weight(32, 64, 3)
    whole = encode_nvfp4("w", values)
    head = encode_nvfp4("w", values[:16], amax=float(np.abs(values).max()))
    assert head.companions[1].raw == whole.companions[1].raw
    assert head.payload == whole.payload[: len(head.payload)]
    assert head.companions[0].raw == whole.companions[0].raw[: len(head.companions[0].raw)]
    assert encode_nvfp4("w", values[:16]).companions[1].raw != whole.companions[1].raw


def test_one_lane_carries_nvfp4_beside_int8(tmp_path: Path) -> None:
    def row(shape: tuple[int, int], seed: int) -> tuple[str, tuple[int, ...], bytes]:
        return "bf16", shape, st.from_f32(weight(*shape, seed), "BF16")

    rows = {
        "transformer": {"blocks.0.ff.weight": row((48, 64), 1), "proj_out.weight": row((8, 64), 2)},
        "text_encoder": {"layers.0.q_proj.weight": row((16, 256), 3)},
    }
    store, source = mint(tmp_path, rows)
    plan = QuantizationPlan(
        components=("transformer", "text_encoder"),
        include=("transformer.blocks.*.weight", "text_encoder.layers.*.weight"),
        encodings=(EncodingRule("transformer.*", ENCODING),),
        output_precision="preserve",
    )
    with NativeExecution(store, tmp_path, "nvfp4", {"source": source}, {"model": BOUND}) as owner:
        artifact = quantize_into(
            owner.context(),
            fake_telemetry(fake_attempt("nvfp4")),
            source=_derive_model(QuantizationSource, source.manifest.digest),
            plan=plan,
            encoding="int8-convrot256/1",
            output="model",
        )
    parsed = header(store, artifact)
    q_proj = parsed["components"]["text_encoder"]["layers.0.q_proj.weight"]
    assert q_proj["encoding"] == dict(tensorfs.seed_digests())["int8-convrot256/1"]
    # The base role set, first of the four the alias files.
    assert next(d for a, d in tensorfs.seed_digests() if a == ENCODING) == SPEC_NVFP4
    ff = parsed["components"]["transformer"]["blocks.0.ff.weight"]
    parts = ff["parts"]
    assert ff["encoding"] == SPEC_NVFP4
    assert (parts["weight"]["dtype"], parts["weight"]["shape"]) == ("u8", [48, 32])
    assert (parts["weight_scale"]["dtype"], parts["weight_scale"]["shape"]) == (
        "f8_e4m3fn",
        [48, 4],
    )
    assert (parts["weight_scale_2"]["dtype"], parts["weight_scale_2"]["shape"]) == ("f32", [])
    assert parsed["components"]["transformer"]["proj_out.weight"]["encoding"] != SPEC_NVFP4


def roles(torch: Any, values: np.ndarray, device: str) -> dict[str, Any]:
    rows, cols = values.shape
    encoded = encode_nvfp4("w", values)
    scale, scale_2 = encoded.companions
    return {
        "weight": torch.frombuffer(bytearray(encoded.payload), dtype=torch.uint8)
        .reshape(rows, cols // 2)
        .to(device),
        "weight_scale": torch.frombuffer(bytearray(scale.raw), dtype=torch.uint8)
        .reshape(rows, cols // 16)
        .view(torch.float8_e4m3fn)
        .to(device),
        "weight_scale_2": torch.frombuffer(bytearray(scale_2.raw), dtype=torch.float32)
        .reshape(())
        .to(device),
    }


def test_decode_is_the_spec_law_and_the_mined_vectors() -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.internal import probe
    from cozy_runtime.internal.encoding import Nvfp4Dequant

    provider = Nvfp4Dequant(SPEC_NVFP4)
    vectors = probe._spec_vectors(torch, provider, torch.device("cpu"))
    assert vectors.passed and not vectors.skipped and len(vectors.cases) == 2
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        assert probe._numerics_nvfp4(torch, provider, torch.device("cpu"), dtype).passed


def test_blocked_scales_round_trip() -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.internal.encoding.leaves import from_blocked, to_blocked

    grid = torch.randint(0, 126, (200, 12), dtype=torch.uint8)
    assert torch.equal(from_blocked(torch, to_blocked(torch, grid), 200, 12), grid)


def test_leaf_matches_its_exact_weight() -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.internal import probe
    from cozy_runtime.internal.encoding import Nvfp4NativeLeaf

    values = weight(256, 512, 5)
    x = torch.randn(3, 40, 512, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    leaf_provider = Nvfp4NativeLeaf(SPEC_NVFP4)
    for device in devices:
        parts = roles(torch, values, device)
        exact = probe._exact_weight(torch, leaf_provider, parts).cpu()
        reference = x @ exact.t()
        for dtype in (torch.float32, torch.bfloat16):
            leaf = leaf_provider.leaf(torch, parts, probe._StandIn(512, 256), dtype)
            got = leaf(x.to(device, dtype))
            assert got.shape == (3, 40, 256) and got.dtype == dtype
            error = (got.double().cpu() - reference).norm() / reference.norm()
            # W4A4 on the kitchen path, the exact weight times an unquantized input elsewhere.
            assert error < 0.15, (device, dtype, float(error))
