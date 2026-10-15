"""`int8-convrot256/1`: producer, recipe selection over real TensorFS channels, decode and leaf."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorfs

from cozy_runtime.author import UnsupportedInput
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author.fakes import fake_attempt, fake_telemetry
from cozy_runtime.derive import safetensors_io as st
from cozy_runtime.derive.microscale import decode_int8_convrot256, encode_int8_convrot256
from cozy_runtime.derive.operations import QuantizationPlan, QuantizationSource, quantize_into
from cozy_runtime.internal.encoding import SPEC_INT8_CONVROT256
from native_weights import NativeExecution
from test_derive_quantize_facade import BOUND, header, mint

ENCODING = "int8-convrot256/1"


def weight(rows: int, cols: int, seed: int) -> np.ndarray:
    values = np.random.default_rng(seed).standard_normal((rows, cols)).astype(np.float32) * 0.02
    values[1, :] = 0.0
    values[2, 7] = 3.0  # one outlier: the row's rotation spreads it over its 256-group
    return values


def test_producer_round_trip_and_scale_floor() -> None:
    values = weight(16, 512, 3)
    encoded = encode_int8_convrot256("w", values)
    payload = np.frombuffer(encoded.payload, dtype=np.int8).reshape(16, 512)
    scale = np.frombuffer(encoded.companions[0].raw, dtype=np.float32)
    assert (encoded.payload_dtype, encoded.saturated) == ("I8", 0)
    assert encoded.companions[0].shape == [16] and scale[1] == np.float32(1e-12)
    assert not payload[1].any() and np.abs(payload).max() == 127
    recon = decode_int8_convrot256(payload, scale)
    assert np.abs(recon - values).max() == pytest.approx(encoded.max_abs_err)
    assert encoded.relative_frobenius() < 0.01
    with pytest.raises(UnsupportedInput):
        encode_int8_convrot256("w", weight(4, 320, 1))


@pytest.mark.skipif(
    ENCODING not in dict(tensorfs.seed_digests()), reason="TensorFS before 0.9.2 lacks the spec"
)
def test_include_recipe_encodes_exactly_its_weights(tmp_path: Path) -> None:
    def row(shape: tuple[int, int], seed: int) -> tuple[str, tuple[int, ...], bytes]:
        return "bf16", shape, st.from_f32(weight(*shape, seed), "BF16")

    rows = {
        "transformer": {
            "blocks.0.attn.to_q.weight": row((8, 256), 1),
            "blocks.0.mlp.out.weight": row((16, 512), 2),
            "proj_out.weight": row((8, 256), 3),
        },
        "vae": {"conv.weight": row((8, 256), 4)},
    }
    store, source = mint(tmp_path, rows)
    plan = QuantizationPlan(
        components=("transformer",),
        include=("transformer.blocks.*.weight",),
        output_precision="preserve",
    )
    with NativeExecution(store, tmp_path, "int8", {"source": source}, {"model": BOUND}) as owner:
        ctx = owner.context()
        artifact = quantize_into(
            ctx,
            fake_telemetry(fake_attempt("int8")),
            source=_derive_model(QuantizationSource, source.manifest.digest),
            plan=plan,
            encoding=ENCODING,
            output="model",
        )
    parsed = header(store, artifact)
    spec = dict(tensorfs.seed_digests())[ENCODING]
    assert spec == SPEC_INT8_CONVROT256
    encoded = {
        (component, key)
        for component, tensors in parsed["components"].items()
        for key, tensor in tensors.items()
        if tensor["encoding"] == spec
    }
    to_q = parsed["components"]["transformer"]["blocks.0.attn.to_q.weight"]["parts"]
    assert (to_q["data"]["dtype"], to_q["scale"]["shape"]) == ("i8", [8])
    assert encoded == {
        ("transformer", "blocks.0.attn.to_q.weight"),
        ("transformer", "blocks.0.mlp.out.weight"),
    }
    typo = QuantizationPlan(components=("transformer",), include=("transformer.nope.*",))
    with (
        pytest.raises(UnsupportedInput, match="match no weight"),
        NativeExecution(store, tmp_path, "typo", {"source": source}, {"model": BOUND}) as owner,
    ):
        quantize_into(
            owner.context(),
            fake_telemetry(fake_attempt("typo")),
            source=_derive_model(QuantizationSource, source.manifest.digest),
            plan=typo,
            encoding=ENCODING,
            output="model",
        )


torch: Any = pytest.importorskip("torch")


def test_decode_is_the_spec_law_and_the_mined_vectors() -> None:
    from cozy_runtime.internal import probe
    from cozy_runtime.internal.encoding import Int8ConvRotDequant

    provider = Int8ConvRotDequant(SPEC_INT8_CONVROT256)
    vectors = probe._spec_vectors(torch, provider, torch.device("cpu"))
    assert vectors.passed and len(vectors.cases) == 3
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        assert probe._numerics_int8_convrot(torch, provider, torch.device("cpu"), dtype).passed


def test_leaf_matches_its_exact_weight() -> None:
    from cozy_runtime.internal import int8_linear

    values = weight(64, 512, 5)
    encoded = encode_int8_convrot256("w", values)
    data = torch.frombuffer(bytearray(encoded.payload), dtype=torch.int8).reshape(64, 512)
    scale = torch.frombuffer(bytearray(encoded.companions[0].raw), dtype=torch.float32)
    exact = torch.from_numpy(decode_int8_convrot256(data.numpy(), scale.numpy()))
    x = torch.randn(3, 40, 512, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    reference = x @ exact.t()
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for dtype in (torch.float32, torch.bfloat16):
            got = int8_linear.linear(
                torch, x.to(device, dtype), data.to(device), scale.to(device), None, dtype
            )
            assert got.shape == (3, 40, 64) and got.dtype == dtype
            error = (got.double().cpu() - reference).norm() / reference.norm()
            assert error < 0.02, (device, dtype, float(error))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the int8 op runs on a CUDA card")
def test_native_op_is_built_by_the_compile_manager_and_keeps_the_law(tmp_path: Path) -> None:
    """The op as a pod gets it: the machine's compile manager builds `cozy-int8` for this card
    from the pinned sources, the leaf loads it, and every path (GEMV rows, the GEMM, a bias,
    fp16 and bf16, an unaligned view) agrees with the eager W8A8 law."""
    from cozy_runtime.internal import int8_linear, kernel_cache, kernel_compile, kernel_sources

    store = kernel_cache.Store(tmp_path / "kernels")
    kernel_cache.configure(store)
    major, minor = torch.cuda.get_device_capability(0)
    state = kernel_compile.wait(store, kernel_sources.job(int8_linear.OP, major * 10 + minor))
    assert state.state == "ready", state.line()
    assert int8_linear.native(torch, 0) is not None
    generator = torch.Generator().manual_seed(11)
    for rows, out_features, in_features in (
        (1, 1024, 4096),
        (3, 768, 512),
        (77, 1280, 2048),
        (300, 4096, 1280),
    ):
        values = torch.randn(out_features, in_features, generator=generator).numpy() * 0.02
        encoded = encode_int8_convrot256("w", values)
        data = torch.frombuffer(bytearray(encoded.payload), dtype=torch.int8)
        data = data.reshape(out_features, in_features).cuda()
        scale = torch.frombuffer(bytearray(encoded.companions[0].raw), dtype=torch.float32).cuda()
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.randn(1, rows + 1, in_features, generator=generator).to("cuda", dtype)[:, 1:]
            bias = torch.randn(out_features, generator=generator).to("cuda", dtype)
            got = int8_linear.linear(torch, x, data, scale, bias, dtype)
            want = int8_linear.eager(torch, x, data, scale, torch.float32) + bias.float()
            assert got.shape == (1, rows, out_features) and got.dtype == dtype
            error = float((got.float() - want).norm() / want.norm())
            assert error < 0.01, (rows, out_features, in_features, dtype, error)
