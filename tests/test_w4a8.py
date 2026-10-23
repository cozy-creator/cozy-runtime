"""`w4a8.convrot` (xs-062): the producer into the store, TensorFS's decode of it, the leaf
and the weight plane's bind, and on a card the fused GEMM against the law."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tensorfs import raw

from cozy_runtime.author import EncodingRule, QuantizationPlan, Weights
from cozy_runtime.author.fakes import fake_attempt, fake_telemetry
from cozy_runtime.author.kernels import INT8_CONVROT
from cozy_runtime.derive.microscale import (
    decode_int8_convrot256,
    e4m3_decode,
    encode_nvfp4,
    encode_w4a8,
    unpack_nibbles,
)
from cozy_runtime.derive.operations import quantize_into
from cozy_runtime.internal import int8_linear, kernel_cache, probe, w4a8_linear
from cozy_runtime.internal.encoding import LeafProvider, W4a8NativeLeaf, params_of
from cozy_runtime.internal.encoding.leaves import OPERANDS
from cozy_runtime.internal.prebuilt import W4a8, unpack
from cozy_runtime.internal.weights import _bound, _derived
from store_fixtures import adopted, checkpoint, context, home

ENCODING = "w4a8.convrot"
PARAMS = params_of({"group_size": 16, "rotation": "kron_regular_h4", "rotation_size": 256})


def weight(rows: int, cols: int, seed: int) -> np.ndarray:
    values = np.random.default_rng(seed).standard_normal((rows, cols)).astype(np.float32) * 0.02
    values[1, :] = 0.0
    values[rows - 1, 5] = 0.5  # an outlier the rotation spreads
    return values


def levels(encoded: Any, rows: int, cols: int) -> np.ndarray:
    """The int8 levels the stored codes stand for, by the spec's law."""
    group_scale, _, codebook = encoded.companions
    book = np.frombuffer(codebook.raw, np.float32)
    rel = e4m3_decode(np.frombuffer(group_scale.raw, np.uint8)).reshape(rows, cols // 16)
    codes = unpack_nibbles(np.frombuffer(encoded.payload, np.uint8).reshape(rows, cols // 2), cols)
    grid = book[codes].reshape(rows, cols // 16, 16) * rel[..., None]
    out: np.ndarray = np.clip(np.rint(grid), -127, 127).reshape(rows, cols)
    return out


def roles(torch: Any, values: np.ndarray, device: str) -> dict[str, Any]:
    rows, cols = values.shape
    encoded = encode_w4a8("w", values)
    group_scale, scale, codebook = encoded.companions
    return {
        "weight": torch.frombuffer(bytearray(encoded.payload), dtype=torch.int8)
        .reshape(rows, cols // 2)
        .to(device),
        "group_scale": torch.frombuffer(bytearray(group_scale.raw), dtype=torch.uint8)
        .reshape(rows, cols // 16)
        .view(torch.float8_e4m3fn)
        .to(device),
        "scale": torch.frombuffer(bytearray(scale.raw), dtype=torch.float32).to(device),
        "codebook": torch.frombuffer(bytearray(codebook.raw), dtype=torch.float32).to(device),
    }


def test_the_leaf_is_a_leaf_provider_taking_int8_convrots_operand() -> None:

    assert isinstance(W4a8NativeLeaf(), LeafProvider)
    assert OPERANDS[W4a8NativeLeaf().name] == INT8_CONVROT
    assert W4a8NativeLeaf().supports([64, 512], dict(PARAMS)) == ""
    assert "group_size" in W4a8NativeLeaf().supports([64, 512], {**dict(PARAMS), "group_size": 32})


def test_four_bits_with_eight_bit_levels_beat_nvfp4_on_the_same_weights() -> None:
    values = weight(64, 1024, 7)
    w4a8, nvfp4 = encode_w4a8("w", values), encode_nvfp4("w", values)
    assert len(w4a8.payload) == len(nvfp4.payload) == 64 * 512
    assert 0.02 < w4a8.relative_frobenius() < nvfp4.relative_frobenius()
    assert w4a8.saturated < values.size // 1000  # ALS may clip a group's outlier to its top level
    exact = decode_int8_convrot256(
        levels(w4a8, 64, 1024), np.frombuffer(w4a8.companions[1].raw, np.float32)
    )
    error = np.linalg.norm(exact - values) / np.linalg.norm(values)
    assert abs(error - w4a8.relative_frobenius()) < 1e-6


def test_the_producer_writes_kitchens_layout_and_tensorfs_decodes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lane with its blocks in w4a8 beside int8 ones, written in chunks (12 MiB of source rows
    against the writer's 8 MiB reads): TensorFS's own decoder reads back exactly the levels and
    scales the producer chose, one codebook for the whole tensor."""
    torch = pytest.importorskip("torch")
    store = home(tmp_path / "tensorfs", monkeypatch)
    shown: list[tuple[str, str, str, str, str]] = []
    big = weight(2048, 1536, 1)
    source = checkpoint(
        store,
        {
            "transformer": {"blocks.0.ff.weight": big, "proj_out.weight": weight(8, 256, 2)},
            "text_encoder": {"layers.0.q_proj.weight": weight(16, 256, 3)},
        },
    )
    plan = QuantizationPlan(
        encoding="int8.convrot",
        components=("transformer", "text_encoder"),
        include=("transformer.blocks.*.weight", "text_encoder.layers.*.weight"),
        encodings=(EncodingRule("transformer.*", ENCODING),),
        output_precision="preserve",
    )
    quantize_into(
        context(store, shown=shown, outputs={"model": 256 << 20}),
        fake_telemetry(fake_attempt("w4a8")), source=Weights(source), plan=plan, output="model",
    )  # fmt: skip
    out = adopted(store, shown)
    ff = out.components["transformer"].tensors["blocks.0.ff.weight"]
    assert ff.encoding is not None and (ff.encoding.scheme, ff.encoding.params) == (
        ENCODING,
        dict(PARAMS),
    )
    q_proj = out.components["text_encoder"].tensors["layers.0.q_proj.weight"]
    assert q_proj.encoding is not None and q_proj.encoding.scheme == "int8.convrot"
    parts = {
        p: (d.dtype, d.shape)
        for p, d in raw.parts(out, "transformer", "blocks.0.ff.weight").items()
    }
    assert parts == {
        "weight": ("I8", (2048, 768)),
        "group_scale": ("F8_E4M3", (2048, 96)),
        "scale": ("F32", (2048,)),
        "codebook": ("F32", (16,)),
    }
    tensors = out.components["transformer"].tensors
    decoded = torch.from_dlpack(tensors.read("blocks.0.ff.weight", decode=True)).numpy()
    whole = encode_w4a8("w", big)
    exact = decode_int8_convrot256(
        levels(whole, 2048, 1536), np.frombuffer(whole.companions[1].raw, np.float32)
    )
    np.testing.assert_allclose(decoded.astype(np.float64), exact, rtol=2e-6, atol=1e-9)


def test_the_leaf_is_the_int8_law_on_the_unpacked_levels() -> None:
    """Off a card the leaf unpacks the codes and the int8 lane's GEMM serves: its output is the
    int8 leaf's on those levels, and within W8A8's activation error of the exact weight."""
    torch = pytest.importorskip("torch")

    provider = W4a8NativeLeaf(params=PARAMS)
    values = weight(256, 512, 5)
    parts = roles(torch, values, "cpu")
    exact = probe._exact_weight(torch, provider, parts)
    x = torch.randn(3, 40, 512, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    reference = x @ exact.t()
    data = torch.from_numpy(levels(encode_w4a8("w", values), 256, 512)).to(torch.int8)
    for dtype in (torch.float32, torch.bfloat16):
        leaf = provider.leaf(torch, parts, probe._StandIn(512, 256), dtype)
        got = leaf(x.to(dtype))
        assert got.shape == (3, 40, 256) and got.dtype == dtype
        assert torch.equal(
            got, int8_linear.linear(torch, x.to(dtype), data, parts["scale"], None, dtype)
        )
        error = (got.double() - reference).norm() / reference.norm()
        assert error < 0.02, (dtype, float(error))
    agreement = probe._native_agreement(torch, provider, torch.device("cpu"), torch.bfloat16)
    assert agreement.passed and 0.03 < agreement.deviation_rel < 0.15, agreement


def test_the_weight_plane_binds_a_stored_tensor_into_the_leaf_it_planned() -> None:
    torch = pytest.importorskip("torch")

    provider, shape = W4a8NativeLeaf(params=PARAMS), probe._StandIn(512, 256)
    stored = roles(torch, weight(256, 512, 5), "cpu")
    meta = {role: tensor.to("meta") for role, tensor in stored.items()}
    planned = provider.leaf(torch, meta, shape, torch.bfloat16)
    made = _bound(
        planned,
        _derived(planned, meta),
        stored,
        lambda: provider.leaf(torch, stored, shape, torch.bfloat16),
    )
    direct = provider.leaf(torch, stored, shape, torch.bfloat16)
    assert made.keys() == {"data", "group_scale", "scale", "codebook"}
    for name, tensor in made.items():
        planned._buffers[name] = tensor
        assert tensor is stored[{"data": "weight"}.get(name, name)], name
    planned._buffers["_logical_dtype"] = direct._buffers["_logical_dtype"]
    x = torch.randn(3, 512, generator=torch.Generator().manual_seed(9)).to(torch.bfloat16)
    assert torch.equal(planned(x), direct(x))


def test_the_fused_gemm_verifies_on_the_card_and_keeps_the_law(tmp_path: Path) -> None:
    """The GEMM as a machine gets it: the Runtime wheel's, verified on this card; every tile, a
    bias, fp16 and bf16 and an unaligned activation view agree with the int8 law on the levels."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the w4a8 GEMM runs on a CUDA card")

    kernel_cache.configure(kernel_cache.Store(tmp_path / "kernels"))
    assert w4a8_linear.native(torch, 0) is not None, w4a8_linear.why(0)
    generator = torch.Generator().manual_seed(11)
    for rows, out_features, in_features in ((1, 1024, 4096), (40, 776, 512), (300, 3072, 2048)):
        values = torch.randn(out_features, in_features, generator=generator).numpy() * 0.02
        parts = roles(torch, values, "cuda")
        w = W4a8(parts["weight"], parts["group_scale"], parts["scale"], parts["codebook"])
        data = torch.from_numpy(levels(encode_w4a8("w", values), out_features, in_features))
        data = data.to("cuda", torch.int8)
        assert torch.equal(unpack(torch, w), data), "the torch law is the encoder's levels"
        ops = w4a8_linear.native(torch, 0)
        assert ops is not None, "the prebuilt 8-bit op serves this card"
        for dtype in (torch.bfloat16, torch.float16):
            q, scales = ops.int8_quantize(
                torch.randn(rows, in_features, generator=generator).to("cuda", dtype)
            )
            bias = torch.randn(out_features, generator=generator).to("cuda", dtype)
            for b in (None, bias):  # every config bit for bit the 8-bit GEMM on the same levels
                want = ops.int8_gemm(q, scales, data, w.scale, b, dtype)
                for config in range(4):
                    got = ops.w4a8_gemm(q, scales, w, b, dtype, config)
                    assert torch.equal(got, want), (config, b is not None)
            x = torch.randn(1, rows + 1, in_features, generator=generator).to("cuda", dtype)[:, 1:]
            got = w4a8_linear.linear(torch, x, w, bias, dtype)
            assert w4a8_linear.serving(0) == w4a8_linear.OP
            want = int8_linear.eager(torch, x, data, parts["scale"], torch.float32) + bias.float()
            assert got.shape == (1, rows, out_features) and got.dtype == dtype
            error = float((got.float() - want).norm() / want.norm())
            assert error < 0.01, (rows, out_features, in_features, dtype, error)
