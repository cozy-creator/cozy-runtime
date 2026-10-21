"""`int8-convrot`: producer, recipe selection into the store, decode and leaf."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tensorfs import raw

from cozy_runtime.author import UnsupportedInput, Weights
from cozy_runtime.author.fakes import fake_attempt, fake_telemetry
from cozy_runtime.derive.microscale import decode_int8_convrot256, encode_int8_convrot256
from cozy_runtime.derive.operations import QuantizationPlan, quantize_into
from cozy_runtime.internal.encoding import params_of
from store_fixtures import adopted, checkpoint, context, home

ENCODING = "int8.convrot"


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


def test_include_recipe_encodes_exactly_its_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = home(tmp_path / "tensorfs", monkeypatch)
    shown: list[tuple[str, str, str, str, str]] = []
    source = checkpoint(
        store,
        {
            "transformer": {
                "blocks.0.attn.to_q.weight": weight(8, 256, 1),
                "blocks.0.mlp.out.weight": weight(16, 512, 2),
                "proj_out.weight": weight(8, 256, 3),
            },
            "vae": {"conv.weight": weight(8, 256, 4)},
        },
    )
    plan = QuantizationPlan(
        components=("transformer",),
        include=("transformer.blocks.*.weight",),
        output_precision="preserve",
    )
    quantize_into(
        context(store, shown=shown), fake_telemetry(fake_attempt("int8")), source=Weights(source),
        plan=plan, encoding=ENCODING, output="model",
    )  # fmt: skip
    out = adopted(store, shown)
    encoded = {
        (component, key)
        for component, held in out.components.items()
        for key, info in held.tensors.items()
        if info.encoding is not None
    }
    assert encoded == {
        ("transformer", "blocks.0.attn.to_q.weight"),
        ("transformer", "blocks.0.mlp.out.weight"),
    }
    to_q = out.components["transformer"].tensors["blocks.0.attn.to_q.weight"]
    assert to_q.encoding is not None and to_q.encoding.scheme == ENCODING
    parts = raw.parts(out, "transformer", "blocks.0.attn.to_q.weight")
    assert (parts["weight"].dtype, parts["scale"].shape) == ("I8", (8,))
    assert out.components["transformer"].tensors["proj_out.weight"].dtype == "F32", "preserved"
    typo = QuantizationPlan(components=("transformer",), include=("transformer.nope.*",))
    with pytest.raises(UnsupportedInput, match="match no weight"):
        quantize_into(
            context(store, "typo"), fake_telemetry(fake_attempt("typo")), source=Weights(source),
            plan=typo, encoding=ENCODING, output="model",
        )  # fmt: skip


torch: Any = pytest.importorskip("torch")


def test_decode_is_the_spec_law_and_the_mined_vectors() -> None:
    from cozy_runtime.internal import probe
    from cozy_runtime.internal.encoding import Int8ConvRotDequant

    provider = Int8ConvRotDequant(
        params=params_of({"rotation": "kron_regular_h4", "rotation_size": 256})
    )
    vectors = probe._spec_vectors(torch, provider, torch.device("cpu"))
    assert vectors.passed and len(vectors.cases) == 4
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


def test_a_gemm_that_raises_is_served_by_torch_and_the_run_says_so() -> None:
    """Run 9141: a GEMM that raised ended its run and poisoned the executor. The op's GEMM of a
    quantized operand that raises (here: no op for CPU tensors) is served by torch's int8 GEMM
    at the eager path's exact law, and the run carries one warning per reason."""
    from cozy_runtime.author._model import WARNINGS
    from cozy_runtime.author.kernels import INT8_CONVROT, PreQuantized
    from cozy_runtime.internal import int8_linear
    from cozy_runtime.internal.encoding.formats import hadamard256

    values = weight(64, 512, 5)
    encoded = encode_int8_convrot256("w", values)
    data = torch.frombuffer(bytearray(encoded.payload), dtype=torch.int8).reshape(64, 512)
    scale = torch.frombuffer(bytearray(encoded.companions[0].raw), dtype=torch.float32)
    x = torch.randn(40, 512, generator=torch.Generator().manual_seed(9))
    rotated = hadamard256(torch, x).mul_(1.0 / 16)
    row_scale = (rotated.abs().amax(-1) / 127.0).clamp_(min=1e-12)
    payload = torch.round(rotated / row_scale[:, None]).clamp_(-127, 127).to(torch.int8)
    operand = PreQuantized(payload, row_scale, tuple(x.shape), INT8_CONVROT)
    WARNINGS.clear()
    for _ in range(2):
        got = int8_linear.gemm(torch, operand, data, scale, None, torch.float32)
        assert torch.equal(got, int8_linear.eager(torch, x, data, scale, torch.float32))
    [(message, fields)] = WARNINGS
    assert message == "encoded GEMM failed, a slower path served"
    assert fields["op"] == int8_linear.OP and fields["device"] == -1
    WARNINGS.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the int8 op runs on a CUDA card")
def test_native_op_verifies_on_the_card_and_keeps_the_law(tmp_path: Path) -> None:
    """The op as a machine gets it: the Runtime wheel's prebuilt one, verified on this card and kept
    in its store; every path (GEMV rows, the GEMM, a bias, fp16 and bf16, an unaligned view)
    agrees with the eager W8A8 law."""
    from cozy_runtime.internal import int8_linear, kernel_cache

    kernel_cache.configure(kernel_cache.Store(tmp_path / "kernels"))
    assert int8_linear.native(torch, 0) is not None
    assert any((tmp_path / "kernels" / "verified").iterdir())
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
            assert int8_linear.serving(0) == int8_linear.OP
            want = int8_linear.eager(torch, x, data, scale, torch.float32) + bias.float()
            assert got.shape == (1, rows, out_features) and got.dtype == dtype
            error = float((got.float() - want).norm() / want.norm())
            assert error < 0.01, (rows, out_features, in_features, dtype, error)
