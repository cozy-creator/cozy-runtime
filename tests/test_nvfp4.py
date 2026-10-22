"""`nvfp4`: the tensor-scaled producer into the store, decode and leaf."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tensorfs import raw

from cozy_runtime.author import Weights
from cozy_runtime.author.fakes import fake_attempt, fake_telemetry
from cozy_runtime.derive.microscale import encode_nvfp4
from cozy_runtime.derive.operations import EncodingRule, QuantizationPlan, quantize_into
from cozy_runtime.internal.encoding import params_of
from store_fixtures import adopted, checkpoint, context, home

ENCODING = "nvfp4"
PARAMS = params_of(
    {
        "block_size": 16,
        "global_scale": "multiply",
        "nibble_order": "low_first",
        "scale_layout": "row_major",
    }
)


def weight(rows: int, cols: int, seed: int) -> np.ndarray:
    values = np.random.default_rng(seed).standard_normal((rows, cols)).astype(np.float32) * 0.02
    values[1, :] = 0.0
    values[rows - 1, 5] = 3.0  # the tensor amax sits in the LAST row chunk
    return values


def test_the_leaf_is_a_leaf_provider() -> None:
    from cozy_runtime.internal.encoding import LeafProvider, Nvfp4NativeLeaf

    # Qualification runs a native oracle only for a LeafProvider; a missing member skips it.
    assert isinstance(Nvfp4NativeLeaf(), LeafProvider)


def test_chunked_encode_uses_the_tensor_amax() -> None:
    values = weight(32, 64, 3)
    whole = encode_nvfp4("w", values)
    head = encode_nvfp4("w", values[:16], amax=float(np.abs(values).max()))
    assert head.companions[1].raw == whole.companions[1].raw
    assert head.payload == whole.payload[: len(head.payload)]
    assert head.companions[0].raw == whole.companions[0].raw[: len(head.companions[0].raw)]
    assert encode_nvfp4("w", values[:16]).companions[1].raw != whole.companions[1].raw


def test_one_lane_carries_nvfp4_beside_int8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = home(tmp_path / "tensorfs", monkeypatch)
    shown: list[tuple[str, str, str, str, str]] = []
    source = checkpoint(
        store,
        {
            "transformer": {
                "blocks.0.ff.weight": weight(48, 64, 1),
                "proj_out.weight": weight(8, 64, 2),
            },
            "text_encoder": {"layers.0.q_proj.weight": weight(16, 256, 3)},
        },
    )
    plan = QuantizationPlan(
        components=("transformer", "text_encoder"),
        include=("transformer.blocks.*.weight", "text_encoder.layers.*.weight"),
        encodings=(EncodingRule("transformer.*", ENCODING),),
        output_precision="preserve",
    )
    quantize_into(
        context(store, shown=shown), fake_telemetry(fake_attempt("nvfp4")), source=Weights(source),
        plan=plan, encoding="int8.convrot", output="model",
    )  # fmt: skip
    out = adopted(store, shown)
    q_proj = out.components["text_encoder"].tensors["layers.0.q_proj.weight"]
    assert q_proj.encoding is not None and q_proj.encoding.scheme == "int8.convrot"
    ff = out.components["transformer"].tensors["blocks.0.ff.weight"]
    assert ff.encoding is not None and (ff.encoding.scheme, ff.encoding.params) == (
        ENCODING,
        {
            "block_size": 16,
            "global_scale": "multiply",
            "nibble_order": "low_first",
            "scale_layout": "row_major",
        },
    )
    parts = {
        role: (p.dtype, p.shape)
        for role, p in raw.parts(out, "transformer", "blocks.0.ff.weight").items()
    }
    assert parts == {
        "weight": ("F4", (48, 64)),
        "scale": ("F8_E4M3", (48, 4)),
        "scale_2": ("F32", ()),
    }
    assert out.components["transformer"].tensors["proj_out.weight"].encoding is None


def roles(torch: Any, values: np.ndarray, device: str) -> dict[str, Any]:
    rows, cols = values.shape
    encoded = encode_nvfp4("w", values)
    scale, scale_2 = encoded.companions
    return {
        "weight": torch.frombuffer(bytearray(encoded.payload), dtype=torch.uint8)
        .reshape(rows, cols // 2)
        .to(device),
        "scale": torch.frombuffer(bytearray(scale.raw), dtype=torch.uint8)
        .reshape(rows, cols // 16)
        .view(torch.float8_e4m3fn)
        .to(device),
        "scale_2": torch.frombuffer(bytearray(scale_2.raw), dtype=torch.float32)
        .reshape(())
        .to(device),
    }


def test_decode_is_the_spec_law_and_the_mined_vectors() -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.internal import probe
    from cozy_runtime.internal.encoding import Nvfp4Dequant

    provider = Nvfp4Dequant(params=PARAMS)
    vectors = probe._spec_vectors(torch, provider, torch.device("cpu"))
    nvfp4 = [c for c in probe.vector_cases() if c.encoding.scheme == "nvfp4" and not c.refuse]
    assert vectors.passed and not vectors.skipped and len(vectors.cases) == len(nvfp4) >= 7
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        assert probe._numerics_nvfp4(torch, provider, torch.device("cpu"), dtype).passed


def test_blocked_scales_round_trip() -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.internal.encoding.formats import from_blocked, to_blocked

    grid = torch.randint(0, 126, (200, 12), dtype=torch.uint8)
    assert torch.equal(from_blocked(torch, to_blocked(torch, grid), 200, 12), grid)


def test_leaf_matches_its_exact_weight() -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.internal import probe
    from cozy_runtime.internal.encoding import Nvfp4NativeLeaf

    values = weight(256, 512, 5)
    x = torch.randn(3, 40, 512, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    leaf_provider = Nvfp4NativeLeaf(params=PARAMS)
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


def test_the_weight_plane_binds_a_stored_tensor_into_the_leaf_it_planned() -> None:
    """`_Routes`: the leaf built on meta at plan time takes one bind's stored roles in the buffers
    its kernels read, the derived ones made again from the stored bytes, a stored role no leaf
    reads left out. H3's nvfp4 lane failed here with `KeyError: 'scale'` (runs 13145, 13185):
    the leaf named its roles after its buffers."""
    torch = pytest.importorskip("torch")
    from cozy_runtime.internal import probe
    from cozy_runtime.internal.encoding import Nvfp4NativeLeaf
    from cozy_runtime.internal.weights import _bound, _derived

    provider, shape = Nvfp4NativeLeaf(params=PARAMS), probe._StandIn(512, 256)
    stored = roles(torch, weight(256, 512, 5), "cpu")
    stored["input_scale"] = torch.ones((), dtype=torch.float32)
    meta = {role: tensor.to("meta") for role, tensor in stored.items()}
    planned = provider.leaf(torch, meta, shape, torch.bfloat16)
    derived = _derived(planned, meta)
    made = _bound(planned, derived, stored, lambda: provider.leaf(torch, stored, shape, torch.bfloat16))
    direct = provider.leaf(torch, stored, shape, torch.bfloat16)
    assert made.keys() == {"weight", "weight_scale", "weight_scale_2"}
    for name, tensor in made.items():
        planned._buffers[name] = tensor
        assert torch.equal(tensor.view(torch.uint8), direct._buffers[name].view(torch.uint8)), name
    planned._buffers["_logical_dtype"] = direct._buffers["_logical_dtype"]
    x = torch.randn(3, 512, generator=torch.Generator().manual_seed(9)).to(torch.bfloat16)
    assert torch.equal(planned(x), direct(x))
