"""LoRA bakes on real stored representations (CPU): the lane's own encoders are the oracle."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cozy_runtime.derive.microscale import (
    decode_int8_convrot256,
    e4m3_decode,
    encode_fp8_rowwise,
    encode_int8_convrot256,
    encode_mxfp8,
)
from cozy_runtime.internal import lora_bake

torch = pytest.importorskip("torch")

ROWS, COLS, RANK = 96, 256, 4


def _layer(seed: int = 7) -> tuple[Any, Any, Any]:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(ROWS, COLS, generator=generator) * 0.05
    down = torch.randn(RANK, COLS, generator=generator) * 0.02
    up = torch.randn(ROWS, RANK, generator=generator) * 0.02
    return weight.numpy(), down, up


def _rowwise(values: np.ndarray) -> tuple[Any, Any]:
    encoded = encode_fp8_rowwise("w", values)
    data = torch.frombuffer(bytearray(encoded.payload), dtype=torch.uint8).view(ROWS, COLS)
    scale = torch.frombuffer(bytearray(encoded.companions[0].raw), dtype=torch.float32)
    return data.view(torch.float8_e4m3fn), scale


def _mx(values: np.ndarray) -> tuple[Any, Any]:
    encoded = encode_mxfp8("w", values)
    data = torch.frombuffer(bytearray(encoded.payload), dtype=torch.uint8).view(ROWS, COLS)
    scale = torch.frombuffer(bytearray(encoded.companions[0].raw), dtype=torch.uint8)
    return data.view(torch.float8_e4m3fn), scale.view(ROWS, COLS // 32)


def _decoded_rowwise(data: Any, scale: Any) -> Any:
    return e4m3_decode(data.view(torch.uint8).numpy()) * scale.numpy()[:, None]


def _decoded_mx(data: Any, scale: Any) -> Any:
    mult = np.power(np.float32(2), scale.numpy().astype(np.float32) - 127).repeat(32, axis=1)
    return e4m3_decode(data.view(torch.uint8).numpy()) * mult


@pytest.mark.parametrize("rounding", lora_bake.ROUNDINGS)
@pytest.mark.parametrize("refit", [True, False])
def test_a_zero_strength_bake_changes_no_byte(rounding: lora_bake.Rounding, refit: bool) -> None:
    values, down, up = _layer()
    data, scale = _rowwise(values)
    before = data.view(torch.uint8).clone(), scale.clone()
    grid: lora_bake.Grid = "refit" if refit else "keep"
    lora_bake.bake_rowwise(torch, data, scale, down, up, 0.0, rounding=rounding, grid=grid)
    assert torch.equal(data.view(torch.uint8), before[0]) and torch.equal(scale, before[1])
    data, scale = _mx(values)
    before = data.view(torch.uint8).clone(), scale.clone()
    lora_bake.bake_mx(torch, data, scale, down, up, 0.0, rounding=rounding, refit=refit)
    assert torch.equal(data.view(torch.uint8), before[0]) and torch.equal(scale, before[1])


def test_a_nearest_refit_bake_is_the_lane_encoder_on_the_merged_weight() -> None:
    values, down, up = _layer()
    delta = (up @ down).numpy() * 0.75

    data, scale = _rowwise(values)
    merged = _decoded_rowwise(data, scale) + delta
    stats = lora_bake.Stats()
    lora_bake.bake_rowwise(torch, data, scale, down, up, 0.75, stats=stats)
    want = encode_fp8_rowwise("w", merged)
    assert data.view(torch.uint8).numpy().tobytes() == want.payload
    assert scale.numpy().tobytes() == want.companions[0].raw
    assert stats.saturated == 0 and stats.layers == 1 and stats.elements == ROWS * COLS
    assert stats.error_energy == pytest.approx(want.sse, rel=1e-4)

    data, scale = _mx(values)
    merged = _decoded_mx(data, scale) + delta
    lora_bake.bake_mx(torch, data, scale, down, up, 0.75)
    want = encode_mxfp8("w", merged)
    assert data.view(torch.uint8).numpy().tobytes() == want.payload
    assert scale.numpy().tobytes() == want.companions[0].raw


def test_nearest_on_the_kept_grid_loses_a_delta_below_half_a_step() -> None:
    values, down, up = _layer()
    alpha = 0.3  # most of the delta is below half of E4M3's step at these weights
    kept, effect, noise = {}, {}, {}
    cells: list[tuple[lora_bake.Rounding, lora_bake.Grid]] = [
        ("nearest", "keep"),
        ("nearest", "refit"),
        ("feedback", "keep"),
        ("stochastic", "keep"),
        ("nearest", "regrid"),
    ]
    for rounding, grid in cells:
        data, scale = _rowwise(values)
        stats = lora_bake.Stats()
        lora_bake.bake_rowwise(
            torch,
            data,
            scale,
            down,
            up,
            alpha,
            rounding=rounding,
            grid=grid,
            generator=torch.Generator().manual_seed(1),
            stats=stats,
        )
        kept[rounding, grid], effect[rounding, grid] = stats.retention, stats.effect_snr_db
        noise[rounding, grid] = stats.sqnr_db
    assert kept["nearest", "keep"] < 0.5 and effect["nearest", "keep"] < 3
    assert kept["nearest", "refit"] < 0.5
    # a run's sum survives error feedback; the delta's own direction does not
    assert kept["feedback", "keep"] < 0.5
    # unbiased, and less error than quantizing the row afresh
    assert 0.8 < kept["stochastic", "keep"] < 1.2
    assert effect["stochastic", "keep"] > effect["nearest", "keep"] + 3
    assert noise["stochastic", "keep"] > noise["nearest", "regrid"] + 3


def test_feedback_keeps_each_runs_sum_and_bakes_the_same_bytes_every_time() -> None:
    values, down, up = _layer()
    baked = []
    for _ in range(2):
        data, scale = _rowwise(values)
        base = _decoded_rowwise(data, scale)
        lora_bake.bake_rowwise(torch, data, scale, down, up, 0.02, rounding="feedback", grid="keep")
        baked.append(data.view(torch.uint8).clone())
    assert torch.equal(baked[0], baked[1])
    target = base + (up @ down).numpy() * 0.02
    stored = _decoded_rowwise(data, scale)
    runs = (stored - target).reshape(ROWS, -1, lora_bake.FEEDBACK_RUN).sum(axis=-1)
    step = np.abs(target).max(axis=1, keepdims=True) / 8  # an E4M3 step is at most 1/8 of the value
    assert np.all(np.abs(runs) <= step)


def test_a_kept_scale_saturates_what_the_delta_pushed_past_it() -> None:
    values, down, up = _layer()
    data, scale = _rowwise(values)
    stats = lora_bake.Stats()
    lora_bake.bake_rowwise(torch, data, scale, down, up, 400.0, grid="keep", stats=stats)
    assert stats.saturated > 0
    data, scale = _rowwise(values)
    stats = lora_bake.Stats()
    lora_bake.bake_rowwise(torch, data, scale, down, up, 400.0, grid="refit", stats=stats)
    assert stats.saturated == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_a_plain_bake_rounds_the_merged_weight_once(dtype: Any) -> None:
    values, down, up = _layer()
    weight = torch.from_numpy(values).to(dtype)
    want = (weight.float() + (up @ down) * 0.5).to(dtype)
    stats = lora_bake.Stats()
    lora_bake.bake_plain(torch, weight, down.to(dtype), up.to(dtype), 0.5, stats=stats)
    assert torch.allclose(weight.float(), want.float(), atol=float(want.abs().max()) * 2**-7)
    assert stats.retention == pytest.approx(1.0, abs=0.05)


def test_accumulate_adds_the_update_to_the_base_output() -> None:
    values, down, up = _layer()
    x = torch.randn(3, 5, COLS, generator=torch.Generator().manual_seed(3))
    weight = torch.from_numpy(values)
    out = torch.nn.functional.linear(x, weight)
    want = out + 0.5 * torch.nn.functional.linear(torch.nn.functional.linear(x, down), up)
    lora_bake.accumulate(torch, x, out, down, up, 0.5)
    assert torch.allclose(out, want, atol=1e-5)


def test_a_stochastic_bf16_bake_keeps_a_delta_below_half_an_ulp() -> None:
    values, down, up = _layer()
    kept = {}
    roundings: tuple[lora_bake.Rounding, ...] = ("nearest", "stochastic")
    for rounding in roundings:
        weight = torch.from_numpy(values).to(torch.bfloat16)
        before = weight.float()
        stats = lora_bake.Stats()
        lora_bake.bake_plain(
            torch,
            weight,
            down,
            up,
            0.02,
            rounding=rounding,
            generator=torch.Generator().manual_seed(2),
            stats=stats,
        )
        kept[rounding] = stats.retention
        # every stored value is one of the two bf16 neighbours of the merged weight
        merged = before + (up @ down) * 0.02
        assert torch.all((weight.float() - merged).abs() <= merged.abs() * 2**-7)
    assert kept["nearest"] < 0.5
    assert 0.8 < kept["stochastic"] < 1.2


def test_a_request_serves_its_loras_per_step_only_when_it_says_so() -> None:
    import msgspec

    from cozy_runtime.author._invoke import lora_mode

    class Plain(msgspec.Struct):
        prompt: str

    class Declares(msgspec.Struct):
        prompt: str
        lora_mode: str = ""

    assert lora_mode(Plain, {"prompt": "a"}) == ({"prompt": "a"}, "bake")
    assert lora_mode(Plain, {"prompt": "a", "lora_mode": "per_step"}) == (
        {"prompt": "a"},
        "per_step",
    )
    # an unknown value stays in the request: the ordinary ignored-field warning names it
    assert lora_mode(Plain, {"prompt": "a", "lora_mode": "fast"})[1] == "bake"
    wire = {"prompt": "a", "lora_mode": "per_step"}
    assert lora_mode(Declares, wire) == (wire, "bake")


def _int8(values: np.ndarray) -> tuple[Any, Any]:
    encoded = encode_int8_convrot256("w", values)
    data = torch.frombuffer(bytearray(encoded.payload), dtype=torch.int8).view(ROWS, COLS)
    scale = torch.frombuffer(bytearray(encoded.companions[0].raw), dtype=torch.float32)
    return data.clone(), scale.clone()


def test_an_int8_convrot_bake_keeps_the_delta_in_its_rotated_grid() -> None:
    """Stochastic rounding of the rotated rows keeps the LoRA in expectation; a zero delta is a
    fixed point; the stored weight decodes to the merged one within the lane's own error."""
    values, down, up = _layer(11)
    assert lora_bake.bakeable({"data": "i8", "scale": "f32"})
    data, scale = _int8(values)
    zero = {"data": data.clone(), "scale": scale}
    lora_bake.bake(torch, zero, down * 0, up, seed=3, stats=lora_bake.Stats())
    assert torch.equal(zero["data"], data)
    delta = (up.double() @ down.double()).numpy()
    target = values + delta
    base = decode_int8_convrot256(data.numpy(), scale.numpy())
    retained = []
    for seed in range(4):
        stored, stats = {"data": data.clone(), "scale": scale}, lora_bake.Stats()
        lora_bake.bake(torch, stored, down * 8, up, seed=seed, stats=stats)
        moved = decode_int8_convrot256(stored["data"].numpy(), scale.numpy()) - base
        retained.append(float((moved * delta * 8).sum() / np.square(delta * 8).sum()))
        assert abs(stats.retention - retained[-1]) < 0.05 and stats.saturated == 0
    assert abs(sum(retained) / 4 - 1) < 0.05
    stored = {"data": data.clone(), "scale": scale}
    lora_bake.bake(torch, stored, down, up, seed=0)
    merged = decode_int8_convrot256(stored["data"].numpy(), scale.numpy())
    lane = encode_int8_convrot256("w", target.astype(np.float32)).relative_frobenius()
    assert np.linalg.norm(merged - target) / np.linalg.norm(target) < 2 * lane + 0.01
