"""LoRA bakes on real stored representations (CPU): the lane's own encoders are the oracle."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cozy_runtime.derive.microscale import e4m3_decode, encode_fp8_rowwise, encode_mxfp8
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
