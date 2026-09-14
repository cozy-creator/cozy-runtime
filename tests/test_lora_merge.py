"""Preparation arithmetic against independent small dense and FP8 format oracles."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from cozy_runtime.internal.encoding.formats import SPEC_ROWWISE, SPEC_ROWWISE_KEEPDIM
from cozy_runtime.internal.lora_merge import MergeAdapter, merge_rowwise_fp8

torch = pytest.importorskip("torch")


def _fp8_codes(values: NDArray[np.float32]) -> NDArray[np.uint8]:
    # Independently enumerate the E4M3FN value table, including subnormals.
    levels = np.array(
        [
            (code & 7) * 2.0**-9
            if code >> 3 == 0
            else (1 + (code & 7) / 8) * 2.0 ** ((code >> 3) - 7)
            for code in range(127)
        ]
    )
    magnitude = np.minimum(np.abs(values.astype(np.float64)), 448)
    high = np.minimum(np.searchsorted(levels, magnitude), 126)
    low = np.maximum(high - 1, 0)
    left, right = magnitude - levels[low], levels[high] - magnitude
    pick_low = (left < right) | ((left == right) & ((low & 1) == 0))
    code = np.where(pick_low, low, high).astype(np.uint8)
    return (code | (np.signbit(values).astype(np.uint8) << 7)).astype(np.uint8)


def _base() -> tuple[Any, Any]:
    data = torch.tensor([[1, 2, -1, 0], [0.5, -0.5, 3, -2], [4, 0, 0, 1]])
    return data.to(torch.float8_e4m3fn), torch.tensor([0.25, 0.5, 0.125])


def test_canonical_orientation_alpha_and_fp8_bytes() -> None:
    data, scale = _base()
    a = torch.tensor([[1, 0, -1, 2], [0, 1, 1, -1]], dtype=torch.float16)
    b = torch.tensor([[1, 2], [-2, 0.5], [3, -1]], dtype=torch.float32)
    originals = (data.view(torch.uint8).clone(), scale.clone(), a.clone(), b.clone())
    tiles = merge_rowwise_fp8(
        data,
        scale,
        [MergeAdapter(a, b, strength=0.25, alpha=4)],
        encoding_spec=SPEC_ROWWISE,
        scale_policy="recalibrate",
        tile_rows=2,
    )
    assert tiles is not None
    result = list(tiles)
    assert [row.start for row in result] == [0, 2]
    expected = data.float().numpy() * scale.numpy()[:, None]
    expected += 0.5 * (b.double().numpy() @ a.double().numpy()).astype(np.float32)
    expected_scale = np.max(np.abs(expected), axis=1) / np.float32(448)
    expected_codes = _fp8_codes(expected / expected_scale[:, None])
    codes = torch.cat([row.data for row in result]).view(torch.uint8).numpy()
    scales = torch.cat([row.scale for row in result]).numpy()
    np.testing.assert_array_equal(codes, expected_codes)
    np.testing.assert_array_equal(scales, expected_scale)
    decoded = torch.cat([row.data for row in result]).float().numpy() * scales[:, None]
    difference = decoded.astype(np.float64) - expected.astype(np.float64)
    assert sum(row.error.squared_error for row in result) == pytest.approx(
        np.square(difference).sum(), abs=1e-12
    )
    for actual, original in zip((data.view(torch.uint8), scale, a, b), originals, strict=True):
        assert torch.equal(actual, original)


def test_independent_stack_strengths_and_chunk_invariance() -> None:
    data, scale = _base()
    a = torch.tensor([[1.0, 0.5, -0.25, 2.0]], dtype=torch.float32)
    b = torch.tensor([[2.0], [-1.0], [0.5]], dtype=torch.bfloat16)
    stack = [MergeAdapter(a, b, 0.3, 2.0), MergeAdapter(a, b, -0.125)]
    outputs = []
    for rows in (1, 2, 128):
        iterator = merge_rowwise_fp8(
            data,
            scale,
            stack,
            encoding_spec=SPEC_ROWWISE,
            scale_policy="recalibrate",
            tile_rows=rows,
        )
        assert iterator is not None
        tiles = list(iterator)
        outputs.append((torch.cat([x.data for x in tiles]), torch.cat([x.scale for x in tiles])))
    for codes, scales in outputs[1:]:
        assert torch.equal(codes.view(torch.uint8), outputs[0][0].view(torch.uint8))
        assert torch.equal(scales, outputs[0][1])
    expected = data.double() * scale.double()[:, None]
    expected += (0.3 * 2.0 - 0.125) * (b.double() @ a.double())
    actual = outputs[0][0].double() * outputs[0][1].double()[:, None]
    assert (
        float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))
        < 0.04
    )


def test_fp32_factor_precision_and_keepdim_scale() -> None:
    data = torch.zeros(1, 1).to(torch.float8_e4m3fn)
    scale = torch.ones(1, 1)
    value = 1 + 2.0**-12  # BF16 downcasting loses this real FP32 factor contribution.
    a, b = torch.tensor([[value]]), torch.ones(1, 1)
    iterator = merge_rowwise_fp8(
        data,
        scale,
        [MergeAdapter(a, b)],
        encoding_spec=SPEC_ROWWISE_KEEPDIM,
        scale_policy="recalibrate",
    )
    assert iterator is not None
    result = list(iterator)
    assert result[0].scale.shape == (1, 1)
    assert float(result[0].data.float() * result[0].scale) == pytest.approx(value, abs=1e-7)


def test_zero_strength_inherits_but_nonfinite_factors_still_refuse() -> None:
    data, scale = _base()
    adapter = MergeAdapter(torch.ones(1, 4), torch.ones(3, 1), strength=0)
    assert (
        merge_rowwise_fp8(
            data, scale, [adapter], encoding_spec=SPEC_ROWWISE, scale_policy="recalibrate"
        )
        is None
    )
    bad = MergeAdapter(torch.full((1, 4), float("nan")), torch.ones(3, 1), strength=0)
    with pytest.raises(ValueError, match="LoRA A contains nonfinite"):
        merge_rowwise_fp8(
            data, scale, [bad], encoding_spec=SPEC_ROWWISE, scale_policy="recalibrate"
        )


def test_ordered_fp32_sums_are_not_coalesced() -> None:
    data, scale = torch.zeros(1, 1).to(torch.float8_e4m3fn), torch.ones(1)
    b = torch.ones(1, 1)
    big = MergeAdapter(torch.tensor([[1e8]]), b)
    small = MergeAdapter(torch.ones(1, 1), b)
    negative = MergeAdapter(big.a, b, -1)
    values = []
    for stack in ([big, small, negative], [big, negative, small]):
        iterator = merge_rowwise_fp8(
            data, scale, stack, encoding_spec=SPEC_ROWWISE, scale_policy="recalibrate"
        )
        assert iterator is not None
        tile = next(iterator)
        values.append(float(tile.data.float() * tile.scale[:, None]))
    assert values == [0.0, 1.0]


def test_shape_encoding_budget_and_overflow_refusals() -> None:
    data, scale = _base()
    adapter = MergeAdapter(torch.ones(2, 4), torch.ones(3, 2))
    with pytest.raises(ValueError, match="explicitly reviewed"):
        merge_rowwise_fp8(
            data, scale, [adapter], encoding_spec="fp8-rowwise", scale_policy="recalibrate"
        )
    with pytest.raises(ValueError, match="scale shape"):
        merge_rowwise_fp8(
            data, scale, [adapter], encoding_spec=SPEC_ROWWISE_KEEPDIM, scale_policy="recalibrate"
        )
    with pytest.raises(ValueError, match="orientation"):
        merge_rowwise_fp8(
            data,
            scale,
            [MergeAdapter(adapter.a.t(), adapter.b)],
            encoding_spec=SPEC_ROWWISE,
            scale_policy="recalibrate",
        )
    with pytest.raises(ValueError, match="scratch budget"):
        merge_rowwise_fp8(
            data,
            scale,
            [adapter],
            encoding_spec=SPEC_ROWWISE,
            scale_policy="recalibrate",
            max_scratch_bytes=4096,
        )
    iterator = merge_rowwise_fp8(
        data,
        scale,
        [adapter],
        encoding_spec=SPEC_ROWWISE,
        scale_policy="recalibrate",
        max_scratch_bytes=4400,
    )
    assert iterator is not None
    assert [x.start for x in iterator] == [0, 1, 2]
    huge = MergeAdapter(torch.full((1, 4), 1e30), torch.full((3, 1), 1e30))
    iterator = merge_rowwise_fp8(
        data, scale, [huge], encoding_spec=SPEC_ROWWISE, scale_policy="recalibrate"
    )
    assert iterator is not None
    with pytest.raises(ValueError, match="overflowed"):
        list(iterator)
    with pytest.raises(ValueError, match="must be finite"):
        merge_rowwise_fp8(
            data,
            scale,
            [MergeAdapter(adapter.a, adapter.b, math.inf)],
            encoding_spec=SPEC_ROWWISE,
            scale_policy="recalibrate",
        )


def test_original_grid_control_and_growth_refusal() -> None:
    data = torch.tensor([[1.0, 1.25, 1.5, 2.0]]).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.01])
    # Nonzero coefficient with zero factor values exercises only the codec.
    # Actual zero strength still returns the original-part graft, without this work.
    zero_update = [MergeAdapter(torch.zeros(1, 4), torch.ones(1, 1))]
    decoded = {}
    for policy in ("preserve", "grow", "recalibrate"):
        iterator = merge_rowwise_fp8(
            data, scale, zero_update, encoding_spec=SPEC_ROWWISE, scale_policy=policy
        )
        assert iterator is not None
        tile = next(iterator)
        decoded[policy] = tile.data.float() * tile.scale[:, None]
        if policy in ("preserve", "grow"):
            assert torch.equal(tile.data.view(torch.uint8), data.view(torch.uint8))
            assert torch.equal(tile.scale, scale)
            assert tile.error.squared_error == 0 and tile.error.changed_scale_rows == 0
        else:
            assert tile.error.squared_error > 0 and tile.error.changed_scale_rows == 1
    assert not torch.equal(decoded["recalibrate"], decoded["preserve"])
    saturated = torch.tensor([[448.0]]).to(torch.float8_e4m3fn)
    addition = [MergeAdapter(torch.ones(1, 1), torch.ones(1, 1))]
    iterator = merge_rowwise_fp8(
        saturated, torch.ones(1), addition, encoding_spec=SPEC_ROWWISE, scale_policy="preserve"
    )
    assert iterator is not None
    with pytest.raises(ValueError, match="original FP8 scales overflow"):
        list(iterator)
    iterator = merge_rowwise_fp8(
        saturated, torch.ones(1), addition, encoding_spec=SPEC_ROWWISE, scale_policy="grow"
    )
    assert iterator is not None
    tile = next(iterator)
    assert float(tile.data.float() * tile.scale[:, None]) == pytest.approx(449.0)
    assert tile.error.changed_scale_rows == 1 and float(tile.scale) > 1


def test_grow_preserves_valid_subnormal_grid_without_unnecessary_enlargement() -> None:
    data = torch.ones(1, 1).to(torch.float8_e4m3fn)
    scale = torch.tensor([2.0**-130])
    iterator = merge_rowwise_fp8(
        data,
        scale,
        [MergeAdapter(torch.zeros(1, 1), torch.ones(1, 1))],
        encoding_spec=SPEC_ROWWISE,
        scale_policy="grow",
    )
    assert iterator is not None
    tile = next(iterator)
    assert torch.equal(tile.data.view(torch.uint8), data.view(torch.uint8))
    assert torch.equal(tile.scale, scale)
    assert tile.error.changed_scale_rows == 0
