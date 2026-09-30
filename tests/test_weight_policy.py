"""The weight-plane policy on SDXL's real region spans (87 UNet blocks, fp16, meta-measured)."""

from __future__ import annotations

import pytest

from cozy_runtime.internal import weight_policy as policy

MIB = 1 << 20
#: SDXL UNet: 104 MiB common, blocks from 6 to 94 MiB (the innermost no-split partition).
UNET = policy.Layout(
    common=104 * MIB, blocks=tuple([6 * MIB] * 30 + [40 * MIB] * 40 + [94 * MIB] * 17)
)
TEXT = policy.Layout(common=200 * MIB, blocks=tuple([12 * MIB] * 12))


def test_a_cyclic_stage_keeps_the_largest_prefix_beside_two_slots() -> None:
    placed = policy.stage({"unet": UNET}, 3 << 30, cyclic={"unet"})["unet"]
    assert placed.window == 2
    assert placed.vram_bytes(UNET) <= 3 << 30
    assert placed.resident == tuple(range(len(placed.resident)))
    assert set(placed.resident) | set(placed.streamed) == set(range(87))
    # everything fits: no slots at all
    assert policy.stage({"unet": UNET}, UNET.total, cyclic={"unet"})["unet"].streamed == ()


def test_a_run_once_stage_streams_whole_or_stays_whole() -> None:
    both = policy.stage({"unet": UNET, "text": TEXT}, 3 << 30, cyclic={"unet"})
    assert both["text"].resident == () and both["text"].window == 2
    roomy = policy.stage({"text": TEXT}, TEXT.total)
    assert roomy["text"].streamed == ()


def test_below_two_slots_it_streams_serially_and_below_one_it_refuses_with_numbers() -> None:
    one = UNET.common + UNET.largest
    assert policy.stage({"unet": UNET}, one, cyclic={"unet"})["unet"].window == 1
    with pytest.raises(policy.BelowFloor) as refused:
        policy.stage({"unet": UNET}, one - 1, cyclic={"unet"})
    assert (refused.value.needed, refused.value.available) == (one, one - 1)


def test_activations_come_first() -> None:
    assert (
        policy.plane_budget(8 << 30, 3 << 30, 300 * MIB)
        == (8 << 30) - (3 << 30) - 300 * MIB - policy.MARGIN
    )
    assert policy.decode_mode(free=1 << 30, evictable=3 << 30, untiled_bytes=3 << 30) == "untiled"
    assert policy.decode_mode(free=1 << 30, evictable=0, untiled_bytes=3 << 30) == "tiled"
    assert policy.window(94 * MIB, 12e9, 0.010) == 2
    assert policy.window(94 * MIB, 3e9, 0.005) == 8
