"""The weight-plane policy on SDXL's measured blocks.

`testdata/sdxl_blocks.csv` is the proving study's C2 input (tensorfs PR #268): the 87 UNet
blocks' bytes from the checkpoint header and their GPU time per step, from the R19 3 GiB trace
at 1605 MHz scaled to the clean 592.9 ms step, in execution order.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cozy_runtime.internal import weight_policy as policy

ROWS = list(csv.DictReader((Path(__file__).parent / "testdata" / "sdxl_blocks.csv").open()))
BLOCKS = [row for row in ROWS if not row["name"].startswith("common")]
UNET = policy.Layout(
    common=sum(int(row["bytes"]) for row in ROWS if row["name"].startswith("common")),
    blocks=tuple(int(row["bytes"]) for row in BLOCKS),
)
COMPUTE = {i: float(row["compute_ms"]) / 1000 for i, row in enumerate(BLOCKS)}
COMMON_S = sum(float(row["compute_ms"]) for row in ROWS if row["name"].startswith("common")) / 1000
#: 3 GiB of free VRAM less the 0.25 GB context and SDXL's measured 0.65 GB activations.
BUDGET = (3 << 30) - 900_000_000
LINK = 6.3e9  # the laptop's measured in-situ H2D rate (PCIe Gen3 x8 at P4)


def step_s(placed: policy.Residence, link: float = LINK) -> float:
    streamed = set(placed.streamed)
    flags = [i in streamed for i in range(len(BLOCKS))]
    compute = [COMPUTE[i] for i in range(len(BLOCKS))]
    return policy.simulate(UNET.blocks, compute, flags, max(placed.window, 1), link) + COMMON_S


def test_sdxl_at_3_gib_streams_a_spread_set_within_the_g4_target() -> None:
    placed = policy.residence(UNET, BUDGET, costs=policy.Costs(compute=COMPUTE, link_bps=LINK))
    assert placed.vram_bytes(UNET) <= BUDGET
    assert placed.window >= 2
    assert step_s(placed) <= 0.70  # G4: C2's model 0.666 + 5%; ComfyUI measured 0.888
    # A resident prefix with a streamed tail of the same bytes idles the copy stream.
    count = len(placed.streamed)
    tail = policy.Residence(
        resident=tuple(range(len(BLOCKS) - count)),
        streamed=tuple(range(len(BLOCKS) - count, len(BLOCKS))),
        window=placed.window,
    )
    assert step_s(tail) > 1.1 * step_s(placed)


def test_unmeasured_costs_place_a_spread_set_two_slots_wide() -> None:
    placed = policy.residence(UNET, BUDGET)
    assert placed.window == policy.DEFAULT_WINDOW
    assert placed.resident and placed.streamed
    assert min(placed.streamed) < len(BLOCKS) // 2 < max(placed.streamed)
    assert policy.residence(UNET, UNET.total).streamed == ()


def test_run_once_is_whole_or_streamed_and_below_one_slot_refuses_with_numbers() -> None:
    text = policy.Layout(common=200 << 20, blocks=(12 << 20,) * 12)
    both = policy.stage({"unet": UNET, "text": text}, BUDGET, cyclic={"unet"})
    assert both["text"].resident == () or both["text"].streamed == ()
    one = UNET.common + UNET.largest
    assert policy.stage({"unet": UNET}, one, cyclic={"unet"})["unet"].window == 1
    with pytest.raises(policy.BelowFloor) as refused:
        policy.stage({"unet": UNET}, one - 1, cyclic={"unet"})
    assert (refused.value.needed, refused.value.available) == (one, one - 1)


def test_activations_come_first() -> None:
    assert policy.plane_budget(8 << 30, 3 << 30, 300 << 20) == (5 << 30) - (300 << 20) - (
        policy.MARGIN
    )
    assert policy.decode_mode(free=1 << 30, evictable=3 << 30, untiled_bytes=3 << 30) == "untiled"
    assert policy.decode_mode(free=1 << 30, evictable=0, untiled_bytes=3 << 30) == "tiled"
