"""The weight-plane policy on SDXL's measured blocks.

`testdata/sdxl_blocks.csv` is the proving study's C2 input (tensorfs PR #268): the 87 UNet
blocks' bytes from the checkpoint header and their GPU time per step, from the R19 3 GiB trace
at 1605 MHz scaled to the clean 592.9 ms step, in execution order.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import weight_policy as policy
from cozy_runtime.internal.weights import Component, Region, _paid, _unpaid

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


def test_h3_bf16_mlp_leaves_can_stream_one_slot_beside_activations_at_one_gib() -> None:
    # Actual H3 Qwen text MLP leaf: 25600 x 5120 bf16 = 250 MiB. Two copies do not
    # fit beside this request's context, activation growth and allocation margin.
    mib = 1 << 20
    layout = policy.Layout(common=2 * mib, blocks=(25600 * 5120 * 2,) * 6)
    available, activation = (1 << 30) - 300 * mib, 350 * mib
    budget = policy.plane_budget(available, activation)
    placed = policy.stage({"h3": layout}, budget, single_slot={"h3"})["h3"]
    assert placed.window == 1
    assert placed.vram_bytes(layout) + activation + policy.MARGIN <= available
    # Unknown first-pass activations still use one slot for this actual degraded path,
    # even where two slots technically fit the unmeasured weight budget.
    unknown = policy.stage({"h3": layout}, available, single_slot={"h3"})["h3"]
    assert unknown.window == 1


def test_attached_regions_are_resident_and_cost_only_their_decoded_copies() -> None:
    """A region the machine keeps (`weights.attach/1`) is on the device outside the budget: the
    stage pays only for the rest, never streams an attached block, and so never drops one."""
    mib = 1 << 20

    unused: Any = None  # module, root and weight set: the arithmetic never touches them

    def region(index: int, span: int) -> Region:
        return Region(index=index, path=f"blocks.{index}", module=unused, weights=[], span=span)

    common, blocks = region(0, 2 * mib), [region(i, 64 * mib) for i in range(1, 5)]
    layout = policy.Layout(common=2 * mib, blocks=(64 * mib,) * 4)
    component = Component(
        name="unet",
        key="sdxl/unet",
        root=unused,
        ws=unused,
        device=0,
        regions=[common, *blocks],
        common=common,
        blocks=blocks,
        layout=layout,
        sealed=False,
    )
    component.shared = {0, 2, 3}  # common, and the blocks at positions 1 and 2
    paid, kept = _paid(component, layout)
    assert kept == (0, 3)
    assert paid == policy.Layout(common=0, blocks=(64 * mib, 64 * mib))

    whole = _unpaid(component, kept, policy.residence(paid, 128 * mib))
    assert sorted(whole.resident) == [0, 1, 2, 3] and whole.streamed == () and whole.ring == 0

    tight = policy.residence(paid, 96 * mib, cyclic=True)
    assert tight.streamed, "the budget holds one paid block and a ring, not both"
    placed = _unpaid(component, kept, tight)
    assert set(placed.streamed) <= {0, 3} and {1, 2} <= set(placed.resident)
    assert placed.ring == tight.ring
