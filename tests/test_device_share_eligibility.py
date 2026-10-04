"""CPU metadata checks for the real attach/export selection policy, not CUDA proof."""

from __future__ import annotations

from typing import Any

import pytest

from cozy_runtime.internal.weights import Bake, Region, _shareable


@pytest.mark.parametrize("mode", ["none", "per_step", "bake", "baked", "dirty"])
def test_lora_regions_stay_private_across_strategy_transitions(mode: str) -> None:
    unused: Any = None
    # Module bytes are not touched by eligibility; the data is the registered plane region.
    region = Region(0, "attention.to_q", unused, [])
    if mode != "none":
        region.bakes = [Bake(weight=0)]
    region.bake = mode in ("bake", "baked", "dirty")
    region.baked = mode == "baked"
    region.dirty = mode == "dirty"
    assert _shareable(region) is (mode == "none")
    if mode == "per_step":
        assert not region.bake and not region.baked
        # The previous export test (`not baked`) admitted this exact churn case.
        assert region.bakes


def test_modified_bytes_cannot_share_even_if_their_hook_was_removed() -> None:
    unused: Any = None
    region = Region(0, "attention.to_q", unused, [], baked=True)
    assert not _shareable(region)
    region.baked, region.dirty = False, True
    assert not _shareable(region)


def test_plain_regions_remain_shareable_next_to_private_lora_regions() -> None:
    unused: Any = None
    plain = Region(0, "norm", unused, [])
    lora = Region(1, "attention.to_q", unused, [], bakes=[Bake(weight=0)])
    assert [region.index for region in (plain, lora) if _shareable(region)] == [0]
