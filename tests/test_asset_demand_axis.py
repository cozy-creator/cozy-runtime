"""Listed input assets are a demand axis: one and ten references are different cells."""

from __future__ import annotations

import msgspec

from cozy_runtime.author import ImageAsset
from cozy_runtime.author._assets import asset_dec_hook
from cozy_runtime.author._demand import normalize


class Edit(msgspec.Struct):
    prompt: str
    width: int = 1024
    height: int = 1024
    reference_images: list[ImageAsset] = []


def _features(count: int) -> dict[str, int]:
    refs = [f"input:{index}" for index in range(count)]
    payload = msgspec.convert(
        {"prompt": "x", "reference_images": refs}, type=Edit, dec_hook=asset_dec_hook
    )
    return dict(normalize(payload).values)


def test_reference_count_separates_demand_cells() -> None:
    assert "assets" not in _features(0)
    assert _features(1)["assets"] == 1
    assert _features(10)["assets"] == 10
    assert _features(10)["pixels"] == 1024 * 1024
