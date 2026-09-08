"""Module-level constants an interface reaches through names, arithmetic and sibling imports."""

from __future__ import annotations

from typing import Annotated

import msgspec

from cozy_runtime.author import AssetBound, AssetLimits, Assets, ImageAsset

_MIB = 1 << 20
MAX_EDGE = 2048
LANE_BYTES = 16 << 30
LANE_ENCODINGS: dict[str, str] = {"fp8": "fp8-rowwise/1", "mxfp8": "mxfp8/1"}
IMAGE_BOUND = AssetBound(max_bytes=64 * _MIB, max_decoded_bytes=256 * _MIB)
AUDIO_BOUND = AssetBound(max_bytes=128 * _MIB, media_types=("audio/wav",))
Prompt = Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
#: One declaration, used by every handler that takes stills — a parameter may name it.
Stills = Annotated[Assets[Annotated[ImageAsset, IMAGE_BOUND]], AssetLimits(images=2)]
