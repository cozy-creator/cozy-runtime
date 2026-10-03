"""The image VAEs whose decode the weight plane sizes (diffusers `AutoencoderKL` and
`AutoencoderKLQwenImage`): each family's scale and how it is told to decode in tiles.
A model's own decode path lives in its package (Anima's in `anima/vae.py`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, cast


class _KlConfig(Protocol):
    block_out_channels: Sequence[int]


class _Kl(Protocol):
    config: _KlConfig
    tile_sample_min_size: int
    tile_latent_min_size: int

    def enable_tiling(self) -> None: ...


class _Qwen(Protocol):
    spatial_compression_ratio: int

    def enable_tiling(self, height: int, width: int, stride_h: int, stride_w: int) -> None: ...


@dataclass(frozen=True)
class Geometry:
    """Pixels per latent side, and how the VAE is told to decode in tiles of a pixel side."""

    scale: int
    tile: Callable[[int], None]


def _family(root: object) -> type | None:
    """The diffusers image VAE class `root` is an instance of."""
    for kind in type(root).__mro__:
        if kind.__module__.startswith("diffusers.") and kind.__name__ in (
            "AutoencoderKL",
            "AutoencoderKLQwenImage",
        ):
            return kind
    return None


def geometry(root: object) -> Geometry | None:
    """`root`'s tile geometry; None when it is not an image VAE the plane sizes."""
    kind = _family(root)
    if kind is None:
        return None
    if kind.__name__ == "AutoencoderKLQwenImage":
        qwen = cast("_Qwen", root)
        return Geometry(
            int(qwen.spatial_compression_ratio),
            lambda side: qwen.enable_tiling(side, side, side * 3 // 4, side * 3 // 4),
        )
    kl = cast("_Kl", root)
    scale = 2 ** (len(kl.config.block_out_channels) - 1)

    def tile(side: int) -> None:
        kl.enable_tiling()
        kl.tile_sample_min_size = side
        kl.tile_latent_min_size = side // scale

    return Geometry(scale, tile)
