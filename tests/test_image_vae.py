"""Image VAE tiles: both diffusers families take a tile side in pixels.

Real diffusers modules with seeded weights on the CPU; nothing is patched.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from cozy_runtime.internal import image_vae  # noqa: E402


def qwen() -> Any:
    torch.manual_seed(11)
    return diffusers.AutoencoderKLQwenImage(base_dim=16, num_res_blocks=1).eval()


def test_each_family_takes_a_tile_side_in_pixels() -> None:
    vae = qwen()
    geometry = image_vae.geometry(vae)
    assert geometry is not None and geometry.scale == 8
    geometry.tile(512)
    assert vae.use_tiling
    assert (vae.tile_sample_min_height, vae.tile_sample_min_width) == (512, 512)
    assert (vae.tile_sample_stride_height, vae.tile_sample_stride_width) == (384, 384)

    kl = diffusers.AutoencoderKL(block_out_channels=(32, 32, 32), latent_channels=4)
    geometry = image_vae.geometry(kl)
    assert geometry is not None and geometry.scale == 4
    geometry.tile(256)
    assert kl.use_tiling and kl.tile_sample_min_size == 256 and kl.tile_latent_min_size == 64

    assert image_vae.geometry(torch.nn.Linear(2, 2)) is None
