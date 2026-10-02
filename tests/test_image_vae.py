"""Image VAE decode: the lone-frame path of diffusers' Qwen VAE and both families' tiles.

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


def temporal_taps(vae: Any, z: Any) -> set[int]:
    """The temporal kernel sizes of the 3D convolutions one decode of `z` ran."""
    with torch.profiler.profile(record_shapes=True) as profile, torch.inference_mode():
        vae.decode(z)
    return {e.input_shapes[1][2] for e in profile.events() if e.name == "aten::conv3d"}


def test_a_lone_frame_decodes_to_the_stock_image_on_one_kernel_tap() -> None:
    vae = qwen()
    z = torch.randn(1, 16, 1, 24, 16)
    with torch.inference_mode():
        stock = vae.decode(z).sample
    assert temporal_taps(vae, z) == {1, 3}

    assert image_vae.fast_decode(vae)
    with torch.inference_mode():
        lone = vae.decode(z).sample
        (bare,) = vae.decode(z, return_dict=False)
    assert lone.shape == stock.shape == (1, 3, 1, 192, 128)
    assert torch.allclose(lone, stock, atol=1e-5)
    assert torch.equal(bare, lone)
    assert temporal_taps(vae, z) == {1}
    # no feature cache is filled for a frame that has no successor
    assert all(row is None for row in vae._feat_map)


def test_tiles_and_videos_keep_the_stock_decode() -> None:
    vae, plain = qwen(), qwen()
    assert image_vae.fast_decode(vae)
    z = torch.randn(1, 16, 1, 24, 24)
    for each in (vae, plain):
        each.enable_tiling(128, 128, 96, 96)
    with torch.inference_mode():
        assert torch.allclose(vae.decode(z).sample, plain.decode(z).sample, atol=1e-5)
    for each in (vae, plain):
        each.disable_tiling()
    video = torch.randn(1, 16, 2, 8, 8)
    with torch.inference_mode():
        assert torch.allclose(vae.decode(video).sample, plain.decode(video).sample, atol=1e-5)


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
    assert not image_vae.fast_decode(kl)

    assert image_vae.geometry(torch.nn.Linear(2, 2)) is None
