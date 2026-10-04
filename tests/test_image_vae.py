"""Image VAE decode: the lone-frame path of diffusers' Qwen VAE, and both families' exact
decode in row bands against diffusers' own untiled one.

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


def kl() -> Any:
    torch.manual_seed(7)
    return diffusers.AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",) * 3,
        up_block_types=("UpDecoderBlock2D",) * 3,
        block_out_channels=(32, 32, 64),
        norm_num_groups=8,
    ).eval()


#: bands of one row everywhere, two rows at full resolution, rows that vary by layer, and one
#: band per layer
BANDS = (1, 2 * 44 * 32, 5000, image_vae.BAND)


@pytest.mark.parametrize("band", BANDS)
def test_a_chunked_kl_decode_is_the_untiled_one(band: int) -> None:
    vae = kl()
    z = torch.randn(2, 4, 7, 11)  # odd sizes, two images
    with torch.inference_mode():
        stock = vae.decode(z).sample
        decode = image_vae.Decode(vae, z, band=band)
        (chunked,) = image_vae.Decode(vae, z, return_dict=False, band=band).run()
        assert decode.rows == {1: 1, BANDS[1]: 2}.get(band, decode.rows)
        assert torch.equal(decode.run().sample, chunked)
    assert chunked.shape == stock.shape == (2, 3, 28, 44)
    assert (chunked - stock).abs().max() <= 1e-5


@pytest.mark.parametrize("band", BANDS)
@pytest.mark.parametrize("fast", [True, False])
def test_a_chunked_qwen_decode_is_the_untiled_one(band: int, fast: bool) -> None:
    vae = qwen()
    assert image_vae.fast_decode(vae) or not fast
    z = torch.randn(1, 16, 1, 5, 3)
    with torch.inference_mode():
        stock = vae.decode(z).sample
        chunked = image_vae.plan(vae, z).run().sample  # type: ignore[union-attr]
    assert chunked.shape == stock.shape == (1, 3, 1, 40, 24)
    assert (chunked - stock).abs().max() <= 1e-5


def test_a_failed_step_runs_again_from_its_input() -> None:
    """A step that fails (out of memory, say) leaves its input; the decode runs again from it
    to the uninterrupted decode's bits."""
    vae, z = kl(), torch.randn(1, 4, 6, 6)
    calls = []

    def within(module: Any, body: Any) -> Any:
        calls.append(module)
        if len(calls) == 9:
            raise RuntimeError("out of memory")
        return body()

    with torch.inference_mode():
        clean = image_vae.Decode(vae, z, band=100).run().sample
        decode = image_vae.Decode(vae, z, band=100)
        with pytest.raises(RuntimeError):
            decode.run(within)
        assert 0 < decode.step < len(decode.steps)
        assert torch.equal(decode.run(within, host=True).sample, clean)


def test_only_the_reviewed_decode_is_exact() -> None:
    vae = kl()
    assert image_vae.exact(vae) and image_vae.exact(qwen())

    class Own(diffusers.AutoencoderKL):
        def _decode(self, z: Any, return_dict: bool = True) -> Any:
            return super()._decode(z * 2, return_dict)

    assert not image_vae.exact(Own(block_out_channels=(32,), norm_num_groups=8))
    assert image_vae.plan(qwen(), torch.randn(1, 16, 2, 4, 4)) is None  # a video
    assert not image_vae.exact(torch.nn.Linear(2, 2))
