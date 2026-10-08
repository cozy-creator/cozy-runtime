"""Image VAE decode: both families' exact decode in row bands against diffusers' own
untiled one, and the Qwen family's one-tap path for a lone frame.

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


def test_a_lone_qwen_frame_decodes_on_one_kernel_tap_with_no_cache() -> None:
    """Diffusers convolves a lone frame with all three temporal taps (two meet zero padding)
    and caches every activation for a next frame; the exact decode runs the one tap that
    meets data as a 2D convolution and keeps no cache."""
    vae, z = qwen(), torch.randn(1, 16, 1, 24, 16)
    with torch.profiler.profile(record_shapes=True) as profile, torch.inference_mode():
        image_vae.plan(vae, z).run()  # type: ignore[union-attr]
    taps = {e.input_shapes[1][2] for e in profile.events() if e.name == "aten::conv3d"}
    assert taps <= {1}  # post_quant_conv's own single tap at most
    assert getattr(vae, "_feat_map", None) is None  # never made: no cache for a lone frame


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
def test_a_chunked_qwen_decode_is_the_untiled_one(band: int) -> None:
    vae = qwen()
    z = torch.randn(1, 16, 1, 5, 3)
    with torch.inference_mode():
        stock = vae.decode(z).sample
        chunked = image_vae.Decode(vae, z, band=band).run().sample
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


@pytest.mark.parametrize("vae, z", [(kl, (1, 4, 6, 6)), (qwen, (1, 16, 1, 5, 3))])
def test_a_block_that_runs_twice_writes_the_same(vae: Any, z: tuple[int, ...]) -> None:
    """The plane runs a block again when it ran out of memory part way: every pass, run twice,
    leaves the bits of one run (no pass reads the map it writes, and its statistics start
    over)."""
    root, latent = vae(), torch.randn(*z)

    def twice(module: Any, body: Any) -> Any:
        body()
        return body()

    with torch.inference_mode():
        clean = image_vae.Decode(root, latent, band=100).run().sample
        assert torch.equal(image_vae.Decode(root, latent, band=100).run(twice).sample, clean)


def test_only_the_reviewed_decode_is_exact() -> None:
    vae = kl()
    assert image_vae.exact(vae) and image_vae.exact(qwen())

    class Own(diffusers.AutoencoderKL):  # type: ignore[misc,name-defined]
        def _decode(self, z: Any, return_dict: bool = True) -> Any:
            return super()._decode(z * 2, return_dict)

    assert not image_vae.exact(Own(block_out_channels=(32,), norm_num_groups=8))
    assert image_vae.plan(qwen(), torch.randn(1, 16, 2, 4, 4)) is None  # a video
    assert not image_vae.exact(torch.nn.Linear(2, 2))
