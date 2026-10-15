"""Ops in bands of rows: every op a VAE decode runs, cut wherever the room says, gives the
whole op's result. Real diffusers VAEs with seeded weights on the CPU, nothing patched.

The device is a `Card`: the room it reports sets the band height, from one row up to the
whole op, so the same decode runs in every cut. Every op runs as one that just ran out of
memory (`short`): the bands it can run in, or whole.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from cozy_runtime.internal.bands import Bands  # noqa: E402


class Card:
    """The executor's memory as bands read it: `room` bytes free, nothing to unmap, no host
    memory to put a result in."""

    grouped = False

    def __init__(self, room: int) -> None:
        self.torch, self.device, self.room = torch, torch.device("cpu"), room
        self.modes: dict[str, str] = {}

    def activation_room(self) -> tuple[int, int]:
        return self.room, 0

    def free_for(self, need: int) -> None:
        pass

    def lockable(self, want: int) -> int:
        return 0  # no host memory apart: results stay on the "device"

    def spill(self, need: int, operands: Any) -> bool:
        return False


class Short(TorchDispatchMode):  # type: ignore[misc]
    """Every op as one out of memory whole: in bands when it can be, else as it is."""

    def __init__(self, room: int, short: bool = True) -> None:
        super().__init__()
        self.bands, self.short = Bands(Card(room)), short  # type: ignore[arg-type]
        self.ran: list[str] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        out = self.bands.run(func, args, kwargs, short=self.short)
        if out is NotImplemented:
            return func(*args, **kwargs)
        self.ran.append(str(func.overloadpacket))
        return out


class Kernels(TorchDispatchMode):  # type: ignore[misc]
    """The ops that run on the device, under the bands (entered before them)."""

    def __init__(self) -> None:
        super().__init__()
        self.ran: list[tuple[Any, Any]] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        if not any(getattr(a, "is_meta", False) for a in args):
            self.ran.append((func, args))
        return func(*args, **(kwargs or {}))


#: one-row bands, a few rows, and the whole op in one band
ROOMS = (1, 6000, 1 << 30)


def kl() -> Any:
    torch.manual_seed(7)
    return diffusers.AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",) * 3,
        up_block_types=("UpDecoderBlock2D",) * 3,
        block_out_channels=(32, 32, 64),
        norm_num_groups=8,
    ).eval()


def qwen() -> Any:
    torch.manual_seed(11)
    return diffusers.AutoencoderKLQwenImage(base_dim=16, num_res_blocks=1).eval()


def banded(call: Callable[[], Any], room: int, short: bool = True) -> tuple[Any, list[str]]:
    with torch.inference_mode():
        stock = call()
        with Short(room, short) as mode:
            out = call()
    assert out.shape == stock.shape
    assert (out - stock).abs().max() <= 1e-5
    return out, mode.ran


@pytest.mark.parametrize("room", ROOMS)
def test_a_kl_decode_in_bands_is_the_whole_one(room: int) -> None:
    """Group norms gather their statistics over every band; convolutions read their reach,
    upsampling its source rows; the mid-block attention runs whole."""
    vae, z = kl(), torch.randn(2, 4, 7, 11)  # odd sizes, two images
    _, ran = banded(lambda: vae.decode(z).sample, room)
    assert {"aten.conv2d", "aten.group_norm", "aten.upsample_nearest2d", "aten.silu"} <= set(ran)


@pytest.mark.parametrize("room", ROOMS)
def test_a_kl_encode_in_bands_is_the_whole_one(room: int) -> None:
    """Strided convolutions after an uneven padding (diffusers' downsampler)."""
    vae, x = kl(), torch.randn(1, 3, 29, 18)
    banded(lambda: vae.encode(x[..., :28, :16]).latent_dist.mean, room)


@pytest.mark.parametrize("room", ROOMS)
def test_a_qwen_decode_in_bands_is_the_whole_one(room: int) -> None:
    """Causal 3D convolutions after their padding, RMS norms over channels, nearest-exact
    upsampling in float32 and back."""
    vae, z = qwen(), torch.randn(1, 16, 1, 5, 3)
    _, ran = banded(lambda: vae.decode(z).sample, room)
    assert {"aten.conv3d", "aten.pad", "aten.linalg_vector_norm"} <= set(ran)


def test_a_lone_frame_convolves_only_the_taps_that_meet_it() -> None:
    """Diffusers convolves a lone frame with all three temporal taps of a causal convolution,
    two of them on its zero padding. With room, the convolution after that padding reads the
    frame itself with the one tap that meets it, a 2D convolution: the same image."""
    vae, z = qwen(), torch.randn(1, 16, 1, 24, 16)
    with torch.inference_mode():
        stock = vae.decode(z).sample
        with Kernels() as kernels, Short(0, short=False):
            image = vae.decode(z).sample
    assert (image - stock).abs().max() <= 1e-5
    weights = [args[1].dim() for func, args in kernels.ran if "conv" in str(func)]
    assert weights and set(weights) == {4}  # every 3D one went 2D


def test_a_padding_that_another_op_reads_first_is_not_folded() -> None:
    """A convolution folds the padding just before it only: an op between them may have
    changed the padded tensor, so the convolution reads it as it is."""
    torch.manual_seed(1)
    x, w = torch.randn(1, 2, 1, 30, 20), torch.randn(3, 2, 3, 3, 3)

    def call() -> Any:
        padded = torch.nn.functional.pad(x, (1, 1, 1, 1, 2, 0))
        padded.mul_(2.0)
        return torch.nn.functional.conv3d(padded, w)

    _, ran = banded(call, 1 << 30, short=False)
    assert "aten.conv3d" not in ran


@pytest.mark.parametrize("room", ROOMS)
def test_ops_with_broadcast_operands_band_the_rows_they_span(room: int) -> None:
    torch.manual_seed(2)
    x, gamma = torch.randn(2, 5, 13, 7), torch.randn(5, 1, 1)
    row = torch.randn(1, 1, 13, 1)
    _, ran = banded(lambda: torch.nn.functional.silu(x * gamma + row).float() / 3, room)
    assert {"aten.mul", "aten.add", "aten.silu", "aten.div"} <= set(ran)


@pytest.mark.parametrize("room", ROOMS)
def test_a_qwen_image21_decoder_in_bands_is_the_whole_one(room: int) -> None:
    """Qwen-Image 2.1's VAE, decoded as qwen-image-2 decodes one frame (the decoder alone):
    2D convolutions after explicit paddings, RMS norms, nearest-exact upsampling."""
    kind = getattr(diffusers, "AutoencoderKLQwenImage21", None)
    if kind is None:
        pytest.skip("this diffusers has no AutoencoderKLQwenImage21")
    torch.manual_seed(13)
    vae = kind(
        base_dim=16,
        decoder_base_dim=16,
        z_dim=8,
        dim_mult=[1, 2, 2],
        num_res_blocks=1,
        temperal_downsample=[False, True],
        latents_mean=[0.0] * 8,
        latents_std=[1.0] * 8,
    ).eval()
    z = torch.randn(1, 8, 1, 5, 3)
    _, ran = banded(lambda: vae.decoder(vae.post_quant_conv(z), first_chunk=True), room)
    assert {"aten.conv2d", "aten.pad", "aten._upsample_nearest_exact2d"} <= set(ran)


@pytest.mark.parametrize("room", ROOMS)
def test_joins_repeats_and_copies_band_along_other_dims(room: int) -> None:
    """A join or repeat along channels, a copy of a permuted view: the rows are the rows."""
    torch.manual_seed(4)
    x, y = torch.randn(1, 6, 2, 11, 9), torch.randn(1, 2, 2, 11, 9)

    def call() -> Any:
        joined = torch.cat([x, y], dim=1).repeat_interleave(2, dim=1)
        shuffled = joined.view(1, 4, 2, 2, 2, 11, 9).permute(0, 1, 5, 2, 6, 3, 4).contiguous()
        return shuffled.view(1, 4, 11 * 2, 9 * 2, 2)

    _, ran = banded(call, room)
    assert {"aten.cat", "aten.repeat_interleave", "aten.contiguous"} <= set(ran)
