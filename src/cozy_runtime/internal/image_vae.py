"""The image VAEs whose decode the weight plane sizes (diffusers `AutoencoderKL` and
`AutoencoderKLQwenImage`): each family's tile geometry, and the Qwen decoder's fast path.

Decoding one image, ComfyUI's Wan VAE beats diffusers' by 2x at the same size: a causal 3D
convolution pads two zero frames in front of a lone frame and convolves all three, so only
the kernel's last temporal tap meets data; the frame-to-frame feature cache copies every
activation for a successor that never comes; and norms and upsampling run in float32.
`fast_decode` does what ComfyUI does: the last tap as a 2D convolution, no cache for a lone
frame, and norms and nearest upsampling in the activation dtype.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from types import MethodType, ModuleType
from typing import Protocol, cast

#: diffusers 0.40.0's `autoencoder_kl_qwenimage.py`, the source `fast_decode` was read against.
#: Another source keeps its own decode.
QWEN_SOURCE = "db8b9869c16b6e5274b5c4db20b488cf68a2f6a78bc1bee08936f2ae65fb8042"


class _Tensor(Protocol):
    shape: tuple[int, ...]

    def __getitem__(self, index: object) -> _Tensor: ...
    def unsqueeze(self, dim: int) -> _Tensor: ...
    def __mul__(self, other: object) -> _Tensor: ...
    def __add__(self, other: object) -> _Tensor: ...


class _Conv(Protocol):
    weight: _Tensor
    bias: _Tensor | None
    kernel_size: tuple[int, int, int]
    stride: tuple[int, int, int]
    dilation: tuple[int, int, int]
    groups: int
    #: (left, right, top, bottom, front, back): the causal padding, all of time in front
    _padding: tuple[int, int, int, int, int, int]
    forward: Callable[..., _Tensor]


class _KlConfig(Protocol):
    block_out_channels: Sequence[int]


class _Norm(Protocol):
    channel_first: bool
    scale: float
    gamma: _Tensor
    bias: _Tensor | float


class _Upsample(Protocol):
    mode: str


class _Kl(Protocol):
    config: _KlConfig
    tile_sample_min_size: int
    tile_latent_min_size: int

    def enable_tiling(self) -> None: ...


class _Decoder(Protocol):
    def __call__(self, x: _Tensor) -> _Tensor: ...
    def modules(self) -> Iterable[object]: ...


class _Qwen(Protocol):
    spatial_compression_ratio: int
    use_tiling: bool
    decoder: _Decoder
    post_quant_conv: Callable[[_Tensor], _Tensor]
    _decode: Callable[..., object]

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


def fast_decode(root: object) -> bool:
    """Give a Qwen image VAE's decoder ComfyUI's fast path. False leaves `root` as it is: not
    that family, or not the reviewed source. Call before anything wraps a module's forward."""
    kind = _family(root)
    if kind is None or kind.__name__ != "AutoencoderKLQwenImage":
        return False
    module = inspect.getmodule(kind)
    if module is None or _digest(module) != QWEN_SOURCE:
        return False
    vae = cast("_Qwen", root)
    functional = module.F
    causal = module.QwenImageCausalConv3d

    def tap(conv: _Conv, x: _Tensor, cache_x: object = None) -> _Tensor:
        if cache_x is not None or x.shape[2] != 1:
            return cast("_Tensor", causal.forward(conv, x, cache_x))
        left, _, top, _, _, _ = conv._padding
        return cast(
            "_Tensor",
            functional.conv2d(
                x[:, :, 0],
                conv.weight[:, :, -1],
                conv.bias,
                conv.stride[1:],
                (top, left),
                conv.dilation[1:],
                conv.groups,
            ),
        ).unsqueeze(2)

    def norm(norm: _Norm, x: _Tensor) -> _Tensor:
        unit = cast("_Tensor", functional.normalize(x, dim=1 if norm.channel_first else -1))
        return unit * norm.scale * norm.gamma + norm.bias

    def upsample(upsample: object, x: _Tensor) -> _Tensor:  # nearest copies: exact in any dtype
        return cast("_Tensor", module.nn.Upsample.forward(upsample, x))

    for layer in vae.decoder.modules():
        if isinstance(layer, causal):
            conv = cast("_Conv", layer)
            if (
                conv.stride[0] == conv.dilation[0] == 1
                and conv._padding[4] == conv.kernel_size[0] - 1 > 0
            ):
                conv.forward = MethodType(tap, conv)
        elif isinstance(layer, module.QwenImageRMS_norm):
            layer.forward = MethodType(norm, layer)
        elif isinstance(layer, module.QwenImageUpsample) and cast("_Upsample", layer).mode in (
            "nearest",
            "nearest-exact",
        ):
            layer.forward = MethodType(upsample, layer)

    stock = vae._decode

    def decode(vae: _Qwen, z: _Tensor, return_dict: bool = True) -> object:
        if vae.use_tiling or z.shape[2] != 1:
            return stock(z, return_dict)
        out = module.torch.clamp(vae.decoder(vae.post_quant_conv(z)), min=-1.0, max=1.0)
        return module.DecoderOutput(sample=out) if return_dict else (out,)

    vae._decode = MethodType(decode, vae)
    return True


def _digest(module: ModuleType) -> str:
    try:
        return hashlib.sha256(inspect.getsource(module).encode()).hexdigest()
    except (OSError, TypeError):
        return ""
