"""The image VAEs whose decode the weight plane runs (diffusers `AutoencoderKL` and
`AutoencoderKLQwenImage`): one exact decode, layer by layer in row bands, and the Qwen
decoder's fast path.

`Decode` runs the decoder one layer at a time over the whole image, each layer's full output
map kept on the device or in host memory. A convolution runs in full-width row bands, each
reading its rows plus the kernel's reach (zero rows only past the image's edges) with no
vertical padding; a GroupNorm normalizes every band with the whole map's statistics, merged
in float32; the mid-block attention runs whole, at latent resolution. The band partition
follows the image size and the model only: memory decides only where the maps live.

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
from collections.abc import Callable, Iterable, Iterator
from functools import cache, partial
from types import MethodType, ModuleType
from typing import Any, Protocol, cast

#: diffusers 0.40.0's `autoencoder_kl_qwenimage.py`, the source `fast_decode` was read against.
#: Another source keeps its own decode.
QWEN_SOURCE = "db8b9869c16b6e5274b5c4db20b488cf68a2f6a78bc1bee08936f2ae65fb8042"
#: diffusers 0.40.0's `_decode` and decoder block forwards that `Decode` mirrors, per family.
REVIEWED = {
    "AutoencoderKL": "95af1233ae21a54a7abcde7699f1e2b80a425f68955e1150028023e00a23b9aa",
    "AutoencoderKLQwenImage": "bd7a59502b15cd394a6e11dd5847614ed6aa81110d2c75dee2d35bc7b83c33fb",
}
#: The family's decode entries a subclass must not override.
ENTRIES = ("decode", "_decode")
#: Blocks that run whole, called as they are: their source is not mirrored.
WHOLE = ("Attention", "QwenImageAttentionBlock")
#: Elements in a band of a layer's wider map, input or output: the fixed partition.
BAND = 1 << 22

#: (module, body) -> body(), with the module's weights bound for all of it
type Within = Callable[[Any, Callable[[], Any]], Any]


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


class _Norm(Protocol):
    channel_first: bool
    scale: float
    gamma: _Tensor
    bias: _Tensor | float


class _Upsample(Protocol):
    mode: str


class _Decoder(Protocol):
    def __call__(self, x: _Tensor) -> _Tensor: ...
    def modules(self) -> Iterable[object]: ...


class _Qwen(Protocol):
    use_tiling: bool
    decoder: _Decoder
    post_quant_conv: Callable[[_Tensor], _Tensor]
    _decode: Callable[..., object]


def _family(root: object) -> type | None:
    """The diffusers image VAE class `root` is an instance of."""
    for kind in type(root).__mro__:
        if kind.__module__.startswith("diffusers.") and kind.__name__ in REVIEWED:
            return kind
    return None


def exact(root: object) -> bool:
    """Whether the plane decodes `root` with `Decode`."""
    return _reviewed(root) is not None


def plan(root: object, z: Any, *args: object, **kwargs: object) -> Decode | None:
    """`root.decode(z, *args, **kwargs)` as a `Decode`; None for a Qwen video (more than one
    frame), which keeps the stock decode."""
    if len(z.shape) == 5 and z.shape[2] != 1:
        return None
    return Decode(root, z, bool(kwargs.get("return_dict", args[0] if args else True)))


def _reviewed(root: object) -> ModuleType | None:
    """The family's diffusers module when `root` runs exactly the decode `Decode` mirrors:
    neither decode overridden, and every decoder block of a reviewed class and branch."""
    kind = _family(root)
    if kind is None or any(getattr(type(root), n) is not getattr(kind, n) for n in ENTRIES):
        return None
    module = inspect.getmodule(kind)
    decoder = getattr(root, "decoder", None)
    if module is None or decoder is None or not all(map(_plain, decoder.modules())):
        return None
    mirrored = frozenset(
        type(m).forward
        for m in decoder.modules()
        if type(m).__module__.startswith("diffusers.") and type(m).__name__ not in WHOLE
    )
    digest = _sources(cast("Any", kind)._decode, mirrored)
    return module if digest == REVIEWED[kind.__name__] else None


@cache
def _sources(entry: Callable[..., Any], mirrored: frozenset[Callable[..., Any]]) -> str:
    try:
        text = "".join(sorted(inspect.getsource(f) for f in (entry, *mirrored)))
    except (OSError, TypeError):
        return ""
    return hashlib.sha256(text.encode()).hexdigest()


def _plain(m: Any) -> bool:
    """`m` takes the branch `Decode` mirrors: a stride-1 zero-padded convolution (causal in
    time), no time embedding or resampling in a residual block, no live dropout."""
    name = type(m).__name__
    if name in ("Conv2d", "QwenImageCausalConv3d"):
        if any(n != 1 for n in (*m.stride, *m.dilation)) or m.padding_mode != "zeros":
            return False
        if name == "Conv2d":
            return not isinstance(m.padding, str)
        left, right, top, bottom, front, back = m._padding
        return bool(left == right and top == bottom and front == m.kernel_size[0] - 1 and back == 0)
    if name == "ResnetBlock2D":
        return m.upsample is None and m.downsample is None and m.time_emb_proj is None
    if name == "Upsample2D":
        return m.norm is None and not m.use_conv_transpose and m.interpolate and m.use_conv
    if name == "Dropout":
        return not m.training or m.p == 0
    return name != "SpatialNorm"


class _Stats:
    """A GroupNorm's mean and variance over a whole map, in float32 from each band's own
    (the parallel formula: stable, and the same bands in the same order give the same bits)."""

    def __init__(self, torch: ModuleType, groups: int) -> None:
        self.torch, self.groups = torch, groups
        self.bands: list[tuple[Any, Any, int]] = []
        self.affine: tuple[Any, Any] | None = None

    def add(self, band: Any) -> None:
        x = band.float().reshape(band.shape[0], self.groups, -1)
        self.bands.append((*self.torch.var_mean(x, dim=-1, correction=0), x.shape[-1]))

    def norm(self, norm: Any, band: Any) -> Any:
        """`band` as `norm` (an nn.GroupNorm) normalizes the whole map: in float32,
        x * (rstd * gamma) + (beta - mean * rstd * gamma), back in the band's dtype."""
        if self.affine is None:
            torch = self.torch
            var, mean = (torch.stack(c) for c in list(zip(*self.bands, strict=True))[:2])
            counts = [count for _, _, count in self.bands]
            n = torch.full((len(counts), 1, 1), counts[0] / sum(counts), device=var.device)
            for index, count in enumerate(counts):  # the last band may be shorter
                if count != counts[0]:
                    n[index] = count / sum(counts)
            whole = (n * mean).sum(0)
            variance = (n * (var + (mean - whole) ** 2)).sum(0)
            per = norm.num_channels // self.groups
            rstd = (variance + norm.eps).rsqrt().repeat_interleave(per, dim=1)
            scale = rstd * norm.weight.float()
            shift = norm.bias.float() - whole.repeat_interleave(per, dim=1) * scale
            self.affine = scale[..., None, None], shift[..., None, None]
        scale, shift = self.affine
        return self.torch.addcmul(shift, band, scale, out=self.torch.empty_like(band))


class Decode:
    """One exact decode (see the module docstring), resumable: a step that fails leaves its
    input map as it was, and `run` starts again at that step."""

    def __init__(self, root: Any, z: Any, return_dict: bool = True, band: int = BAND) -> None:
        module = _reviewed(root)
        assert module is not None, "not a reviewed image VAE"
        self.module, self.torch, self.F = module, module.torch, module.torch.nn.functional
        self.root, self.return_dict, self.band = root, return_dict, band
        self.device, self.dtype = z.device, z.dtype
        self.inputs = list(z.split(1))
        self.done: list[Any] = []
        self.step = 0
        self.x: Any = self.inputs[0]
        self.within: Within = _direct
        self.host = False
        #: bytes of the most full maps one step holds, and of a block that runs whole (input,
        #: queries, keys, values); elements of the largest map; rows of a full-resolution band
        self.maps = self.whole = self.biggest = self.rows = 0
        self.steps: list[Callable[[Any], Any]] = []
        height, width = int(z.shape[-2]), int(z.shape[-1])
        if module.__name__.endswith("qwenimage"):
            self._qwen(height, width)
        else:
            self._kl(height, width)
        #: a band's working bytes: its float32 copy, normalized and activated rows, output
        self.working: int = min(band, self.biggest) * (8 + 5 * int(z.element_size()))

    def peak(self, host: bool) -> int:
        """Device bytes the decode needs at most, its maps on the device or in host memory."""
        return self.working + (self.whole if host else self.maps)

    def run(self, within: Within | None = None, host: bool = False) -> Any:
        """The decode's output, as the family's `decode` returns it. New maps go to pinned host
        memory when `host`, and so does the map the step that runs first reads."""
        self.within, self.host = within or _direct, host
        if host and self.step > 0 and self.x.dim() == 4 and self.device.type == "cuda":
            x, (channels, height, width) = self.x, _dims(self.x)
            self.x = self._new(channels, height, width)
            self._put(self.x, 0, x)
        while True:
            while self.step < len(self.steps):
                self.x = self.steps[self.step](self.x)
                self.step += 1
            self.done.append(self.x)
            if len(self.done) == len(self.inputs):
                break
            self.step, self.x = 0, self.inputs[len(self.done)]
        sample = self.done[0] if len(self.done) == 1 else self.torch.cat(self.done)
        return self.module.DecoderOutput(sample=sample) if self.return_dict else (sample,)

    # -- maps and bands

    def _new(self, channels: int, height: int, width: int, final: bool = False) -> Any:
        """A map: on the device, or in pinned host memory rows outermost (a band of rows is
        one contiguous copy each way)."""
        if self.host and not final and self.device.type == "cuda":
            return self.torch.empty((height, channels, width), dtype=self.dtype, pin_memory=True)
        return self.torch.empty((1, channels, height, width), dtype=self.dtype, device=self.device)

    def _rows(self, x: Any, lo: int, hi: int) -> Any:
        """Rows [lo, hi) of map `x` on the device (a view of a device map)."""
        if x.dim() == 3:
            band = x[lo:hi].to(self.device, non_blocking=True)
            return band.permute(1, 0, 2).unsqueeze(0).contiguous()
        return x[:, :, lo:hi]

    def _put(self, out: Any, a: int, y: Any) -> None:
        if out.dim() == 3:
            out[a : a + y.shape[2]].copy_(y[0].permute(1, 0, 2), non_blocking=True)
        else:
            out[:, :, a : a + y.shape[2]].copy_(y)

    def _bands(self, height: int, width: int, channels: int) -> Iterator[tuple[int, int]]:
        rows = _rows_of(self.band, height, width, channels)
        return ((a, min(a + rows, height)) for a in range(0, height, rows))

    def _size(self, *maps: tuple[int, int, int]) -> None:
        self.maps = max(self.maps, sum(c * h * w for c, h, w in maps) * self.dtype.itemsize)
        self.biggest = max(self.biggest, *(c * h * w for c, h, w in maps))

    def _convolve(
        self, conv: Any, source: Callable[[int, int], Any], height: int, a: int, b: int
    ) -> Any:
        """Rows [a, b) of `conv` over a map `height` rows tall whose rows `source(lo, hi)`
        gives: its halo read, zero rows only past the image's edges, no vertical padding."""
        if type(conv).__name__ == "QwenImageCausalConv3d":  # a lone frame: the last tap
            left, _, top, _, _, _ = conv._padding
            weight, padding = conv.weight[:, :, -1], (top, left)
        else:
            weight, padding = conv.weight, conv.padding
        lo, hi = a - padding[0], b - padding[0] + weight.shape[2] - 1
        x = source(max(lo, 0), min(hi, height))
        if lo < 0 or hi > height:
            x = self.F.pad(x, (0, 0, max(-lo, 0), max(hi - height, 0)))
        return self.F.conv2d(x, weight, conv.bias, 1, (0, padding[1]), 1, conv.groups)

    def _pass(
        self,
        conv: Any,
        source: Callable[[int, int], Any],
        out: Any,
        residual: Callable[[int, int], Any] | None = None,
        scale: float = 1.0,
        stats: _Stats | None = None,
    ) -> Any:
        """`conv` band by band into map `out` with its weights bound, each band plus its
        `residual` rows and over `scale` (written in place into a map on the device); `stats`
        gathers the stored bands' statistics."""
        channels, height, width = _dims(out)

        def body() -> None:
            for a, b in self._bands(height, width, max(channels, _in(conv))):
                y = self._convolve(conv, source, height, a, b)
                if residual is not None:
                    there = out[:, :, a:b] if out.dim() == 4 else y
                    y = self.torch.add(y, residual(a, b), out=there)
                    if scale != 1:  # x / 1 is x: diffusers' division changes nothing
                        y.div_(scale)
                if stats is not None:
                    stats.add(y)
                if residual is None or out.dim() == 3:
                    self._put(out, a, y)

        self.within(conv, body)
        return out

    def _source(self, x: Any) -> Callable[[int, int], Any]:
        return lambda lo, hi: self._rows(x, lo, hi)

    def _stats(self, x: Any, groups: int) -> _Stats:
        stats = _Stats(self.torch, groups)
        channels, height, width = _dims(x)
        for a, b in self._bands(height, width, channels):
            stats.add(self._rows(x, a, b))
        return stats

    def _upsample(self, x: Any, up: Callable[[Any], Any]) -> Callable[[int, int], Any]:
        """Rows of the 2x nearest upsampled map: a band upsamples only the rows it reads."""

        def source(lo: int, hi: int) -> Any:
            first = lo // 2
            return up(self._rows(x, first, (hi + 1) // 2))[:, :, lo - 2 * first : hi - 2 * first]

        return source

    def _decoder(self, height: int, width: int, resnet: Callable[..., None]) -> None:
        """The decoder's layers in its forward's order: conv_in, the mid block's resnets and
        attentions, each up block's resnets and upsampler, then the head."""
        decoder = self.root.decoder
        channels = decoder.conv_in.out_channels
        self._size((decoder.conv_in.in_channels, height, width), (channels, height, width))
        out = (channels, height, width)
        self.steps.append(lambda x: self._pass(decoder.conv_in, self._source(x), self._new(*out)))
        mid = decoder.mid_block
        resnet(mid.resnets[0], height, width)
        for attention, block in zip(mid.attentions, mid.resnets[1:], strict=True):
            if attention is not None:
                self._size((channels, height, width), (channels, height, width))
                self.whole = max(self.whole, 4 * channels * height * width * self.dtype.itemsize)
                self.steps.append(partial(self._whole, attention))
            resnet(block, height, width)
        for up in decoder.up_blocks:
            for block in up.resnets:
                resnet(block, height, width)
            for upsampler in up.upsamplers or ():
                height, width = height * 2, width * 2
                self._upsampler(upsampler, height, width)
        head = decoder.conv_out.in_channels
        self.rows = _rows_of(self.band, height, width, head)
        self._size((head, height, width))
        self.steps.append(self._head)

    # -- AutoencoderKL (diffusers 0.40.0: `_decode`, `Decoder`, `UNetMidBlock2D`,
    #    `UpDecoderBlock2D`, `ResnetBlock2D`, `Upsample2D`)

    def _kl(self, height: int, width: int) -> None:
        if self.root.post_quant_conv is not None:
            self.steps.append(self.root.post_quant_conv)
        self._decoder(height, width, self._kl_resnet)

    def _whole(self, module: Any, x: Any) -> Any:
        """A block that runs whole (the mid-block attention), on the device."""
        x = self._rows(x, 0, _dims(x)[1])
        if type(module).__name__ == "QwenImageAttentionBlock":
            return module(x.unsqueeze(2))[:, :, 0]
        return module(x)

    def _kl_resnet(self, resnet: Any, height: int, width: int) -> None:
        """`ResnetBlock2D.forward` with no time embedding: a stats pass over the input, conv1
        gathering its output's statistics, conv2, then the shortcut, all scaled as
        `(input_tensor + hidden_states) / output_scale_factor`."""
        c_in, c_out = resnet.in_channels, resnet.conv2.out_channels
        self._size((c_in, height, width), (c_out, height, width), (c_out, height, width))
        act, scale, shortcut = resnet.nonlinearity, resnet.output_scale_factor, resnet.conv_shortcut

        def body(x: Any) -> Any:
            first = self._stats(x, resnet.norm1.num_groups)
            second = _Stats(self.torch, resnet.norm2.num_groups)
            h = self._pass(
                resnet.conv1,
                lambda lo, hi: act(first.norm(resnet.norm1, self._rows(x, lo, hi))),
                self._new(resnet.conv1.out_channels, height, width),
                stats=second,
            )
            out = self._pass(
                resnet.conv2,
                lambda lo, hi: resnet.dropout(
                    act(second.norm(resnet.norm2, self._rows(h, lo, hi)))
                ),
                self._new(c_out, height, width),
                None if shortcut is not None else self._source(x),
                scale,
            )
            if shortcut is None:
                return out
            return self._pass(shortcut, self._source(x), out, self._source(out), scale)

        self.steps.append(lambda x: self.within(resnet, lambda: body(x)))

    def _upsampler(self, upsampler: Any, height: int, width: int) -> None:
        """`Upsample2D.forward` (nearest 2x, then its convolution) or, for one frame with no
        feature cache, `QwenImageResample.forward` (nearest-exact 2x, then its convolution:
        an upsample3d's time convolution never runs on a first frame)."""
        if type(upsampler).__name__ == "Upsample2D":
            conv = upsampler.conv
            up = partial(self.F.interpolate, scale_factor=2.0, mode="nearest")
        else:
            up, conv = upsampler.resample[0], upsampler.resample[1]
        self._size((_in(conv), height // 2, width // 2), (conv.out_channels, height, width))
        self.steps.append(
            lambda x: self.within(
                upsampler,
                lambda: self._pass(
                    conv, self._upsample(x, up), self._new(conv.out_channels, height, width)
                ),
            )
        )

    def _head(self, x: Any) -> Any:
        """The decoder's head: `Decoder.forward`'s conv_norm_out, conv_act, conv_out; or the
        Qwen decoder's norm_out, nonlinearity, conv_out and `_decode`'s clamp."""
        decoder = self.root.decoder
        _, height, width = _dims(x)
        out = self._new(decoder.conv_out.out_channels, height, width, final=True)
        if not hasattr(decoder, "norm_out"):
            stats = self._stats(x, decoder.conv_norm_out.num_groups)
            norm = decoder.conv_norm_out
            source = lambda lo, hi: decoder.conv_act(stats.norm(norm, self._rows(x, lo, hi)))  # noqa: E731
            return self._pass(decoder.conv_out, source, out)
        self._pass(decoder.conv_out, self._local(decoder.norm_out, decoder.nonlinearity, x), out)
        return self.torch.clamp(out, min=-1.0, max=1.0).unsqueeze(2)

    # -- AutoencoderKLQwenImage, one frame (diffusers 0.40.0: `_decode`, `QwenImageDecoder3d`
    #    and its blocks with no feature cache, as `fast_decode` runs it)

    def _qwen(self, height: int, width: int) -> None:
        post = self.root.post_quant_conv
        self.steps.append(lambda z: post(z)[:, :, 0])
        self._decoder(height, width, self._qwen_resnet)

    def _local(self, norm: Any, act: Any, x: Any) -> Callable[[int, int], Any]:
        """Rows of `act(norm(x))`: an RMS norm is per position."""
        return lambda lo, hi: act(norm(self._rows(x, lo, hi).unsqueeze(2))[:, :, 0])

    def _qwen_resnet(self, resnet: Any, height: int, width: int) -> None:
        """`QwenImageResidualBlock.forward`: the shortcut, conv1, then conv2 plus the
        shortcut."""
        c_out = resnet.out_dim
        self._size((resnet.in_dim, height, width), (c_out, height, width), (c_out, height, width))
        shortcut = resnet.conv_shortcut
        conv = type(shortcut).__name__ != "Identity"

        def body(x: Any) -> Any:
            out = self._new(c_out, height, width)
            if conv:
                self._pass(shortcut, self._source(x), out)
            h = self._pass(
                resnet.conv1,
                self._local(resnet.norm1, resnet.nonlinearity, x),
                self._new(resnet.conv1.out_channels, height, width),
            )
            second = self._local(resnet.norm2, resnet.nonlinearity, h)
            return self._pass(
                resnet.conv2,
                lambda lo, hi: resnet.dropout(second(lo, hi)),
                out,
                self._source(out if conv else x),
            )

        self.steps.append(lambda x: self.within(resnet, lambda: body(x)))


def _dims(x: Any) -> tuple[int, int, int]:
    """A map's (channels, height, width): (1, C, H, W) on the device, (H, C, W) on the host."""
    if x.dim() == 3:
        return int(x.shape[1]), int(x.shape[0]), int(x.shape[2])
    return int(x.shape[1]), int(x.shape[2]), int(x.shape[3])


def _rows_of(band: int, height: int, width: int, channels: int) -> int:
    return min(max(band // (width * channels), 1), height)


def _in(conv: Any) -> int:
    return int(conv.weight.shape[1]) * int(conv.groups)


def _direct(_: Any, body: Callable[[], Any]) -> Any:
    return body()


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
