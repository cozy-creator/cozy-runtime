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

from cozy_runtime.author.kernels import INT8_CONVROT, PreQuantized  # noqa: E402
from cozy_runtime.derive.microscale import encode_int8_convrot256  # noqa: E402
from cozy_runtime.internal import attention_upstream as upstream  # noqa: E402
from cozy_runtime.internal import spill  # noqa: E402
from cozy_runtime.internal.attention_ulysses import Options, _call  # noqa: E402
from cozy_runtime.internal.bands import Bands  # noqa: E402
from cozy_runtime.internal.encoding.formats import hadamard256  # noqa: E402
from cozy_runtime.internal.encoding.leaves import _native_leaf_class  # noqa: E402


class Card:
    """The executor's memory as bands read it: `room` bytes free, nothing to unmap, no host
    memory to put a result in."""

    unified = False

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
        return False  # no other tensors to move


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


def products(x: Any, w: Any, v: Any, bias: Any, gamma: Any) -> Any:
    """A transformer's row ops: linears with and without a bias, layer and RMS norms, a
    softmax over scores, batched products with a bias, a log softmax."""
    f = torch.nn.functional
    h = f.layer_norm(f.linear(x, w, bias), (40,), gamma, bias)
    h = f.rms_norm(f.linear(h, v), (40,), gamma, 1e-6)
    heads = h.view(2, 37, 4, 10).transpose(1, 2)
    scores = (heads @ heads.transpose(-1, -2)).softmax(-1)
    h = (scores @ heads).transpose(1, 2).reshape(h.shape)
    h = torch.baddbmm(h, h, v.expand(2, -1, -1), alpha=0.5)
    return torch.log_softmax(torch.addmm(bias, h[0], v.t()), -1)


@pytest.mark.parametrize("inference", (False, True))
@pytest.mark.parametrize("room", ROOMS)
def test_products_softmaxes_and_norms_band_by_rows(room: int, inference: bool) -> None:
    """Output rows read the same rows of a product's left operand (its right one whole) and
    of a norm's or softmax's input. Under inference mode torch hands the mode `linear`,
    `matmul`, `softmax` and the norms whole; else their decompositions."""
    torch.manual_seed(5)
    x, w, v = torch.randn(2, 37, 24), torch.randn(40, 24), torch.randn(40, 40) / 6
    bias, gamma = torch.randn(40), torch.rand(40)
    mode = torch.inference_mode if inference else torch.no_grad
    with mode():
        stock = products(x, w, v, bias, gamma)
        with Short(room) as short:
            out = products(x, w, v, bias, gamma)
    torch.testing.assert_close(out, stock, rtol=1e-5, atol=1e-5)
    ran = set(short.ran)
    if inference:
        assert {"aten.linear", "aten.matmul", "aten.layer_norm", "aten.rms_norm"} <= ran
        assert {"aten.softmax", "aten.log_softmax"} <= ran
    else:
        assert {"aten.addmm", "aten.mm", "aten.bmm", "aten.native_layer_norm"} <= ran
        assert {"aten._softmax", "aten._log_softmax"} <= ran
    assert {"aten.baddbmm"} <= ran


@pytest.mark.parametrize("room", ROOMS)
def test_row_ops_in_bands_are_bit_exact_and_products_close(room: int) -> None:
    """A softmax or a norm computes each row alone: the same bits in any band. A product's
    kernel may change with the band's height (CPU and cuBLAS alike): float32 within rounding,
    bfloat16 within its own."""
    torch.manual_seed(6)
    x, w = torch.randn(3, 50, 64), torch.randn(96, 64)
    aten = torch.ops.aten
    exact = {
        "softmax": lambda: torch.softmax(x, -1),
        "layer_norm": lambda: aten.native_layer_norm(x, [64], w[0], w[1], 1e-5),
        "rms_norm": lambda: torch.nn.functional.rms_norm(x, (64,), w[0], 1e-6),
        "fused_rms_norm": lambda: aten._fused_rms_norm(x, [64], w[0], 1e-6),
    }
    e4m3, rows, cols = x[0].to(torch.float8_e4m3fn), x[0, :, :1].abs() + 0.5, w[None, :, 0].abs()
    close = {
        "linear": lambda: torch.nn.functional.linear(x, w),
        "bf16": lambda: torch.nn.functional.linear(x.bfloat16(), w.bfloat16(), w[:, 0].bfloat16()),
        "fp8 rowwise": lambda: torch._scaled_mm(
            e4m3, w.to(torch.float8_e4m3fn).t(), rows, cols, out_dtype=torch.bfloat16
        ),
    }
    ran: set[str] = set()
    with torch.inference_mode():
        for name, call in {**exact, **close}.items():
            stock = call()
            with Short(room) as short:
                out = call()
            ran |= set(short.ran)
            if name in exact:
                pairs = zip(
                    *(t if isinstance(t, tuple) else (t,) for t in (out, stock)), strict=True
                )
                assert all(torch.equal(got, want) for got, want in pairs), name
            else:  # bfloat16 within one of its steps at this magnitude (about 8)
                loose = {} if out.dtype == torch.float32 else {"atol": 2**-4, "rtol": 2**-7}
                torch.testing.assert_close(out, stock, **loose)
    names = ("softmax", "native_layer_norm", "rms_norm", "_fused_rms_norm", "linear", "_scaled_mm")
    assert {f"aten.{name}" for name in names} <= ran


@pytest.mark.parametrize("room", ROOMS)
def test_a_flux2_transformer_in_bands_is_the_whole_one(room: int) -> None:
    """A Flux.2 (Klein) transformer, seeded on the CPU: its linears, norms and modulations
    in bands; its attention whole."""
    torch.manual_seed(14)
    model = diffusers.Flux2Transformer2DModel(
        in_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=24,
        timestep_guidance_channels=16,
        axes_dims_rope=(4, 4, 4, 4),
        guidance_embeds=False,
    ).eval()
    image, text = torch.randn(1, 30, 8), torch.randn(1, 7, 24)
    ids = torch.zeros(1, 30, 4)
    ids[..., 2], ids[..., 3] = torch.arange(30) // 6, torch.arange(30) % 6

    def call() -> Any:
        return model(image, text, torch.tensor([0.5]), ids, torch.zeros(1, 7, 4)).sample

    with torch.inference_mode():
        stock = call()
        with Short(room) as short:
            out = call()
    torch.testing.assert_close(out, stock, rtol=1e-4, atol=1e-5)
    assert "aten.linear" in short.ran


@pytest.mark.parametrize("room", ROOMS)
def test_a_padding_taller_than_a_band_fills_the_bands_wholly_inside_it(room: int) -> None:
    """Run 10683, H3's second segment on an RTX 5090: a constant padding ran in one-row bands,
    and a band wholly inside the padding was cut from no rows with a negative pad, which
    torch refuses: "narrow(): length must be non-negative"."""
    torch.manual_seed(5)
    x = torch.randn(1, 3, 4, 7)
    _, ran = banded(lambda: torch.nn.functional.pad(x, (1, 2, 6, 5), value=0.5) * 2, room)
    assert "aten.pad" in ran or "aten.constant_pad_nd" in ran


class Tight(TorchDispatchMode):  # type: ignore[misc]
    """A card whose attention kernel fits `heads` query heads of `rows` query rows at once: a
    larger call runs out of device memory, as on a full GPU. Notes each call's (heads, rows)."""

    def __init__(self, heads: int, rows: int) -> None:
        super().__init__()
        self.heads, self.rows, self.ran = heads, rows, set[tuple[int, int]]()

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        if "scaled_dot_product" in str(func):
            size = (int(args[0].shape[1]), int(args[0].shape[2]))  # [batch, heads, rows, dim]
            if size[0] > self.heads or size[1] > self.rows:
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB")
            self.ran.add(size)
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize(
    ("heads", "rows", "masking"),
    [(4, 1000, "none"), (2, 1000, "causal"), (1, 128, "none"), (4, 128, "mask"), (8, 128, "mask")],
)
def test_attention_out_of_memory_runs_in_head_groups_then_query_rows(
    heads: int, rows: int, masking: str
) -> None:
    """The Runtime's attention dispatch, out of device memory whole: query heads with their K/V
    head (4 share one), whole groups first, then query rows in 128s beside every key, a mask
    cut with them; causal by heads only. Each piece computes its rows as the whole does."""
    torch.manual_seed(8)
    keys = 300 if masking == "causal" else 250
    q, k, v = torch.randn(2, 300, 8, 16), torch.randn(2, keys, 2, 16), torch.randn(2, keys, 2, 16)
    mask = torch.rand(2, 1, 300, keys) > 0.3 if masking == "mask" else None
    causal = masking == "causal"
    options = Options(attn_mask=mask, dropout_p=0.0, is_causal=causal, scale=None, enable_gqa=True)
    card = Bands(Card(1 << 30))  # the dispatch finds a device's bands while they live
    with torch.inference_mode():
        whole = upstream.sdpa(q, k, v, **options)
        with Tight(heads, rows) as tight:
            out = _call("_cozy_sdpa", upstream.SDPA, q, k, v, options)
    if rows >= 300:  # a head's rows as the whole call computes them, to the bit
        assert torch.equal(out, whole)
    else:  # a shorter query may take another CPU GEMM tiling (a GPU kernel's 128-row tiles not)
        torch.testing.assert_close(out, whole, rtol=1e-6, atol=1e-6)
    assert max(h for h, _ in tight.ran) <= heads and max(r for _, r in tight.ran) <= rows
    assert card.weights.modes["bands"].startswith("1 ops (1 attention)")


class Rows(TorchDispatchMode):  # type: ignore[misc]
    """A card whose products fit `rows` rows of their left operand at once."""

    def __init__(self, rows: int) -> None:
        super().__init__()
        self.rows, self.ran = rows, list[int]()

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        if func.overloadpacket in (torch.ops.aten.mm, torch.ops.aten.matmul):
            if args[0].shape[0] > self.rows:
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB")
            self.ran.append(int(args[0].shape[0]))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("operand", ("activation", "prequantized"))
def test_an_encoded_linear_out_of_memory_runs_in_rows(operand: str) -> None:
    """An int8 convrot leaf (its torch-ops law here) that runs out of device memory whole:
    each piece of rows is rotated, quantized and multiplied on its own, the whole call's bits;
    a producer's quantized operand is cut with its row scales."""
    torch.manual_seed(9)
    w, x = torch.randn(48, 256), torch.randn(2, 37, 256)
    encoded = encode_int8_convrot256("w", w.numpy())
    data = torch.frombuffer(bytearray(encoded.payload), dtype=torch.int8).view(48, 256)
    scale = torch.frombuffer(bytearray(encoded.companions[0].raw), dtype=torch.float32)
    roles = {"data": data, "scale": scale}
    leaf = _native_leaf_class(torch)(
        None, roles, torch.randn(48), torch.float32, INT8_CONVROT, "int8", 256, 48
    )
    given: Any = x
    if operand == "prequantized":
        rotated = hadamard256(torch, x.reshape(-1, 256)).mul_(1 / 16)
        rows = (rotated.abs().amax(-1, keepdim=True) / 127).clamp_(min=1e-12)
        q = torch.round(rotated / rows).clamp_(-127, 127).to(torch.int8)
        given = PreQuantized(q, rows, tuple(x.shape), INT8_CONVROT)
    card = Bands(Card(1 << 30))  # the leaf finds a device's bands while they live
    with torch.inference_mode():
        whole = leaf(given)
        with Rows(16) as tight:
            out = leaf(given)
    assert torch.equal(out, whole) and out.shape == (2, 37, 48)
    assert max(tight.ran) <= 16 and "1 encoded linear" in card.weights.modes["bands"]


class Crowded(Card):
    """A card `room` bytes free beside `others` bytes of tensors no op reads: `spill` moves
    what it is asked of them to host memory, never the op's `operands`."""

    def __init__(self, room: int, others: int, host: int = 0, device: str = "cpu") -> None:
        super().__init__(room)
        self.others, self.host, self.device = others, host, torch.device(device)
        self.asked: list[tuple[int, Any]] = []

    def lockable(self, want: int) -> int:
        return self.host

    def spill(self, need: int, operands: Any) -> bool:
        self.asked.append((need, operands))
        moved = min(need, self.others)
        self.room, self.others = self.room + moved, self.others - moved
        return moved > 0


def crowded_product(card: Crowded) -> tuple[Any, Any, int]:
    """A product out of device memory whole on `card`: its result, the whole one's, and the
    bytes its result and two pieces of rows need."""
    torch.manual_seed(12)
    x, w = torch.randn(64, 32, device=card.device), torch.randn(32, 128, device=card.device)
    with torch.inference_mode():
        whole = x @ w
        bands = Bands(card)  # type: ignore[arg-type]
        out = bands.run(torch.ops.aten.mm.default, (x, w), {}, short=True)
    assert [ops for _, ops in card.asked] == [(x, w)]
    return out, whole, whole.nbytes + 2 * (whole.nbytes + x.nbytes) // 64


@pytest.mark.parametrize("others", (1 << 20, 0))
def test_a_result_without_room_moves_other_tensors_off_first(others: int) -> None:
    """Run 11340 (0.28.1, 8 GB H3): feed-forward results went to host memory while tensors no
    op read kept the card, 1.48 TB over the bus, and the LoRA found no room. Those tensors move
    first, just what the result and two pieces lack, the op's operands never. Nothing to move
    and no host memory: the result takes the device all the same."""
    card = Crowded(room=1024, others=others, host=1 << 30 if others else 0)
    out, whole, least = crowded_product(card)
    assert card.asked[0][0] == least - 1024 and card.others == max(others - least + 1024, 0)
    assert out.device == card.device and "host memory" not in card.modes["bands"]
    torch.testing.assert_close(out, whole)  # pieces may take another CPU GEMM tiling


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a host block is page-locked memory")
def test_a_result_goes_to_host_memory_only_when_nothing_moves() -> None:
    """No other tensor to move: the result takes host memory, which the bands write."""
    card = Crowded(room=1024, others=0, host=1 << 30, device="cuda")
    out, whole, _ = crowded_product(card)
    assert spill.hosted(out) and "B of results in host memory" in card.modes["bands"]
    torch.testing.assert_close(out, whole)
    del out
    spill.release(torch)
