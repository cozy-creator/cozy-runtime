"""Feed-forward modules run in token pieces when their intermediate would not fit, and the
pieces join into exactly the whole module's output (memory v2 §3 rung 3, cut first).

Real diffusers and torch modules on the CPU. The device is the `Card` of `test_split`: it holds
`limit` input elements of a piece and runs out of memory past that, as a GPU does.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

torch = pytest.importorskip("torch")
diffusers_attention = pytest.importorskip("diffusers.models.attention")

from cozy_runtime.author._split import install  # noqa: E402
from cozy_runtime.internal import ffn  # noqa: E402
from cozy_runtime.internal.pieces import Pieces  # noqa: E402
from cozy_runtime.internal.weights import Weights  # noqa: E402
from test_split import Card  # noqa: E402

FeedForward = diffusers_attention.FeedForward


class GatedMLP(torch.nn.Module):  # type: ignore[misc,name-defined]
    """A transformers-style MLP: gate, up and down projections."""

    def __init__(self, dim: int, width: int) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(dim, width, bias=False)
        self.up_proj = torch.nn.Linear(dim, width, bias=False)
        self.down_proj = torch.nn.Linear(width, dim, bias=False)

    def forward(self, x: Any) -> Any:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class OwnFeedForward(FeedForward):  # type: ignore[misc,valid-type]
    """A package's subclass of diffusers' module."""


class FeedForwardLookalike(torch.nn.Module):  # type: ignore[misc,name-defined]
    """Named like it, but nobody says its rows are independent."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)


def joined(cut: Any, whole: Any) -> bool:
    """Equal up to the GEMM's own rounding: a piece's row count can pick another kernel, so
    sums of the same products may round apart, never by more than a few units of the scale."""
    scale = whole.abs().max()
    return bool((cut - whole).abs().max() <= 8 * torch.finfo(whole.dtype).eps * scale)


@pytest.fixture
def card() -> Iterator[Any]:
    def plug(device: Card) -> Card:
        install(device)
        return device

    yield plug
    install(None)


def test_feed_forward_modules_are_found_by_class_and_by_nothing_else() -> None:
    assert ffn.widest(FeedForward(32, inner_dim=64, activation_fn="swiglu")) == 128
    assert ffn.widest(OwnFeedForward(32, inner_dim=96, activation_fn="gelu")) == 96
    assert ffn.widest(GatedMLP(32, 80)) == 80
    assert ffn.widest(FeedForwardLookalike()) == 0
    linear, gelu = torch.nn.Linear, torch.nn.GELU
    assert ffn.widest(torch.nn.Sequential(linear(8, 32), gelu(), linear(32, 8))) == 0


@pytest.mark.parametrize("module", ["diffusers", "gated"])
def test_a_feed_forward_short_of_room_runs_in_token_pieces_that_join_exactly(
    card: Any, module: str
) -> None:
    torch.manual_seed(5)
    net = (
        FeedForward(32, inner_dim=64, activation_fn="swiglu")
        if module == "diffusers"
        else GatedMLP(32, 64)
    ).double()
    x = torch.randn(2, 37, 32, dtype=torch.float64)
    with torch.inference_mode():
        whole = net(x)
    device = card(Card(limit=2 * 9 * 32, ratio=None, room=2 * 9 * 32))
    run = ffn.pieces(net.forward, ffn.widest(net), fits=lambda nbytes: False)
    with torch.inference_mode():
        cut = run(x)
    assert len(device.pieces) > 1 and max(device.pieces) <= device.limit
    assert joined(cut, whole)


def test_a_feed_forward_with_room_runs_once_whole(card: Any) -> None:
    device = card(Card(limit=10**9, ratio=1.0, room=10**9))
    net = FeedForward(16, inner_dim=32, activation_fn="swiglu").double()
    calls: list[tuple[int, ...]] = []
    asked: list[int] = []

    def forward(x: Any) -> Any:
        calls.append(tuple(x.shape))
        return net(x)

    def fits(nbytes: int) -> bool:
        asked.append(nbytes)
        return True

    x = torch.randn(1, 11, 16, dtype=torch.float64)
    run = ffn.pieces(forward, ffn.widest(net), fits)
    with torch.inference_mode():
        assert torch.equal(run(x), net(x))
    assert calls == [(1, 11, 16)] and not device.pieces and not device.runs
    assert asked == [11 * 2 * 64 * 8], "two of the widest intermediates per token"


def test_a_whole_call_that_runs_out_of_memory_runs_again_in_pieces(card: Any) -> None:
    device = card(Card(limit=10**9, ratio=None, room=4 * 16))
    net = GatedMLP(16, 48).double()
    tried: list[int] = []

    def forward(x: Any) -> Any:
        tried.append(int(x.shape[-2]))
        if x.shape[-2] > 8:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return net(x)

    x = torch.randn(1, 30, 16, dtype=torch.float64)
    with torch.inference_mode():
        whole = net(x)
        cut = ffn.pieces(forward, ffn.widest(net), fits=lambda nbytes: True)(x)
    assert tried[0] == 30 and max(tried[1:]) <= 8 and len(device.pieces) > 1
    assert joined(cut, whole)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA card")
def test_on_the_card_a_feed_forward_beside_too_little_room_runs_in_pieces() -> None:
    """The Runtime's own side: `Weights.fits` reads torch's limit, `Pieces` cuts by the room
    the ladder finds, and the module's whole intermediate never has to exist."""
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    install(Pieces(weights))
    try:
        torch.manual_seed(7)
        net = FeedForward(1024, inner_dim=8192, activation_fn="swiglu").cuda().half()
        x = torch.randn(1, 8192, 1024, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            whole = net(x)
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            # room for the input, the output and a few pieces, not the [8192, 16384] projection
            ballast = torch.empty(free - (160 << 20), dtype=torch.uint8, device="cuda")
            weights.limit_torch()
            cut = ffn.pieces(net.forward, ffn.widest(net), weights.fits)(x)
            del ballast
        assert weights.modes.get("FeedForward.forward.pieces"), weights.modes
        assert joined(cut.float(), whole.float())
    finally:
        install(None)
        torch.cuda.empty_cache()
