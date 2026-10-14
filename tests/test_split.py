"""`author.split`: an op the package declares splittable runs whole, or in the pieces the
Runtime's memory allows, and the pieces put together are the whole op's result.

Real torch modules on the CPU. The device is a `Card`: it holds `limit` input elements of a
piece at once and runs out of memory past that, as a GPU does, so the piece count follows it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.author import Cut, split  # noqa: E402
from cozy_runtime.author._errors import RuntimeFailure  # noqa: E402
from cozy_runtime.author._split import install  # noqa: E402


class Card:
    def __init__(self, limit: int, ratio: float | None = None, room: int = 0) -> None:
        self.limit, self.ratio, self.free = limit, ratio, room
        self.pieces: list[int] = []
        self.runs: list[int] = []

    def room(self, key: str) -> tuple[float | None, int]:
        return self.ratio, self.free

    @contextmanager
    def piece(self, key: str, elements: int, need: int) -> Iterator[None]:
        if elements > self.limit:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        self.pieces.append(elements)
        yield

    def ran(self, name: str, pieces: int) -> None:
        self.runs.append(pieces)


@pytest.fixture
def card() -> Iterator[Any]:
    def plug(device: Card) -> Card:
        install(device)
        return device

    yield plug
    install(None)


def decoder() -> Any:
    """Latent -> 2x pixels, each pixel from its own latent element: every window computes its
    pixels as the whole does, so the cross-fade must give back exactly the whole."""
    torch.manual_seed(3)
    return torch.nn.Sequential(
        torch.nn.Conv2d(4, 8, 1),
        torch.nn.SiLU(),
        torch.nn.Upsample(scale_factor=2),
        torch.nn.Conv2d(8, 3, 1),
    ).double()


def test_without_a_runtime_the_op_runs_whole() -> None:
    calls: list[tuple[int, ...]] = []
    x = torch.randn(3, 4)
    assert torch.equal(split(lambda t: calls.append(tuple(t.shape)) or t * 2, x, Cut(0)), x * 2)
    assert calls == [(3, 4)]


def test_a_batch_cut_runs_in_pieces_that_join_exactly(card: Any) -> None:
    device = card(Card(limit=10**9, ratio=1.0, room=2 * 4 * 6 * 6))
    net, x = decoder(), torch.randn(5, 4, 6, 6, dtype=torch.float64)
    with torch.inference_mode():
        assert torch.equal(split(net, x, Cut(0)), net(x))
    assert device.pieces == [2 * 4 * 36] * 2 + [4 * 36] and device.runs == [3]


def test_an_unmeasured_op_runs_in_small_pieces_first(card: Any) -> None:
    device = card(Card(limit=10**9))
    split(lambda t: t, torch.zeros(16, 3), Cut(0))
    assert device.pieces == [2 * 3] * 8


@pytest.mark.parametrize("limit", [10**9, 4 * 20 * 14, 4 * 12 * 12])
def test_overlapping_tiles_cross_fade_into_the_whole_result(card: Any, limit: int) -> None:
    device = card(Card(limit=limit, ratio=1.0, room=10**9))
    net, z = decoder(), torch.randn(1, 4, 37, 29, dtype=torch.float64)
    with torch.inference_mode():
        tiled = split(net, z, Cut(-2, scale=2, overlap=6), Cut(-1, scale=2, overlap=6))
        whole = net(z)
    assert tiled.shape == whole.shape
    torch.testing.assert_close(tiled, whole, rtol=1e-9, atol=1e-9)
    assert max(device.pieces) <= limit and (len(device.pieces) > 1) == (limit < z.numel())


def test_an_encode_cuts_on_aligned_pixels(card: Any) -> None:
    card(Card(limit=3 * 24 * 24, ratio=1.0, room=10**9))
    torch.manual_seed(5)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 6, 1), torch.nn.SiLU(), torch.nn.Conv2d(6, 4, 2, stride=2)
    ).double()
    x = torch.randn(1, 3, 64, 48, dtype=torch.float64)
    with torch.inference_mode():
        tiled = split(net, x, Cut(-2, 0.5, overlap=12, align=2), Cut(-1, 0.5, overlap=12, align=2))
        torch.testing.assert_close(tiled, net(x), rtol=1e-9, atol=1e-9)


def test_a_measured_op_starts_at_the_pieces_that_fit(card: Any) -> None:
    """Bytes per element times the largest piece within the room: no failed attempt first."""
    device = card(Card(limit=10**9, ratio=100.0, room=100 * 2 * 4 * 36))
    x = torch.randn(8, 4, 6, 6)
    split(lambda t: t, x, Cut(0))
    assert device.pieces == [2 * 4 * 36] * 4


def test_a_refusal_is_a_reason_to_cut_finer_and_other_failures_are_not(card: Any) -> None:
    class Refusing(Card):
        @contextmanager
        def piece(self, key: str, elements: int, need: int) -> Iterator[None]:
            if elements > self.limit:
                raise RuntimeFailure("no room", code="operation_exceeds_device")
            self.pieces.append(elements)
            yield

    device = card(Refusing(limit=8, ratio=1.0, room=16))
    assert torch.equal(split(lambda t: t + 1, torch.zeros(4, 4), Cut(0)), torch.ones(4, 4))
    assert device.pieces == [8, 8]

    def broken(t: Any) -> Any:
        raise ValueError("not memory")

    with pytest.raises(ValueError):
        split(broken, torch.zeros(4, 4), Cut(0))
    card(Card(limit=1, ratio=1.0, room=16))
    with pytest.raises(torch.OutOfMemoryError):  # one row is the finest it cuts
        split(lambda t: t, torch.zeros(4, 4), Cut(0))
