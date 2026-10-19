"""`author.split`: an op the package declares splittable runs whole, or in the windows the
memory allows, and the windows put together are the whole op's result.

Real torch modules on the CPU. The memory is `Sizes`: it runs the split at each window size it
is given, as varena's `cuts.split` does after each out-of-memory (its lattice is varena's own
test, `scripts/varena-cuts.py`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.author import Cut, split  # noqa: E402
from cozy_runtime.author._split import install  # noqa: E402


class Sizes:
    """Runs a split in windows of at most `most` elements, recording what it was asked."""

    def __init__(self, most: int) -> None:
        self.most = most
        self.asked: list[tuple[str, int, int]] = []
        self.pieces: list[int] = []

    def __call__(
        self, name: str, key: str, device: Any, total: int, finest: int, run: Callable[[int], Any]
    ) -> Any:
        self.asked.append((name, total, finest))
        return run(self.most)


@pytest.fixture
def sizes() -> Iterator[Callable[[int], Sizes]]:
    def plug(most: int) -> Sizes:
        memory = Sizes(most)
        install(memory)
        return memory

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


def counted(fn: Callable[[Any], Any], seen: list[tuple[int, ...]]) -> Callable[[Any], Any]:
    def call(t: Any) -> Any:
        seen.append(tuple(t.shape))
        return fn(t)

    return call


def test_without_a_memory_the_op_runs_whole() -> None:
    seen: list[tuple[int, ...]] = []
    x = torch.randn(3, 4)
    assert torch.equal(split(counted(lambda t: t * 2, seen), x, Cut(0)), x * 2)
    assert seen == [(3, 4)]


def test_a_batch_cut_runs_in_windows_that_join_exactly(sizes: Any) -> None:
    memory = sizes(2 * 4 * 36)
    net, x = decoder(), torch.randn(5, 4, 6, 6, dtype=torch.float64)
    seen: list[tuple[int, ...]] = []
    with torch.inference_mode():
        assert torch.equal(split(counted(net, seen), x, Cut(0)), net(x))
    assert [s[0] for s in seen] == [2, 2, 1]
    assert memory.asked[0][1:] == (5 * 4 * 36, 4 * 36)


def test_whole_when_the_memory_says_it_fits(sizes: Any) -> None:
    sizes(10**9)
    seen: list[tuple[int, ...]] = []
    split(counted(lambda t: t, seen), torch.zeros(16, 3), Cut(0))
    assert seen == [(16, 3)]


@pytest.mark.parametrize("most", [10**9, 4 * 20 * 14, 4 * 12 * 12])
def test_overlapping_tiles_cross_fade_into_the_whole_result(sizes: Any, most: int) -> None:
    sizes(most)
    net, z = decoder(), torch.randn(1, 4, 37, 29, dtype=torch.float64)
    seen: list[tuple[int, ...]] = []
    with torch.inference_mode():
        tiled = split(
            counted(net, seen), z, Cut(-2, scale=2, overlap=6), Cut(-1, scale=2, overlap=6)
        )
        whole = net(z)
    assert tiled.shape == whole.shape
    torch.testing.assert_close(tiled, whole, rtol=1e-9, atol=1e-9)
    elements = [s[1] * s[2] * s[3] for s in seen]
    assert max(elements) <= most and (len(seen) > 1) == (most < z.numel())


def test_an_encode_cuts_on_aligned_pixels(sizes: Any) -> None:
    sizes(3 * 24 * 24)
    torch.manual_seed(5)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 6, 1), torch.nn.SiLU(), torch.nn.Conv2d(6, 4, 2, stride=2)
    ).double()
    x = torch.randn(1, 3, 64, 48, dtype=torch.float64)
    with torch.inference_mode():
        tiled = split(net, x, Cut(-2, 0.5, overlap=12, align=2), Cut(-1, 0.5, overlap=12, align=2))
        torch.testing.assert_close(tiled, net(x), rtol=1e-9, atol=1e-9)


def test_the_finest_window_is_the_memory_s_floor(sizes: Any) -> None:
    """Twelve latent frames cut on time: one frame is the least a window takes."""
    memory = sizes(1)
    seen: list[tuple[int, ...]] = []
    split(counted(lambda t: t, seen), torch.zeros(12, 6, 10, 3), Cut(0))
    assert memory.asked[0][2] == 6 * 10 * 3 and seen == [(1, 6, 10, 3)] * 12
