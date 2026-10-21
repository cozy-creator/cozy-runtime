"""The execution layer (varena-slim §2) on real torch ops on the CPU, over varena's host pages:
a request's ops answer to Rust's `r.oom`, cuts read `r.cut`, and each result is the uncut run's.

A `Card` (execution_models) runs out of memory: at seeded random ops, or for every op whose
result and largest operand pass its limit. The models are loaded through `Memory.load` with the
catch installed, exactly as the executor loads them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from torch import nn  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from cozy_runtime.author import Cut, split  # noqa: E402
from cozy_runtime.author._errors import MEMORY_EXHAUSTED, Fault, classify, fault  # noqa: E402
from cozy_runtime.author._split import install as install_split  # noqa: E402
from cozy_runtime.internal import attention_ulysses, execution  # noqa: E402
from cozy_runtime.internal.execution import catch  # noqa: E402
from execution_models import (  # noqa: E402
    Card,
    checkpoint,
    memory,
    placed,
    transformer,
    vae,
)

BF16 = torch.bfloat16
AGREEMENT = Path(__file__).parent / "testdata" / "collective_agreement.py"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}


@pytest.fixture(autouse=True)
def installed() -> Iterator[None]:
    attention_ulysses.install(execution)
    install_split(execution.split)
    yield
    attention_ulysses.install(None)
    install_split(None)


def tokens(seed: int, rows: int = 96) -> Any:
    return torch.randn(1, rows, 16, generator=torch.Generator().manual_seed(seed)).to(BF16)


def test_varena_s_exhausted_is_the_one_memory_error() -> None:
    """With nothing evictable and nothing to cut, Rust's answers end in Exhausted: raised as
    it is, it fails the attempt as capacity and never the process."""
    mem, x = memory(1 << 20), torch.ones(64)
    with mem.request("exhausted") as r, execution.serving(r), Card(limit=1):
        with pytest.raises(Exception) as raised:
            catch.settle(lambda: x * 2, None, (x,))
    assert fault(raised.value) is Fault.EXHAUSTED
    assert classify(raised.value).code == MEMORY_EXHAUSTED


def test_without_an_oom_the_catch_changes_nothing(tmp_path: Path) -> None:
    plain = transformer()
    mem = memory()
    model = placed(mem, checkpoint(tmp_path, "dit", plain), "dit", transformer)
    x = tokens(0)
    with torch.no_grad():
        want = plain(x)
        with mem.request("whole") as r, execution.serving(r):
            got = model(x)
    assert torch.equal(got, want)
    assert r.stats().leftover == 0


@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_ooms_at_seeded_random_ops_in_a_transformer_change_nothing(
    tmp_path: Path, seed: int
) -> None:
    plain = transformer()
    mem = memory()
    model = placed(mem, checkpoint(tmp_path, "dit", plain), "dit", transformer)
    card = Card(seed=seed, rate=0.04)
    with torch.no_grad():
        want = [plain(tokens(n)) for n in range(3)]
        with mem.request(f"random-{seed}") as r, execution.serving(r), card:
            got = [model(tokens(n)) for n in range(3)]
    assert card.refused, "the seed ran nothing out"
    assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))
    stats = r.stats()
    assert stats.retries + stats.cuts + stats.parks >= 1 and stats.leftover == 0


def test_a_small_card_cuts_ffn_tokens_and_attention_and_keeps_the_cuts(tmp_path: Path) -> None:
    """At 64 KiB a block's whole [96, 256] FFN projection (with its weight) and its [4, 96, 96]
    scores run out: Rust cuts the FFN's tokens and the attention's heads, the result is the
    whole one, and the next request runs at those levels with no OOM at all."""
    plain = transformer()
    mem = memory()
    model = placed(mem, checkpoint(tmp_path, "dit", plain), "dit", transformer)
    x = tokens(1)
    with torch.no_grad():
        want = plain(x)
        first, again = Card(limit=64 << 10), Card(limit=64 << 10)
        with mem.request("short") as r, execution.serving(r), first:
            got = model(x)
        assert first.refused and r.stats().cuts >= 2
        assert torch.equal(got, want)
        with mem.request("short again") as r, execution.serving(r), again:
            got = model(x)
    assert torch.equal(got, want)
    assert again.refused == 0, "the cuts that fit were not applied up front"


def test_a_vae_decode_runs_in_conv_norm_and_elementwise_bands(tmp_path: Path) -> None:
    """At 36 KiB no [16, 32, 32] activation of the decoder fits beside its input: its convs,
    group norms, SiLUs, adds and nearest upsampling run in row bands, results on the device."""
    plain = vae()
    mem = memory()
    model = placed(mem, checkpoint(tmp_path, "vae", plain), "vae", vae)
    z = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(2)).to(BF16)
    card = Card(limit=36 << 10)
    with torch.no_grad():
        want = plain.decode(z).sample
        with mem.request("decode") as r, execution.serving(r), card:
            got = model.decode(z).sample
    assert card.refused and r.stats().cuts >= 1
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [21, 22])
def test_ooms_at_seeded_random_ops_in_a_vae_decode_change_nothing(
    tmp_path: Path, seed: int
) -> None:
    plain = vae()
    mem = memory()
    model = placed(mem, checkpoint(tmp_path, "vae", plain), "vae", vae)
    z = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(3)).to(BF16)
    card = Card(seed=seed, rate=0.05)
    with torch.no_grad():
        want = plain.decode(z).sample
        with mem.request(f"decode-{seed}") as r, execution.serving(r), card:
            got = model.decode(z).sample
    assert card.refused
    assert torch.equal(got, want)


class Tiles(nn.Module):
    """A decoder whose forward splits itself by batch (`author.split`)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4, 8, 1), nn.SiLU(), nn.Conv2d(8, 3, 1))

    def forward(self, x: Any) -> Any:
        return split(self.net, x, Cut(0))


def tiles() -> Tiles:
    torch.manual_seed(4)
    return Tiles().to(BF16).eval()


def test_a_package_split_runs_in_the_windows_rust_cuts(tmp_path: Path) -> None:
    plain = tiles()
    mem = memory()
    model = placed(mem, checkpoint(tmp_path, "tiles", plain), "tiles", tiles)
    x = torch.randn(8, 4, 16, 16, generator=torch.Generator().manual_seed(5)).to(BF16)
    card = Card(limit=24 << 10)  # a [8, 8, 16, 16] activation is 32 KiB
    with torch.no_grad():
        want = plain(x)
        with mem.request("split") as r, execution.serving(r), card:
            got = model(x)
    assert card.refused and r.stats().cuts >= 1
    assert torch.equal(got, want)


# ------------------------------------------------------------------- mutation-safe replay


class FailsAfter(TorchDispatchMode):  # type: ignore[misc]
    """Runs the op, then raises an allocation failure the first time `target` runs: a write
    that changed its destination and then ran out (an out resize, an in-place op)."""

    def __init__(self, target: Any) -> None:
        super().__init__()
        self.target, self.attempts = target, 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        if func is self.target:
            self.attempts += 1
            if self.attempts == 1:
                raise torch.OutOfMemoryError("out of memory. Tried to allocate 32 MiB")
        return out


class Accumulates(nn.Module):
    """Adds its input into a tensor that existed before it ran (a KV cache's shape)."""

    def __init__(self, held: Any) -> None:
        super().__init__()
        self.held = held

    def forward(self, x: Any) -> Any:
        self.held.add_(x)
        return self.held * 2


class Fresh(nn.Module):
    """Writes only into what it made, and draws from the default generator."""

    def forward(self, x: Any) -> Any:
        h = x * 2
        h.add_(torch.randn(x.shape))
        return h.sin()


def wrapped(module: nn.Module, root: nn.Module | None = None, state: tuple[str, ...] = ()) -> Any:
    block = catch.Block("block", module if root is None else root, state)
    catch.wrap(module, block)
    return block


def test_a_failed_write_is_never_run_again(tmp_path: Path) -> None:
    held = torch.zeros(8)
    module = Accumulates(held)
    block = wrapped(module)
    mode = FailsAfter(torch.ops.aten.add_.Tensor)
    mem = memory()
    with torch.no_grad(), mem.request("write") as r, execution.serving(r):
        assert torch.equal(module(torch.ones(8)), torch.full((8,), 2.0))  # first run: observed
        assert block.bare is False, "a block writing what it did not make ran bare"
        with pytest.raises(torch.OutOfMemoryError), mode:
            module(torch.ones(8))
    assert mode.attempts == 1 and torch.equal(held, torch.full((8,), 2.0))


def test_a_bare_block_runs_again_with_its_generators_and_declared_state_back() -> None:
    root = nn.Module()
    root.seen = []  # type: ignore[assignment]
    fresh = Fresh()

    def forward(x: Any) -> Any:
        root.seen.append(1)  # type: ignore[operator, union-attr]
        return Fresh.forward(fresh, x)

    fresh.forward = forward  # type: ignore[method-assign]
    block = wrapped(fresh, root, ("seen",))
    x = torch.ones(16)
    torch.manual_seed(7)
    reference = Fresh()(x)
    mem = memory()
    with torch.no_grad(), mem.request("bare") as r, execution.serving(r):
        fresh(x)
        assert block.bare is True, "a block writing only what it made stayed under the layer"
        torch.manual_seed(7)
        with FailsAfter(torch.ops.aten.sin.default) as mode:  # out of memory at its last op
            got = fresh(x)
    assert mode.attempts == 2 and torch.equal(got, reference)
    assert block.bare is False and root.seen == [1, 1], root.seen  # type: ignore[attr-defined]


# ----------------------------------------------------------------------------- collectives


@pytest.mark.parametrize("short", [0, 1])
def test_an_oom_on_one_rank_becomes_the_same_cut_on_every_rank(tmp_path: Path, short: int) -> None:
    """Two gloo ranks: one runs out allocating its exchange's buffers in the first pass. It
    recovers locally mid-pass (both cross every head at once), and from the next pass both
    agree its finer level and cross the heads in halves. Every pass is degree 1's result."""
    argv = [sys.executable, str(AGREEMENT), str(tmp_path), "--degree", "2", "--short", str(short)]
    subprocess.run(argv, env=ENV, check=True)
    rows = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert [row["refused"] for row in rows] == [int(rank == short) for rank in range(2)]
    for row in rows:
        assert all(p["byte_equal"] for p in row["passes"]), row
        heads = [p["exchanged_heads"] for p in row["passes"]]
        assert heads == [[8], [4, 4], [4, 4]], row
