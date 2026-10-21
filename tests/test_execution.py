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
from cozy_runtime.internal.execution import catch, cuts  # noqa: E402
from execution_models import (  # noqa: E402
    Card,
    checkpoint,
    memory,
    placed,
    scores,
    transformer,
    vae,
)

BF16 = torch.bfloat16
AGREEMENT = Path(__file__).parent / "testdata" / "collective_agreement.py"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}


@pytest.fixture(autouse=True)
def installed() -> Iterator[None]:
    attention_ulysses.install(execution)
    install_split(execution)
    yield
    attention_ulysses.install(None)
    install_split(None)


def tokens(seed: int, rows: int = 96) -> Any:
    return torch.randn(1, rows, 16, generator=torch.Generator().manual_seed(seed)).to(BF16)


def test_varena_s_exhausted_is_the_one_memory_error() -> None:
    """With nothing evictable and nothing to cut, Rust's answers end in Exhausted: raised as
    it is, it fails the attempt as capacity and never the process."""
    mem, x = memory(1 << 20), torch.ones(64)
    with (
        Card(limit=1),
        mem.request("exhausted") as r,
        execution.serving(r),
        pytest.raises(Exception) as raised,
    ):
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
    card, xs = Card(seed=seed, rate=0.04), [tokens(n) for n in range(3)]
    with torch.no_grad():
        want = [plain(x) for x in xs]
        with card, mem.request(f"random-{seed}") as r, execution.serving(r):
            got = [model(x) for x in xs]
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
        with first, mem.request("short") as r, execution.serving(r):
            got = model(x)
        assert first.refused and r.stats().cuts >= 2
        assert torch.equal(got, want)
        with again, mem.request("short again") as r, execution.serving(r):
            got = model(x)
    assert torch.equal(got, want)
    assert again.refused == 0, "the cuts that fit were not applied up front"


def banded(func: Any) -> bool:
    """The ops a band cut serves: their working memory spans their operands (`Card`)."""
    return func.overloadpacket in catch.RULES


def test_a_vae_decode_runs_in_conv_norm_and_upsampling_bands(tmp_path: Path) -> None:
    """At 36 KiB no [16, 32, 32] conv, group norm or nearest upsampling of the decoder fits
    with its working memory: they run in channel bands, each the whole op's. Each op's site is
    its block's path and its place among the block's ops of its kind, the same in every pass:
    the next request starts each at the level Rust kept, and nothing runs out."""
    plain = vae()
    mem = memory()
    model = placed(mem, checkpoint(tmp_path, "vae", plain), "vae", vae)
    z = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(2)).to(BF16)
    cold, warm = Card(limit=36 << 10, working=banded), Card(limit=36 << 10, working=banded)
    with torch.no_grad():
        want = plain.decode(z).sample
        with cold, mem.request("decode") as r, execution.serving(r):
            got = model.decode(z).sample
        assert cold.refused and r.stats().cuts >= 1
        assert torch.equal(got, want)
        with warm, mem.request("decode again") as r, execution.serving(r):
            got = model.decode(z).sample
    assert warm.refused == 0 and r.stats().retries == r.stats().cuts == 0
    assert torch.equal(got, want)


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
        with card, mem.request(f"decode-{seed}") as r, execution.serving(r):
            got = model.decode(z).sample
    assert card.refused
    torch.testing.assert_close(got, want, rtol=2**-7, atol=2**-7)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_each_band_rule_is_the_whole_op(dtype: Any) -> None:
    """Every op band at every level against the op whole on real CPU kernels, each one's max
    error printed: nearest upsampling at a non-integer scale and to a size no scale divides
    (the whole op's index mapping in every band) and group norm by whole groups and by images
    (eps 0 stays 0) are bit-exact; conv by images and by output channels (whole groups of a
    grouped one) is in bfloat16, and in float32 within the rounding of another blocking."""
    aten = torch.ops.aten
    gen = torch.Generator().manual_seed(9)
    x = torch.randn(1, 16, 32, 32, generator=gen).to(dtype)
    four = torch.randn(4, 16, 8, 8, generator=gen).to(dtype)
    w, b = torch.randn(16, 16, 3, 3, generator=gen).to(dtype), torch.randn(16).to(dtype)
    grouped = torch.randn(16, 4, 3, 3, generator=gen).to(dtype)
    conv = ([1, 1], [1, 1], [1, 1], False, [0, 0])
    cases = {
        "conv by channels": (aten.convolution.default, (x, w, b, *conv, 1)),
        "grouped conv by groups": (aten.convolution.default, (x, grouped, b, *conv, 4)),
        "conv by images": (aten.convolution.default, (four, w, None, *conv, 1)),
        "upsample x1.5": (aten.upsample_nearest2d.vec, (x, None, [1.5, 1.5])),
        "exact upsample to 45": (aten._upsample_nearest_exact2d.vec, (four, [45, 45], None)),
        "upsample x2": (aten.upsample_nearest2d.default, (x, [64, 64], 2.0, 2.0)),
        "group norm by groups": (aten.native_group_norm.default, (x, b, b, 1, 16, 1024, 4, 1e-6)),
        "group norm eps 0": (aten.native_group_norm.default, (four, b, b, 4, 16, 64, 4, 0.0)),
    }
    for name, (func, args) in cases.items():
        whole = func(*args)
        plan = catch.RULES[func.overloadpacket](func, args, {})
        assert plan is not None and plan.finest >= 2, name
        for level in range(1, plan.finest + 1):
            got = plan(catch.Scope(name, plan.finest, level))
            for a, b in zip(catch.tensors(got), catch.tensors(whole), strict=True):
                error = (a.float() - b.float()).abs().max().item()
                print(f"{dtype} {name} level {level}: max error {error:.3g} against whole")
                if name.startswith(("conv", "grouped")):  # a CPU kernel blocks fewer channels
                    torch.testing.assert_close(a, b, rtol=2**-16, atol=2**-14)  # otherwise
                else:
                    assert torch.equal(a, b), (name, level)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ffn_tokens_and_attention_pieces_against_the_whole(dtype: Any) -> None:
    """FFN token pieces and attention head groups and query rows at levels 1-3 against the call
    whole on real CPU kernels, each one's measured max error printed. A GEMM of fewer rows may
    sum in another order on a CPU, so pieces are equal up to rounding (the output bar)."""
    gen = torch.Generator().manual_seed(10)
    ffn = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64)).to(dtype)
    x = torch.randn(2, 48, 64, generator=gen).to(dtype)
    q, k, v = (torch.randn(1, 64, 8, 16, generator=gen).to(dtype) for _ in range(3))
    tolerance = 2**-6 if dtype == torch.bfloat16 else 2**-18
    with torch.no_grad():
        whole_ffn, whole_attention = ffn(x), scores(q, k, v)
        for level in (1, 2, 3):
            heads = cuts.pieces(8, level)  # 2^level pieces of whole K/V heads
            pieces = {
                "ffn tokens": (cuts._tokens(ffn, x, 96, level), whole_ffn),
                "attention heads": (
                    cuts._attend(lambda *t: scores(*t[:3]), q, k, v, None, 1, heads, 64),
                    whole_attention,
                ),
                "attention rows": (
                    cuts._attend(lambda *t: scores(*t[:3]), q, k, v, None, 1, 8, 64 >> level),
                    whole_attention,
                ),
            }
            for name, (got, want) in pieces.items():
                assert isinstance(got, torch.Tensor)
                error = (got.float() - want.float()).abs().max().item()
                print(f"{dtype} {name} level {level}: max error {error:.3g} against whole")
                torch.testing.assert_close(got, want, rtol=tolerance, atol=tolerance)


class Tiles(nn.Module):  # type: ignore[misc]
    """A decoder whose forward splits itself by batch (`author.split`)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4, 8, 1), nn.SiLU(), nn.Conv2d(8, 3, 1))

    def forward(self, x: Any) -> Any:
        return split(self.net, x, Cut(0))


def tiles() -> Tiles:
    torch.manual_seed(4)
    tiles = Tiles()
    tiles.to(BF16).eval()
    return tiles


def test_a_package_split_runs_in_the_windows_rust_cuts(tmp_path: Path) -> None:
    plain = tiles()
    mem = memory()
    model = placed(mem, checkpoint(tmp_path, "tiles", plain), "tiles", tiles)
    x = torch.randn(8, 4, 16, 16, generator=torch.Generator().manual_seed(5)).to(BF16)
    card = Card(limit=24 << 10)  # a [8, 8, 16, 16] activation is 32 KiB
    with torch.no_grad():
        want = plain(x)
        with card, mem.request("split") as r, execution.serving(r):
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


class Accumulates(nn.Module):  # type: ignore[misc]
    """Adds its input into a tensor that existed before it ran (a KV cache's shape)."""

    def __init__(self, held: Any) -> None:
        super().__init__()
        self.held = held

    def forward(self, x: Any) -> Any:
        self.held.add_(x)
        return self.held * 2


class Fresh(nn.Module):  # type: ignore[misc]
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
    root.seen = []
    fresh = Fresh()

    def forward(x: Any) -> Any:
        root.seen.append(1)
        return Fresh.forward(fresh, x)

    fresh.forward = forward  # type: ignore[method-assign]
    block = wrapped(fresh, root, ("seen",))
    x = torch.ones(16)
    torch.manual_seed(7)
    reference = Fresh()(x)
    mem = memory()
    with torch.no_grad(), mem.request("bare") as r, execution.serving(r):
        fresh(x)
        bare = [block.bare]  # a block writing only what it made runs bare
        torch.manual_seed(7)
        with FailsAfter(torch.ops.aten.sin.default) as mode:  # out of memory at its last op
            got = fresh(x)
        bare.append(block.bare)  # it ran short: under the layer from now on
    assert mode.attempts == 2 and torch.equal(got, reference)
    assert bare == [True, False] and root.seen == [1, 1], (bare, root.seen)


# ----------------------------------------------------------------------------- collectives


@pytest.mark.parametrize("short", [0, 1])
def test_an_oom_on_one_rank_becomes_the_same_cut_on_every_rank(tmp_path: Path, short: int) -> None:
    """Two gloo ranks: one runs out allocating its exchange's buffers in the first pass. It
    recovers locally mid-pass (both cross every head at once), and from the next pass both
    agree its finer level and cross the heads in halves. Measured, the collective block runs
    bare; out of memory there both fail typed with no replay, and the next pass at those shapes
    runs under the layer. Every pass that completes is degree 1's result. The ranks' rows cross
    once, in the first pass: the later ones have its shapes."""
    argv = [sys.executable, str(AGREEMENT), str(tmp_path), "--degree", "2", "--short", str(short)]
    subprocess.run(argv, env=ENV, check=True)
    rows = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert [row["refused"] for row in rows] == [1 + int(rank == short) for rank in range(2)]
    for row in rows:
        passes = row["passes"]
        assert all(p["byte_equal"] for n, p in enumerate(passes) if n != 3), row
        assert passes[3]["failed"] == "device_out_of_memory", row
        heads = [p["exchanged_heads"] for n, p in enumerate(passes) if n != 3]
        assert heads == [[8], [4, 4], [4, 4], [4, 4]], row
        assert [p["layered"] for p in passes] == [True, False, False, False, True], row
        assert [p["gathered"] for p in passes] == [1, 0, 0, 0, 0], row


# ------------------------------------------------------------ in place, park, package code


class UpBlock(nn.Module):  # type: ignore[misc]
    """A VAE up-block's shape: norm, SiLU, conv; the SiLU's input is dead after it."""

    def __init__(self) -> None:
        super().__init__()
        self.norm, self.conv = nn.GroupNorm(4, 16), nn.Conv2d(16, 16, 3, padding=1)

    def forward(self, x: Any) -> Any:
        return self.conv(torch.nn.functional.silu(self.norm(x)))


class Decoder(nn.Module):  # type: ignore[misc]
    _no_split_modules = ("UpBlock",)

    def __init__(self) -> None:
        super().__init__()
        self.up = UpBlock()

    def forward(self, x: Any) -> Any:
        return self.up(torch.nn.functional.interpolate(x, scale_factor=2.0))


def decoder() -> Decoder:
    torch.manual_seed(12)
    model = Decoder()
    model.to(BF16).eval()
    return model


def test_a_silu_whose_input_is_dead_runs_in_place_else_needs_the_park(tmp_path: Path) -> None:
    """At 40 KiB the up-block's [16, 32, 32] SiLU beside its input does not fit. Cold, nothing
    has shown its input dead: Rust's answers reach Park, and a CPU has nothing to park, so the
    request is exhausted. Once a run has shown the input dead, the SiLU runs in place, the norm
    and conv in bands, and the result is the whole one."""
    plain = decoder()
    ck = checkpoint(tmp_path, "decoder", plain)
    x = torch.randn(1, 16, 16, 16, generator=torch.Generator().manual_seed(13)).to(BF16)
    with torch.no_grad():
        want = plain(x)
        cold = memory()
        model = placed(cold, ck, "decoder", decoder)
        with (
            pytest.raises(Exception) as raised,
            Card(limit=40 << 10),
            cold.request("cold") as r,
            execution.serving(r),
        ):
            model(x)
        assert fault(raised.value) is Fault.EXHAUSTED and r.stats().parks >= 1
        warm = memory()
        model = placed(warm, ck, "decoder", decoder)
        with warm.request("warm") as r, execution.serving(r):
            assert torch.equal(model(x), want)
        card = Card(limit=40 << 10)
        with card, warm.request("short") as r, execution.serving(r):
            got = model(x)
    assert card.refused and r.stats().parks == 0
    error = (got.float() - want.float()).abs().max().item()
    print(f"up-block, SiLU in place, norm and conv in bands: max error {error:.3g} against whole")
    torch.testing.assert_close(got, want, rtol=2**-7, atol=2**-7)


class Short(TorchDispatchMode):  # type: ignore[misc]
    """The next `target` op runs out of memory once while `armed`, as an allocation would."""

    def __init__(self, target: Any) -> None:
        super().__init__()
        self.target, self.armed, self.attempts = target, False, 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        if self.armed and func is self.target:
            self.attempts += 1
            if self.attempts == 1:
                raise torch.OutOfMemoryError("out of memory. Tried to allocate 24 KiB")
        return func(*args, **(kwargs or {}))


def test_a_package_s_code_between_passes_is_caught(tmp_path: Path) -> None:
    """An op outside every block (a package's code between passes) runs under the request's
    layer: out of memory it asks Rust like any op and runs again."""
    plain = transformer()
    mem = memory()
    model = placed(mem, checkpoint(tmp_path, "dit", plain), "dit", transformer)
    x, short = tokens(3), Short(torch.ops.aten.mul.Tensor)
    with torch.no_grad():
        want = plain(x) * 2
        with short, mem.request("between") as r, execution.serving(r):
            h = model(x)
            short.armed = True
            got = h * 2
    assert short.attempts == 2 and torch.equal(got, want)
